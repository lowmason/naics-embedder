'''
The text stage's data (Req 10; spec 4.3): two streams over every code and every task query.

One epoch reads every code once as an anchor and every task query once. ``StepDataset`` permutes
both streams per epoch from (seed, epoch) and cuts the two orders into the same S even chunks, so
step s reads code chunk s and query chunk s. Nothing is pre-drawn, and no step carries a candidate
pool: every candidate a term scores comes from the model's code cache.

``NAICSDataModule`` builds the token cache, the task queries and the step dataset from the
validated bundle, and holds every code's token rows in codebook order for that cache. Its one
loader hands each step through whole (P12), and it has no validation loader: validation is the
outcome monitor (spec 4.4). ``TrainDatasetEpochCallback`` sets the epoch at each epoch start
(P27). ``stack_text_inputs`` batches the token rows of every encode.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import logging
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pyl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from naics_embedder.supervision.artifacts import (
    ValidatedSupervisionBundle,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.code_targets import CodeTargets
from naics_embedder.supervision.queries import TaskQuery, build_task_queries
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model.dataloader.tokenization_cache import (
    load_verified_tokenization_cache,
    tokenization_cache,
)
from naics_embedder.text_model.fields import CHANNELS, QUERY, tokenize_field
from naics_embedder.utils.config import TokenizationConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Token rows into batches
# -------------------------------------------------------------------------------------------------

def stack_text_inputs(
    embeddings: Sequence[Mapping[str, Mapping[str, Any]]],
    fields: Sequence[str] = CHANNELS,
) -> Dict[str, Dict[str, torch.Tensor]]:
    '''
    Stack token rows into one batch: per field, ``input_ids``, ``attention_mask`` and a boolean
    ``present`` of shape (B,).

    Every code batch is built here (the training steps, the export and the HGCN feeder), and a query
    batch too, under the field ``query``. The encoder reads presence from ``present``, never from
    the attention mask.

    Raises:
        ValueError: If a row's field has no ``present`` flag.
    '''

    for embedding in embeddings:
        for field in fields:
            if 'present' not in embedding[field]:
                raise ValueError(
                    f'a {field!r} token row has no present flag; rebuild the tokenization cache'
                )
    return {
        field: {
            'input_ids': torch.stack([embedding[field]['input_ids'] for embedding in embeddings]),
            'attention_mask': torch.stack(
                [embedding[field]['attention_mask'] for embedding in embeddings]
            ),
            'present': torch.tensor(
                [bool(embedding[field]['present']) for embedding in embeddings],
                dtype=torch.bool,
            ),
        }
        for field in fields
    }

# -------------------------------------------------------------------------------------------------
# Two-stream epochs (Req 10; spec 4.3)
#
# One epoch reads every code once as an anchor and every task query once. Each stream is permuted
# per epoch from (seed, epoch), and both orders are cut into the same S even chunks: step s reads
# code chunk s and query chunk s. Nothing is pre-drawn, and no step carries a candidate pool.
# -------------------------------------------------------------------------------------------------

# The two streams an epoch permutes, each under its own generator seed
CODE_STREAM = 'codes'
QUERY_STREAM = 'queries'
STREAMS = (CODE_STREAM, QUERY_STREAM)

# A permutation's generator seed: 63 bits of a sha256, so a positive int64
_GENERATOR_SEED_MASK = (1 << 63) - 1

def steps_per_epoch(n_queries: int, queries_per_step: int) -> int:
    '''
    S, the steps of one epoch: ⌈n_queries / queries_per_step⌉ (spec 4.3).

    Args:
        n_queries: The task queries one epoch reads.
        queries_per_step: The most queries one step reads.

    Returns:
        The number of steps. Each reads one chunk of the queries and one of the codes.

    Raises:
        ValueError: If ``queries_per_step`` is below 1 or ``n_queries`` is negative.
    '''

    if queries_per_step < 1:
        raise ValueError(f'queries_per_step must be at least 1, not {queries_per_step}')
    if n_queries < 0:
        raise ValueError(f'the number of task queries cannot be negative, not {n_queries}')
    return (n_queries + queries_per_step - 1) // queries_per_step

def epoch_permutation(seed: int, epoch: int, n: int, stream: str) -> torch.Tensor:
    '''
    One stream's order in one epoch: a permutation of ``range(n)`` that depends only on the
    arguments.

    The CPU generator that draws it is seeded by the first eight bytes of the sha256 of
    ``f'{seed}:{epoch}:{stream}'``, kept to 63 bits. So the order is the same in every process,
    on every platform and after any history, and no global random state is read or moved.

    Args:
        seed: The run's seed.
        epoch: The epoch, from 0.
        n: The stream's length.
        stream: ``'codes'`` or ``'queries'``.

    Returns:
        The permutation, a CPU int64 tensor of shape (n,).

    Raises:
        ValueError: If the stream is neither, or the epoch or ``n`` is negative.
    '''

    if stream not in STREAMS:
        raise ValueError(f'unknown stream {stream!r}; the streams are {list(STREAMS)}')
    seed, epoch, n = operator.index(seed), operator.index(epoch), operator.index(n)
    if epoch < 0:
        raise ValueError(f'epoch must be at least 0, not {epoch}')
    if n < 0:
        raise ValueError(f'a stream cannot have a negative length, not {n}')
    key = f'{seed}:{epoch}:{stream}'.encode('utf-8')
    generator_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') & _GENERATOR_SEED_MASK
    generator = torch.Generator(device='cpu').manual_seed(generator_seed)
    return torch.randperm(n, generator=generator)

def even_chunks(order: torch.Tensor, steps: int) -> List[torch.Tensor]:
    '''
    Cut an order into exactly ``steps`` contiguous chunks whose sizes differ by at most one.

    ``torch.tensor_split`` makes the first ``len(order) % steps`` chunks one longer than the rest.
    ``torch.chunk`` is never used: it cuts chunks of ⌈len / steps⌉, so it can return fewer than
    ``steps`` (2,125 codes over 87 steps give 85), which would leave steps with no chunk.

    Args:
        order: A one-dimensional order, such as an epoch's permutation.
        steps: The number of chunks.

    Returns:
        The chunks, in order: views of ``order`` that concatenate to it.

    Raises:
        ValueError: If ``steps`` is below 1 or ``order`` is not one-dimensional.
    '''

    if steps < 1:
        raise ValueError(f'an epoch needs at least one step, not {steps}')
    if order.dim() != 1:
        raise ValueError(f'an order is one-dimensional, not of shape {tuple(order.shape)}')
    chunks = list(torch.tensor_split(order, steps))
    assert len(chunks) == steps, f'tensor_split cut {len(chunks)} chunks for {steps} steps'
    return chunks

# eq=False: a generated __eq__ would compare the token rows' tensors, which have no single truth
# value, and its __hash__ would hash the rows' dicts
@dataclass(frozen=True, eq=False)
class TokenizedQueries:
    '''
    The task queries, each tokenized once as a ``query:`` text, with its level and code ids.

    Built by ``tokenize_task_queries``. Every field is in query order.

    Attributes:
        texts: Each query's text.
        tokens: Each query's ``query`` token row, as ``tokenize_field`` returns it.
        levels: Each query's level.
        target_ids: The code ids of each query's targets, T.
        negative_ids: The code ids of each query's forced negatives, N.
    '''

    texts: Tuple[str, ...]
    tokens: Tuple[Dict[str, Any], ...]
    levels: Tuple[int, ...]
    target_ids: Tuple[Tuple[int, ...], ...]
    negative_ids: Tuple[Tuple[int, ...], ...]

    def __len__(self) -> int:
        '''The number of queries.'''

        return len(self.texts)

def tokenize_task_queries(
    queries: Sequence[TaskQuery],
    tokenizer: Any,
    max_length: int,
    code_ids: Mapping[str, int],
) -> TokenizedQueries:
    '''
    Tokenize every task query once, as a ``query:`` text, and name its codes by their ids.

    A query is tokenized as ``ArmEncoder.encode_queries`` tokenizes a read's query texts: by
    ``tokenize_field`` under the field ``query``, padded and truncated to ``max_length``.

    Args:
        queries: The task queries, as ``build_task_queries`` returns them.
        tokenizer: The backbone's tokenizer.
        max_length: Tokens kept: the window the code channels are tokenized at.
        code_ids: Each code's id, its row in the codebook.

    Returns:
        The tokenized queries, in the order given.

    Raises:
        ValueError: If a query names a code that has no code id.
    '''

    for query in queries:
        unknown = sorted({code for code in query.targets + query.negatives if code not in code_ids})
        if unknown:
            raise ValueError(
                f'task query {query.text!r} at level {query.level} names codes with no code '
                f'id: {unknown}'
            )
    tokenized = TokenizedQueries(
        texts=tuple(query.text for query in queries),
        tokens=tuple(tokenize_field(tokenizer, QUERY, query.text, max_length) for query in queries),
        levels=tuple(query.level for query in queries),
        target_ids=tuple(tuple(int(code_ids[code]) for code in query.targets) for query in queries),
        negative_ids=tuple(
            tuple(int(code_ids[code]) for code in query.negatives) for query in queries
        ),
    )
    logger.info(
        f'Tokenized {len(tokenized):,} task queries as query: texts at a {max_length}-token window'
    )
    return tokenized

def _refuse_unscorable_queries(queries: TokenizedQueries, code_levels: Sequence[int]) -> None:
    '''
    Refuse a query no step could score: one that names a code id outside the codes, has no
    target, or has a target at a level other than its own, so outside its candidates (spec 4.1).
    '''

    n_codes = len(code_levels)
    for text, level, targets, negatives in zip(
        queries.texts, queries.levels, queries.target_ids, queries.negative_ids, strict=True
    ):
        name = f'task query {text!r} at level {level}'
        outside = sorted({code_id for code_id in targets + negatives if not 0 <= code_id < n_codes})
        if outside:
            raise ValueError(f'{name} names code ids {outside}, outside the {n_codes} codes')
        if not targets:
            raise ValueError(f'{name} has no target')
        elsewhere = sorted({code_id for code_id in targets if code_levels[code_id] != level})
        if elsewhere:
            raise ValueError(f'{name} has targets at another level: code ids {elsewhere}')

class StepDataset(Dataset):
    '''
    The steps of one epoch over two streams: every code once as an anchor, and every task query
    once (Req 10; spec 4.3).

    Each epoch permutes the codes and the queries (``epoch_permutation``, under the run's seed and
    the epoch) and cuts both orders into the same S even chunks (``even_chunks``), with S from
    ``steps_per_epoch``: step s reads code chunk s and query chunk s. A step carries only its
    anchors' token rows and its queries'. Every candidate a term scores comes from the model's
    code cache, so no step carries a candidate pool.

    The epoch comes only from ``set_epoch``, which ``TrainDatasetEpochCallback`` calls at each
    epoch start, and its two permutations are drawn at its first step and kept for the others.
    ``set_epoch`` reaches only the main process's copy of the dataset, so the steps are read
    there, as ``DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)`` reads them.

    A step is a dict of two dicts:

    - ``codes``: ``inputs``, the anchors' token rows through ``stack_text_inputs``; and ``ids``
      and ``levels``, int64 (A,).
    - ``queries``: ``inputs``, the queries' token rows under the field ``query``; ``levels``,
      int64 (Q,); and ``targets`` and ``negatives``, bool (Q, N) masks over the codes of each
      query's T and N.

    Args:
        code_rows: Every code's token row, in codebook order, as the tokenization cache holds it.
        code_levels: Every code's level, in codebook order.
        queries: The tokenized task queries.
        n_codes: N, the number of codes.
        seed: The run's seed.
        queries_per_step: The most queries one step reads.

    Raises:
        ValueError: If the code rows or levels are not one per code; if there is no task query,
            or there are more steps than codes, since every step needs at least one anchor; or if
            a query names a code id outside the codes, has no target, or has a target at another
            level.
    '''

    def __init__(
        self,
        code_rows: Sequence[Mapping[str, Any]],
        code_levels: Sequence[int],
        queries: TokenizedQueries,
        n_codes: int,
        seed: int,
        queries_per_step: int,
    ):
        n_codes = operator.index(n_codes)
        if len(code_rows) != n_codes:
            raise ValueError(f'{len(code_rows)} code token rows for {n_codes} codes')
        levels = torch.tensor(np.asarray(code_levels, dtype=np.int64))
        if levels.dim() != 1:
            raise ValueError(f'code levels are one-dimensional, not of shape {tuple(levels.shape)}')
        if len(levels) != n_codes:
            raise ValueError(f'{len(levels)} code levels for {n_codes} codes')
        steps = steps_per_epoch(len(queries), queries_per_step)
        if steps < 1:
            raise ValueError(
                'there are no task queries, so an epoch would have no step and no code would be '
                'an anchor'
            )
        if steps > n_codes:
            raise ValueError(
                f'{steps} steps for {n_codes} codes: every step needs at least one anchor, so '
                'raise queries_per_step'
            )
        _refuse_unscorable_queries(queries, levels.tolist())

        self.code_rows = tuple(code_rows)
        self.code_levels = levels
        self.queries = queries
        self.query_levels = torch.tensor(queries.levels, dtype=torch.long)
        self.n_codes = n_codes
        self.seed = operator.index(seed)
        self.queries_per_step = queries_per_step
        self.steps = steps
        # Set only by set_epoch, and the chunks of the epoch drawn last
        self.epoch: Optional[int] = None
        self._drawn: Optional[Tuple[int, List[torch.Tensor], List[torch.Tensor]]] = None
        logger.info(
            f'Two-stream epochs: {steps:,} steps over {n_codes:,} codes and {len(queries):,} task '
            f'queries, at most {queries_per_step:,} queries a step'
        )

    def set_epoch(self, epoch: int) -> None:
        '''
        Make ``epoch`` the epoch the steps read; its permutations are drawn at its first step.

        Raises:
            ValueError: If the epoch is negative.
        '''

        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError(f'epoch must be at least 0, not {epoch}')
        self.epoch = epoch

    def __len__(self) -> int:
        '''S, the steps of an epoch.'''

        return self.steps

    def _epoch_chunks(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''The current epoch's code and query chunks, drawn at its first step and then kept.'''

        if self.epoch is None:
            raise RuntimeError(
                'StepDataset has no epoch: call set_epoch first (TrainDatasetEpochCallback '
                'calls it at each epoch start)'
            )
        if self._drawn is None or self._drawn[0] != self.epoch:
            codes = epoch_permutation(self.seed, self.epoch, self.n_codes, CODE_STREAM)
            queries = epoch_permutation(self.seed, self.epoch, len(self.queries), QUERY_STREAM)
            self._drawn = (
                self.epoch,
                even_chunks(codes, self.steps),
                even_chunks(queries, self.steps),
            )
        return self._drawn[1], self._drawn[2]

    def _mask(self, code_ids: Sequence[Tuple[int, ...]], rows: torch.Tensor) -> torch.Tensor:
        '''A bool (Q, N) mask whose row q marks the code ids of query ``rows[q]``.'''

        mask = torch.zeros((len(rows), self.n_codes), dtype=torch.bool)
        for row, query in enumerate(rows.tolist()):
            mask[row, list(code_ids[query])] = True
        return mask

    def __getitem__(self, step: int) -> Dict[str, Dict[str, Any]]:
        '''
        Step ``step`` of the current epoch: its anchors and its queries.

        Raises:
            IndexError: If the step is outside the epoch.
            RuntimeError: If no epoch has been set.
        '''

        step = operator.index(step)
        if not 0 <= step < self.steps:
            raise IndexError(f'step {step} is outside the {self.steps} steps of an epoch')
        code_chunks, query_chunks = self._epoch_chunks()
        ids = code_chunks[step].clone()
        rows = query_chunks[step]
        query_rows = [{QUERY: self.queries.tokens[row]} for row in rows.tolist()]
        return {
            'codes': {
                'inputs': stack_text_inputs([self.code_rows[code_id] for code_id in ids.tolist()]),
                'ids': ids,
                'levels': self.code_levels[ids],
            },
            'queries': {
                'inputs': stack_text_inputs(query_rows, fields=(QUERY, )),
                'levels': self.query_levels[rows],
                'targets': self._mask(self.queries.target_ids, rows),
                'negatives': self._mask(self.queries.negative_ids, rows),
            },
        }

# -------------------------------------------------------------------------------------------------
# The DataModule (spec 4.3)
# -------------------------------------------------------------------------------------------------

# The most task queries one step reads when the run sets no other: spec 4.3's 128 (R7)
DEFAULT_QUERIES_PER_STEP = 128

def _cache_fingerprints(bundle: ValidatedSupervisionBundle) -> Dict[str, str]:
    '''What a tokenization cache of the bundle records: its descriptions' and codebook's hashes.'''

    manifest = bundle.manifest
    return {
        'description_fingerprint': manifest.description_fingerprint,
        'codebook_fingerprint': manifest.codebook_fingerprint,
    }

def _codebook_rows(
    cache: Mapping[int, Mapping[str, Any]],
    codes: Sequence[str],
) -> Tuple[Mapping[str, Any], ...]:
    '''
    Every code's token rows in codebook order: row i is code id i's.

    The cache is keyed by the descriptions' ``index``, which the bundle's codebook takes as its
    code ids, and each row names its code. A row that names another code is refused, since the
    code cache would encode one code's texts under another's id.

    Raises:
        ValueError: If the cache does not hold one row per code, or a code id's row names
            another code.
    '''

    if len(cache) != len(codes):
        raise ValueError(
            f"the token cache holds {len(cache)} rows for the codebook's {len(codes)} codes"
        )
    rows = []
    for code_id, code in enumerate(codes):
        row = cache.get(code_id)
        named = None if row is None else row.get('code')
        if named != code:
            raise ValueError(
                f'the token cache row for code id {code_id} names {named!r}, not the '
                f"codebook's {code!r}"
            )
        rows.append(row)
    return tuple(rows)

class NAICSDataModule(LightningDataModule):
    '''
    The text stage's data: two streams over every code and every task query (Req 10; spec 4.3).

    ``prepare_data`` builds the tokenization cache, or keeps one built from the same inputs.
    ``setup`` builds the ``StepDataset`` once: every code's cached token rows in codebook order,
    and the bundle's task queries, each tokenized once as a ``query:`` text with the cache's
    tokenizer at its window. ``code_rows`` holds those rows for the model's code cache, which
    encodes them at fit start and at each epoch's end.

    The one loader is ``train_dataloader`` (P12). There is no validation loader, so Lightning
    runs no validation loop: validation is the outcome monitor (spec 4.4). Construction reads
    nothing; the bundle is loaded, and refused, in ``prepare_data`` and ``setup``.

    Args:
        token_config: The tokenization cache to build and read: the descriptions, the tokenizer,
            the window and the file. ``train`` passes ``code_token_config``'s, the cache the
            export and the reads load.
        seed: The run's seed, from which each epoch's two permutations are drawn.
        queries_per_step: The most task queries one step reads.
        supervision_manifest_path: The bundle's manifest, read when no bundle is given.
        supervision_contract_version: The contract the manifest must carry.
        supervision_bundle: The validated bundle, when the caller has loaded it.

    Raises:
        ValueError: If ``queries_per_step`` is below 1.
    '''

    def __init__(
        self,
        token_config: TokenizationConfig,
        *,
        seed: int,
        queries_per_step: int = DEFAULT_QUERIES_PER_STEP,
        supervision_manifest_path: Optional[str] = None,
        supervision_contract_version: str = CONTRACT_VERSION,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
    ):
        super().__init__()
        queries_per_step = operator.index(queries_per_step)
        if queries_per_step < 1:
            raise ValueError(f'queries_per_step must be at least 1, not {queries_per_step}')

        self.token_config = token_config
        self.seed = operator.index(seed)
        self.queries_per_step = queries_per_step
        self.supervision_manifest_path = supervision_manifest_path
        self.supervision_contract_version = supervision_contract_version
        self._bundle: Optional[ValidatedSupervisionBundle] = supervision_bundle
        # Built by setup
        self.train_dataset: Optional[StepDataset] = None

    # ---------------------------------------------------------------------------------------------
    # The bundle
    # ---------------------------------------------------------------------------------------------

    def _validated_bundle(self) -> ValidatedSupervisionBundle:
        '''
        The validated bundle, loaded from the manifest when none was given; refused unless the
        descriptions are the bundle's own.
        '''

        if self._bundle is None:
            if not self.supervision_manifest_path:
                raise ValueError(
                    'Repaired Stage-3 data loading requires a supervision manifest: run '
                    '`naics-embedder data supervision` and set supervision.manifest_path'
                )
            self._bundle = load_validated_bundle(
                self.supervision_manifest_path,
                expected_contract=self.supervision_contract_version,
            )
        descriptions = Path(self.token_config.descriptions_parquet)
        found = sha256_file(descriptions)
        expected = self._bundle.manifest.description_fingerprint
        if found != expected:
            raise ValueError(
                f'descriptions input {descriptions} does not match supervision bundle '
                f'{self._bundle.manifest.bundle_id}: expected {expected}, found {found}'
            )
        return self._bundle

    # ---------------------------------------------------------------------------------------------
    # The two streams
    # ---------------------------------------------------------------------------------------------

    def prepare_data(self) -> None:
        '''
        Build the tokenization cache of the bundle's descriptions, or keep one built from the same
        inputs.

        Raises:
            ValueError: If the bundle is refused.
        '''

        bundle = self._validated_bundle()
        logger.info('Preparing the tokenization cache...')
        tokenization_cache(self.token_config, **_cache_fingerprints(bundle))

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        Build the step dataset from the bundle, once (spec 4.3); a later call keeps it.

        Args:
            stage: Lightning's stage. Every stage reads the same steps.

        Raises:
            RuntimeError: If the tokenization cache is missing or was built from other inputs.
            ValueError: If the bundle is refused, the cache does not hold the codebook's codes
                in order, or ``StepDataset`` refuses the two streams.
        '''

        if self.train_dataset is not None:
            return
        bundle = self._validated_bundle()
        targets = CodeTargets.from_bundle(bundle)
        cache = load_verified_tokenization_cache(self.token_config, **_cache_fingerprints(bundle))
        code_rows = _codebook_rows(cache, targets.codes)
        tokenizer = AutoTokenizer.from_pretrained(self.token_config.tokenizer_name)
        code_ids = {code: code_id for code_id, code in enumerate(targets.codes)}
        queries = tokenize_task_queries(
            build_task_queries(bundle), tokenizer, self.token_config.max_length, code_ids
        )
        self.train_dataset = StepDataset(
            code_rows=code_rows,
            code_levels=targets.levels,
            queries=queries,
            n_codes=len(targets.codes),
            seed=self.seed,
            queries_per_step=self.queries_per_step,
        )

    def _step_dataset(self) -> StepDataset:
        '''The step dataset ``setup`` built.'''

        if self.train_dataset is None:
            raise RuntimeError('NAICSDataModule has no step dataset before setup; call setup first')
        return self.train_dataset

    @property
    def code_rows(self) -> Tuple[Mapping[str, Any], ...]:
        '''
        Every code's token rows in codebook order, which the model's code cache encodes at fit
        start and at each epoch's end (spec 4.3): the rows the steps read their anchors from.

        Raises:
            RuntimeError: Before ``setup``.
        '''

        return self._step_dataset().code_rows

    def train_dataloader(self) -> DataLoader:
        '''
        The epoch's steps, one batch each, in step order (P12).

        ``batch_size=None`` hands each step through whole, and ``shuffle=False`` keeps step order,
        since the epoch's permutations are the order. With ``num_workers=0`` every step is read
        in the main process, whose dataset ``set_train_epoch`` reaches. The loader never reads
        the trainer's epoch, which can be stale on a resume; ``TrainDatasetEpochCallback`` sets
        it at each epoch start (P27).

        Raises:
            RuntimeError: Before ``setup``.
        '''

        return DataLoader(self._step_dataset(), batch_size=None, shuffle=False, num_workers=0)

    def set_train_epoch(self, epoch: int) -> None:
        '''
        Make ``epoch`` the epoch the steps read: ``TrainDatasetEpochCallback`` calls it at each
        epoch start.

        Raises:
            RuntimeError: Before ``setup``.
            ValueError: If the epoch is negative.
        '''

        self._step_dataset().set_epoch(epoch)
        logger.debug(f'The steps read epoch {epoch}')

# -------------------------------------------------------------------------------------------------
# Callback setting the training epoch on the datamodule
# -------------------------------------------------------------------------------------------------

class TrainDatasetEpochCallback(pyl.Callback):
    '''
    Set the step dataset's epoch to the trainer's at each epoch start (P27).

    Register it on every Trainer that fits a NAICSDataModule: Lightning dispatches
    on_train_epoch_start to callbacks and the LightningModule, never to a LightningDataModule,
    and the step dataset refuses a step read before any epoch is set. Lightning never prefetches
    from a loader whose length it knows, so the epoch is set before the epoch's first step is read.
    '''

    def on_train_epoch_start(self, trainer: pyl.Trainer, pl_module: pyl.LightningModule) -> None:
        datamodule = trainer.datamodule
        if isinstance(datamodule, NAICSDataModule):
            datamodule.set_train_epoch(trainer.current_epoch)
