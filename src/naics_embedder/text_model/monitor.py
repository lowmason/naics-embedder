'''
D6's monitor (spec 4.4): the outcome panel's validation split, read on the live model each epoch.

Training holds a cache of every code's point, refreshed at fit start and after each epoch's last
step (spec 4.3). After the end-of-epoch refresh the monitor scores the validation split through
``OutcomePanel.score_logged``, the path every read takes, so the selection log records the read.
Its ``LiveEncoder`` encodes queries through the live model and decodes codes from the cache, and
both go through the head's float64 read map, as ``ArmEncoder``'s do, under the head's distance
(Req 12). The cache is encoded as the export encodes the code table, in the same batches and
order, so on the CPU a read of the live model and a read of the table exported from its
checkpoint agree exactly.

Each read's record, as logged, goes with its MRR to ``monitor_reads.jsonl`` in the run's checkpoint
directory: the durable carrier that takes a run's validation reads into its decision records
(Req 4). An exact resume keeps the records through the restored checkpoint's epoch, so the file
holds each epoch of the surviving run exactly once.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import contextlib
import json
import logging
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.text_only import matrix_fingerprint
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.export import (
    ENCODE_BATCH_SIZE,
    encode_query_texts,
    encode_token_rows,
)
from naics_embedder.text_model.heads import head_of

logger = logging.getLogger(__name__)

# The file each training run keeps its monitor reads in, beside its checkpoints
MONITOR_RECORDS = 'monitor_reads.jsonl'

# -------------------------------------------------------------------------------------------------
# Training flags and the float32 encode
# -------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def preserve_training_flags(module: torch.nn.Module) -> Iterator[None]:
    '''
    Record every submodule's ``training`` flag, and put each back on exit, an error's included.

    Lightning never calls ``train()`` in its fit loop, so an encode in a training hook that leaves
    the model in eval mode, as ``encode_token_rows`` does, would turn dropout and gradient
    checkpointing off for the rest of the run, with no error. Each flag is put back as it was, so a
    submodule that was in eval mode stays there.

    Args:
        module: The module whose flags, and its submodules', are kept.
    '''

    flags = [(submodule, submodule.training) for submodule in module.modules()]
    try:
        yield
    finally:
        for submodule, training in flags:
            submodule.training = training

@contextlib.contextmanager
def _live_encode(model: torch.nn.Module) -> Iterator[None]:
    '''An encode of the live model: its flags kept, and autocast off, so it all runs in float32.'''

    device_type = next(model.parameters()).device.type
    with preserve_training_flags(model), torch.autocast(device_type=device_type, enabled=False):
        yield

# -------------------------------------------------------------------------------------------------
# The code cache
# -------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class CodeCache:
    '''
    Every code's point on the live model at its last refresh, in codebook order (spec 4.3).

    Attributes:
        codes: The codes, in codebook order.
        radius: Each code's radius r, (N,) float32 on the model's device.
        direction: Each code's direction û, (N, d) float32 on the model's device.
        tangent: Each code's coordinates in its arm's export form, the head's ``tangent``, (N, d)
            float64 on the CPU: r · û in the hyperbolic arm, v in the Euclidean and û in the
            spherical (Req 12). It is what the monitor decodes against and what the export writes.
    '''

    codes: Tuple[str, ...]
    radius: torch.Tensor
    direction: torch.Tensor
    tangent: torch.Tensor

    def with_live(
        self,
        ids: torch.Tensor,
        radius: torch.Tensor,
        direction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Every code's radius and direction, with the anchors' rows replaced by their live points.

        The rows are replaced out of place, so the cache is unchanged. Every other row is a
        constant of the last refresh, so gradient flows through the anchors' rows only, whether a
        term reads them as anchors or as candidates (spec 4.3).

        Args:
            ids: The anchors' code ids (A,), distinct: their rows in ``codes``.
            radius: The anchors' live radii (A,).
            direction: The anchors' live directions (A, d).

        Returns:
            The radii (N,) and the directions (N, d).

        Raises:
            ValueError: If the ids, radii and directions do not fit the cache.
        '''

        width = self.direction.shape[1]
        expected = [(len(ids), ), (len(ids), width)] if ids.dim() == 1 else None
        if expected is None or [tuple(radius.shape), tuple(direction.shape)] != expected:
            raise ValueError(
                f'with_live takes anchor ids (A,), radii (A,) and directions (A, {width}); got '
                f'{tuple(ids.shape)}, {tuple(radius.shape)} and {tuple(direction.shape)}'
            )
        rows = (ids.to(self.radius.device), )
        return self.radius.index_put(rows, radius), self.direction.index_put(rows, direction)

def refresh_code_cache(
    model: torch.nn.Module,
    code_rows: Sequence[Mapping[str, Mapping[str, Any]]],
    codes: Sequence[str],
    batch_size: int = ENCODE_BATCH_SIZE,
) -> CodeCache:
    '''
    Encode every code into a new cache: in eval mode, without gradient, in float32 with autocast
    off, in batches in codebook order (spec 4.3).

    The codes go through ``encode_token_rows`` as the export's do, in the same batches and order,
    so on the CPU the cache's tangents are the exported table's, bit for bit. The model's
    training flags are put back afterwards.

    Args:
        model: The live model: the Lightning module, or its shared encoder.
        code_rows: Every code's cached token rows, in codebook order.
        codes: The codes, in the same order.
        batch_size: Codes per forward pass.

    Returns:
        The cache.

    Raises:
        ValueError: If the rows are not one per code, or a code repeats.
    '''

    codes = tuple(str(code) for code in codes)
    if len(code_rows) != len(codes):
        raise ValueError(
            f'{len(code_rows)} token rows for {len(codes)} codes: a refresh encodes one row per '
            'code, in codebook order'
        )
    seen = set()
    for code in codes:
        if code in seen:
            raise ValueError(f'the code cache repeats the code {code!r}')
        seen.add(code)
    device = next(model.parameters()).device
    with _live_encode(model):
        encoded = encode_token_rows(model, code_rows, batch_size=batch_size)
    # float32 values survive the float64 round trip exactly. Cast on the CPU, then move: MPS has
    # no float64
    return CodeCache(
        codes=codes,
        radius=encoded['radius'].to(torch.float32).to(device),
        direction=encoded['direction'].to(torch.float32).to(device),
        tangent=encoded['tangent'],
    )

# -------------------------------------------------------------------------------------------------
# The live encoder
# -------------------------------------------------------------------------------------------------

class LiveEncoder:
    '''
    ``QueryCodeEncoder`` for the live model (spec 4.4): queries through the model, codes from the
    code cache, and both through the head's read map (``read_points``).

    A query is encoded as ``ArmEncoder`` encodes one (``encode_query_texts``), in eval mode and
    in float32 with autocast off, and the model's training flags are put back afterwards. A code
    is decoded from its cached tangent, as ``ArmEncoder`` decodes one from the exported table.

    Args:
        model: The live model.
        cache: The code cache of its last refresh.
        tokenizer: The token cache's tokenizer.
        max_length: The token cache's window.
        batch_size: Queries per forward pass.

    Attributes:
        head: The model's geometry head.
        distance: The head's decoding distance, as ``ArmEncoder``'s: ``'lorentz'`` in the
            hyperbolic arm, ``'euclidean'`` or ``'cosine'`` in the flat arms (Req 12).
    '''

    def __init__(
        self,
        model: torch.nn.Module,
        cache: CodeCache,
        tokenizer: Any,
        max_length: int,
        batch_size: int = ENCODE_BATCH_SIZE,
    ):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.head = head_of(model)
        self.distance = self.head.distance
        self._rows = {code: row for row, code in enumerate(cache.codes)}

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the live model, then the head's read map: float64.'''

        with _live_encode(self.model):
            tangent = encode_query_texts(
                self.model, self.tokenizer, texts, self.max_length, batch_size=self.batch_size
            )
        return self.head.read_points(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' cached coordinates through the head's read map: float64.

        Raises:
            ValueError: If a code has no row in the cache.
        '''

        unknown = sorted(set(codes) - set(self._rows))
        if unknown:
            raise ValueError(f'the code cache has no row for {unknown[:5]} ({len(unknown)} codes)')
        return self.head.read_points(self.cache.tangent[[self._rows[code] for code in codes]])

# -------------------------------------------------------------------------------------------------
# The monitor
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitorRead:
    '''
    One epoch's monitor read.

    Attributes:
        epoch: The epoch the read scored.
        mrr: Its validation MRR: the one value training logs as ``val/outcome_mrr``, steps the
            learning rate on and records.
        record: The read's selection-log record, as the log appended it.
    '''

    epoch: int
    mrr: float
    record: Dict[str, Any]

class OutcomeMonitor:
    '''
    D6's monitor: one logged read of the outcome panel's validation split per epoch, kept in the
    run's ``monitor_reads.jsonl`` (spec 4.4).

    Args:
        panel: The outcome panel of the run's bundle (``OutcomePanel.from_bundle``), whose log is
            ``conf/data/outcome_panel.yaml``'s ``selection_log``.
        tokenizer: The token cache's tokenizer.
        max_length: The token cache's window.
        records_path: The run's ``monitor_reads.jsonl``, in its checkpoint directory.
        purpose: Why the reads happen; the selection log records it with each read.

    Raises:
        ValueError: If ``purpose`` is blank, which the log would refuse only at the first read,
            an epoch into the run.
    '''

    def __init__(
        self,
        panel: OutcomePanel,
        tokenizer: Any,
        max_length: int,
        records_path: Union[str, Path],
        purpose: str,
    ):
        if not purpose.strip():
            raise ValueError(
                'a monitor needs a purpose: the selection log records one with every read'
            )
        self.panel = panel
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records_path = Path(records_path)
        self.purpose = purpose

    def start(self, *, resumed_epoch: Optional[int]) -> None:
        '''
        Ready the records file when a fit starts.

        A fresh run starts a new file, which its first read writes. An exact resume continues the
        file it resumed with: it keeps the records through the restored checkpoint's epoch and
        drops any later one, which an interrupted segment left (that read stays in the
        segment's selection log). The kept lines are rewritten as they were, and the file is
        replaced in one step.

        Args:
            resumed_epoch: The restored checkpoint's ``epoch``, the last epoch it completed; None
                for a fresh run.

        Raises:
            ValueError: If a fresh run finds a records file; or if a resume finds none, which
                would start a file without the run's earlier reads (spec 4.6), or its epoch is
                negative.
        '''

        path = self.records_path
        if resumed_epoch is None:
            if path.exists():
                raise ValueError(
                    f'{path} already holds monitor records: a fresh run writes a new file; resume '
                    'the run that wrote it, or train into another checkpoint directory'
                )
            return
        resumed_epoch = operator.index(resumed_epoch)
        if resumed_epoch < 0:
            raise ValueError(f'a resumed epoch is never negative, not {resumed_epoch}')
        if not path.exists():
            raise ValueError(
                f'an exact resume from epoch {resumed_epoch} continues its monitor records, but '
                f'{path} does not exist: bring it with last.ckpt (spec 4.6)'
            )
        lines = _record_lines(path)
        kept = [line for line, record in lines if _record_epoch(record) <= resumed_epoch]
        staged = path.with_name(f'{path.name}.tmp')
        staged.write_text(''.join(f'{line}\n' for line in kept), encoding='utf-8')
        os.replace(staged, path)
        logger.info(
            f'Monitor records: kept {len(kept)} through epoch {resumed_epoch} and dropped '
            f'{len(lines) - len(kept)} later ({path})'
        )

    def read(
        self,
        model: torch.nn.Module,
        cache: CodeCache,
        *,
        training_run: str,
        seed: int,
        epoch: int,
    ) -> MonitorRead:
        '''
        Score the validation split on the live model under ``'lorentz'``, logging the read.

        Every refusal comes before the read is logged. The read's detail names the training run,
        the seed, the epoch and the epoch's code table: the ``matrix_fingerprint`` of the cache's
        tangents, the name a read of the exported table logs it by.

        Args:
            model: The live model, just after its end-of-epoch refresh.
            cache: The code cache of that refresh.
            training_run: The training run's id, which every checkpoint saves.
            seed: The run's seed.
            epoch: The epoch the read scores.

        Returns:
            The read: its epoch, its MRR, and its record as logged.

        Raises:
            ValueError: If the training run is blank, the epoch is negative, or a candidate has
                no row in the cache.
            TypeError: If the seed or the epoch is not an integer.
        '''

        # JSON-native integers, whatever integer type the caller holds
        seed, epoch = operator.index(seed), operator.index(epoch)
        if not isinstance(training_run, str) or not training_run.strip():
            raise ValueError(f'a monitor read names its training run, not {training_run!r}')
        if epoch < 0:
            raise ValueError(f'an epoch is never negative, not {epoch}')
        missing = sorted(set(self.panel.candidates) - set(cache.codes))
        if missing:
            raise ValueError(
                f'the code cache has no row for the candidates {missing[:5]} ({len(missing)} codes)'
            )
        encoder = LiveEncoder(model, cache, self.tokenizer, self.max_length)
        detail = {
            'training_run': training_run,
            'seed': seed,
            'epoch': epoch,
            'table': matrix_fingerprint(cache.codes, cache.tangent.numpy()),
        }
        result, record = self.panel.score_logged(
            encoder, IndexRole.VALIDATION, self.purpose, distance=encoder.distance, detail=detail
        )
        return MonitorRead(epoch=epoch, mrr=float(result.summary['mrr']), record=record)

    def append(self, read: MonitorRead) -> None:
        '''
        Append one read to the records file, as ``{"mrr": …, "read": <its record>}``.

        Raises:
            ValueError: If the file already records the read's epoch or a later one: each epoch
                of the surviving run appears once, in order.
        '''

        path = self.records_path
        recorded: List[int] = []
        if path.exists():
            recorded = [_record_epoch(record) for _, record in _record_lines(path)]
        if recorded and max(recorded) >= read.epoch:
            raise ValueError(
                f'{path} already records epoch {max(recorded)}: a read of epoch {read.epoch} '
                "would repeat or reorder the run's epochs"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'mrr': read.mrr, 'read': read.record}, sort_keys=True) + '\n')

# -------------------------------------------------------------------------------------------------
# The records file
# -------------------------------------------------------------------------------------------------

def read_monitor_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    '''
    The records of a ``monitor_reads.jsonl``, oldest first, each ``{'mrr': …, 'read': …}``.

    Blank lines are skipped.

    Raises:
        ValueError: If a line is not a monitor record: a JSON object with exactly ``mrr``, a
            number, and ``read``, a record whose detail names a non-negative integer epoch.
        FileNotFoundError: If the file does not exist.
    '''

    return [record for _, record in _record_lines(Path(path))]

def _record_lines(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    '''Each non-blank line of a records file, as written, with its record, checked.'''

    lines = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path} line {number} is not JSON: {error}') from error
        problem = _record_problem(record)
        if problem is not None:
            raise ValueError(f'{path} line {number} is not a monitor record: {problem}')
        lines.append((line, record))
    return lines

def _is_integer(value: Any) -> bool:
    '''JSON's integers; a boolean is not one.'''

    return isinstance(value, int) and not isinstance(value, bool)

def _record_problem(record: Any) -> Optional[str]:
    '''What keeps a parsed line from being a monitor record, or None.'''

    if not isinstance(record, dict) or set(record) != {'mrr', 'read'}:
        return "it must be an object with exactly 'mrr' and 'read'"
    if isinstance(record['mrr'], bool) or not isinstance(record['mrr'], (int, float)):
        return 'its mrr is not a number'
    read = record['read']
    detail = read.get('detail') if isinstance(read, dict) else None
    epoch = detail.get('epoch') if isinstance(detail, dict) else None
    if not _is_integer(epoch) or epoch < 0:
        return 'its read names no non-negative integer epoch'
    return None

def _record_epoch(record: Mapping[str, Any]) -> int:
    '''The epoch a checked record's read scored.'''

    return record['read']['detail']['epoch']
