'''
Unit tests for the text stage's data path (Req 10; spec 4.3).

Tests cover:
- stack_text_inputs, which batches the token rows of every encode
- The repaired training rows of streaming_dataset
- Two-stream epochs: the steps, the permutations, the even chunks, the tokenized task queries and
  the step dataset
- NAICSDataModule: the two streams from the bundle, the code rows in codebook order, the one train
  loader (P12), no validation loader, and the epoch set only by TrainDatasetEpochCallback (P27),
  under a real Trainer too
'''

import dataclasses
import hashlib
import inspect
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from naics_embedder.supervision.code_targets import CodeTargets
from naics_embedder.supervision.queries import build_task_queries
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model.dataloader import datamodule as two_stream
from naics_embedder.text_model.dataloader.datamodule import (
    NAICSDataModule,
    TrainDatasetEpochCallback,
    stack_text_inputs,
)
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS, QUERY, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from tests.fixtures.shared_encoder import MINILM, REFERENCE_QUERIES_PER_STEP, reference_step_dataset

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def channels():
    '''Standard text channels.'''
    return ['title', 'description', 'excluded', 'examples']

@pytest.fixture
def make_embedding(channels):
    '''Factory to create mock embeddings for all channels.'''

    def _make(seq_len=128):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
                'present': True,
            }
            for ch in channels
        }

    return _make

# -------------------------------------------------------------------------------------------------
# Token rows into batches
# -------------------------------------------------------------------------------------------------

def test_stack_text_inputs_carries_a_boolean_present_per_channel(make_embedding):
    absent = make_embedding()
    absent['excluded']['present'] = False

    batch = stack_text_inputs([make_embedding(), absent])

    assert batch['excluded']['present'].dtype == torch.bool
    assert batch['excluded']['present'].tolist() == [True, False]
    assert batch['title']['present'].tolist() == [True, True]

def test_stack_text_inputs_refuses_a_row_without_present(make_embedding):
    row = make_embedding()
    del row['title']['present']

    with pytest.raises(ValueError, match='no present flag'):
        stack_text_inputs([row])

def test_stack_text_inputs_builds_a_query_batch():
    row = {
        'query': {
            'input_ids': torch.tensor([101, 102]),
            'attention_mask': torch.ones(2, dtype=torch.long),
            'present': True,
        }
    }

    batch = stack_text_inputs([row], fields=('query', ))

    assert list(batch) == ['query']
    assert batch['query']['present'].tolist() == [True]

# -------------------------------------------------------------------------------------------------
# The repaired training rows of streaming_dataset
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def hierarchy_bundle(hierarchy_manifest):
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.supervision.index import SupervisionIndex

    bundle = load_validated_bundle(hierarchy_manifest)
    return bundle, SupervisionIndex.from_bundle(bundle)

@pytest.fixture
def repaired_streaming_config(hierarchy_descriptions_parquet):
    from naics_embedder.utils.config import StreamingConfig

    return StreamingConfig(
        descriptions_parquet=hierarchy_descriptions_parquet,
        n_negatives=3,
        n_candidates=4,
        n_negatives_phase1=3,
        seed=5,
    )

def test_repaired_rows_never_use_an_exclusion_as_the_positive(
    hierarchy_bundle, repaired_streaming_config
):
    from naics_embedder.text_model.dataloader.streaming_dataset import (
        build_repaired_triplet_rows,
    )
    from naics_embedder.utils.config import SamplingConfig

    bundle, index = hierarchy_bundle
    rows = build_repaired_triplet_rows(
        repaired_streaming_config, SamplingConfig(), bundle, index, sampling_epoch=0
    )

    for row in rows:
        assert row['positive_code_id'] not in index.exclusion_code_ids(row['anchor_code_id'])
        assert row['raw_candidates']

# -------------------------------------------------------------------------------------------------
# Two-stream epochs (Req 10; spec 4.3; section 6, "Coverage", the data half)
#
# A synthetic row's token ids name where it came from: a code row's ids are its code id in every
# channel, and a query row's are QUERY_TOKEN_OFFSET plus the query's index.
# -------------------------------------------------------------------------------------------------

# Twelve codes in codebook order, at least two at each level
SYNTHETIC_LEVELS = (2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6, 6)
QUERY_TOKEN_OFFSET = 1000
# The shipped window: the reference bundle's longest marked channel text is 44 tokens
REFERENCE_WINDOW = 128

def _token_row(value: int) -> Dict[str, Any]:
    '''A present two-token row whose ids are ``value``.'''

    return {
        'input_ids': torch.full((2, ), value, dtype=torch.long),
        'attention_mask': torch.ones(2, dtype=torch.long),
        'present': True,
    }

def _code_rows(n_codes: int) -> List[Dict[str, Any]]:
    '''Token rows of the codes 0..n-1 in codebook order, as the tokenization cache holds them.'''

    rows = []
    for code_id in range(n_codes):
        row: Dict[str, Any] = {channel: _token_row(code_id) for channel in CHANNELS}
        row['code'] = f'code-{code_id}'
        rows.append(row)
    return rows

def _synthetic_queries(levels: Sequence[int], n_queries: int):
    '''
    Query q reads at the level of code a = q mod N with a as a target, and for odd q also the next
    code at that level. Every third query has no forced negative; the others have a code at
    another level, where there is one.
    '''

    by_level: Dict[int, List[int]] = defaultdict(list)
    rank: Dict[int, int] = {}
    for code, level in enumerate(levels):
        rank[code] = len(by_level[level])
        by_level[level].append(code)
    elsewhere = {
        level: [code for code, other in enumerate(levels) if other != level]
        for level in by_level
    }
    targets, negatives = [], []
    for query in range(n_queries):
        anchor = query % len(levels)
        same = by_level[levels[anchor]]
        partner = same[(rank[anchor] + 1) % len(same)]
        others = elsewhere[levels[anchor]]
        targets.append(tuple(sorted({anchor, partner})) if query % 2 else (anchor, ))
        negatives.append((others[query % len(others)], ) if others and query % 3 else ())
    return two_stream.TokenizedQueries(
        texts=tuple(f'query {query}' for query in range(n_queries)),
        tokens=tuple(_token_row(QUERY_TOKEN_OFFSET + query) for query in range(n_queries)),
        levels=tuple(levels[query % len(levels)] for query in range(n_queries)),
        target_ids=tuple(targets),
        negative_ids=tuple(negatives),
    )

def _step_dataset(
    *,
    n_queries: int = 10,
    queries_per_step: int = 3,
    seed: int = 0,
    levels: Sequence[int] = SYNTHETIC_LEVELS,
    queries: Any = None,
    code_rows: Optional[List[Dict[str, Any]]] = None,
    code_levels: Optional[Sequence[int]] = None,
):
    '''A step dataset over the synthetic codes and queries (12 codes and 10 queries by default).'''

    return two_stream.StepDataset(
        code_rows=_code_rows(len(levels)) if code_rows is None else code_rows,
        code_levels=levels if code_levels is None else code_levels,
        queries=_synthetic_queries(levels, n_queries) if queries is None else queries,
        n_codes=len(levels),
        seed=seed,
        queries_per_step=queries_per_step,
    )

def _read_epoch(dataset, epoch: int) -> List[Dict[str, Any]]:
    '''Every step of one epoch, in step order.'''

    dataset.set_epoch(epoch)
    return [dataset[step] for step in range(len(dataset))]

def _query_indices(step: Dict[str, Any]) -> List[int]:
    '''The synthetic queries a step reads, named by their token ids.'''

    return (step['queries']['inputs'][QUERY]['input_ids'][:, 0] - QUERY_TOKEN_OFFSET).tolist()

def _marked(mask_row: torch.Tensor) -> List[int]:
    '''The code ids a mask row marks, ascending.'''

    return torch.nonzero(mask_row).flatten().tolist()

def _same(left: Any, right: Any) -> bool:
    '''Equal nested dicts of equal tensors (dtype included) and plain values.'''

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype:
            return False
        return torch.equal(left, right)
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_same(left[key], right[key]) for key in left)
    return left == right

def _expected_permutation(seed: int, epoch: int, n: int, stream: str) -> torch.Tensor:
    '''P12's rule written out: randperm on a CPU generator seeded by 63 bits of a sha256.'''

    digest = hashlib.sha256(f'{seed}:{epoch}:{stream}'.encode('utf-8')).digest()
    generator_seed = int.from_bytes(digest[:8], 'big') & (2**63 - 1)
    return torch.randperm(n, generator=torch.Generator().manual_seed(generator_seed))

# -------------------------------------------------------------------------------------------------
# Two-stream epochs: the steps and their chunks
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('n_queries', 'queries_per_step', 'steps'),
    [(11039, 128, 87), (11, 3, 4), (12, 3, 4), (11, 1, 11), (11, 128, 1), (0, 128, 0)],
)
def test_steps_per_epoch_is_the_ceiling_of_queries_over_queries_per_step(
    n_queries, queries_per_step, steps
):
    assert two_stream.steps_per_epoch(n_queries, queries_per_step) == steps

@pytest.mark.parametrize('queries_per_step', [0, -1])
def test_steps_per_epoch_refuses_fewer_than_one_query_per_step(queries_per_step):
    with pytest.raises(
        ValueError, match=f'queries_per_step must be at least 1, not {queries_per_step}'
    ):
        two_stream.steps_per_epoch(11, queries_per_step)

def test_steps_per_epoch_refuses_a_negative_query_count():
    with pytest.raises(ValueError, match='cannot be negative, not -1'):
        two_stream.steps_per_epoch(-1, 128)

@pytest.mark.parametrize(
    ('n', 'steps', 'torch_chunks'),
    [(2125, 87, 85), (17, 11, 9), (6, 4, 3)],
    ids=['the-shipped-codes', 'the-reference-codes-at-one-query-per-step', 'six-over-four'],
)
def test_even_chunks_cuts_exactly_the_steps_where_torch_chunk_cuts_fewer(n, steps, torch_chunks):
    order = torch.randperm(n, generator=torch.Generator().manual_seed(n))

    chunks = two_stream.even_chunks(order, steps)

    # The hazard is real: torch.chunk cuts chunks of ceil(n / steps) and runs out of elements
    assert len(torch.chunk(order, steps)) == torch_chunks < steps
    assert len(chunks) == steps
    sizes = [len(chunk) for chunk in chunks]
    assert min(sizes) >= 1 and max(sizes) - min(sizes) <= 1
    assert torch.equal(torch.cat(chunks), order)

@pytest.mark.parametrize('steps', [0, -1])
def test_even_chunks_refuses_fewer_than_one_step(steps):
    with pytest.raises(ValueError, match='at least one step'):
        two_stream.even_chunks(torch.arange(5), steps)

def test_even_chunks_refuses_an_order_that_is_not_one_dimensional():
    with pytest.raises(ValueError, match=re.escape('one-dimensional, not of shape (2, 3)')):
        two_stream.even_chunks(torch.arange(6).reshape(2, 3), 2)

def test_the_shipped_epoch_has_87_steps_of_126_or_127_queries_and_24_or_25_codes():
    '''Spec 4.3: 11,039 queries at 128 a step and 2,125 codes, both cut into S = 87 chunks.'''

    dataset = _step_dataset(n_queries=11039, queries_per_step=128, levels=(6, ) * 2125)

    read = _read_epoch(dataset, 0)

    assert len(dataset) == len(read) == 87
    assert Counter(len(step['codes']['ids']) for step in read) == {25: 37, 24: 50}
    assert Counter(len(step['queries']['levels']) for step in read) == {127: 77, 126: 10}
    anchors = torch.cat([step['codes']['ids'] for step in read])
    assert sorted(anchors.tolist()) == list(range(2125))
    assert sorted(index for step in read for index in _query_indices(step)) == list(range(11039))

# -------------------------------------------------------------------------------------------------
# Two-stream epochs: the permutations
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('seed', 'epoch', 'stream'),
    [(0, 0, 'codes'), (42, 3, 'queries'), (7, 39, 'codes')],
)
def test_an_epoch_permutation_is_drawn_under_the_sha256_of_seed_epoch_and_stream(
    seed, epoch, stream
):
    permutation = two_stream.epoch_permutation(seed, epoch, 2125, stream)

    assert permutation.dtype == torch.int64
    assert torch.equal(permutation, _expected_permutation(seed, epoch, 2125, stream))
    assert torch.equal(permutation.sort().values, torch.arange(2125))

def test_epoch_permutations_depend_only_on_seed_epoch_and_stream():
    first = two_stream.epoch_permutation(42, 3, 2125, 'codes')
    # The conftest reseeds torch before every test, so move the global generator on first
    torch.manual_seed(1234)
    torch.rand(100)
    two_stream.epoch_permutation(42, 4, 2125, 'codes')

    assert torch.equal(two_stream.epoch_permutation(42, 3, 2125, 'codes'), first)
    for seed, epoch, stream in [(43, 3, 'codes'), (42, 4, 'codes'), (42, 3, 'queries')]:
        assert not torch.equal(two_stream.epoch_permutation(seed, epoch, 2125, stream), first)

def test_drawing_an_epoch_permutation_leaves_the_global_generator_alone():
    state = torch.get_rng_state()

    two_stream.epoch_permutation(42, 3, 2125, 'codes')

    assert torch.equal(torch.get_rng_state(), state)

@pytest.mark.parametrize(
    ('stream', 'epoch', 'match'),
    [('code', 0, "unknown stream 'code'"), ('codes', -1, 'epoch must be at least 0, not -1')],
    ids=['an-unknown-stream', 'a-negative-epoch'],
)
def test_an_epoch_permutation_refuses_an_unknown_stream_or_a_negative_epoch(stream, epoch, match):
    with pytest.raises(ValueError, match=match):
        two_stream.epoch_permutation(0, epoch, 5, stream)

def test_an_epoch_permutation_refuses_a_negative_length():
    with pytest.raises(ValueError, match='a stream cannot have a negative length, not -1'):
        two_stream.epoch_permutation(0, 0, -1, 'codes')

# -------------------------------------------------------------------------------------------------
# Two-stream epochs: the step dataset
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(('queries_per_step', 'steps'), [(1, 10), (3, 4), (10, 1), (128, 1)])
def test_one_epoch_reads_every_code_as_an_anchor_once_and_every_query_once(queries_per_step, steps):
    dataset = _step_dataset(n_queries=10, queries_per_step=queries_per_step)

    assert len(dataset) == steps
    for epoch in (0, 1, 7):
        read = _read_epoch(dataset, epoch)
        anchors = torch.cat([step['codes']['ids'] for step in read])
        assert sorted(anchors.tolist()) == list(range(12))
        assert sorted(index for step in read for index in _query_indices(step)) == list(range(10))
        assert min(len(step['codes']['ids']) for step in read) >= 1
        assert max(len(step['queries']['levels']) for step in read) <= queries_per_step

def test_each_row_of_a_step_carries_its_own_tokens_level_targets_and_negatives():
    queries = _synthetic_queries(SYNTHETIC_LEVELS, 10)
    dataset = _step_dataset(queries=queries, queries_per_step=3)

    for step in _read_epoch(dataset, 2):
        codes, reads = step['codes'], step['queries']
        assert codes['ids'].dtype == codes['levels'].dtype == torch.int64
        assert codes['levels'].tolist() == [
            SYNTHETIC_LEVELS[code] for code in codes['ids'].tolist()
        ]
        for channel in CHANNELS:
            ids = codes['inputs'][channel]['input_ids']
            assert torch.equal(ids, codes['ids'][:, None].expand_as(ids))
            assert codes['inputs'][channel]['present'].all()
        indices = _query_indices(step)
        assert reads['levels'].dtype == torch.int64
        assert reads['levels'].tolist() == [queries.levels[index] for index in indices]
        assert reads['targets'].dtype == reads['negatives'].dtype == torch.bool
        assert reads['targets'].shape == reads['negatives'].shape == (len(indices), 12)
        for row, index in enumerate(indices):
            assert _marked(reads['targets'][row]) == list(queries.target_ids[index])
            assert _marked(reads['negatives'][row]) == list(queries.negative_ids[index])

def test_a_step_carries_no_candidate_pool():
    '''Spec 4.3: a step is its anchors and its queries; the candidates come from the code cache.'''

    dataset = _step_dataset()
    dataset.set_epoch(0)

    step = dataset[0]

    assert set(step) == {'codes', 'queries'}
    assert set(step['codes']) == {'inputs', 'ids', 'levels'}
    assert set(step['queries']) == {'inputs', 'levels', 'targets', 'negatives'}
    # The only code texts a step carries are its anchors'
    assert set(step['codes']['inputs']) == set(CHANNELS)
    for channel in CHANNELS:
        assert len(step['codes']['inputs'][channel]['input_ids']) == len(step['codes']['ids'])
    assert set(step['queries']['inputs']) == {QUERY}
    assert len(step['queries']['inputs'][QUERY]['input_ids']) == len(step['queries']['levels'])

def test_an_epochs_steps_are_its_two_permutations_cut_into_even_chunks():
    dataset = _step_dataset(seed=5)
    codes = two_stream.even_chunks(two_stream.epoch_permutation(5, 4, 12, 'codes'), 4)
    queries = two_stream.even_chunks(two_stream.epoch_permutation(5, 4, 10, 'queries'), 4)

    read = _read_epoch(dataset, 4)

    assert len(read) == 4
    for index, step in enumerate(read):
        assert torch.equal(step['codes']['ids'], codes[index])
        assert _query_indices(step) == queries[index].tolist()

def test_an_epochs_steps_depend_only_on_the_seed_and_the_epoch():
    '''Exact resume needs this: epoch k's steps are the same whatever epochs were read before.'''

    def orders(dataset, epoch):
        return [
            (step['codes']['ids'].tolist(), _query_indices(step))
            for step in _read_epoch(dataset, epoch)
        ]

    resumed = _step_dataset(seed=5)
    for epoch in range(3):
        _read_epoch(resumed, epoch)

    assert orders(resumed, 3) == orders(_step_dataset(seed=5), 3)
    assert orders(_step_dataset(seed=6), 3) != orders(_step_dataset(seed=5), 3)
    assert orders(resumed, 4) != orders(resumed, 3)

def test_each_epochs_two_permutations_are_drawn_once_at_its_first_step(monkeypatch):
    drawn = []
    real = two_stream.epoch_permutation

    def recording(seed, epoch, n, stream):
        drawn.append((epoch, stream))
        return real(seed, epoch, n, stream)

    monkeypatch.setattr(two_stream, 'epoch_permutation', recording)
    dataset = _step_dataset()

    dataset.set_epoch(0)
    assert drawn == []
    _read_epoch(dataset, 0)
    _read_epoch(dataset, 0)
    assert sorted(drawn) == [(0, 'codes'), (0, 'queries')]
    _read_epoch(dataset, 1)
    assert sorted(drawn) == [(0, 'codes'), (0, 'queries'), (1, 'codes'), (1, 'queries')]

def test_changing_a_steps_anchor_ids_leaves_the_epochs_order_alone():
    dataset = _step_dataset()
    dataset.set_epoch(0)
    ids = dataset[0]['codes']['ids']
    expected = ids.clone()

    ids.fill_(-1)

    assert torch.equal(dataset[0]['codes']['ids'], expected)

def test_a_negative_epoch_is_refused():
    with pytest.raises(ValueError, match='epoch must be at least 0, not -1'):
        _step_dataset().set_epoch(-1)

def test_a_step_read_before_any_epoch_is_set_is_refused():
    '''P27: the epoch comes only from set_epoch, which TrainDatasetEpochCallback calls.'''

    with pytest.raises(RuntimeError, match='set_epoch'):
        _step_dataset()[0]

@pytest.mark.parametrize('step', [4, -1])
def test_a_step_outside_the_epoch_is_refused(step):
    dataset = _step_dataset()
    dataset.set_epoch(0)

    with pytest.raises(IndexError, match=f'step {step} is outside the 4 steps of an epoch'):
        dataset[step]

def test_more_steps_than_codes_are_refused():
    '''No step may be empty: code_code_loss and radial_loss refuse a step with no anchor.'''

    with pytest.raises(
        ValueError, match='13 steps for 12 codes: every step needs at least one anchor'
    ):
        _step_dataset(n_queries=13, queries_per_step=1)

def test_as_many_steps_as_codes_give_each_step_one_anchor():
    dataset = _step_dataset(n_queries=12, queries_per_step=1)

    read = _read_epoch(dataset, 0)

    assert [len(step['codes']['ids']) for step in read] == [1] * 12

def test_an_epoch_without_task_queries_is_refused():
    '''With no query there is no step, so no code would be an anchor (Req 10).'''

    with pytest.raises(ValueError, match='no task queries'):
        _step_dataset(n_queries=0)

@pytest.mark.parametrize(
    ('n_rows', 'n_levels', 'match'),
    [(11, 12, '11 code token rows for 12 codes'), (12, 11, '11 code levels for 12 codes')],
    ids=['a-row-short', 'a-level-short'],
)
def test_the_code_rows_and_levels_must_be_one_per_code(n_rows, n_levels, match):
    with pytest.raises(ValueError, match=match):
        _step_dataset(code_rows=_code_rows(n_rows), code_levels=SYNTHETIC_LEVELS[:n_levels])

def test_the_code_levels_must_be_one_dimensional():
    with pytest.raises(ValueError, match=re.escape('one-dimensional, not of shape (1, 12)')):
        _step_dataset(code_levels=[list(SYNTHETIC_LEVELS)])

@pytest.mark.parametrize(
    ('field', 'ids', 'problem'),
    [
        ('target_ids', (12, ), 'names code ids [12], outside the 12 codes'),
        ('negative_ids', (-1, ), 'names code ids [-1], outside the 12 codes'),
        ('target_ids', (), 'has no target'),
        ('target_ids', (1, ), 'has targets at another level: code ids [1]'),
    ],
    ids=[
        'a-target-outside-the-codes',
        'a-negative-outside-the-codes',
        'no-target',
        'a-target-at-another-level',
    ],
)
def test_a_query_no_step_could_score_is_refused(field, ids, problem):
    queries = _synthetic_queries(SYNTHETIC_LEVELS, 10)
    broken = dataclasses.replace(queries, **{field: (ids, ) + getattr(queries, field)[1:]})

    # Query 0 reads at the level of code 0, level 2
    with pytest.raises(ValueError, match=re.escape(f"task query 'query 0' at level 2 {problem}")):
        _step_dataset(queries=broken)

def test_the_train_loader_hands_each_step_through_whole():
    '''P12's loader: with batch_size None, each step is one batch, unchanged, in step order.'''

    dataset = _step_dataset()
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    batches = list(loader)

    assert len(loader) == len(batches) == len(dataset) == 4
    for index, batch in enumerate(batches):
        assert _same(batch, dataset[index])

# -------------------------------------------------------------------------------------------------
# Two-stream epochs: the reference bundle, end to end
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def minilm_tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

@pytest.fixture
def reference_code_rows(tmp_path, reference_bundle) -> List[Dict[str, Any]]:
    '''The reference codes' cached token rows at the shipped window, in codebook order.'''

    parameters = reference_bundle.manifest.generation_parameters
    config = TokenizationConfig(
        descriptions_parquet=parameters['descriptions_parquet'],
        tokenizer_name=MINILM,
        max_length=REFERENCE_WINDOW,
        output_path=str(tmp_path / 'token_cache' / 'token_cache.pt'),
    )
    cache = tokenization_cache(
        config,
        description_fingerprint=reference_bundle.manifest.description_fingerprint,
        codebook_fingerprint=reference_bundle.manifest.codebook_fingerprint,
    )
    return [cache[code_id] for code_id in range(len(cache))]

def _code_ids(bundle) -> Dict[str, int]:
    '''Each code's id: its row in the codebook.'''

    return {code: code_id for code_id, code in enumerate(CodeTargets.from_bundle(bundle).codes)}

def test_task_queries_are_tokenized_once_as_query_texts_and_named_by_code_id(
    reference_bundle, minilm_tokenizer
):
    queries = build_task_queries(reference_bundle)
    codes = CodeTargets.from_bundle(reference_bundle).codes
    # 'query: ' as tokens, which follow [CLS]
    marker_ids = minilm_tokenizer(marker(QUERY), add_special_tokens=False)['input_ids']

    tokenized = two_stream.tokenize_task_queries(
        queries, minilm_tokenizer, REFERENCE_WINDOW, _code_ids(reference_bundle)
    )

    assert len(tokenized) == len(queries) == 11
    assert tokenized.texts == tuple(query.text for query in queries)
    assert tokenized.levels == tuple(query.level for query in queries)
    for index, query in enumerate(queries):
        tokens = tokenized.tokens[index]
        expected = tokenize_field(minilm_tokenizer, QUERY, query.text, REFERENCE_WINDOW)
        assert torch.equal(tokens['input_ids'], expected['input_ids'])
        assert torch.equal(tokens['attention_mask'], expected['attention_mask'])
        assert tokens['present'] is True
        assert tokens['input_ids'][1:1 + len(marker_ids)].tolist() == marker_ids
        assert [codes[code_id] for code_id in tokenized.target_ids[index]] == list(query.targets)
        assert [codes[code_id]
                for code_id in tokenized.negative_ids[index]] == list(query.negatives)

def test_a_task_query_naming_a_code_without_a_code_id_is_refused(
    reference_bundle, minilm_tokenizer
):
    code_ids = _code_ids(reference_bundle)
    del code_ids['321111']

    # The first query, in (level, text) order, to name 321111 sends wood flour grinding away
    with pytest.raises(
        ValueError,
        match=re.escape(
            "task query 'Wood flour grinding' at level 4 names codes with no code id: ['321111']"
        ),
    ):
        two_stream.tokenize_task_queries(
            build_task_queries(reference_bundle), minilm_tokenizer, REFERENCE_WINDOW, code_ids
        )

@pytest.mark.parametrize(('queries_per_step', 'steps'), [(1, 11), (2, 6), (11, 1)])
def test_one_epoch_of_the_reference_bundle_reads_each_code_and_each_task_query_once(
    reference_bundle, reference_code_rows, minilm_tokenizer, queries_per_step, steps
):
    targets = CodeTargets.from_bundle(reference_bundle)
    queries = build_task_queries(reference_bundle)
    tokenized = two_stream.tokenize_task_queries(
        queries, minilm_tokenizer, REFERENCE_WINDOW, _code_ids(reference_bundle)
    )
    dataset = two_stream.StepDataset(
        code_rows=reference_code_rows,
        code_levels=targets.levels,
        queries=tokenized,
        n_codes=len(targets.codes),
        seed=0,
        queries_per_step=queries_per_step,
    )
    # A query is named by its tokens and its level: two texts are queries at two levels each
    by_content = {
        (tuple(tokens['input_ids'].tolist()), query.level): query
        for tokens, query in zip(tokenized.tokens, queries)
    }
    assert len(by_content) == len(queries) == 11
    assert [row['code'] for row in reference_code_rows] == list(targets.codes)

    assert len(dataset) == steps
    for epoch in (0, 1):
        anchors, read = [], []
        for step in _read_epoch(dataset, epoch):
            ids = step['codes']['ids'].tolist()
            anchors.extend(ids)
            assert step['codes']['levels'].tolist() == targets.levels[ids].tolist()
            for channel in CHANNELS:
                expected = torch.stack(
                    [reference_code_rows[code_id][channel]['input_ids'] for code_id in ids]
                )
                assert torch.equal(step['codes']['inputs'][channel]['input_ids'], expected)
            reads = step['queries']
            rows = zip(reads['inputs'][QUERY]['input_ids'], reads['levels'].tolist())
            for row, (tokens, level) in enumerate(rows):
                query = by_content[tuple(tokens.tolist()), level]
                read.append(query)
                assert [targets.codes[code_id]
                        for code_id in _marked(reads['targets'][row])] == list(query.targets)
                assert [targets.codes[code_id]
                        for code_id in _marked(reads['negatives'][row])] == list(query.negatives)
        assert sorted(anchors) == list(range(17))
        assert sorted(read, key=lambda query: (query.level, query.text)) == queries

# -------------------------------------------------------------------------------------------------
# The datamodule (spec 4.3): the two streams from the bundle, the code rows, the one train loader
# (P12), no validation loader, and the epoch only from TrainDatasetEpochCallback (P27)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def datamodule_token_config(tmp_path, reference_bundle) -> TokenizationConfig:
    '''
    The reference codes' token cache at the shipped window, under a path of its own, so the
    datamodule builds it rather than reusing the one ``reference_arm_code_rows`` built.
    '''

    parameters = reference_bundle.manifest.generation_parameters
    return TokenizationConfig(
        descriptions_parquet=parameters['descriptions_parquet'],
        tokenizer_name=MINILM,
        max_length=REFERENCE_WINDOW,
        output_path=str(tmp_path / 'datamodule_token_cache' / 'token_cache.pt'),
    )

def _reference_datamodule(
    token_config: TokenizationConfig, bundle: Any, **overrides: Any
) -> NAICSDataModule:
    '''The reference bundle's datamodule, as constructed: 4 queries a step, under seed 0.'''

    arguments: Dict[str, Any] = {
        'seed': 0,
        'queries_per_step': REFERENCE_QUERIES_PER_STEP,
        'supervision_bundle': bundle,
    }
    arguments.update(overrides)
    return NAICSDataModule(token_config, **arguments)

@pytest.fixture
def reference_datamodule(datamodule_token_config, reference_bundle) -> NAICSDataModule:
    '''The reference bundle's datamodule after ``prepare_data`` and ``setup``.'''

    datamodule = _reference_datamodule(datamodule_token_config, reference_bundle)
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule

def _same_steps(read: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, Any]]) -> bool:
    '''Two runs of steps, equal step by step.'''

    if len(read) != len(expected):
        return False
    return all(_same(step, other) for step, other in zip(read, expected))

# Each read the datamodule refuses before setup has built the step dataset
BEFORE_SETUP = {
    'code_rows': lambda datamodule: datamodule.code_rows,
    'train_dataloader': lambda datamodule: datamodule.train_dataloader(),
    'set_train_epoch': lambda datamodule: datamodule.set_train_epoch(0),
}

class _StepRecorder(pyl.LightningModule):
    '''Records each step it is handed, by epoch; its one parameter gives a loss to step on.'''

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.steps: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def training_step(self, batch, batch_idx):
        self.steps[self.current_epoch].append(batch)
        return self.weight.sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

def test_the_constructor_takes_exactly_its_arguments_with_their_defaults():
    '''No **kwargs: a stray argument, such as the old data path's batch size, is refused.'''

    parameters = inspect.signature(NAICSDataModule.__init__).parameters
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword = inspect.Parameter.KEYWORD_ONLY
    empty = inspect.Parameter.empty

    assert [(name, parameter.kind, parameter.default)
            for name, parameter in parameters.items()] == [
                ('self', positional, empty),
                ('token_config', positional, empty),
                ('seed', keyword, empty),
                ('queries_per_step', keyword, 128),
                ('supervision_manifest_path', keyword, None),
                ('supervision_contract_version', keyword, CONTRACT_VERSION),
                ('supervision_bundle', keyword, None),
            ]
    with pytest.raises(TypeError, match='batch_size'):
        NAICSDataModule(TokenizationConfig(), seed=0, batch_size=16)

@pytest.mark.parametrize('queries_per_step', [0, -1])
def test_fewer_than_one_query_per_step_is_refused_at_construction(queries_per_step):
    with pytest.raises(
        ValueError, match=f'queries_per_step must be at least 1, not {queries_per_step}'
    ):
        NAICSDataModule(TokenizationConfig(), seed=0, queries_per_step=queries_per_step)

def test_construction_reads_nothing(tmp_path):
    '''The bundle and the token cache are read in prepare_data and setup, never at construction.'''

    missing = tmp_path / 'missing'
    token_config = TokenizationConfig(
        descriptions_parquet=str(missing / 'descriptions.parquet'),
        output_path=str(missing / 'token_cache.pt'),
    )

    datamodule = NAICSDataModule(
        token_config, seed=0, supervision_manifest_path=str(missing / 'manifest.json')
    )

    assert datamodule.train_dataset is None
    assert not missing.exists()

@pytest.mark.parametrize('read', sorted(BEFORE_SETUP))
def test_before_setup_a_read_is_refused(datamodule_token_config, reference_bundle, read):
    datamodule = _reference_datamodule(datamodule_token_config, reference_bundle)

    with pytest.raises(RuntimeError, match='before setup'):
        BEFORE_SETUP[read](datamodule)

def test_setup_builds_the_two_streams_from_the_bundle(
    reference_datamodule, reference_bundle, reference_arm_code_rows, minilm_tokenizer
):
    '''
    Spec 4.3: every code once as an anchor and every task query once. The steps are the step
    dataset of the bundle's codes and task queries, epoch by epoch, as training builds it.
    '''

    dataset = reference_datamodule.train_dataset
    expected = reference_step_dataset(reference_bundle, reference_arm_code_rows, minilm_tokenizer)

    assert isinstance(dataset, two_stream.StepDataset)
    assert (len(dataset), dataset.n_codes, len(dataset.queries)) == (3, 17, 11)
    assert (dataset.seed, dataset.queries_per_step) == (0, REFERENCE_QUERIES_PER_STEP)
    assert dataset.queries.texts == expected.queries.texts
    for epoch in (0, 1):
        assert _same_steps(_read_epoch(dataset, epoch), _read_epoch(expected, epoch))

def test_the_steps_follow_the_runs_seed(
    datamodule_token_config, reference_bundle, reference_arm_code_rows, minilm_tokenizer
):
    datamodule = _reference_datamodule(datamodule_token_config, reference_bundle, seed=7)
    datamodule.prepare_data()
    datamodule.setup('fit')

    read = _read_epoch(datamodule.train_dataset, 0)

    rows = reference_arm_code_rows
    seven = reference_step_dataset(reference_bundle, rows, minilm_tokenizer, seed=7)
    zero = reference_step_dataset(reference_bundle, rows, minilm_tokenizer, seed=0)
    assert _same_steps(read, _read_epoch(seven, 0))
    assert not _same_steps(read, _read_epoch(zero, 0))

def test_code_rows_are_every_codes_token_rows_in_codebook_order(
    reference_datamodule, reference_bundle, reference_arm_code_rows
):
    '''The rows the model's code cache encodes (spec 4.3): row i is code id i's, as cached.'''

    rows = reference_datamodule.code_rows

    assert [row['code'] for row in rows] == list(reference_bundle.manifest.codebook_order)
    assert len(rows) == len(reference_arm_code_rows) == 17
    for row, expected in zip(rows, reference_arm_code_rows):
        assert _same(row, expected)
    # The steps read their anchors from the same rows
    assert rows is reference_datamodule.train_dataset.code_rows

@pytest.mark.parametrize(
    ('damage', 'problem'),
    [
        ('swap', "the token cache row for code id 0 names '311', not the codebook's '31'"),
        ('drop', "the token cache holds 16 rows for the codebook's 17 codes"),
    ],
    ids=['two-rows-swapped', 'a-row-missing'],
)
def test_a_token_cache_that_is_not_the_codebooks_is_refused(
    monkeypatch, datamodule_token_config, reference_bundle, damage, problem
):
    '''Row i must be code id i's, or the code cache would encode a code under another's id.'''

    load = two_stream.load_verified_tokenization_cache

    def damaged(token_config, **fingerprints):
        cache = dict(load(token_config, **fingerprints))
        if damage == 'swap':
            cache[0], cache[1] = cache[1], cache[0]
        else:
            del cache[len(cache) - 1]
        return cache

    monkeypatch.setattr(two_stream, 'load_verified_tokenization_cache', damaged)
    datamodule = _reference_datamodule(datamodule_token_config, reference_bundle)
    datamodule.prepare_data()

    with pytest.raises(ValueError, match=re.escape(problem)):
        datamodule.setup('fit')
    assert datamodule.train_dataset is None

def test_descriptions_other_than_the_bundles_are_refused_before_any_cache_is_built(
    tmp_path, datamodule_token_config, reference_bundle
):
    other = tmp_path / 'other_descriptions.parquet'
    descriptions = pl.read_parquet(datamodule_token_config.descriptions_parquet)
    descriptions.with_columns(pl.col('title') + ' (edited)').write_parquet(other)
    token_config = datamodule_token_config.model_copy(update={'descriptions_parquet': str(other)})
    datamodule = _reference_datamodule(token_config, reference_bundle)

    for stage in (datamodule.prepare_data, lambda: datamodule.setup('fit')):
        with pytest.raises(ValueError, match='does not match supervision bundle'):
            stage()
    assert not Path(token_config.output_path).exists()

def test_without_a_bundle_the_datamodule_loads_the_one_its_manifest_names(
    datamodule_token_config,
    reference_manifest,
    reference_bundle,
    reference_arm_code_rows,
    minilm_tokenizer,
):
    datamodule = NAICSDataModule(
        datamodule_token_config,
        seed=0,
        queries_per_step=REFERENCE_QUERIES_PER_STEP,
        supervision_manifest_path=str(reference_manifest),
    )

    datamodule.prepare_data()
    datamodule.setup('fit')

    expected = reference_step_dataset(reference_bundle, reference_arm_code_rows, minilm_tokenizer)
    codes = [row['code'] for row in datamodule.code_rows]
    assert codes == list(reference_bundle.manifest.codebook_order)
    assert _same_steps(_read_epoch(datamodule.train_dataset, 0), _read_epoch(expected, 0))

@pytest.mark.parametrize(
    ('manifest', 'contract', 'problem'),
    [
        (False, CONTRACT_VERSION, 'requires a supervision manifest'),
        (True, 'stage3-supervision-v0', 'expected supervision contract stage3-supervision-v0'),
    ],
    ids=['no-manifest', 'another-contract'],
)
def test_without_a_bundle_a_missing_manifest_or_another_contract_is_refused(
    datamodule_token_config, reference_manifest, manifest, contract, problem
):
    datamodule = NAICSDataModule(
        datamodule_token_config,
        seed=0,
        supervision_manifest_path=str(reference_manifest) if manifest else None,
        supervision_contract_version=contract,
    )

    with pytest.raises(ValueError, match=problem):
        datamodule.prepare_data()

def test_there_is_no_validation_loader():
    '''Spec 4.3: validation is the outcome monitor, so Lightning runs no validation loop.'''

    datamodule = NAICSDataModule(TokenizationConfig(), seed=0)

    for hook in ('val_dataloader', 'test_dataloader', 'predict_dataloader'):
        assert not is_overridden(hook, datamodule)
    with pytest.raises(MisconfigurationException, match='val_dataloader'):
        datamodule.val_dataloader()

def test_the_train_loader_hands_each_step_through_whole_in_step_order(
    reference_datamodule, reference_bundle, reference_arm_code_rows, minilm_tokenizer
):
    '''P12: DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0) over the steps.'''

    loader = reference_datamodule.train_dataloader()
    expected = reference_step_dataset(reference_bundle, reference_arm_code_rows, minilm_tokenizer)

    assert loader.dataset is reference_datamodule.train_dataset
    assert loader.batch_size is None
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.num_workers == 0
    assert not loader.persistent_workers
    for epoch in (0, 1):
        reference_datamodule.set_train_epoch(epoch)
        assert len(loader) == 3
        assert _same_steps(list(loader), _read_epoch(expected, epoch))

def test_the_train_loader_never_takes_the_epoch_from_the_trainer(reference_datamodule):
    '''
    P27: when Lightning asks for the loader on a resume, the trainer's epoch can be stale, so the
    loader never reads it. Only TrainDatasetEpochCallback sets the epoch, and a step read before
    it has is refused.
    '''

    reference_datamodule.trainer = SimpleNamespace(current_epoch=3)

    loader = reference_datamodule.train_dataloader()

    assert reference_datamodule.train_dataset.epoch is None
    with pytest.raises(RuntimeError, match='set_epoch'):
        next(iter(loader))

def test_the_epoch_callback_hands_the_trainers_epoch_to_the_step_dataset(reference_datamodule):
    '''Lightning dispatches on_train_epoch_start to callbacks, never to a LightningDataModule.'''

    callback = TrainDatasetEpochCallback()

    callback.on_train_epoch_start(
        SimpleNamespace(datamodule=reference_datamodule, current_epoch=2), None
    )
    assert reference_datamodule.train_dataset.epoch == 2

    # A trainer without a NAICSDataModule is left alone
    callback.on_train_epoch_start(SimpleNamespace(datamodule=None, current_epoch=5), None)
    assert reference_datamodule.train_dataset.epoch == 2

def test_a_second_setup_keeps_what_the_first_built(reference_datamodule, monkeypatch):
    dataset = reference_datamodule.train_dataset

    def reread(*args, **kwargs):
        raise AssertionError('a second setup read the token cache again')

    monkeypatch.setattr(two_stream, 'load_verified_tokenization_cache', reread)

    reference_datamodule.setup('fit')

    assert reference_datamodule.train_dataset is dataset

@pytest.mark.unit
def test_train_dataset_epoch_advances_under_real_trainer(
    tmp_path, datamodule_token_config, reference_bundle, reference_arm_code_rows, minilm_tokenizer
):
    '''
    Lightning never calls a LightningDataModule's epoch hooks: TrainDatasetEpochCallback sets each
    epoch before its first step is read, so each epoch reads its own two permutations (P27).
    '''

    datamodule = _reference_datamodule(datamodule_token_config, reference_bundle)
    model = _StepRecorder()
    trainer = pyl.Trainer(
        max_epochs=2,
        accelerator='cpu',
        devices=1,
        callbacks=[TrainDatasetEpochCallback()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, datamodule=datamodule)

    expected = reference_step_dataset(reference_bundle, reference_arm_code_rows, minilm_tokenizer)
    assert sorted(model.steps) == [0, 1]
    for epoch in (0, 1):
        assert _same_steps(model.steps[epoch], _read_epoch(expected, epoch))
    assert not _same_steps(model.steps[0], model.steps[1])
    assert datamodule.train_dataset.epoch == 1
    # prepare_data built the datamodule's own token cache
    assert Path(datamodule_token_config.output_path).exists()
