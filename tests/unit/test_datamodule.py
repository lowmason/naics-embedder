'''
Unit tests for NAICSDataModule and collate_fn.

Tests cover:
- collate_fn batching over candidate pools
- Multi-level supervision expansion
- Sampling metadata accumulation
- NAICSMapDataset indexing and __getitem__
- Train-epoch propagation to epoch-aware datasets under a real Trainer (incl. persistent workers)
- Two-stream epochs (Req 10; spec 4.3): the steps, the permutations, the even chunks, the
  tokenized task queries and the step dataset
'''

import copy
import dataclasses
import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set

import pytest
import pytorch_lightning as pyl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from naics_embedder.supervision.code_targets import CodeTargets
from naics_embedder.supervision.queries import build_task_queries
from naics_embedder.text_model.dataloader import datamodule as two_stream
from naics_embedder.text_model.dataloader.datamodule import (
    NAICSDataModule,
    NAICSMapDataset,
    TrainDatasetEpochCallback,
    collate_fn,
    stack_text_inputs,
)
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS, QUERY, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from tests.fixtures.epoch_datasets import EpochRecordingDataset
from tests.fixtures.shared_encoder import MINILM

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

@pytest.fixture
def make_batch_item(make_embedding):
    '''Factory to create a single batch item.'''

    def _create(anchor_code, positive_code, negative_codes, seq_len=128):
        return {
            'anchor_code': anchor_code,
            'anchor_embedding': make_embedding(seq_len),
            'positive_code': positive_code,
            'positive_embedding': make_embedding(seq_len),
            'negatives': [
                {
                    'negative_code': nc,
                    'negative_idx': i,
                    'negative_embedding': make_embedding(seq_len),
                    'relation_margin': 0,
                    'distance_margin': 4,
                    'explicit_exclusion': False,
                } for i, nc in enumerate(negative_codes)
            ],
        }

    return _create

@pytest.fixture
def make_repaired_batch_item():
    channels = ('title', 'description', 'excluded', 'examples')

    def encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            channel: {
                'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
                'attention_mask': torch.ones(2, dtype=torch.long),
                'present': True,
            }
            for channel in channels
        }

    def make(
        candidate_code_ids: list[int],
        anchor_code_id: int = 100,
        positive_code_id: int = 104,
        positive_structural_distance: float = 1.0,
    ) -> dict[str, Any]:
        candidates = [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_embedding': encoded(code_id),
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            } for code_id in candidate_code_ids
        ]
        difficulty_order = sorted(
            range(len(candidate_code_ids)),
            key=lambda index: candidate_code_ids[index],
        )
        return {
            'anchor_code_id': anchor_code_id,
            'anchor_code': str(anchor_code_id),
            'anchor_embedding': encoded(anchor_code_id),
            'positive_code_id': positive_code_id,
            'positive_code': str(positive_code_id),
            'positive_embedding': encoded(positive_code_id),
            'positive_structural_distance': positive_structural_distance,
            'positive_structural_relation_id': 1,
            'candidate_pool': candidates,
            'difficulty_proposal_indices': difficulty_order,
            'selection_k': min(3, len(candidates)),
        }

    return make

# -------------------------------------------------------------------------------------------------
# Repaired candidate-pool collation
# -------------------------------------------------------------------------------------------------

def test_collate_does_not_mutate_input_and_uses_invalid_rows(make_repaired_batch_item):
    short = make_repaired_batch_item(candidate_code_ids=[101])
    long = make_repaired_batch_item(candidate_code_ids=[201, 202, 203])
    original_short = copy.deepcopy(short)

    batch = collate_fn([short, long])

    assert len(short['candidate_pool']) == 1
    assert short['candidate_pool'][0]['negative_code_id'] == 101
    assert torch.equal(
        short['candidate_pool'][0]['negative_embedding']['title']['input_ids'],
        original_short['candidate_pool'][0]['negative_embedding']['title']['input_ids'],
    )
    assert batch['candidate_code_id'].tolist()[0] == [101, -1, -1]
    assert batch['candidate_valid_mask'].tolist()[0] == [True, False, False]
    assert batch['candidate_source_slot'].tolist()[0] == [0, -1, -1]
    assert batch['candidate_inputs']['title']['attention_mask'][1].count_nonzero() == 0
    assert batch['candidate_inputs']['title']['attention_mask'][2].count_nonzero() == 0

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

def test_repaired_collate_marks_invalid_rows_absent(make_repaired_batch_item):
    batch = collate_fn(
        [make_repaired_batch_item([101]),
         make_repaired_batch_item([201, 202, 203])],
    )

    for channel in ('title', 'description', 'excluded', 'examples'):
        assert batch['candidate_inputs'][channel]['present'].tolist() == [
            True, False, False, True, True, True
        ]

def test_collate_carries_every_candidate_field_in_one_order(make_repaired_batch_item):
    item = make_repaired_batch_item(candidate_code_ids=[103, 101, 102])

    batch = collate_fn([item])

    assert batch['candidate_code_id'].tolist() == [[103, 101, 102]]
    assert batch['candidate_sampling_provenance_id'].tolist() == [[2, 2, 2]]
    assert batch['difficulty_proposal_indices'].tolist() == [[1, 2, 0]]
    assert batch['positive_code_id'].tolist() == [104]
    assert batch['positive_structural_distance'].tolist() == [1.0]

def test_repaired_collate_pads_proposals_and_flattens_candidates_row_major(
    make_repaired_batch_item,
):
    short = make_repaired_batch_item(candidate_code_ids=[7])
    long = make_repaired_batch_item(candidate_code_ids=[9, 8])

    batch = collate_fn([short, long])

    assert batch['k_candidates'] == 2
    assert batch['batch_size'] == 2
    assert batch['difficulty_proposal_indices'].tolist() == [[0, -1], [1, 0]]
    assert batch['candidate_inputs']['title']['input_ids'][:, 0].tolist() == [7, 0, 9, 8]
    assert 'negatives' not in batch and 'negative_codes' not in batch
    assert 'all_candidates' not in batch

def test_repaired_collate_rejects_legacy_items(make_batch_item):
    with pytest.raises(ValueError, match='candidate_pool'):
        collate_fn([make_batch_item('111', '11', ['222'])])

def test_repaired_collate_uses_the_smallest_selection_k(make_repaired_batch_item):
    first = make_repaired_batch_item(candidate_code_ids=[1, 2, 3])
    second = make_repaired_batch_item(candidate_code_ids=[4, 5])

    batch = collate_fn([first, second])

    assert batch['selection_k'] == 2

def test_repaired_collate_requires_a_positive_selection_k(make_repaired_batch_item):
    item = make_repaired_batch_item(candidate_code_ids=[1, 2])
    item['selection_k'] = 0

    with pytest.raises(ValueError, match='selection_k'):
        collate_fn([item])

def test_collate_takes_no_supervision_mode(make_repaired_batch_item):
    '''D2: every batch is repaired, so the collate has no mode to choose.'''

    with pytest.raises(TypeError, match='supervision_mode'):
        collate_fn([make_repaired_batch_item([1])], supervision_mode='repaired')

# -------------------------------------------------------------------------------------------------
# Bundle-backed repaired datasets
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def hierarchy_bundle(hierarchy_manifest):
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.supervision.index import SupervisionIndex

    bundle = load_validated_bundle(hierarchy_manifest)
    return bundle, SupervisionIndex.from_bundle(bundle)

@pytest.fixture
def hierarchy_token_cache(hierarchy_descriptions):
    channels = ('title', 'description', 'excluded', 'examples')
    return {
        int(index): {
            'code': code,
            **{
                channel: {
                    'input_ids': torch.full((4, ), int(index), dtype=torch.long),
                    'attention_mask': torch.ones(4, dtype=torch.long),
                    'present': True,
                }
                for channel in channels
            },
        }
        for index, code in hierarchy_descriptions.select('index', 'code').rows()
    }

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

def _assert_repaired_item(item, index, selection_k):
    codes = [candidate['negative_code_id'] for candidate in item['candidate_pool']]
    assert 'negatives' not in item and 'all_candidates' not in item
    assert item['selection_k'] == selection_k
    assert len(set(codes)) == len(codes)
    assert item['anchor_code_id'] not in codes
    assert item['positive_code_id'] not in codes
    exclusions = set(index.exclusion_code_ids(item['anchor_code_id']))
    assert exclusions - {item['positive_code_id']} <= set(codes)
    assert item['positive_code_id'] not in exclusions
    for candidate in item['candidate_pool']:
        assert candidate['negative_is_explicit_exclusion'] == (
            candidate['negative_code_id'] in exclusions
        )
        assert torch.equal(
            candidate['negative_embedding']['title']['input_ids'],
            torch.full((4, ), candidate['negative_code_id'], dtype=torch.long),
        )
    assert item['positive_structural_distance'] == pytest.approx(
        float(index.structural_distance[item['anchor_code_id'], item['positive_code_id']])
    )

def test_repaired_map_dataset_emits_one_bundle_backed_pool(
    hierarchy_bundle, hierarchy_token_cache, repaired_streaming_config
):
    from naics_embedder.text_model.dataloader.datamodule import RepairedMapDataset
    from naics_embedder.text_model.dataloader.streaming_dataset import (
        build_repaired_triplet_rows,
    )
    from naics_embedder.utils.config import SamplingConfig

    bundle, index = hierarchy_bundle
    rows = build_repaired_triplet_rows(
        repaired_streaming_config, SamplingConfig(), bundle, index, sampling_epoch=0
    )
    dataset = RepairedMapDataset(
        rows, hierarchy_token_cache, index, repaired_streaming_config, selection_k=3
    )

    assert len(dataset) > 0
    items = [dataset[position] for position in range(len(dataset))]
    for item in items:
        _assert_repaired_item(item, index, selection_k=3)
        assert item['difficulty_proposal_indices'] == list(range(len(item['candidate_pool'])))

    batch = collate_fn(items)
    assert batch['candidate_valid_mask'].sum().item() == sum(
        len(item['candidate_pool']) for item in items
    )

def test_repaired_phase1_dataset_proposes_by_difficulty_over_the_pool(
    hierarchy_bundle, hierarchy_token_cache, repaired_streaming_config
):
    from naics_embedder.text_model.dataloader.datamodule import RepairedPhase1Dataset
    from naics_embedder.utils.config import SamplingConfig

    bundle, index = hierarchy_bundle
    dataset = RepairedPhase1Dataset(
        cfg=repaired_streaming_config,
        sampling_cfg=SamplingConfig(),
        token_cache=hierarchy_token_cache,
        bundle=bundle,
        index=index,
        phase1_end_epoch=4,
    )

    assert len(dataset) > 0
    for position in range(len(dataset)):
        item = dataset[position]
        if item is None:
            continue
        _assert_repaired_item(item, index, selection_k=3)
        proposals = item['difficulty_proposal_indices']
        assert len(set(proposals)) == len(proposals)
        assert all(
            not item['candidate_pool'][slot]['negative_is_explicit_exclusion'] for slot in proposals
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

def test_validation_pools_are_stable_across_training_epochs(
    tmp_path,
    hierarchy_bundle,
    hierarchy_token_cache,
    hierarchy_descriptions_parquet,
    repaired_streaming_config,
):
    # val/contrastive_loss scores whole validation pools and drives checkpointing, so epoch
    # progress may only change training pools.

    from naics_embedder.text_model.dataloader.datamodule import (
        NAICSDataModule,
        RepairedPhase1Dataset,
    )
    from naics_embedder.utils.config import SamplingConfig

    bundle, index = hierarchy_bundle

    def on_the_fly_dataset():
        return RepairedPhase1Dataset(
            cfg=repaired_streaming_config,
            sampling_cfg=SamplingConfig(),
            token_cache=hierarchy_token_cache,
            bundle=bundle,
            index=index,
            phase1_end_epoch=4,
        )

    def pools(dataset):
        return [
            None if item is None else [c['negative_code_id'] for c in item['candidate_pool']]
            for item in (dataset[position] for position in range(len(dataset)))
        ]

    datamodule = NAICSDataModule(
        descriptions_path=hierarchy_descriptions_parquet,
        triplets_path=str(tmp_path / 'unused_triplets'),
        batch_size=2,
        num_workers=0,
    )
    datamodule.train_dataset = on_the_fly_dataset()
    datamodule.val_dataset = on_the_fly_dataset()
    before = pools(datamodule.val_dataset)
    assert any(before)

    datamodule.set_train_epoch(3)

    assert datamodule.train_dataset.epoch == 3
    assert datamodule.val_dataset.epoch == 0
    assert pools(datamodule.val_dataset) == before

# -------------------------------------------------------------------------------------------------
# Positive Level Tests
# -------------------------------------------------------------------------------------------------

def test_collate_extracts_positive_level(make_repaired_batch_item):
    '''collate_fn should extract positive_level from batch items.'''
    item = make_repaired_batch_item([1])
    item['positive_level'] = 5

    result = collate_fn([item])

    assert 'positive_levels' in result
    assert result['positive_levels'] == [5]

def test_collate_infers_positive_level_from_code_length(make_repaired_batch_item):
    '''collate_fn should infer positive_level from positive_code length if not present.'''
    item = make_repaired_batch_item([1])
    item['positive_code'] = '3111'

    result = collate_fn([item])

    assert 'positive_levels' in result
    # Default is len(positive_code) = 4
    assert result['positive_levels'] == [4]

def test_collate_multiple_positive_levels(make_repaired_batch_item):
    '''collate_fn should track positive_levels for multiple items.'''
    batch = [make_repaired_batch_item([1]), make_repaired_batch_item([2])]
    batch[0]['positive_level'] = 5
    batch[1]['positive_level'] = 4

    result = collate_fn(batch)

    assert result['positive_levels'] == [5, 4]

# -------------------------------------------------------------------------------------------------
# Sampling Metadata Tests
# -------------------------------------------------------------------------------------------------

def test_collate_accumulates_sampling_metadata(make_repaired_batch_item):
    '''Sampling metadata should be accumulated across batch items.'''
    batch = [make_repaired_batch_item([1]), make_repaired_batch_item([2])]

    # Add sampling metadata
    batch[0]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 10,
        'candidates_far': 5,
        'sampled_near': 2,
        'sampled_far': 1,
        'effective_near_weight': 0.6,
        'effective_far_weight': 0.4,
    }
    batch[1]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 8,
        'candidates_far': 7,
        'sampled_near': 1,
        'sampled_far': 2,
        'effective_near_weight': 0.5,
        'effective_far_weight': 0.5,
    }

    result = collate_fn(batch)

    assert 'sampling_metadata' in result
    assert result['sampling_metadata']['candidates_near'] == 18
    assert result['sampling_metadata']['candidates_far'] == 12
    assert result['sampling_metadata']['sampled_near'] == 3
    assert result['sampling_metadata']['sampled_far'] == 3

def test_collate_computes_average_weights(make_repaired_batch_item):
    '''Effective weights should be averaged across records.'''
    batch = [make_repaired_batch_item([1]), make_repaired_batch_item([2])]

    batch[0]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 10,
        'candidates_far': 5,
        'sampled_near': 2,
        'sampled_far': 1,
        'effective_near_weight': 0.8,
        'effective_far_weight': 0.2,
    }
    batch[1]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 8,
        'candidates_far': 7,
        'sampled_near': 1,
        'sampled_far': 2,
        'effective_near_weight': 0.4,
        'effective_far_weight': 0.6,
    }

    result = collate_fn(batch)

    # Average weights: (0.8 + 0.4) / 2 = 0.6, (0.2 + 0.6) / 2 = 0.4
    assert abs(result['sampling_metadata']['avg_effective_near_weight'] - 0.6) < 1e-6
    assert abs(result['sampling_metadata']['avg_effective_far_weight'] - 0.4) < 1e-6

def test_collate_no_metadata_when_missing(make_repaired_batch_item):
    '''No sampling_metadata key when items have no metadata.'''
    result = collate_fn([make_repaired_batch_item([1])])

    assert 'sampling_metadata' not in result

# -------------------------------------------------------------------------------------------------
# NAICSMapDataset Tests
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def mock_token_cache():
    '''Create a mock token cache with embeddings for indices 0-4.'''
    channels = ['title', 'description', 'excluded', 'examples']

    def make_embedding(idx):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
                'present': True,
            }
            for ch in channels
        }

    return {i: {'code': f'{i:06d}', **make_embedding(i)} for i in range(5)}

@pytest.fixture
def mock_triplet_rows():
    '''Create mock triplet rows for testing.'''
    return [
        {
            'anchor_idx': 0,
            'anchor_code': '000000',
            'positive_idx': 1,
            'positive_code': '000001',
            'positive_level': 6,
            'stratum_id': 0,
            'stratum_wgt': 1.0,
            'negatives': [
                {
                    'negative_idx': 2,
                    'negative_code': '000002',
                    'relation_margin': 0,
                    'distance_margin': 4,
                },
                {
                    'negative_idx': 3,
                    'negative_code': '000003',
                    'relation_margin': 0,
                    'distance_margin': 4,
                },
            ],
        },
        {
            'anchor_idx': 1,
            'anchor_code': '000001',
            'positive_idx': 0,
            'positive_code': '000000',
            'positive_level': 6,
            'stratum_id': 1,
            'stratum_wgt': 1.0,
            'negatives': [
                {
                    'negative_idx': 4,
                    'negative_code': '000004',
                    'relation_margin': 0,
                    'distance_margin': 4,
                },
            ],
        },
    ]

def test_map_dataset_len(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should return correct length.'''
    dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)
    assert len(dataset) == 2

def test_map_dataset_getitem_returns_correct_structure(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset __getitem__ should return correctly structured item.'''
    dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)

    item = dataset[0]

    assert item is not None
    assert item['anchor_idx'] == 0
    assert item['anchor_code'] == '000000'
    assert item['positive_idx'] == 1
    assert item['positive_code'] == '000001'
    assert 'anchor_embedding' in item
    assert 'positive_embedding' in item
    assert len(item['negatives']) == 2

def test_map_dataset_getitem_extracts_embeddings(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should extract embeddings from token cache.'''
    dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)

    item = dataset[0]
    assert item is not None

    # Check that embeddings have all channels
    for channel in ['title', 'description', 'excluded', 'examples']:
        assert channel in item['anchor_embedding']
        assert channel in item['positive_embedding']
        assert 'input_ids' in item['anchor_embedding'][channel]
        assert 'attention_mask' in item['anchor_embedding'][channel]

def test_map_dataset_getitem_excludes_code_from_embedding(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should exclude 'code' field from embeddings.'''
    dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)

    item = dataset[0]
    assert item is not None

    assert 'code' not in item['anchor_embedding']
    assert 'code' not in item['positive_embedding']

def test_map_dataset_getitem_returns_none_for_missing_anchor(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should return None if anchor not in token cache.'''
    # Remove anchor idx 0 from token cache
    token_cache_missing = {k: v for k, v in mock_token_cache.items() if k != 0}
    dataset = NAICSMapDataset(mock_triplet_rows, token_cache_missing)

    item = dataset[0]
    assert item is None

def test_map_dataset_getitem_returns_none_for_missing_positive(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should return None if positive not in token cache.'''
    # Remove positive idx 1 from token cache
    token_cache_missing = {k: v for k, v in mock_token_cache.items() if k != 1}
    dataset = NAICSMapDataset(mock_triplet_rows, token_cache_missing)

    item = dataset[0]
    assert item is None

def test_map_dataset_getitem_filters_missing_negatives(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should filter out negatives not in token cache.'''
    # Remove negative idx 2 from token cache
    token_cache_missing = {k: v for k, v in mock_token_cache.items() if k != 2}
    dataset = NAICSMapDataset(mock_triplet_rows, token_cache_missing)

    item = dataset[0]

    # Should have 1 negative instead of 2
    assert item is not None
    assert len(item['negatives']) == 1
    assert item['negatives'][0]['negative_idx'] == 3

def test_map_dataset_getitem_returns_none_for_no_negatives(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should return None if all negatives are missing.'''
    # Remove all negative indices from token cache
    token_cache_missing = {k: v for k, v in mock_token_cache.items() if k not in [2, 3]}
    dataset = NAICSMapDataset(mock_triplet_rows, token_cache_missing)

    item = dataset[0]
    assert item is None

def test_map_dataset_random_access(mock_triplet_rows, mock_token_cache):
    '''NAICSMapDataset should support random access (any order).'''
    dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)

    # Access in reverse order
    item1 = dataset[1]
    item0 = dataset[0]

    assert item0 is not None
    assert item1 is not None
    assert item0['anchor_idx'] == 0
    assert item1['anchor_idx'] == 1

def test_map_dataset_includes_sampling_metadata(mock_token_cache):
    '''NAICSMapDataset should include sampling_metadata if present.'''
    triplet_rows = [
        {
            'anchor_idx': 0,
            'anchor_code': '000000',
            'positive_idx': 1,
            'positive_code': '000001',
            'negatives': [
                {
                    'negative_idx': 2,
                    'negative_code': '000002',
                    'relation_margin': 0,
                    'distance_margin': 4,
                },
            ],
            'sampling_metadata': {
                'strategy': 'sans_static',
                'sampled_near': 1
            },
        },
    ]
    dataset = NAICSMapDataset(triplet_rows, mock_token_cache)

    item = dataset[0]
    assert item is not None

    assert 'sampling_metadata' in item
    assert item['sampling_metadata']['strategy'] == 'sans_static'

# -------------------------------------------------------------------------------------------------
# Edge Cases
# -------------------------------------------------------------------------------------------------

def test_collate_different_sequence_lengths(make_repaired_batch_item, make_embedding):
    '''Batch items with same sequence length should collate.'''

    def make_item(seq_len):
        item = make_repaired_batch_item([1])
        item['anchor_embedding'] = make_embedding(seq_len)
        item['positive_embedding'] = make_embedding(seq_len)
        item['candidate_pool'][0]['negative_embedding'] = make_embedding(seq_len)
        return item

    # Same sequence length should work
    result = collate_fn([make_item(64), make_item(64)])
    assert result['anchor']['title']['input_ids'].shape == (2, 64)

def test_collate_preserves_tensor_dtype(make_repaired_batch_item):
    '''Tensor dtypes should be preserved after collation.'''
    result = collate_fn([make_repaired_batch_item([1])])

    assert result['anchor']['title']['input_ids'].dtype == torch.long
    assert result['anchor']['title']['attention_mask'].dtype == torch.long

def test_collate_large_batch(make_repaired_batch_item):
    '''Should handle larger batches efficiently.'''
    batch = [make_repaired_batch_item([i + 100]) for i in range(64)]

    result = collate_fn(batch)

    assert result['batch_size'] == 64
    assert result['anchor']['title']['input_ids'].shape == (64, 2)

# -------------------------------------------------------------------------------------------------
# NAICSDataModule Setup Tests
# -------------------------------------------------------------------------------------------------

class TestNAICSDataModuleSetup:
    '''Test suite for NAICSDataModule initialization and setup.'''

    @pytest.fixture
    def mock_descriptions_parquet(self, tmp_path):
        '''Create mock descriptions parquet file.'''
        import polars as pl

        data = {
            'index': [0, 1, 2],
            'code': ['311111', '311112', '321111'],
            'level': [6, 6, 6],
            'title': ['Dog Food', 'Cat Food', 'Sawmills'],
            'description': ['Make dog food', 'Make cat food', 'Cut wood'],
            'excluded': ['', '', ''],
            'examples': ['', '', ''],
            'excluded_codes': [None, None, None],
        }
        df = pl.DataFrame(data)
        path = tmp_path / 'descriptions.parquet'
        df.write_parquet(path)
        return str(path)

    @pytest.fixture
    def mock_triplets_dir(self, tmp_path):
        '''Create mock triplets directory with parquet files.'''
        import polars as pl

        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        # Create anchor subdirectory
        anchor_dir = triplets_dir / 'anchor=0'
        anchor_dir.mkdir()

        data = {
            'anchor_idx': [0, 0],
            'anchor_code': ['311111', '311111'],
            'anchor_level': [6, 6],
            'positive_idx': [1, 1],
            'positive_code': ['311112', '311112'],
            'positive_level': [6, 6],
            'negative_idx': [2, 2],
            'negative_code': ['321111', '321111'],
            'negative_level': [6, 6],
            'relation_margin': [0, 0],
            'distance_margin': [4, 4],
            'positive_relation': [1, 1],
            'positive_distance': [2, 2],
            'negative_relation': [3, 3],
            'negative_distance': [8, 8],
        }
        df = pl.DataFrame(data)
        path = anchor_dir / 'part0.parquet'
        df.write_parquet(path)
        return str(triplets_dir)

    def test_datamodule_init_default_params(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test NAICSDataModule initializes with default parameters.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.batch_size == 4
        assert datamodule.num_workers == 0
        assert datamodule.descriptions_path == mock_descriptions_parquet
        assert datamodule.triplets_path == mock_triplets_dir

    def test_datamodule_init_custom_streaming_config(
        self, mock_descriptions_parquet, mock_triplets_dir
    ):
        '''Test NAICSDataModule initializes with custom streaming config.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        streaming_config = {
            'n_negatives': 8,
            'seed': 123,
        }

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            streaming_config=streaming_config,
            batch_size=8,
            num_workers=0,
            seed=100,  # Explicit seed for validation config
        )

        assert datamodule.train_streaming_cfg.n_negatives == 8
        assert datamodule.train_streaming_cfg.seed == 123
        # Validation config uses (seed + 1000) from NAICSDataModule.__init__ seed param
        assert datamodule.val_streaming_cfg.seed == 1100  # 100 + 1000

    def test_datamodule_init_custom_sampling_config(
        self, mock_descriptions_parquet, mock_triplets_dir
    ):
        '''Test NAICSDataModule initializes with custom sampling config.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        sampling_config = {'strategy': 'sans_static'}

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            sampling_config=sampling_config,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.sampling_cfg.strategy == 'sans_static'

    def test_datamodule_train_dataset_none_before_setup(
        self, mock_descriptions_parquet, mock_triplets_dir
    ):
        '''Test that train_dataset is None before setup() is called.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
        )

        # Datasets are None until setup() is called
        assert datamodule.train_dataset is None
        assert datamodule.val_dataset is None

    def test_datamodule_n_epochs_parameter(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test that n_epochs parameter is stored correctly.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
            n_epochs=50,
        )

        assert datamodule.n_epochs == 50

    def test_datamodule_tokenization_config(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test that tokenization config is set correctly.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.tokenization_cfg.descriptions_parquet == mock_descriptions_parquet
        assert datamodule.tokenization_cfg.tokenizer_name == 'sentence-transformers/all-MiniLM-L6-v2'

# -------------------------------------------------------------------------------------------------
# Train/Val DataLoader Creation Tests
# -------------------------------------------------------------------------------------------------

class TestDataLoaderCreation:
    '''Test suite for train and validation dataloader creation.'''

    @pytest.fixture
    def mock_datamodule_with_datasets(self, tmp_path):
        '''Create a NAICSDataModule with mock datasets for testing DataLoader creation.'''
        import polars as pl

        from naics_embedder.text_model.dataloader.datamodule import (
            NAICSDataModule,
            NAICSMapDataset,
        )

        # Create descriptions
        desc_data = {
            'index': [0, 1, 2],
            'code': ['311111', '311112', '321111'],
            'level': [6, 6, 6],
            'title': ['Dog Food', 'Cat Food', 'Sawmills'],
            'description': ['Make dog food', 'Make cat food', 'Cut wood'],
            'excluded': ['', '', ''],
            'examples': ['', '', ''],
            'excluded_codes': [None, None, None],
        }
        desc_df = pl.DataFrame(desc_data)
        desc_path = tmp_path / 'descriptions.parquet'
        desc_df.write_parquet(desc_path)

        # Create triplets
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()
        anchor_dir = triplets_dir / 'anchor=0'
        anchor_dir.mkdir()

        triplet_data = {
            'anchor_idx': [0],
            'anchor_code': ['311111'],
            'anchor_level': [6],
            'positive_idx': [1],
            'positive_code': ['311112'],
            'positive_level': [6],
            'negative_idx': [2],
            'negative_code': ['321111'],
            'negative_level': [6],
            'relation_margin': [0],
            'distance_margin': [4],
            'positive_relation': [1],
            'positive_distance': [2],
            'negative_relation': [3],
            'negative_distance': [8],
        }
        triplet_df = pl.DataFrame(triplet_data)
        triplet_path = anchor_dir / 'part0.parquet'
        triplet_df.write_parquet(triplet_path)

        datamodule = NAICSDataModule(
            descriptions_path=str(desc_path),
            triplets_path=str(triplets_dir),
            batch_size=2,
            num_workers=0,
        )

        # Create mock token cache and triplet rows
        channels = ['title', 'description', 'excluded', 'examples']

        def make_embedding():
            return {
                ch: {
                    'input_ids': torch.randint(0, 1000, (128, )),
                    'attention_mask': torch.ones(128, dtype=torch.long),
                    'present': True,
                }
                for ch in channels
            }

        mock_token_cache = {i: {'code': f'{i:06d}', **make_embedding()} for i in range(3)}
        mock_triplet_rows = [
            {
                'anchor_idx': 0,
                'anchor_code': '311111',
                'positive_idx': 1,
                'positive_code': '311112',
                'positive_level': 6,
                'stratum_id': 0,
                'stratum_wgt': 1.0,
                'negatives': [
                    {
                        'negative_idx': 2,
                        'negative_code': '321111',
                        'relation_margin': 0,
                        'distance_margin': 4,
                    }
                ],
            },
        ]

        # Manually set datasets (bypassing setup())
        datamodule.train_dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)
        datamodule.val_dataset = NAICSMapDataset(mock_triplet_rows, mock_token_cache)

        return datamodule

    def test_train_dataloader_returns_dataloader(self, mock_datamodule_with_datasets):
        '''Test that train_dataloader returns a DataLoader instance.'''
        from torch.utils.data import DataLoader

        train_loader = mock_datamodule_with_datasets.train_dataloader()

        assert isinstance(train_loader, DataLoader)

    def test_train_dataloader_batch_size(self, mock_datamodule_with_datasets):
        '''Test that train_dataloader uses correct batch size.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()

        assert train_loader.batch_size == 2

    def test_train_dataloader_num_workers(self, mock_datamodule_with_datasets):
        '''Test that train_dataloader uses correct num_workers.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()

        assert train_loader.num_workers == 0

    def test_train_dataloader_has_shuffle_enabled(self, mock_datamodule_with_datasets):
        '''Test that train_dataloader has shuffle=True for map-style dataset.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()

        # DataLoader with shuffle=True uses a RandomSampler
        from torch.utils.data import RandomSampler

        assert isinstance(train_loader.sampler, RandomSampler)

    def test_val_dataloader_returns_dataloader(self, mock_datamodule_with_datasets):
        '''Test that val_dataloader returns a DataLoader instance.'''
        from torch.utils.data import DataLoader

        val_loader = mock_datamodule_with_datasets.val_dataloader()

        assert isinstance(val_loader, DataLoader)

    def test_val_dataloader_batch_size(self, mock_datamodule_with_datasets):
        '''Test that val_dataloader uses correct batch size.'''
        val_loader = mock_datamodule_with_datasets.val_dataloader()

        assert val_loader.batch_size == 2

    def test_val_dataloader_num_workers(self, mock_datamodule_with_datasets):
        '''Test that val_dataloader uses correct num_workers.'''
        val_loader = mock_datamodule_with_datasets.val_dataloader()

        assert val_loader.num_workers == 0

    def test_val_dataloader_has_shuffle_disabled(self, mock_datamodule_with_datasets):
        '''Test that val_dataloader has shuffle=False.'''
        val_loader = mock_datamodule_with_datasets.val_dataloader()

        # DataLoader with shuffle=False uses a SequentialSampler
        from torch.utils.data import SequentialSampler

        assert isinstance(val_loader.sampler, SequentialSampler)

    def test_train_val_dataloaders_are_different(self, mock_datamodule_with_datasets):
        '''Test that train and val dataloaders are distinct.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()
        val_loader = mock_datamodule_with_datasets.val_dataloader()

        # They should be different objects
        assert train_loader is not val_loader
        # They should use different datasets
        assert train_loader.dataset is not val_loader.dataset

    def test_persistent_workers_disabled_when_zero_workers(self, mock_datamodule_with_datasets):
        '''Test persistent_workers is False when num_workers=0.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()

        # persistent_workers should be False since num_workers=0
        assert train_loader.persistent_workers is False

    def test_train_dataloader_keeps_precomputed_dataset(self, mock_datamodule_with_datasets):
        '''The default precomputed path (no set_epoch) loads its dataset unchanged.'''
        train_loader = mock_datamodule_with_datasets.train_dataloader()

        assert train_loader.dataset is mock_datamodule_with_datasets.train_dataset

    def test_train_dataloader_raises_if_dataset_none(self, tmp_path):
        '''Test that train_dataloader raises RuntimeError if setup() not called.'''
        import polars as pl

        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        # Create minimal parquet file
        desc_data = {
            'index': [0],
            'code': ['311111'],
            'level': [6],
            'title': ['Test'],
            'description': [''],
            'excluded': [''],
            'examples': [''],
            'excluded_codes': [None]
        }
        desc_df = pl.DataFrame(desc_data)
        desc_path = tmp_path / 'desc.parquet'
        desc_df.write_parquet(desc_path)

        datamodule = NAICSDataModule(
            descriptions_path=str(desc_path),
            triplets_path=str(tmp_path),
            batch_size=2,
            num_workers=0,
        )

        with pytest.raises(RuntimeError, match='train_dataset is None'):
            datamodule.train_dataloader()

# -------------------------------------------------------------------------------------------------
# Train Epoch Propagation Tests
# -------------------------------------------------------------------------------------------------

class _InjectedDataModule(NAICSDataModule):
    '''NAICSDataModule with injected datasets: skips building the parquet/tokenizer caches.'''

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

def _sampled_epochs(batch: Dict[str, Any]) -> Set[int]:
    '''Epochs that the items of a collated EpochRecordingDataset batch were sampled with.'''
    return {int(code) for code in batch['anchor_code']}

class _EpochRecorder(pyl.LightningModule):
    '''Records, per trainer epoch, the dataset epochs that train and val batches were sampled at.'''

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.train_epochs: Dict[int, Set[int]] = defaultdict(set)
        self.val_epochs: Dict[int, Set[int]] = defaultdict(set)

    def training_step(self, batch, batch_idx):
        self.train_epochs[self.current_epoch] |= _sampled_epochs(batch)
        return self.weight.sum()

    def validation_step(self, batch, batch_idx):
        self.val_epochs[self.current_epoch] |= _sampled_epochs(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

def _epoch_aware_datamodule(tmp_path, num_workers: int) -> NAICSDataModule:
    datamodule = _InjectedDataModule(
        descriptions_path=str(tmp_path / 'descriptions.parquet'),
        triplets_path=str(tmp_path / 'triplets'),
        batch_size=2,
        num_workers=num_workers,
    )
    # More items than one batch, so worker prefetch is in flight when an epoch is cut short
    datamodule.train_dataset = EpochRecordingDataset(n_items=16)
    datamodule.val_dataset = EpochRecordingDataset(n_items=16)
    return datamodule

def _fit(tmp_path, datamodule, max_epochs, ckpt_path=None, extra_callbacks=(), **trainer_kwargs):
    '''Fit an _EpochRecorder with a real Trainer; returns (trainer, model).'''
    options: Dict[str, Any] = {
        'max_epochs': max_epochs,
        'limit_train_batches': 1,
        'limit_val_batches': 0,
        'num_sanity_val_steps': 0,
        'accelerator': 'cpu',
        'callbacks': [TrainDatasetEpochCallback(), *extra_callbacks],
        'logger': False,
        'enable_checkpointing': bool(extra_callbacks),
        'enable_progress_bar': False,
        'enable_model_summary': False,
        'default_root_dir': tmp_path,
    }
    options.update(trainer_kwargs)
    model = _EpochRecorder()
    trainer = pyl.Trainer(**options)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, model

@pytest.mark.unit
def test_train_dataset_epoch_advances_under_real_trainer(tmp_path):
    '''Lightning never calls LightningDataModule epoch hooks; the epoch must still arrive.'''
    datamodule = _epoch_aware_datamodule(tmp_path, num_workers=0)

    _, model = _fit(tmp_path, datamodule, max_epochs=2)

    assert datamodule.train_dataset.epoch == 1
    assert model.train_epochs == {0: {0}, 1: {1}}

@pytest.mark.unit
def test_persistent_workers_sample_with_current_epoch(tmp_path):
    '''Persistent workers keep their own dataset copies; each new epoch must still reach them.'''
    datamodule = _epoch_aware_datamodule(tmp_path, num_workers=2)

    trainer, model = _fit(tmp_path, datamodule, max_epochs=3)

    assert trainer.train_dataloader.persistent_workers
    assert model.train_epochs == {0: {0}, 1: {1}, 2: {2}}

@pytest.mark.unit
def test_validation_pools_stay_at_epoch_zero(tmp_path):
    '''val/contrastive_loss drives checkpointing, so validation must not follow the epoch.'''
    datamodule = _epoch_aware_datamodule(tmp_path, num_workers=2)

    _, model = _fit(tmp_path, datamodule, max_epochs=3, limit_val_batches=1)

    assert model.train_epochs == {0: {0}, 1: {1}, 2: {2}}
    assert model.val_epochs == {0: {0}, 1: {0}, 2: {0}}
    assert datamodule.val_dataset.epoch == 0

@pytest.mark.unit
def test_resume_with_workers_samples_with_restored_epoch(tmp_path):
    '''Workers spawn and prefetch in setup_data(), before any epoch hook of the resumed run.'''
    trainer, _ = _fit(tmp_path, _epoch_aware_datamodule(tmp_path, num_workers=2), max_epochs=2)
    ckpt_path = tmp_path / 'epoch_end.ckpt'
    trainer.save_checkpoint(ckpt_path)

    datamodule = _epoch_aware_datamodule(tmp_path, num_workers=2)
    _, model = _fit(tmp_path, datamodule, max_epochs=3, ckpt_path=ckpt_path)

    assert model.train_epochs == {2: {2}}

@pytest.mark.unit
def test_mid_epoch_resume_samples_with_restored_epoch(tmp_path):
    '''Lightning skips on_train_epoch_start entirely when it resumes mid-epoch.'''
    step_checkpoints = ModelCheckpoint(
        dirpath=tmp_path / 'steps', filename='{epoch}-{step}', every_n_train_steps=1, save_top_k=-1
    )
    _fit(
        tmp_path,
        _epoch_aware_datamodule(tmp_path, num_workers=0),
        max_epochs=2,
        limit_train_batches=2,
        extra_callbacks=[step_checkpoints],
    )

    datamodule = _epoch_aware_datamodule(tmp_path, num_workers=0)
    _, model = _fit(
        tmp_path,
        datamodule,
        max_epochs=2,
        limit_train_batches=2,
        ckpt_path=tmp_path / 'steps' / 'epoch=1-step=3.ckpt',  # after 1 of 2 batches of epoch 1
    )

    assert model.train_epochs == {1: {1}}

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
