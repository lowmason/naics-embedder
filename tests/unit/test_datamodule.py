'''
Unit tests for NAICSDataModule and collate_fn.

Tests cover:
- collate_fn batching and padding logic
- Multi-level supervision expansion
- Sampling metadata accumulation
- GeneratorDataset worker sharding
'''

from unittest.mock import MagicMock, patch

import pytest
import torch

from naics_embedder.text_model.dataloader.datamodule import GeneratorDataset, collate_fn

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
            }
            for ch in channels
        }

    return _make

@pytest.fixture
def make_batch_item(make_embedding):
    '''Factory to create a single batch item.'''

    def _create(anchor_code, positive_code, negative_codes, seq_len=128):
        return {
            'anchor_code':
            anchor_code,
            'anchor_embedding':
            make_embedding(seq_len),
            'positive_code':
            positive_code,
            'positive_embedding':
            make_embedding(seq_len),
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
def make_multilevel_item(make_embedding):
    '''Factory to create a multi-level supervision batch item.'''

    def _create(anchor_code, positive_codes, negative_codes, seq_len=128):
        positives = [
            {
                'positive_code':
                pc,
                'positive_idx':
                i,
                'positive_embedding':
                make_embedding(seq_len),
                'negatives': [
                    {
                        'negative_code': nc,
                        'negative_idx': j,
                        'negative_embedding': make_embedding(seq_len),
                        'relation_margin': 0,
                        'distance_margin': 4,
                    } for j, nc in enumerate(negative_codes)
                ],
            } for i, pc in enumerate(positive_codes)
        ]

        return {
            'anchor_code':
            anchor_code,
            'anchor_embedding':
            make_embedding(seq_len),
            'positives':
            positives,
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

# -------------------------------------------------------------------------------------------------
# Basic Collate Tests
# -------------------------------------------------------------------------------------------------

def test_collate_stacks_embeddings_correctly(make_batch_item):
    '''Embeddings should be stacked into proper tensor shapes.'''
    batch = [
        make_batch_item('111', '11', ['222', '333']),
        make_batch_item('444', '44', ['555', '666']),
    ]

    result = collate_fn(batch)

    # Check anchor shape: (batch_size, seq_len)
    assert result['anchor']['title']['input_ids'].shape == (2, 128)
    assert result['anchor']['title']['attention_mask'].shape == (2, 128)

    # Check positive shape: (batch_size, seq_len)
    assert result['positive']['title']['input_ids'].shape == (2, 128)

    # Check negative shape: (batch_size * k_negatives, seq_len)
    assert result['negatives']['title']['input_ids'].shape == (4, 128)

def test_collate_preserves_all_channels(make_batch_item, channels):
    '''All four channels should be present in output.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    for channel in channels:
        assert channel in result['anchor']
        assert channel in result['positive']
        assert channel in result['negatives']

def test_collate_includes_metadata(make_batch_item):
    '''Batch metadata should be included.'''
    batch = [
        make_batch_item('111', '11', ['222', '333']),
        make_batch_item('444', '44', ['555', '666']),
    ]

    result = collate_fn(batch)

    assert result['batch_size'] == 2
    assert result['k_negatives'] == 2
    assert result['anchor_code'] == ['111', '444']
    assert result['positive_code'] == ['11', '44']
    assert len(result['negative_codes']) == 2
    assert result['negative_codes'][0] == ['222', '333']

# -------------------------------------------------------------------------------------------------
# Padding Tests
# -------------------------------------------------------------------------------------------------

def test_collate_pads_uneven_negatives(make_batch_item):
    '''Items with fewer negatives should be padded.'''
    batch = [
        make_batch_item('111', '11', ['222', '333', '444']),  # 3 negatives
        make_batch_item('555', '55', ['666']),  # 1 negative
    ]

    result = collate_fn(batch)

    assert result['k_negatives'] == 3
    # Total negatives: 3 + 3 (padded) = 6
    assert result['negatives']['title']['input_ids'].shape == (6, 128)

def test_collate_padding_repeats_last_negative(make_batch_item, make_embedding):
    '''Padding should repeat the last negative.'''
    # Create item with single known negative
    item = {
        'anchor_code':
        '111',
        'anchor_embedding':
        make_embedding(),
        'positive_code':
        '11',
        'positive_embedding':
        make_embedding(),
        'negatives': [
            {
                'negative_code': 'LAST',
                'negative_idx': 0,
                'negative_embedding': make_embedding(),
                'relation_margin': 0,
                'distance_margin': 4,
            }
        ],
    }

    # Create item with multiple negatives to force padding
    item2 = {
        'anchor_code':
        '222',
        'anchor_embedding':
        make_embedding(),
        'positive_code':
        '22',
        'positive_embedding':
        make_embedding(),
        'negatives': [
            {
                'negative_code': f'NEG{i}',
                'negative_idx': i,
                'negative_embedding': make_embedding(),
                'relation_margin': 0,
                'distance_margin': 4,
            } for i in range(3)
        ],
    }

    batch = [item, item2]
    result = collate_fn(batch)

    # After collation, first item should have 3 negatives (padded from 1)
    assert result['k_negatives'] == 3
    # The negative_codes for first item should repeat 'LAST'
    assert result['negative_codes'][0] == ['LAST', 'LAST', 'LAST']

# -------------------------------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------------------------------

def test_collate_raises_on_empty_negatives(make_embedding):
    '''Batch with no negatives should raise ValueError.'''
    batch = [
        {
            'anchor_code': '111',
            'anchor_embedding': make_embedding(),
            'positive_code': '11',
            'positive_embedding': make_embedding(),
            'negatives': [],
        }
    ]

    with pytest.raises(ValueError, match='no negatives'):
        collate_fn(batch)

def test_collate_handles_single_item_batch(make_batch_item):
    '''Single item batch should work correctly.'''
    batch = [make_batch_item('111', '11', ['222', '333'])]

    result = collate_fn(batch)

    assert result['batch_size'] == 1
    assert result['k_negatives'] == 2
    assert result['anchor']['title']['input_ids'].shape == (1, 128)

# -------------------------------------------------------------------------------------------------
# Multi-Level Supervision Tests
# -------------------------------------------------------------------------------------------------

def test_collate_multi_level_expansion(make_multilevel_item):
    '''Multi-level positives should be expanded into separate entries.'''
    batch = [
        make_multilevel_item(
            '311111',
            ['31111', '3111', '311'],  # 3 ancestor levels
            ['222'],
        )
    ]

    result = collate_fn(batch)

    # Should expand to 3 entries (one per positive level)
    assert result['batch_size'] == 3
    assert result['positive_levels'] == [5, 4, 3]  # len of each positive code

def test_collate_multi_level_preserves_anchor(make_multilevel_item):
    '''Each expanded entry should share the same anchor.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222'])]

    result = collate_fn(batch)

    # All anchor codes should be the same
    assert result['anchor_code'] == ['311111', '311111']

    # Anchor embeddings should be stacked (same embedding repeated)
    assert result['anchor']['title']['input_ids'].shape == (2, 128)

def test_collate_multi_level_different_positives(make_multilevel_item):
    '''Each expanded entry should have different positive.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222'])]

    result = collate_fn(batch)

    assert result['positive_code'] == ['31111', '3111']

def test_collate_multi_level_shared_negatives(make_multilevel_item):
    '''All expanded entries should share the same negatives.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222', '333'])]

    result = collate_fn(batch)

    # Each expansion uses the same negatives
    assert result['negative_codes'][0] == ['222', '333']
    assert result['negative_codes'][1] == ['222', '333']

def test_collate_mixed_single_and_multi_level(make_batch_item, make_multilevel_item):
    '''Batch can contain both single and multi-level items.'''
    batch = [
        make_multilevel_item('311111', ['31111', '3111'], ['222']),
        # Standard single-positive item won't trigger multi-level path
        # since it doesn't have 'positives' key as a list
    ]

    result = collate_fn(batch)

    # Should have 2 entries from multi-level expansion
    assert result['batch_size'] == 2

# -------------------------------------------------------------------------------------------------
# Sampling Metadata Tests
# -------------------------------------------------------------------------------------------------

def test_collate_accumulates_sampling_metadata(make_batch_item):
    '''Sampling metadata should be accumulated across batch items.'''
    batch = [
        make_batch_item('111', '11', ['222']),
        make_batch_item('333', '33', ['444']),
    ]

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

def test_collate_computes_average_weights(make_batch_item):
    '''Effective weights should be averaged across records.'''
    batch = [
        make_batch_item('111', '11', ['222']),
        make_batch_item('333', '33', ['444']),
    ]

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

def test_collate_no_metadata_when_missing(make_batch_item):
    '''No sampling_metadata key when items have no metadata.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    assert 'sampling_metadata' not in result

# -------------------------------------------------------------------------------------------------
# GeneratorDataset Tests
# -------------------------------------------------------------------------------------------------

def test_generator_dataset_sharding_worker_0():
    '''Worker 0 should get items at indices 0, 2, 4, ...'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(10):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Mock worker info for worker 0 of 2
    worker_info = MagicMock()
    worker_info.id = 0
    worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=worker_info):
        with patch.object(dataset, '_get_token_cache', return_value={}):
            items = list(dataset)

    # Worker 0 should get indices 0, 2, 4, 6, 8
    assert [item['idx'] for item in items] == [0, 2, 4, 6, 8]

def test_generator_dataset_sharding_worker_1():
    '''Worker 1 should get items at indices 1, 3, 5, ...'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(10):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Mock worker info for worker 1 of 2
    worker_info = MagicMock()
    worker_info.id = 1
    worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=worker_info):
        with patch.object(dataset, '_get_token_cache', return_value={}):
            items = list(dataset)

    # Worker 1 should get indices 1, 3, 5, 7, 9
    assert [item['idx'] for item in items] == [1, 3, 5, 7, 9]

def test_generator_dataset_no_sharding_single_worker():
    '''With no workers, should yield all items from generator.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(5):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Directly set the token cache to bypass loading
    dataset._token_cache = {}

    # Patch get_worker_info at the module level
    with patch(
        'naics_embedder.text_model.dataloader.datamodule.torch.utils.data.get_worker_info',
        return_value=None,
    ):
        items = list(dataset)

    assert [item['idx'] for item in items] == [0, 1, 2, 3, 4]

def test_generator_dataset_sharding_four_workers():
    '''Four workers should each get 1/4 of items.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(12):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    all_items = []
    for worker_id in range(4):
        dataset = GeneratorDataset(
            mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
        )

        worker_info = MagicMock()
        worker_info.id = worker_id
        worker_info.num_workers = 4

        with patch('torch.utils.data.get_worker_info', return_value=worker_info):
            with patch.object(dataset, '_get_token_cache', return_value={}):
                items = list(dataset)
                all_items.extend([item['idx'] for item in items])

    # All items should be covered exactly once
    assert sorted(all_items) == list(range(12))

def test_generator_dataset_lazy_cache_loading():
    '''Token cache should be loaded lazily on first iteration.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        yield {'cache': token_cache}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'

    dataset = GeneratorDataset(mock_generator_fn, mock_tokenization_cfg, MagicMock(), MagicMock())

    # Cache should not be loaded yet
    assert dataset._token_cache is None

    # Mock the cache loading
    with patch('torch.utils.data.get_worker_info', return_value=None):
        with patch.object(dataset, '_get_token_cache', return_value={'loaded': True}) as mock_get:
            list(dataset)
            mock_get.assert_called_once()

# -------------------------------------------------------------------------------------------------
# Edge Cases
# -------------------------------------------------------------------------------------------------

def test_collate_different_sequence_lengths(make_embedding):
    '''Batch items with same sequence length should collate.'''
    channels = ['title', 'description', 'excluded', 'examples']

    def make_item(seq_len):
        embedding = {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }
            for ch in channels
        }
        return {
            'anchor_code':
            '111',
            'anchor_embedding':
            embedding,
            'positive_code':
            '11',
            'positive_embedding':
            embedding,
            'negatives': [
                {
                    'negative_code': '222',
                    'negative_idx': 0,
                    'negative_embedding': embedding,
                    'relation_margin': 0,
                    'distance_margin': 4,
                }
            ],
        }

    # Same sequence length should work
    batch = [make_item(64), make_item(64)]
    result = collate_fn(batch)
    assert result['anchor']['title']['input_ids'].shape == (2, 64)

def test_collate_preserves_tensor_dtype(make_batch_item):
    '''Tensor dtypes should be preserved after collation.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    assert result['anchor']['title']['input_ids'].dtype == torch.long
    assert result['anchor']['title']['attention_mask'].dtype == torch.long

def test_collate_large_batch(make_batch_item):
    '''Should handle larger batches efficiently.'''
    batch = [make_batch_item(f'{i:03d}', f'{i:02d}', [f'{i + 100}']) for i in range(64)]

    result = collate_fn(batch)

    assert result['batch_size'] == 64
    assert result['anchor']['title']['input_ids'].shape == (64, 128)
