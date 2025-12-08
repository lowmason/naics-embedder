'''
Unit tests for NAICSDataModule and collate_fn.

Tests cover:
- collate_fn batching and padding logic
- Multi-level supervision expansion
- Sampling metadata accumulation
- NAICSMapDataset indexing and __getitem__
'''

from unittest.mock import MagicMock, patch

import pytest
import torch

from naics_embedder.text_model.dataloader.datamodule import NAICSMapDataset, collate_fn

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
# Positive Level Tests
# -------------------------------------------------------------------------------------------------

def test_collate_extracts_positive_level(make_batch_item):
    '''collate_fn should extract positive_level from batch items.'''
    batch = [make_batch_item('311111', '31111', ['222'])]
    # Add positive_level to the item
    batch[0]['positive_level'] = 5

    result = collate_fn(batch)

    assert 'positive_levels' in result
    assert result['positive_levels'] == [5]

def test_collate_infers_positive_level_from_code_length(make_batch_item):
    '''collate_fn should infer positive_level from positive_code length if not present.'''
    batch = [make_batch_item('311111', '3111', ['222'])]
    # positive_level not explicitly set, should use len(positive_code)

    result = collate_fn(batch)

    assert 'positive_levels' in result
    # Default is len(positive_code) = 4
    assert result['positive_levels'] == [4]

def test_collate_multiple_positive_levels(make_batch_item):
    '''collate_fn should track positive_levels for multiple items.'''
    batch = [
        make_batch_item('311111', '31111', ['222']),
        make_batch_item('321111', '3211', ['333']),
    ]
    batch[0]['positive_level'] = 5
    batch[1]['positive_level'] = 4

    result = collate_fn(batch)

    assert result['positive_levels'] == [5, 4]

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
# NAICSMapDataset Tests
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def mock_token_cache():
    '''Create a mock token cache with embeddings for indices 0-4.'''
    channels = ['title', 'description', 'excluded', 'examples']

    def make_embedding(idx):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128, dtype=torch.long),
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
                {'negative_idx': 2, 'negative_code': '000002', 'relation_margin': 0, 'distance_margin': 4},
                {'negative_idx': 3, 'negative_code': '000003', 'relation_margin': 0, 'distance_margin': 4},
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
                {'negative_idx': 4, 'negative_code': '000004', 'relation_margin': 0, 'distance_margin': 4},
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
                {'negative_idx': 2, 'negative_code': '000002', 'relation_margin': 0, 'distance_margin': 4},
            ],
            'sampling_metadata': {'strategy': 'sans_static', 'sampled_near': 1},
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
                    'input_ids': torch.randint(0, 1000, (128,)),
                    'attention_mask': torch.ones(128, dtype=torch.long),
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

    def test_train_dataloader_raises_if_dataset_none(self, tmp_path):
        '''Test that train_dataloader raises RuntimeError if setup() not called.'''
        import polars as pl

        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        # Create minimal parquet file
        desc_data = {'index': [0], 'code': ['311111'], 'level': [6], 'title': ['Test'],
                     'description': [''], 'excluded': [''], 'examples': [''],
                     'excluded_codes': [None]}
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
