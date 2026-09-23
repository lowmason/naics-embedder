# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from naics_embedder.data.positive_sampling import create_positive_sampler
from naics_embedder.text_model.dataloader.difficulty_sampler import select_by_difficulty
from naics_embedder.text_model.dataloader.streaming_dataset import (
    _get_multi_epoch_cache_path,
    _load_distance_matrix,
    _load_excluded_codes,
    _load_negative_candidates,
    _sample_negatives_phase1,
    build_multi_epoch_triplets,
)
from naics_embedder.utils.config import SamplingConfig, StreamingConfig, TokenizationConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    '''Collate function to batch triplets for training. Each batch item represents a single positive.'''
    channels = ['title', 'description', 'excluded', 'examples']

    # Find maximum number of negatives in batch and pad shorter lists
    max_negatives = max(len(item['negatives']) for item in batch) if batch else 0
    if max_negatives == 0:
        raise ValueError('Batch contains items with no negatives - cannot create training batch')

    for item in batch:
        if len(item['negatives']) < max_negatives:
            # Pad by repeating the last negative
            last_negative = item['negatives'][-1] if item['negatives'] else None
            if last_negative is None:
                anchor_code = item.get('anchor_code', 'unknown')
                raise ValueError(f'Item has no negatives to pad from: {anchor_code}')
            padding_needed = max_negatives - len(item['negatives'])
            item['negatives'].extend([last_negative] * padding_needed)

    sampling_accumulator: Optional[Dict[str, Any]] = None

    # Initialize batch dictionaries
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}

    # Collect codes and indices for evaluation tracking
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    positive_levels: List[int] = []

    # Process each channel
    for channel in channels:
        anchor_ids = []
        anchor_masks = []
        positive_ids = []
        positive_masks = []

        # Collect anchor and positive for this channel
        for item in batch:
            anchor_ids.append(item['anchor_embedding'][channel]['input_ids'])
            anchor_masks.append(item['anchor_embedding'][channel]['attention_mask'])
            positive_ids.append(item['positive_embedding'][channel]['input_ids'])
            positive_masks.append(item['positive_embedding'][channel]['attention_mask'])

        # Stack anchor
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)

        # Stack positive
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)

        # Collect all negatives for this channel
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_dict in item['negatives']:
                all_neg_ids.append(neg_dict['negative_embedding'][channel]['input_ids'])
                all_neg_masks.append(neg_dict['negative_embedding'][channel]['attention_mask'])

        # Stack negatives
        negatives_batch[channel]['input_ids'] = torch.stack(all_neg_ids)
        negatives_batch[channel]['attention_mask'] = torch.stack(all_neg_masks)

    # Extract codes from batch items
    for item in batch:
        anchor_codes.append(item['anchor_code'])
        positive_codes.append(item['positive_code'])
        negative_codes.append([neg_dict['negative_code'] for neg_dict in item['negatives']])
        positive_levels.append(item.get('positive_level', len(item['positive_code'])))

        metadata = item.get('sampling_metadata')
        if metadata:
            if sampling_accumulator is None:
                sampling_accumulator = {
                    'strategy': metadata.get('strategy', 'unknown'),
                    'candidates_near': 0,
                    'candidates_far': 0,
                    'sampled_near': 0,
                    'sampled_far': 0,
                    'effective_near_weight_sum': 0.0,
                    'effective_far_weight_sum': 0.0,
                    'records': 0,
                }

            sampling_accumulator['candidates_near'] += metadata.get('candidates_near', 0)
            sampling_accumulator['candidates_far'] += metadata.get('candidates_far', 0)
            sampling_accumulator['sampled_near'] += metadata.get('sampled_near', 0)
            sampling_accumulator['sampled_far'] += metadata.get('sampled_far', 0)
            sampling_accumulator['effective_near_weight_sum'] += metadata.get(
                'effective_near_weight', 0.0
            )
            sampling_accumulator['effective_far_weight_sum'] += metadata.get(
                'effective_far_weight', 0.0
            )
            sampling_accumulator['records'] += 1

    result = {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': max_negatives,
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes,
    }

    # Add positive_levels for multi-level supervision tracking
    result['positive_levels'] = positive_levels

    if sampling_accumulator and sampling_accumulator['records'] > 0:
        records = sampling_accumulator.pop('records')
        effective_near_avg = sampling_accumulator.pop('effective_near_weight_sum') / records
        effective_far_avg = sampling_accumulator.pop('effective_far_weight_sum') / records
        sampling_accumulator['avg_effective_near_weight'] = effective_near_avg
        sampling_accumulator['avg_effective_far_weight'] = effective_far_avg
        result['sampling_metadata'] = sampling_accumulator

    # Include all_candidates for Phase 2+ hard negative mining if present
    # This contains oversampled negatives from Phase 1 that weren't selected
    # by the difficulty curriculum but are available for HNM
    if batch and batch[0].get('all_candidates'):
        all_candidates_batch = {channel: {} for channel in channels}

        # Find max candidates across batch
        max_candidates = max(len(item.get('all_candidates', [])) for item in batch)

        # Pad all_candidates lists to same length
        for item in batch:
            candidates = item.get('all_candidates', [])
            if len(candidates) < max_candidates and candidates:
                last_candidate = candidates[-1]
                padding_needed = max_candidates - len(candidates)
                candidates.extend([last_candidate] * padding_needed)

        # Collect candidate embeddings for each channel
        for channel in channels:
            all_cand_ids = []
            all_cand_masks = []
            for item in batch:
                for cand_dict in item.get('all_candidates', []):
                    all_cand_ids.append(cand_dict['negative_embedding'][channel]['input_ids'])
                    all_cand_masks.append(cand_dict['negative_embedding'][channel]['attention_mask'])

            if all_cand_ids:
                all_candidates_batch[channel]['input_ids'] = torch.stack(all_cand_ids)
                all_candidates_batch[channel]['attention_mask'] = torch.stack(all_cand_masks)

        result['all_candidates'] = all_candidates_batch
        result['k_candidates'] = max_candidates

        # Also collect all_candidate codes
        all_candidate_codes = []
        for item in batch:
            all_candidate_codes.append(
                [cand['negative_code'] for cand in item.get('all_candidates', [])]
            )
        result['all_candidate_codes'] = all_candidate_codes

    return result

# -------------------------------------------------------------------------------------------------
# Map-style Dataset for pre-sampled triplets
# -------------------------------------------------------------------------------------------------

class NAICSMapDataset(Dataset):
    '''Map-style dataset for pre-sampled triplets with tokenized embeddings.'''

    def __init__(self, triplet_rows: List[Dict[str, Any]], token_cache: Dict[int, Dict[str, Any]]):
        '''
        Initialize the map-style dataset.

        Args:
            triplet_rows: List of triplet dictionaries with anchor/positive/negative info
            token_cache: Dictionary mapping index to tokenized embeddings
        '''
        self.triplet_rows = triplet_rows
        self.token_cache = token_cache

    def __len__(self) -> int:
        return len(self.triplet_rows)

    def _extract_embedding(self, idx: int) -> Optional[Dict[str, Any]]:
        '''Extract embedding from token cache, excluding code field.'''
        try:
            return {k: v for k, v in self.token_cache[idx].items() if k != 'code'}
        except KeyError:
            logger.warning(f'Missing token_cache for index {idx}')
            return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        '''Get a single triplet item by index.'''
        row = self.triplet_rows[idx]

        anchor_idx = int(row['anchor_idx'])
        positive_idx = int(row['positive_idx'])

        anchor_embedding = self._extract_embedding(anchor_idx)
        if anchor_embedding is None:
            # Return a placeholder that will be filtered by collate_fn
            return None

        positive_embedding = self._extract_embedding(positive_idx)
        if positive_embedding is None:
            return None

        negative_entries = []
        for neg in row.get('negatives', []):
            neg_embedding = self._extract_embedding(int(neg['negative_idx']))
            if neg_embedding is None:
                continue

            negative_entries.append(
                {
                    'negative_idx': int(neg['negative_idx']),
                    'negative_code': neg['negative_code'],
                    'negative_embedding': neg_embedding,
                    'relation_margin': neg.get('relation_margin', 0),
                    'distance_margin': neg.get('distance_margin', 0),
                    'explicit_exclusion': neg.get('explicit_exclusion', False),
                }
            )

        if not negative_entries:
            return None

        result: Dict[str, Any] = {
            'anchor_idx': anchor_idx,
            'anchor_code': row['anchor_code'],
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': row['positive_code'],
            'positive_level': row.get('positive_level', len(row['positive_code'])),
            'stratum_id': row.get('stratum_id', 0),
            'stratum_wgt': row.get('stratum_wgt', 1.0),
            'positive_embedding': positive_embedding,
            'negatives': negative_entries,
        }

        sampling_metadata = row.get('sampling_metadata')
        if sampling_metadata:
            result['sampling_metadata'] = sampling_metadata

        return result


# -------------------------------------------------------------------------------------------------
# Phase 1 Map Dataset with On-the-Fly Negative Sampling
# -------------------------------------------------------------------------------------------------

class Phase1MapDataset(Dataset):
    """
    Map-style dataset with on-the-fly Phase 1 negative sampling and difficulty curriculum.

    Instead of pre-computing all negatives for multiple epochs, this dataset:
    1. Pre-computes (anchor, positive) pairs as the fixed index space
    2. Samples negatives on-the-fly in __getitem__() with epoch-aware seeds
    3. Applies difficulty curriculum: easy -> semi-hard -> hard across Phase 1
    4. Provides oversampled candidates for Phase 2+ hard negative mining

    Attributes:
        cfg: Streaming configuration
        sampling_cfg: Sampling strategy configuration
        token_cache: Pre-computed tokenized embeddings
        phase1_end_epoch: Epoch at which Phase 1 ends
        epoch: Current training epoch (updated via set_epoch)
    """

    def __init__(
        self,
        cfg: StreamingConfig,
        sampling_cfg: SamplingConfig,
        token_cache: Dict[int, Dict[str, Any]],
        phase1_end_epoch: int,
    ):
        """
        Initialize the Phase 1 map dataset.

        Args:
            cfg: Streaming configuration with sampling parameters
            sampling_cfg: Sampling strategy configuration
            token_cache: Dictionary mapping index to tokenized embeddings
            phase1_end_epoch: Epoch at which Phase 1 ends (for curriculum progress)
        """
        self.cfg = cfg
        self.sampling_cfg = sampling_cfg
        self.token_cache = token_cache
        self.phase1_end_epoch = max(phase1_end_epoch, 1)
        self.epoch = 0

        # Load code/index mappings
        logger.info('Phase1MapDataset: Loading code/index mappings...')
        code_to_idx_raw = get_indices_codes('code_to_idx')
        idx_to_code_raw = get_indices_codes('idx_to_code')
        assert isinstance(code_to_idx_raw, dict), 'code_to_idx must be a dict'
        assert isinstance(idx_to_code_raw, dict), 'idx_to_code must be a dict'
        self.code_to_idx: Dict[str, int] = code_to_idx_raw  # type: ignore
        self.idx_to_code: Dict[int, str] = idx_to_code_raw  # type: ignore

        # Load tree distance matrix for Phase 1 sampling and difficulty bucketing
        logger.info('Phase1MapDataset: Loading tree distance matrix...')
        self.distance_lookup = _load_distance_matrix(
            cfg.distance_matrix_parquet, self.code_to_idx, self.idx_to_code
        )

        # Load excluded codes for Phase 1 sampling
        logger.info('Phase1MapDataset: Loading excluded codes...')
        self.excluded_map = _load_excluded_codes(cfg.descriptions_parquet, self.code_to_idx)

        # Create positive sampler and build (anchor, positive) pair index
        logger.info('Phase1MapDataset: Creating positive sampler...')
        self.positive_sampler = create_positive_sampler(
            descriptions_parquet=cfg.descriptions_parquet,
            relations_parquet=cfg.relations_parquet,
            max_per_stratum=4,
            seed=cfg.seed,
        )

        # Build the fixed (anchor, positive) pair index
        logger.info('Phase1MapDataset: Building pair index...')
        self.pairs: List[Tuple[int, Dict[str, Any]]] = []
        self._anchor_code_map: Dict[int, str] = {}
        required_pairs: Set[Tuple[int, int]] = set()

        for anchor_idx in self.positive_sampler.anchors:
            anchor_code = self.idx_to_code.get(anchor_idx)
            if anchor_code is None:
                continue

            positives = self.positive_sampler.sample_positives(anchor_idx)
            if not positives:
                continue

            self._anchor_code_map[anchor_idx] = anchor_code
            for positive in positives:
                self.pairs.append((anchor_idx, positive))
                required_pairs.add((anchor_idx, positive['positive_idx']))

        # Load candidate negatives for all required pairs
        logger.info('Phase1MapDataset: Loading negative candidates...')
        self.negative_candidates = _load_negative_candidates(
            cfg.triplets_parquet, required_pairs=required_pairs
        )

        logger.info(
            f'Phase1MapDataset: Initialized with {len(self.pairs):,} (anchor, positive) pairs'
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for different negative sampling."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def _extract_embedding(self, idx: int) -> Optional[Dict[str, Any]]:
        """Extract embedding from token cache, excluding code field."""
        try:
            return {k: v for k, v in self.token_cache[idx].items() if k != 'code'}
        except KeyError:
            logger.warning(f'Missing token_cache for index {idx}')
            return None

    def _attach_embeddings(
        self, negatives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attach embeddings to negative dictionaries."""
        result = []
        for neg in negatives:
            neg_idx = int(neg['negative_idx'])
            neg_embedding = self._extract_embedding(neg_idx)
            if neg_embedding is None:
                continue

            result.append({
                'negative_idx': neg_idx,
                'negative_code': neg['negative_code'],
                'negative_embedding': neg_embedding,
                'relation_margin': neg.get('relation_margin', 0),
                'distance_margin': neg.get('distance_margin', 0),
                'explicit_exclusion': neg.get('explicit_exclusion', False),
            })

        return result

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single triplet item by index with on-the-fly negative sampling.

        Returns:
            Dictionary with anchor, positive, selected negatives (Phase 1),
            and all candidates (for Phase 2+ HNM). Returns None if embeddings
            are missing.
        """
        anchor_idx, positive = self.pairs[idx]
        anchor_code = self._anchor_code_map.get(anchor_idx)
        if anchor_code is None:
            return None

        # Get anchor and positive embeddings
        anchor_embedding = self._extract_embedding(anchor_idx)
        if anchor_embedding is None:
            return None

        positive_idx = positive['positive_idx']
        positive_embedding = self._extract_embedding(positive_idx)
        if positive_embedding is None:
            return None

        # Deterministic seed per (idx, epoch) for reproducibility
        seed = self.cfg.seed + self.epoch * len(self) + idx
        rng = np.random.default_rng(seed)

        # Get raw candidates for this (anchor, positive) pair
        key = (anchor_idx, positive_idx)
        raw_candidates = self.negative_candidates.get(key, [])

        if not raw_candidates:
            return None

        # Sample n_candidates using Phase 1 tree-distance weighting
        all_candidates = _sample_negatives_phase1(
            anchor_code=anchor_code,
            anchor_idx=anchor_idx,
            candidate_negatives=raw_candidates,
            n_negatives=self.cfg.n_candidates,
            distance_lookup=self.distance_lookup,
            excluded_map=self.excluded_map,
            code_to_idx=self.code_to_idx,
            alpha=self.cfg.phase1_alpha,
            exclusion_weight=self.cfg.phase1_exclusion_weight,
            seed=seed,
        )

        if not all_candidates:
            return None

        # Select n_negatives_phase1 using difficulty curriculum
        epoch_progress = min(self.epoch / self.phase1_end_epoch, 1.0)
        selected = select_by_difficulty(
            candidates=all_candidates,
            n_select=self.cfg.n_negatives_phase1,
            distance_lookup=self.distance_lookup,
            anchor_code=anchor_code,
            epoch_progress=epoch_progress,
            cfg=self.cfg,
            rng=rng,
        )

        if not selected:
            return None

        # Attach embeddings to negatives
        selected_with_emb = self._attach_embeddings(selected)
        all_with_emb = self._attach_embeddings(all_candidates)

        if not selected_with_emb:
            return None

        return {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': positive['positive_code'],
            'positive_level': positive.get('positive_level', len(positive['positive_code'])),
            'stratum_id': positive.get('stratum_id', 0),
            'stratum_wgt': positive.get('stratum_wgt', 1.0),
            'positive_embedding': positive_embedding,
            'negatives': selected_with_emb,  # Phase 1 uses these
            'all_candidates': all_with_emb,  # Phase 2+ HNM pool
        }


# -------------------------------------------------------------------------------------------------
# Collate function wrapper to filter None items
# -------------------------------------------------------------------------------------------------

def _filter_none_collate_fn(batch: List[Optional[Dict]]) -> Dict:
    '''Filter out None items before calling the main collate function.'''
    filtered = [item for item in batch if item is not None]
    if not filtered:
        raise ValueError('All items in batch were None - no valid triplets')
    return collate_fn(filtered)


# -------------------------------------------------------------------------------------------------
# Main DataModule for PyTorch Lightning
# -------------------------------------------------------------------------------------------------

class NAICSDataModule(LightningDataModule):
    '''DataModule for NAICS embedding training with pre-sampled or on-the-fly triplets.'''

    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        triplets_path: str = './data/naics_training_pairs',
        tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        streaming_config: Optional[Dict] = None,
        sampling_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        val_split: float = 0.1,
        n_epochs: int = 100,
        max_epochs: int = 30,
        phase1_end: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()

        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_epochs = n_epochs
        self.max_epochs = max_epochs
        self.phase1_end = phase1_end

        # Create streaming configs
        if streaming_config is not None:
            val_streaming_config = streaming_config.copy()
            val_streaming_config['seed'] = seed + 1000  # Large offset for validation
            curriculum = StreamingConfig(**streaming_config)
            val_curriculum = StreamingConfig(**val_streaming_config)
        else:
            curriculum = StreamingConfig()
            val_curriculum = StreamingConfig(seed=seed + 1000)

        self.tokenization_cfg = TokenizationConfig(
            descriptions_parquet=descriptions_path,
            tokenizer_name=tokenizer_name,
            max_length=curriculum.max_length,
        )

        if sampling_config is None:
            self.sampling_cfg = SamplingConfig()
        elif isinstance(sampling_config, SamplingConfig):
            self.sampling_cfg = sampling_config
        else:
            self.sampling_cfg = SamplingConfig(**sampling_config)

        # Store streaming configs for use in prepare_data() and setup()
        self.train_streaming_cfg = curriculum
        self.val_streaming_cfg = val_curriculum

        # Datasets will be created in setup() after prepare_data() builds caches
        # Can be NAICSMapDataset (pre-computed) or Phase1MapDataset (on-the-fly)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self._token_cache: Optional[Dict[int, Dict[str, Any]]] = None

    def prepare_data(self):
        '''Build all caches before worker processes are spawned.'''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache

        # Build tokenization cache
        logger.info('Preparing tokenization cache in main process...')
        tokenization_cache(self.tokenization_cfg)

        # Build codes/indices cache
        logger.info('Preparing codes/indices cache in main process...')
        cache_dir = Path(self.tokenization_cfg.descriptions_parquet).parent / 'codes_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        codes_cache_path = cache_dir / 'codes_indices.pkl'

        if not codes_cache_path.exists():
            logger.info('Loading codes and indices for caching...')
            codes = get_indices_codes('codes')
            code_to_idx = get_indices_codes('code_to_idx')

            with open(codes_cache_path, 'wb') as f:
                pickle.dump({'codes': codes, 'code_to_idx': code_to_idx}, f)
            logger.info(f'Cached codes/indices to {codes_cache_path}')
        else:
            logger.info('Codes/indices cache already exists')

        # Build multi-epoch triplet caches (only if not using on-the-fly sampling)
        if not self.train_streaming_cfg.use_on_the_fly_sampling:
            self._build_multi_epoch_cache(self.train_streaming_cfg, 'training')
            self._build_multi_epoch_cache(self.val_streaming_cfg, 'validation')
        else:
            logger.info('On-the-fly sampling enabled, skipping multi-epoch cache build')

    def _build_multi_epoch_cache(self, cfg: StreamingConfig, name: str):
        '''Build multi-epoch triplet cache for a given config.'''
        logger.info(f'Preparing multi-epoch triplet cache ({name}) in main process...')
        cache_path = _get_multi_epoch_cache_path(cfg, self.n_epochs)

        if cache_path.exists():
            logger.info(f'{name.capitalize()} multi-epoch cache already exists')
            return

        logger.info(f'Building {name} multi-epoch cache for {self.n_epochs} epochs...')
        # This will build and save the cache
        build_multi_epoch_triplets(cfg, self.sampling_cfg, self.n_epochs)
        logger.info(f'{name.capitalize()} multi-epoch cache built successfully')

    def setup(self, stage: Optional[str] = None):
        '''Load caches and create datasets.'''
        from naics_embedder.text_model.dataloader.tokenization_cache import _load_tokenization_cache

        # Load token cache (shared between train and val)
        if self._token_cache is None:
            logger.info('Loading tokenization cache...')
            self._token_cache = _load_tokenization_cache(
                self.tokenization_cfg.output_path, verbose=True
            )
            if self._token_cache is None:
                raise RuntimeError('Failed to load tokenization cache')

        # Calculate Phase 1 end epoch for difficulty curriculum
        phase1_end_epoch = int(self.max_epochs * self.phase1_end)

        # Load and create training dataset
        if self.train_dataset is None:
            if self.train_streaming_cfg.use_on_the_fly_sampling:
                logger.info('Creating on-the-fly Phase1MapDataset for training...')
                self.train_dataset = Phase1MapDataset(
                    cfg=self.train_streaming_cfg,
                    sampling_cfg=self.sampling_cfg,
                    token_cache=self._token_cache,
                    phase1_end_epoch=phase1_end_epoch,
                )
            else:
                logger.info('Loading pre-computed training triplets...')
                train_triplets = build_multi_epoch_triplets(
                    self.train_streaming_cfg, self.sampling_cfg, self.n_epochs
                )
                logger.info(f'  • Creating training dataset with {len(train_triplets):,} triplets')
                self.train_dataset = NAICSMapDataset(train_triplets, self._token_cache)

        # Load and create validation dataset
        # Note: Validation always uses pre-computed for consistency
        if self.val_dataset is None:
            if self.val_streaming_cfg.use_on_the_fly_sampling:
                logger.info('Creating on-the-fly Phase1MapDataset for validation...')
                self.val_dataset = Phase1MapDataset(
                    cfg=self.val_streaming_cfg,
                    sampling_cfg=self.sampling_cfg,
                    token_cache=self._token_cache,
                    phase1_end_epoch=phase1_end_epoch,
                )
            else:
                logger.info('Loading pre-computed validation triplets...')
                val_triplets = build_multi_epoch_triplets(
                    self.val_streaming_cfg, self.sampling_cfg, self.n_epochs
                )
                logger.info(f'  • Creating validation dataset with {len(val_triplets):,} triplets\n')
                self.val_dataset = NAICSMapDataset(val_triplets, self._token_cache)

    def train_dataloader(self) -> DataLoader:
        '''Create training dataloader with shuffling enabled.'''
        if self.train_dataset is None:
            raise RuntimeError('train_dataset is None - call setup() first')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Enable shuffling for map-style dataset
            num_workers=self.num_workers,
            collate_fn=_filter_none_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        '''Create validation dataloader.'''
        if self.val_dataset is None:
            raise RuntimeError('val_dataset is None - call setup() first')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_filter_none_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def on_train_epoch_start(self) -> None:
        '''Update dataset epoch for on-the-fly sampling with difficulty curriculum.'''
        if self.trainer is not None and hasattr(self.train_dataset, 'set_epoch'):
            current_epoch = self.trainer.current_epoch
            self.train_dataset.set_epoch(current_epoch)
            logger.debug(f'Updated train dataset epoch to {current_epoch}')