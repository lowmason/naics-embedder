# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import pytorch_lightning as pyl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from naics_embedder.data.positive_sampling import create_positive_sampler
from naics_embedder.data.supervision_bundle import build_codebook
from naics_embedder.supervision.artifacts import (
    ValidatedSupervisionBundle,
    codebook_fingerprint,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model.dataloader.difficulty_sampler import (
    propose_by_difficulty,
    select_by_difficulty,
)
from naics_embedder.text_model.dataloader.streaming_dataset import (
    IndexDistanceLookup,
    _get_multi_epoch_cache_path,
    _load_distance_matrix,
    _load_excluded_codes,
    _load_negative_candidates,
    _sample_negatives_phase1,
    build_candidate_pool,
    build_multi_epoch_triplets,
    build_repaired_multi_epoch_rows,
    load_bundle_candidates,
    repaired_positive_sampler,
    sample_raw_candidates,
    sampled_positive_pairs,
)
from naics_embedder.utils.config import SamplingConfig, StreamingConfig, TokenizationConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

SUPERVISION_MODES = ('repaired', 'legacy_containment')
CHANNELS = ('title', 'description', 'excluded', 'examples')

# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def _stack_text_inputs(
    embeddings: List[Dict[str, Dict[str, torch.Tensor]]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        channel: {
            'input_ids': torch.stack([embedding[channel]['input_ids'] for embedding in embeddings]),
            'attention_mask':
            torch.stack([embedding[channel]['attention_mask'] for embedding in embeddings]),
        }
        for channel in CHANNELS
    }

def _accumulate_sampling_metadata(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    '''Average SANS sampling diagnostics across the batch items that carry them.'''

    accumulator: Optional[Dict[str, Any]] = None
    for item in batch:
        metadata = item.get('sampling_metadata')
        if not metadata:
            continue
        if accumulator is None:
            accumulator = {
                'strategy': metadata.get('strategy', 'unknown'),
                'candidates_near': 0,
                'candidates_far': 0,
                'sampled_near': 0,
                'sampled_far': 0,
                'effective_near_weight_sum': 0.0,
                'effective_far_weight_sum': 0.0,
                'records': 0,
            }
        accumulator['candidates_near'] += metadata.get('candidates_near', 0)
        accumulator['candidates_far'] += metadata.get('candidates_far', 0)
        accumulator['sampled_near'] += metadata.get('sampled_near', 0)
        accumulator['sampled_far'] += metadata.get('sampled_far', 0)
        accumulator['effective_near_weight_sum'] += metadata.get('effective_near_weight', 0.0)
        accumulator['effective_far_weight_sum'] += metadata.get('effective_far_weight', 0.0)
        accumulator['records'] += 1

    if accumulator is None or accumulator['records'] == 0:
        return None
    records = accumulator.pop('records')
    accumulator['avg_effective_near_weight'] = (
        accumulator.pop('effective_near_weight_sum') / records
    )
    accumulator['avg_effective_far_weight'] = accumulator.pop('effective_far_weight_sum') / records
    return accumulator

def _collate_legacy(batch: List[Dict]) -> Dict:
    '''Legacy-containment collation: local negatives, repeat-last padding, inputs never mutated.'''

    max_negatives = max(len(item['negatives']) for item in batch) if batch else 0
    if max_negatives == 0:
        raise ValueError('Batch contains items with no negatives - cannot create training batch')

    padded_negatives: List[List[Dict[str, Any]]] = []
    for item in batch:
        negatives = list(item['negatives'])
        if not negatives:
            anchor_code = item.get('anchor_code', 'unknown')
            raise ValueError(f'Item has no negatives to pad from: {anchor_code}')
        negatives.extend([negatives[-1]] * (max_negatives - len(negatives)))
        padded_negatives.append(negatives)

    result = {
        'anchor': _stack_text_inputs([item['anchor_embedding'] for item in batch]),
        'positive': _stack_text_inputs([item['positive_embedding'] for item in batch]),
        'negatives': _stack_text_inputs(
            [negative['negative_embedding'] for negatives in padded_negatives
             for negative in negatives]
        ),
        'batch_size': len(batch),
        'k_negatives': max_negatives,
        'anchor_code': [item['anchor_code'] for item in batch],
        'positive_code': [item['positive_code'] for item in batch],
        'negative_codes': [
            [negative['negative_code'] for negative in negatives] for negatives in padded_negatives
        ],
        'positive_levels': [
            item.get('positive_level', len(item['positive_code'])) for item in batch
        ],
    }
    sampling_metadata = _accumulate_sampling_metadata(batch)
    if sampling_metadata:
        result['sampling_metadata'] = sampling_metadata
    return result

def _collate_repaired(batch: List[Dict]) -> Dict:
    '''
    Repaired collation: one candidate pool per item, flattened row-major, with explicit invalid
    rows (zero input IDs and attention mask, code ID -1, source slot -1) instead of repeated
    padding. Pair-dependent supervision is joined later, after any distributed entity gather.
    '''

    if not batch:
        raise ValueError('cannot collate an empty batch')
    for item in batch:
        if 'candidate_pool' not in item:
            raise ValueError(
                'repaired collation requires candidate_pool items; legacy negatives require '
                "supervision_mode='legacy_containment'"
            )
    max_candidates = max(len(item['candidate_pool']) for item in batch)
    if max_candidates == 0:
        raise ValueError('Batch contains items with empty candidate pools')
    selection_k = min(int(item['selection_k']) for item in batch)
    if selection_k < 1:
        raise ValueError('selection_k must be at least one')

    template = next(
        candidate['negative_embedding'] for item in batch for candidate in item['candidate_pool']
    )
    invalid_row = {
        channel: {
            'input_ids': torch.zeros_like(template[channel]['input_ids']),
            'attention_mask': torch.zeros_like(template[channel]['attention_mask']),
        }
        for channel in CHANNELS
    }

    candidate_rows: List[Dict[str, Dict[str, torch.Tensor]]] = []
    candidate_code_ids: List[List[int]] = []
    candidate_valid_mask: List[List[bool]] = []
    candidate_source_slots: List[List[int]] = []
    candidate_roles: List[List[int]] = []
    candidate_provenance: List[List[int]] = []
    for item in batch:
        pool = item['candidate_pool']
        padding = max_candidates - len(pool)
        candidate_rows.extend(candidate['negative_embedding'] for candidate in pool)
        candidate_rows.extend([invalid_row] * padding)
        candidate_code_ids.append(
            [int(candidate['negative_code_id']) for candidate in pool] + [-1] * padding
        )
        candidate_valid_mask.append([True] * len(pool) + [False] * padding)
        candidate_source_slots.append(list(range(len(pool))) + [-1] * padding)
        candidate_roles.append(
            [int(candidate['sampling_role_id']) for candidate in pool] + [0] * padding
        )
        candidate_provenance.append(
            [int(candidate['sampling_provenance_id']) for candidate in pool] + [0] * padding
        )

    proposal_width = max(len(item['difficulty_proposal_indices']) for item in batch)
    difficulty_indices = [
        [int(slot) for slot in item['difficulty_proposal_indices']] +
        [-1] * (proposal_width - len(item['difficulty_proposal_indices'])) for item in batch
    ]

    result = {
        'anchor': _stack_text_inputs([item['anchor_embedding'] for item in batch]),
        'positive': _stack_text_inputs([item['positive_embedding'] for item in batch]),
        'candidate_inputs': _stack_text_inputs(candidate_rows),
        'batch_size': len(batch),
        'k_candidates': max_candidates,
        'selection_k': selection_k,
        'anchor_code_id': torch.tensor(
            [int(item['anchor_code_id']) for item in batch], dtype=torch.long
        ),
        'positive_code_id': torch.tensor(
            [int(item['positive_code_id']) for item in batch], dtype=torch.long
        ),
        'positive_structural_distance': torch.tensor(
            [float(item['positive_structural_distance']) for item in batch], dtype=torch.float32
        ),
        'positive_structural_relation_id': torch.tensor(
            [int(item['positive_structural_relation_id']) for item in batch], dtype=torch.int16
        ),
        'candidate_code_id': torch.tensor(candidate_code_ids, dtype=torch.long),
        'candidate_valid_mask': torch.tensor(candidate_valid_mask, dtype=torch.bool),
        'candidate_source_slot': torch.tensor(candidate_source_slots, dtype=torch.long),
        'candidate_sampling_role_id': torch.tensor(candidate_roles, dtype=torch.int8),
        'candidate_sampling_provenance_id': torch.tensor(candidate_provenance, dtype=torch.int8),
        'difficulty_proposal_indices': torch.tensor(
            difficulty_indices, dtype=torch.long
        ).reshape(len(batch), proposal_width),
        'anchor_code': [item['anchor_code'] for item in batch],
        'positive_code': [item['positive_code'] for item in batch],
        'positive_levels': [
            item.get('positive_level', len(item['positive_code'])) for item in batch
        ],
    }
    sampling_metadata = _accumulate_sampling_metadata(batch)
    if sampling_metadata:
        result['sampling_metadata'] = sampling_metadata
    return result

def collate_fn(batch: List[Dict], supervision_mode: str = 'repaired') -> Dict:
    '''
    Collate batch items; each item represents a single (anchor, positive).

    Args:
        batch: Dataset items.
        supervision_mode: ``'repaired'`` (one candidate pool per item) or the explicit
            ``'legacy_containment'`` mode (local legacy negatives).
    '''

    if supervision_mode == 'repaired':
        return _collate_repaired(batch)
    if supervision_mode == 'legacy_containment':
        return _collate_legacy(batch)
    raise ValueError(f'unknown supervision mode {supervision_mode!r}; expected {SUPERVISION_MODES}')

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
    '''
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
    '''

    def __init__(
        self,
        cfg: StreamingConfig,
        sampling_cfg: SamplingConfig,
        token_cache: Dict[int, Dict[str, Any]],
        phase1_end_epoch: int,
    ):
        '''
        Initialize the Phase 1 map dataset.

        Args:
            cfg: Streaming configuration with sampling parameters
            sampling_cfg: Sampling strategy configuration
            token_cache: Dictionary mapping index to tokenized embeddings
            phase1_end_epoch: Epoch at which Phase 1 ends (for curriculum progress)
        '''
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
        '''Update the current epoch for different negative sampling.'''
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def _extract_embedding(self, idx: int) -> Optional[Dict[str, Any]]:
        '''Extract embedding from token cache, excluding code field.'''
        try:
            return {k: v for k, v in self.token_cache[idx].items() if k != 'code'}
        except KeyError:
            logger.warning(f'Missing token_cache for index {idx}')
            return None

    def _attach_embeddings(
        self, negatives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        '''Attach embeddings to negative dictionaries.'''
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
        '''
        Get a single triplet item by index with on-the-fly negative sampling.

        Returns:
            Dictionary with anchor, positive, selected negatives (Phase 1),
            and all candidates (for Phase 2+ HNM). Returns None if embeddings
            are missing.
        '''
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
# Repaired Stage-3 datasets: one bundle-backed candidate pool per item
# -------------------------------------------------------------------------------------------------

def _token_embedding(token_cache: Dict[int, Dict[str, Any]], code_id: int) -> Dict[str, Any]:
    try:
        entry = token_cache[int(code_id)]
    except KeyError as exc:
        raise KeyError(
            f'token cache has no entry for code ID {code_id}; the cache must match the '
            'supervision codebook'
        ) from exc
    return {key: value for key, value in entry.items() if key != 'code'}

def _repaired_item(
    *,
    anchor_code_id: int,
    positive: Dict[str, Any],
    pool: List[Dict[str, Any]],
    proposals: List[int],
    selection_k: int,
    index: SupervisionIndex,
    token_cache: Dict[int, Dict[str, Any]],
    sampling_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    positive_code_id = int(positive['positive_code_id'])
    item: Dict[str, Any] = {
        'anchor_code_id': anchor_code_id,
        'anchor_code': index.id_to_code[anchor_code_id],
        'anchor_embedding': _token_embedding(token_cache, anchor_code_id),
        'positive_code_id': positive_code_id,
        'positive_code': index.id_to_code[positive_code_id],
        'positive_embedding': _token_embedding(token_cache, positive_code_id),
        'positive_level': positive['positive_level'],
        'stratum_id': positive['stratum_id'],
        'stratum_wgt': positive['stratum_wgt'],
        'positive_structural_distance': float(
            index.structural_distance[anchor_code_id, positive_code_id]
        ),
        'positive_structural_relation_id': int(
            index.structural_relation_id[anchor_code_id, positive_code_id]
        ),
        'candidate_pool': [
            {
                **candidate,
                'negative_embedding': _token_embedding(token_cache, candidate['negative_code_id']),
            } for candidate in pool
        ],
        'difficulty_proposal_indices': proposals,
        'selection_k': selection_k,
    }
    if sampling_metadata:
        item['sampling_metadata'] = sampling_metadata
    return item

class RepairedMapDataset(Dataset):
    '''
    Pre-sampled repaired rows. Each access builds one canonical candidate pool from the row's raw
    candidates (deterministic for the row's sampling epoch) and proposes every pool position.
    '''

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        token_cache: Dict[int, Dict[str, Any]],
        index: SupervisionIndex,
        cfg: StreamingConfig,
        selection_k: int,
    ):
        self.rows = rows
        self.token_cache = token_cache
        self.index = index
        self.cfg = cfg
        self.selection_k = selection_k

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        pool = build_candidate_pool(
            anchor_code_id=int(row['anchor_code_id']),
            positive_code_id=int(row['positive_code_id']),
            raw_candidates=row['raw_candidates'],
            supervision_index=self.index,
            n_candidates=self.cfg.n_negatives,
            final_k=self.selection_k,
            epoch=int(row['sampling_epoch']),
            seed=self.cfg.seed,
        )
        return _repaired_item(
            anchor_code_id=int(row['anchor_code_id']),
            positive=row,
            pool=pool,
            proposals=list(range(len(pool))),
            selection_k=self.selection_k,
            index=self.index,
            token_cache=self.token_cache,
            sampling_metadata=row.get('sampling_metadata'),
        )

class RepairedPhase1Dataset(Dataset):
    '''
    On-the-fly repaired sampling. Each access samples raw candidates for the epoch, builds one
    canonical candidate pool, and proposes pool positions by the difficulty curriculum.
    '''

    def __init__(
        self,
        cfg: StreamingConfig,
        sampling_cfg: SamplingConfig,
        token_cache: Dict[int, Dict[str, Any]],
        bundle: ValidatedSupervisionBundle,
        index: SupervisionIndex,
        phase1_end_epoch: int,
    ):
        self.cfg = cfg
        self.sampling_cfg = sampling_cfg
        self.token_cache = token_cache
        self.index = index
        self.phase1_end_epoch = max(phase1_end_epoch, 1)
        self.epoch = 0
        self.distance_lookup = IndexDistanceLookup(index)

        sampler = repaired_positive_sampler(cfg, bundle, index)
        pairs, dropped = sampled_positive_pairs(sampler, index)
        if dropped:
            logger.info(f'Dropped {dropped:,} sampled positives that are explicit exclusions')
        self.negative_candidates = load_bundle_candidates(
            bundle, {(anchor, int(positive['positive_idx'])) for anchor, positive in pairs}
        )
        self.pairs: List[Tuple[int, Dict[str, Any]]] = [
            (anchor, positive)
            for anchor, positive in pairs
            if self.negative_candidates.get((anchor, int(positive['positive_idx'])))
        ]
        logger.info(f'RepairedPhase1Dataset: {len(self.pairs):,} (anchor, positive) pairs')

    def set_epoch(self, epoch: int) -> None:
        '''Update the current epoch for epoch-dependent sampling.'''
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anchor_code_id, positive = self.pairs[idx]
        positive_code_id = int(positive['positive_idx'])
        seed = self.cfg.seed + self.epoch * len(self) + idx
        raw, metadata = sample_raw_candidates(
            anchor_code=self.index.id_to_code[anchor_code_id],
            anchor_code_id=anchor_code_id,
            candidates=self.negative_candidates[(anchor_code_id, positive_code_id)],
            n_sample=self.cfg.n_candidates,
            cfg=self.cfg,
            sampling_cfg=self.sampling_cfg,
            distance_lookup=self.distance_lookup,
            index=self.index,
            seed=seed,
        )
        pool = build_candidate_pool(
            anchor_code_id=anchor_code_id,
            positive_code_id=positive_code_id,
            raw_candidates=raw,
            supervision_index=self.index,
            n_candidates=self.cfg.n_candidates,
            final_k=self.cfg.n_negatives_phase1,
            epoch=self.epoch,
            seed=self.cfg.seed,
        )
        proposals = propose_by_difficulty(
            candidates=pool,
            n_propose=self.cfg.n_negatives_phase1,
            epoch_progress=min(self.epoch / self.phase1_end_epoch, 1.0),
            cfg=self.cfg,
            rng=np.random.default_rng(seed),
        )
        return _repaired_item(
            anchor_code_id=anchor_code_id,
            positive={**positive, 'positive_code_id': positive_code_id},
            pool=pool,
            proposals=proposals,
            selection_k=self.cfg.n_negatives_phase1,
            index=self.index,
            token_cache=self.token_cache,
            sampling_metadata=metadata,
        )

# -------------------------------------------------------------------------------------------------
# Collate function wrapper to filter None items
# -------------------------------------------------------------------------------------------------

def _filter_none_collate_fn(
    batch: List[Optional[Dict]],
    supervision_mode: str = 'repaired',
) -> Dict:
    '''Filter out None items before calling the main collate function.'''
    filtered = [item for item in batch if item is not None]
    if not filtered:
        raise ValueError('All items in batch were None - no valid triplets')
    return collate_fn(filtered, supervision_mode=supervision_mode)

def legacy_token_fingerprints(descriptions_parquet: str) -> Dict[str, str]:
    '''
    Tokenization-cache fingerprints computed directly from a descriptions file.

    Used only by legacy containment, which has no supervision bundle manifest to read them from.
    '''

    descriptions_path = Path(descriptions_parquet)
    return {
        'description_fingerprint': sha256_file(descriptions_path),
        'codebook_fingerprint': codebook_fingerprint(
            build_codebook(pl.read_parquet(descriptions_path))
        ),
    }


# -------------------------------------------------------------------------------------------------
# Epoch propagation to DataLoader worker processes
# -------------------------------------------------------------------------------------------------

class _EpochSyncedDataset(Dataset):
    '''
    Apply a shared training epoch to an epoch-aware dataset in every process that samples it.

    DataLoader workers (persistent or not; fork, spawn or forkserver) sample from their own copy of
    the dataset, so set_epoch() in the main process never reaches them. The epoch lives in a
    shared-memory tensor read by every copy, and each copy applies it to its own dataset.
    '''

    def __init__(self, dataset: Dataset, shared_epoch: torch.Tensor):
        self.dataset = dataset
        self.shared_epoch = shared_epoch
        self._applied_epoch: Optional[int] = None

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        epoch = int(self.shared_epoch)
        if epoch != self._applied_epoch:
            self.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
            self._applied_epoch = epoch
        return self.dataset[idx]


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
        supervision_mode: str = 'repaired',
        supervision_manifest_path: Optional[str] = None,
        supervision_contract_version: str = CONTRACT_VERSION,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
        **kwargs: Any,
    ):
        super().__init__()

        if supervision_mode not in SUPERVISION_MODES:
            raise ValueError(
                f'unknown supervision mode {supervision_mode!r}; expected {SUPERVISION_MODES}'
            )
        # Repaired mode requires a validated bundle; it is loaded (and fails closed) in
        # prepare_data()/setup(), so construction stays side-effect free.
        self.supervision_mode = supervision_mode
        self.supervision_manifest_path = supervision_manifest_path
        self.supervision_contract_version = supervision_contract_version
        self._bundle: Optional[ValidatedSupervisionBundle] = supervision_bundle
        self._index: Optional[SupervisionIndex] = None

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

        # Training epoch shared with DataLoader worker processes (see _EpochSyncedDataset)
        self._train_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    # ---------------------------------------------------------------------------------------------
    # Supervision identity
    # ---------------------------------------------------------------------------------------------

    def _supervision(self) -> Tuple[ValidatedSupervisionBundle, SupervisionIndex]:
        '''The validated bundle and its index (repaired mode only; fails closed).'''

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
        descriptions_hash = sha256_file(Path(self.tokenization_cfg.descriptions_parquet))
        if descriptions_hash != self._bundle.manifest.description_fingerprint:
            raise ValueError(
                f'descriptions input {self.tokenization_cfg.descriptions_parquet} does not match '
                f'supervision bundle {self._bundle.manifest.bundle_id}: expected '
                f'{self._bundle.manifest.description_fingerprint}, found {descriptions_hash}'
            )
        if self._index is None:
            self._index = SupervisionIndex.from_bundle(self._bundle)
        return self._bundle, self._index

    def _token_fingerprints(self) -> Dict[str, str]:
        '''Fingerprints the tokenization cache must record to be reused.'''

        if self.supervision_mode == 'repaired':
            manifest = self._supervision()[0].manifest
            return {
                'description_fingerprint': manifest.description_fingerprint,
                'codebook_fingerprint': manifest.codebook_fingerprint,
            }
        return legacy_token_fingerprints(self.tokenization_cfg.descriptions_parquet)

    def _collate(self):
        return partial(_filter_none_collate_fn, supervision_mode=self.supervision_mode)

    def prepare_data(self):
        '''Build all caches before worker processes are spawned.'''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache

        # Build tokenization cache
        logger.info('Preparing tokenization cache in main process...')
        tokenization_cache(self.tokenization_cfg, **self._token_fingerprints())

        if self.supervision_mode == 'repaired':
            if not self.train_streaming_cfg.use_on_the_fly_sampling:
                bundle, index = self._supervision()
                for name, cfg in (
                    ('training', self.train_streaming_cfg),
                    ('validation', self.val_streaming_cfg),
                ):
                    logger.info(f'Preparing repaired {name} rows in main process...')
                    build_repaired_multi_epoch_rows(
                        cfg, self.sampling_cfg, bundle, index, self.n_epochs
                    )
            return

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
        from naics_embedder.text_model.dataloader.tokenization_cache import (
            load_verified_tokenization_cache,
        )

        # Load token cache (shared between train and val)
        if self._token_cache is None:
            logger.info('Loading tokenization cache...')
            self._token_cache = load_verified_tokenization_cache(
                self.tokenization_cfg, **self._token_fingerprints()
            )

        # Calculate Phase 1 end epoch for difficulty curriculum
        phase1_end_epoch = int(self.max_epochs * self.phase1_end)

        if self.supervision_mode == 'repaired':
            self._setup_repaired(phase1_end_epoch)
            return

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

    def _repaired_dataset(self, cfg: StreamingConfig, phase1_end_epoch: int) -> Dataset:
        bundle, index = self._supervision()
        assert self._token_cache is not None
        if cfg.use_on_the_fly_sampling:
            return RepairedPhase1Dataset(
                cfg=cfg,
                sampling_cfg=self.sampling_cfg,
                token_cache=self._token_cache,
                bundle=bundle,
                index=index,
                phase1_end_epoch=phase1_end_epoch,
            )
        rows = build_repaired_multi_epoch_rows(cfg, self.sampling_cfg, bundle, index, self.n_epochs)
        return RepairedMapDataset(rows, self._token_cache, index, cfg, selection_k=cfg.n_negatives)

    def _setup_repaired(self, phase1_end_epoch: int) -> None:
        if self.train_dataset is None:
            logger.info('Creating repaired training dataset from the supervision bundle...')
            self.train_dataset = self._repaired_dataset(self.train_streaming_cfg, phase1_end_epoch)
        if self.val_dataset is None:
            logger.info('Creating repaired validation dataset from the supervision bundle...')
            self.val_dataset = self._repaired_dataset(self.val_streaming_cfg, phase1_end_epoch)

    def train_dataloader(self) -> DataLoader:
        '''Create training dataloader with shuffling enabled.'''
        if self.train_dataset is None:
            raise RuntimeError('train_dataset is None - call setup() first')
        if self.trainer is not None:
            # Lightning calls this after restoring a checkpoint's loop state and before creating
            # the first iterator; no epoch hook runs in between (and none at all on a mid-epoch
            # resume), so start from the trainer's epoch here.
            self.set_train_epoch(self.trainer.current_epoch)
        dataset = self.train_dataset
        if hasattr(dataset, 'set_epoch'):
            dataset = _EpochSyncedDataset(dataset, self._train_epoch)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Enable shuffling for map-style dataset
            num_workers=self.num_workers,
            collate_fn=self._collate(),
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
            collate_fn=self._collate(),
            persistent_workers=self.num_workers > 0,
        )

    def set_train_epoch(self, epoch: int) -> None:
        '''
        Set the sampling epoch of an epoch-aware training dataset in every process sampling it.

        The validation dataset is deliberately never updated: its pools must not depend on the
        epoch because val/contrastive_loss drives checkpointing.
        '''
        if hasattr(self.train_dataset, 'set_epoch'):
            self._train_epoch.fill_(epoch)  # read by DataLoader workers
            self.train_dataset.set_epoch(epoch)  # main-process copy
            logger.debug(f'Updated train dataset epoch to {epoch}')


# -------------------------------------------------------------------------------------------------
# Callback propagating the training epoch to the datamodule
# -------------------------------------------------------------------------------------------------

class TrainDatasetEpochCallback(pyl.Callback):
    '''
    Propagate the trainer's epoch to NAICSDataModule's training dataset at each epoch start.

    Register it on every Trainer that fits a NAICSDataModule: Lightning dispatches
    on_train_epoch_start to callbacks and the LightningModule, never to a LightningDataModule,
    so the datamodule cannot advance on-the-fly sampling past epoch 0 by itself.
    '''

    def on_train_epoch_start(self, trainer: pyl.Trainer, pl_module: pyl.LightningModule) -> None:
        datamodule = trainer.datamodule
        if isinstance(datamodule, NAICSDataModule):
            datamodule.set_train_epoch(trainer.current_epoch)
