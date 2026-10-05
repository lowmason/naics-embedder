# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import logging
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pytorch_lightning as pyl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from naics_embedder.data.positive_sampling import create_positive_sampler
from naics_embedder.supervision.artifacts import (
    ValidatedSupervisionBundle,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.queries import TaskQuery
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.text_model.dataloader.difficulty_sampler import (
    propose_by_difficulty,
    select_by_difficulty,
)
from naics_embedder.text_model.dataloader.streaming_dataset import (
    IndexDistanceLookup,
    _load_distance_matrix,
    _load_excluded_codes,
    _load_negative_candidates,
    _sample_negatives_phase1,
    build_candidate_pool,
    build_repaired_multi_epoch_rows,
    load_bundle_candidates,
    repaired_positive_sampler,
    sample_raw_candidates,
    sampled_positive_pairs,
)
from naics_embedder.text_model.fields import CHANNELS, QUERY, tokenize_field
from naics_embedder.utils.config import SamplingConfig, StreamingConfig, TokenizationConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def stack_text_inputs(
    embeddings: Sequence[Mapping[str, Mapping[str, Any]]],
    fields: Sequence[str] = CHANNELS,
) -> Dict[str, Dict[str, torch.Tensor]]:
    '''
    Stack token rows into one batch: per field, ``input_ids``, ``attention_mask`` and a boolean
    ``present`` of shape (B,).

    Every code batch is built here (the collates, the export and the HGCN feeder), and a query
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
                'repaired collation requires candidate_pool items; there are no legacy negatives '
                '(roadmap D2)'
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
            'present': False,
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
        [int(slot) for slot in item['difficulty_proposal_indices']] + [-1] *
        (proposal_width - len(item['difficulty_proposal_indices'])) for item in batch
    ]

    result = {
        'anchor': stack_text_inputs([item['anchor_embedding'] for item in batch]),
        'positive': stack_text_inputs([item['positive_embedding'] for item in batch]),
        'candidate_inputs': stack_text_inputs(candidate_rows),
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
        'difficulty_proposal_indices': torch.tensor(difficulty_indices, dtype=torch.long).reshape(
            len(batch), proposal_width
        ),
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

def collate_fn(batch: List[Dict]) -> Dict:
    '''
    Collate batch items; each item represents a single (anchor, positive) and its candidate pool.

    Args:
        batch: Dataset items.
    '''

    return _collate_repaired(batch)

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

    def _attach_embeddings(self, negatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''Attach embeddings to negative dictionaries.'''
        result = []
        for neg in negatives:
            neg_idx = int(neg['negative_idx'])
            neg_embedding = self._extract_embedding(neg_idx)
            if neg_embedding is None:
                continue

            result.append(
                {
                    'negative_idx': neg_idx,
                    'negative_code': neg['negative_code'],
                    'negative_embedding': neg_embedding,
                    'relation_margin': neg.get('relation_margin', 0),
                    'distance_margin': neg.get('distance_margin', 0),
                    'explicit_exclusion': neg.get('explicit_exclusion', False),
                }
            )

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
            bundle, {(anchor, int(positive['positive_idx']))
                     for anchor, positive in pairs}
        )
        self.pairs: List[Tuple[int, Dict[str, Any]]] = [
            (anchor, positive) for anchor, positive in pairs
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
            positive={
                **positive, 'positive_code_id': positive_code_id
            },
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

def _filter_none_collate_fn(batch: List[Optional[Dict]]) -> Dict:
    '''Filter out None items before calling the main collate function.'''
    filtered = [item for item in batch if item is not None]
    if not filtered:
        raise ValueError('All items in batch were None - no valid triplets')
    return collate_fn(filtered)

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
        supervision_manifest_path: Optional[str] = None,
        supervision_contract_version: str = CONTRACT_VERSION,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
        **kwargs: Any,
    ):
        super().__init__()

        # Training requires a validated bundle; it is loaded (and fails closed) in
        # prepare_data()/setup(), so construction stays side-effect free.
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

        # Datasets will be created in setup() after prepare_data() builds caches: a
        # RepairedMapDataset (pre-computed) or a RepairedPhase1Dataset (on-the-fly)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self._token_cache: Optional[Dict[int, Dict[str, Any]]] = None

        # Training epoch shared with DataLoader worker processes (see _EpochSyncedDataset)
        self._train_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    # ---------------------------------------------------------------------------------------------
    # Supervision identity
    # ---------------------------------------------------------------------------------------------

    def _supervision(self) -> Tuple[ValidatedSupervisionBundle, SupervisionIndex]:
        '''The validated bundle and its index (fails closed).'''

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

        manifest = self._supervision()[0].manifest
        return {
            'description_fingerprint': manifest.description_fingerprint,
            'codebook_fingerprint': manifest.codebook_fingerprint,
        }

    def _collate(self):
        return _filter_none_collate_fn

    def prepare_data(self):
        '''Build all caches before worker processes are spawned.'''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache

        # Build tokenization cache
        logger.info('Preparing tokenization cache in main process...')
        tokenization_cache(self.tokenization_cfg, **self._token_fingerprints())

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
        self._setup_repaired(phase1_end_epoch)

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
