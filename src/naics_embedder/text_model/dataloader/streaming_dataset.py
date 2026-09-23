# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
import pickle
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from naics_embedder.data.positive_sampling import PositiveSampler, create_positive_sampler
from naics_embedder.supervision.artifacts import (
    ValidatedSupervisionBundle,
    aggregate_fingerprint,
)
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.schema import SAMPLING_ROLE_TO_ID, SamplingProvenance, SamplingRole
from naics_embedder.supervision.selection import stable_hash
from naics_embedder.utils.config import SamplingConfig, SansStaticConfig, StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Phase 1 Sampling Utilities
# -------------------------------------------------------------------------------------------------

def _load_distance_matrix(
    distance_matrix_path: str, code_to_idx: Dict[str, int], idx_to_code: Dict[int, str]
) -> Dict[Tuple[str, str], float]:
    '''
    Load distance matrix and create a lookup dictionary.

    The distance matrix has columns like 'idx_0-code_11', 'idx_1-code_111', etc.
    where the column name format is 'idx_{matrix_idx}-code_{code}'.
    Each row corresponds to an anchor code in sorted order.

    Note: The distance matrix uses sorted code order for indices, which may differ
    from the 'index' column in descriptions.parquet. We use code strings as keys.

    Args:
        distance_matrix_path: Path to distance matrix parquet file
        code_to_idx: Mapping from code to index (from descriptions parquet)
        idx_to_code: Mapping from index to code (from descriptions parquet)

    Returns:
        Dictionary mapping (anchor_code, negative_code) -> tree_distance
    '''

    logger.info('Loading distance matrix for Phase 1 sampling...')

    df = pl.read_parquet(distance_matrix_path)

    # Create lookup dictionary using code strings as keys
    distance_lookup: Dict[Tuple[str, str], float] = {}

    # Build mapping from column name to negative code
    col_to_negative_code = {}
    for col in df.columns:
        # Column format: 'idx_{matrix_idx}-code_{code}'
        parts = col.split('-')
        if len(parts) != 2:
            continue

        # Extract code from second part (e.g., 'code_111' -> '111')
        code_str = parts[1].replace('code_', '')

        if code_str in code_to_idx:
            col_to_negative_code[col] = code_str

    # Get all codes sorted by code string (matching distance matrix row order)
    # The distance matrix rows are in sorted code order
    codes_sorted = sorted(code_to_idx.keys())

    # Build lookup: (anchor_code, negative_code) -> distance
    for row_idx, anchor_code in enumerate(codes_sorted):
        if row_idx >= df.height:
            break

        # For each column (negative)
        for col, negative_code in col_to_negative_code.items():
            if col not in df.columns:
                continue

            # Get distance value from the row and column
            distance_val = df.select(pl.col(col)).row(row_idx)[0]

            if distance_val is not None:
                distance_lookup[(anchor_code, negative_code)] = float(distance_val)

    logger.info(f'Loaded {len(distance_lookup):,} distance entries')
    return distance_lookup

def _load_excluded_codes(descriptions_path: str,
                         code_to_idx: Optional[Dict[str, int]] = None) -> Dict[str, Set[str]]:
    '''
    Load excluded codes from descriptions parquet.

    Args:
        descriptions_path: Path to descriptions parquet file

    Returns:
        Dictionary mapping code -> set of excluded codes
    '''
    logger.info('Loading excluded codes for Phase 1 sampling...')

    df = (
        pl.read_parquet(descriptions_path).select('code', 'excluded_codes').filter(
            pl.col('excluded_codes').is_not_null()
        )
    )

    excluded_map: Dict[str, Set[str]] = {}
    unknown_codes = 0
    for row in df.iter_rows(named=True):
        code = row['code']
        excluded_list = row['excluded_codes']
        if excluded_list:
            filtered: Set[str] = set()
            for ex_code in excluded_list:
                if code_to_idx is not None and ex_code not in code_to_idx:
                    unknown_codes += 1
                    continue
                filtered.add(ex_code)

            if filtered:
                excluded_map[code] = filtered

    logger.info(f'Loaded excluded codes for {len(excluded_map):,} codes')
    if unknown_codes:
        logger.warning(f'{unknown_codes:,} excluded codes not in taxonomy were ignored')
    return excluded_map

def _compute_phase1_weights(
    anchor_code: str,
    anchor_idx: int,
    candidate_negatives: List[Dict[str, Any]],
    distance_lookup: Dict[Tuple[str, str], float],
    excluded_map: Dict[str, Set[str]],
    code_to_idx: Dict[str, int],
    alpha: float = 1.5,
    exclusion_weight: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute Phase 1 sampling weights for candidate negatives.

    Args:
        anchor_code: Anchor code
        anchor_idx: Anchor index
        candidate_negatives: List of candidate negative dictionaries with 'negative_code'
        distance_lookup: Dictionary mapping (anchor_idx, negative_idx) -> tree_distance
        excluded_map: Dictionary mapping code -> set of excluded codes
        code_to_idx: Mapping from code to index
        alpha: Exponent for inverse tree distance weighting
        exclusion_weight: Legacy-containment constant weight for excluded codes. ``None`` gives
            exclusions no special weight (repaired training reserves exactly one exclusion slot
            at selection time instead).

    Returns:
        Array of sampling weights (unnormalized)
    '''
    weights = np.zeros(len(candidate_negatives))
    excluded_mask = np.zeros(len(candidate_negatives), dtype=bool)

    for i, neg in enumerate(candidate_negatives):
        negative_code = neg['negative_code']

        # Check if anchor excludes this negative
        if (
            exclusion_weight is not None and anchor_code in excluded_map
            and negative_code in excluded_map[anchor_code]
        ):
            weights[i] = exclusion_weight
            excluded_mask[i] = True
            continue

        # Get tree distance using code strings as keys
        key = (anchor_code, negative_code)
        if key not in distance_lookup:
            # If distance not found, use a default large distance
            tree_distance = 12.0
        else:
            tree_distance = distance_lookup[key]

        # Sibling masking: set weight to 0 if distance == 2 (siblings)
        if tree_distance == 2.0:
            weights[i] = 0.0
            continue

        # Inverse tree distance weighting: P(n) ∝ 1 / D_tree(a, n)^α
        if tree_distance > 0:
            weights[i] = 1.0 / (tree_distance**alpha)
        else:
            weights[i] = 0.0

    return weights, excluded_mask

def _sample_negatives_phase1(
    anchor_code: str,
    anchor_idx: int,
    candidate_negatives: List[Dict[str, Any]],
    n_negatives: int,
    distance_lookup: Dict[Tuple[str, str], float],
    excluded_map: Dict[str, Set[str]],
    code_to_idx: Dict[str, int],
    alpha: float = 1.5,
    exclusion_weight: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    '''
    Sample negatives using Phase 1 tree-distance based sampling.

    Args:
        anchor_code: Anchor code
        anchor_idx: Anchor index
        candidate_negatives: List of candidate negative dictionaries
        n_negatives: Number of negatives to sample
        distance_lookup: Dictionary mapping (anchor_idx, negative_idx) -> tree_distance
        excluded_map: Dictionary mapping code -> set of excluded codes
        code_to_idx: Mapping from code to index
        alpha: Exponent for inverse tree distance weighting
        exclusion_weight: High constant weight for excluded codes
        seed: Random seed for sampling

    Returns:
        List of sampled negative dictionaries
    '''
    if len(candidate_negatives) == 0:
        return []

    # Compute weights and exclusion flags
    weights, excluded_mask = _compute_phase1_weights(
        anchor_code=anchor_code,
        anchor_idx=anchor_idx,
        candidate_negatives=candidate_negatives,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=alpha,
        exclusion_weight=exclusion_weight,
    )

    # Normalize weights
    total_weight = weights.sum()
    if total_weight == 0:
        # If all weights are zero (e.g., all siblings), fall back to uniform sampling
        logger.warning(f'All weights zero for anchor {anchor_code}, using uniform sampling')
        weights = np.ones(len(candidate_negatives))
        total_weight = len(candidate_negatives)

    probabilities = weights / total_weight

    # Sample using numpy
    rng = np.random.default_rng(seed)
    n_sample = min(n_negatives, len(candidate_negatives))
    sampled_indices = rng.choice(
        len(candidate_negatives), size=n_sample, replace=False, p=probabilities
    )

    excluded_chosen = excluded_mask[sampled_indices].sum()
    if n_sample > 0:
        exclusion_ratio = excluded_chosen / n_sample
        logger.debug(
            f'Anchor {anchor_code}: {exclusion_ratio:.2%} negatives from explicit exclusions '
            f'({excluded_chosen}/{n_sample})'
        )

    sampled = []
    for idx in sampled_indices:
        neg = dict(candidate_negatives[idx])
        neg['explicit_exclusion'] = bool(excluded_mask[idx])
        sampled.append(neg)

    return sampled

def _sample_negatives_sans_static(
    anchor_code: str,
    candidate_negatives: List[Dict[str, Any]],
    n_negatives: int,
    distance_lookup: Dict[Tuple[str, str], float],
    sans_cfg: 'SansStaticConfig',
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    '''
    Sample negatives using static near/far buckets (SANS baseline).

    Args:
        anchor_code: Anchor code
        candidate_negatives: Candidate negatives list
        n_negatives: Number of negatives to sample
        distance_lookup: Tree distance lookup (code pairs -> distance)
        sans_cfg: SANS configuration parameters
        seed: Optional RNG seed

    Returns:
        Tuple of sampled negatives and sampling metadata for diagnostics.
    '''

    metadata = {
        'strategy': 'sans_static',
        'candidates_near': 0,
        'candidates_far': 0,
        'sampled_near': 0,
        'sampled_far': 0,
        'effective_near_weight': 0.0,
        'effective_far_weight': 0.0,
    }

    if not candidate_negatives:
        return [], metadata

    distances = []
    for neg in candidate_negatives:
        negative_code = neg['negative_code']
        distance = distance_lookup.get((anchor_code, negative_code))
        if distance is None:
            distance = distance_lookup.get((negative_code, anchor_code), sans_cfg.default_distance)
        distances.append(distance)

    near_indices = [
        idx for idx, dist in enumerate(distances) if dist <= sans_cfg.near_distance_threshold
    ]
    far_indices = [idx for idx in range(len(distances)) if idx not in near_indices]

    metadata['candidates_near'] = len(near_indices)
    metadata['candidates_far'] = len(far_indices)

    near_weight = sans_cfg.near_bucket_weight
    far_weight = sans_cfg.far_bucket_weight
    total_candidates = len(candidate_negatives)

    weights = np.zeros(total_candidates, dtype=np.float64)
    bucket_weight_total = 0.0

    if near_indices and near_weight > 0:
        bucket_weight_total += near_weight
        weights[near_indices] = near_weight / len(near_indices)
    if far_indices and far_weight > 0:
        bucket_weight_total += far_weight
        weights[far_indices] = far_weight / len(far_indices)

    if bucket_weight_total == 0:
        weights[:] = 1.0 / total_candidates
    else:
        weights /= bucket_weight_total

    metadata['effective_near_weight'] = float(weights[near_indices].sum()) if near_indices else 0.0
    metadata['effective_far_weight'] = float(weights[far_indices].sum()) if far_indices else 0.0

    rng = np.random.default_rng(seed)
    n_sample = min(n_negatives, total_candidates)
    sampled_indices = rng.choice(total_candidates, size=n_sample, replace=False, p=weights)

    near_index_set = set(near_indices)

    sampled = []
    for idx in sampled_indices:
        neg = dict(candidate_negatives[idx])
        neg.setdefault('explicit_exclusion', False)
        sampled.append(neg)

        if idx in near_index_set:
            metadata['sampled_near'] += 1
        else:
            metadata['sampled_far'] += 1

    return sampled, metadata

# -------------------------------------------------------------------------------------------------
# Negative candidate loading
# -------------------------------------------------------------------------------------------------

def _load_negative_candidates(
    triplets_parquet: str,
    required_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    '''Load all candidate negatives from triplets parquet files.

    Returns a dictionary mapping (anchor_idx, positive_idx) -> list of negative dicts.
    Each negative dict has: negative_idx, negative_code, relation_margin, distance_margin.

    NOTE: This function loads ALL parquet files at once. For per-anchor streaming,
    use _load_anchor_negative_candidates instead.
    '''
    logger.info('Loading negative candidates from triplets parquet...')

    dataset_files = [str(p) for p in Path(triplets_parquet).glob('**/*.parquet')]
    if not dataset_files:
        logger.warning(f'No parquet files found in {triplets_parquet}')
        return {}

    anchor_filter: Optional[List[int]] = None
    pairs_lazy: Optional[pl.LazyFrame] = None
    if required_pairs is not None:
        if not required_pairs:
            logger.info('No anchor/positive pairs provided; skipping negative candidate load.')
            return {}

        unique_pairs = sorted(required_pairs)
        anchor_filter = sorted({anchor for anchor, _ in unique_pairs})
        pair_df = pl.DataFrame(
            {
                'anchor_idx': [anchor for anchor, _ in unique_pairs],
                'positive_idx': [positive for _, positive in unique_pairs],
            },
            schema={'anchor_idx': pl.UInt32, 'positive_idx': pl.UInt32},
        )
        pairs_lazy = pair_df.lazy()
        logger.info(
            f'Filtering negative candidates to {len(unique_pairs):,} selected (anchor, positive) pairs'
        )

    scan = pl.scan_parquet(dataset_files)
    if anchor_filter:
        scan = scan.filter(pl.col('anchor_idx').is_in(anchor_filter))

    if pairs_lazy is not None:
        scan = scan.join(pairs_lazy, on=['anchor_idx', 'positive_idx'], how='inner')

    df = scan.select(
        'anchor_idx',
        'positive_idx',
        'negative_idx',
        'negative_code',
        'relation_margin',
        'distance_margin',
    ).collect()

    logger.info(f'Loaded {len(df):,} negative candidate rows')

    # Group by (anchor, positive)
    result: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in df.iter_rows(named=True):
        key = (row['anchor_idx'], row['positive_idx'])
        if key not in result:
            result[key] = []
        result[key].append(
            {
                'negative_idx': row['negative_idx'],
                'negative_code': row['negative_code'],
                'relation_margin': row['relation_margin'],
                'distance_margin': row['distance_margin'],
            }
        )

    logger.info(f'Grouped into {len(result):,} (anchor, positive) pairs')
    return result


def _load_anchor_negative_candidates(
    triplets_parquet: str,
    anchor_idx: int,
    required_positive_idxs: Optional[Set[int]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    '''Load negative candidates for a single anchor from its partitioned parquet file.

    Args:
        triplets_parquet: Base path to partitioned triplets directory
        anchor_idx: The anchor index to load candidates for
        required_positive_idxs: Optional set of positive indices to filter to

    Returns:
        Dictionary mapping positive_idx -> list of negative dicts.
        Each negative dict has: negative_idx, negative_code, relation_margin, distance_margin.
    '''
    anchor_dir = Path(triplets_parquet) / f'anchor={anchor_idx}'
    if not anchor_dir.exists():
        return {}

    parquet_files = list(anchor_dir.glob('*.parquet'))
    if not parquet_files:
        return {}

    df = pl.read_parquet(parquet_files)

    # Filter to required positives if specified
    if required_positive_idxs is not None:
        df = df.filter(pl.col('positive_idx').is_in(list(required_positive_idxs)))

    if df.is_empty():
        return {}

    # Group by positive_idx
    result: Dict[int, List[Dict[str, Any]]] = {}
    for row in df.iter_rows(named=True):
        positive_idx = row['positive_idx']
        if positive_idx not in result:
            result[positive_idx] = []
        result[positive_idx].append(
            {
                'negative_idx': row['negative_idx'],
                'negative_code': row['negative_code'],
                'relation_margin': row['relation_margin'],
                'distance_margin': row['distance_margin'],
            }
        )

    return result

# -------------------------------------------------------------------------------------------------
# Triplet materialization helpers
# -------------------------------------------------------------------------------------------------


def _build_triplet_rows(
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    worker_id: str,
) -> List[Dict[str, Any]]:
    '''Materialize triplet rows applying the configured sampling strategy.'''

    # Load indices
    code_to_idx_dict = get_indices_codes('code_to_idx')
    idx_to_code_dict = get_indices_codes('idx_to_code')

    # Type assertions for type checker
    assert isinstance(code_to_idx_dict, dict), 'code_to_idx must be a dict'
    assert isinstance(idx_to_code_dict, dict), 'idx_to_code must be a dict'
    code_to_idx: Dict[str, int] = code_to_idx_dict  # type: ignore
    idx_to_code: Dict[int, str] = idx_to_code_dict  # type: ignore

    sampling_strategy = sampling_cfg.strategy
    sans_cfg: SansStaticConfig = sampling_cfg.sans_static

    # Load Phase 1/SANS sampling data if needed
    distance_lookup: Optional[Dict[Tuple[str, str], float]] = None
    excluded_map: Optional[Dict[str, Set[str]]] = None
    requires_distance_lookup = cfg.use_phase1_sampling or sampling_strategy == 'sans_static'

    if requires_distance_lookup:
        logger.info(f'{worker_id} Loading tree distance matrix for sampling strategy...')
        distance_lookup = _load_distance_matrix(
            cfg.distance_matrix_parquet, code_to_idx, idx_to_code
        )

    if cfg.use_phase1_sampling:
        logger.info(f'{worker_id} Loading excluded codes for Phase 1 sampling...')
        excluded_map = _load_excluded_codes(cfg.descriptions_parquet, code_to_idx)

    # Create positive sampler using taxonomy-based stratification
    logger.info(f'{worker_id} Creating positive sampler...')
    positive_sampler = create_positive_sampler(
        descriptions_parquet=cfg.descriptions_parquet,
        relations_parquet=cfg.relations_parquet,
        max_per_stratum=4,
        seed=cfg.seed,
    )

    # Pre-sample positives for each anchor
    anchor_positive_map: Dict[int, List[Dict[str, Any]]] = {}
    anchor_code_map: Dict[int, str] = {}

    for anchor_idx in positive_sampler.anchors:
        anchor_code = idx_to_code.get(anchor_idx)
        if anchor_code is None:
            continue

        positives = positive_sampler.sample_positives(anchor_idx)
        if not positives:
            continue

        anchor_code_map[anchor_idx] = anchor_code
        anchor_positive_map[anchor_idx] = positives

    if not anchor_positive_map:
        logger.warning(f'{worker_id} Positive sampling produced no anchors with positives.')
        return []

    triplet_rows: List[Dict[str, Any]] = []
    n_anchors = len(anchor_positive_map)

    for i, anchor_idx in enumerate(positive_sampler.anchors):
        positives = anchor_positive_map.get(anchor_idx)
        if not positives:
            continue

        anchor_code = anchor_code_map.get(anchor_idx)
        if anchor_code is None:
            continue

        # Load negative candidates for this anchor only (per-anchor streaming)
        required_positive_idxs = {p['positive_idx'] for p in positives}
        anchor_negatives = _load_anchor_negative_candidates(
            cfg.triplets_parquet, anchor_idx, required_positive_idxs
        )

        if not anchor_negatives:
            continue

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f'{worker_id} Processing anchor {i + 1}/{n_anchors}...')

        for positive in positives:
            positive_idx = positive['positive_idx']
            positive_code = positive['positive_code']

            # Get candidate negatives for this positive
            candidates = anchor_negatives.get(positive_idx, [])

            if not candidates:
                continue

            # Apply sampling strategy
            sampling_metadata: Optional[Dict[str, Any]] = None

            if sampling_strategy == 'sans_static' and distance_lookup is not None:
                sampled_negatives, sampling_metadata = _sample_negatives_sans_static(
                    anchor_code=anchor_code,
                    candidate_negatives=candidates,
                    n_negatives=cfg.n_negatives,
                    distance_lookup=distance_lookup,
                    sans_cfg=sans_cfg,
                    seed=cfg.seed,
                )
            elif cfg.use_phase1_sampling and distance_lookup is not None and excluded_map is not None:
                sampled_negatives = _sample_negatives_phase1(
                    anchor_code=anchor_code,
                    anchor_idx=anchor_idx,
                    candidate_negatives=candidates,
                    n_negatives=cfg.n_negatives,
                    distance_lookup=distance_lookup,
                    excluded_map=excluded_map,
                    code_to_idx=code_to_idx,
                    alpha=cfg.phase1_alpha,
                    exclusion_weight=cfg.phase1_exclusion_weight,
                    seed=cfg.seed,
                )
            else:
                rng = random.Random(cfg.seed)
                n_sample = min(cfg.n_negatives, len(candidates))
                sampled_negatives = rng.sample(candidates, n_sample)
                if sampling_strategy == 'sans_static' and distance_lookup is None:
                    logger.warning(
                        'SANS static sampling requested but tree distances were unavailable; '
                        'falling back to uniform sampling.'
                    )

            if not sampled_negatives:
                continue

            row: Dict[str, Any] = {
                'anchor_idx': anchor_idx,
                'anchor_code': anchor_code,
                'positive_idx': positive_idx,
                'positive_code': positive_code,
                'positive_level': positive['positive_level'],
                'stratum_id': positive['stratum_id'],
                'stratum_wgt': positive['stratum_wgt'],
                'negatives': sampled_negatives,
            }

            if sampling_metadata:
                row['sampling_metadata'] = sampling_metadata

            triplet_rows.append(row)

    logger.info(f'{worker_id} Built {len(triplet_rows):,} triplet rows')
    return triplet_rows

# -------------------------------------------------------------------------------------------------
# Cache utilities
# -------------------------------------------------------------------------------------------------

def _get_final_cache_path(cfg: StreamingConfig) -> Path:
    '''Get the cache file path for streaming query cache.'''

    cache_dict = {
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'relations_parquet': str(cfg.relations_parquet),
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
    }

    config_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    descriptions_path = Path(cfg.descriptions_parquet)
    cache_dir = descriptions_path.parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'streaming_final_{cache_key}.pkl'

def _load_final_cache(cfg: StreamingConfig) -> Optional[List[Dict[str, Any]]]:
    '''Load cached streaming data if available.'''
    cache_path = _get_final_cache_path(cfg)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f'Loaded streaming cache from {cache_path}')
        return data
    except Exception as e:
        logger.warning(f'Failed to load cache: {e}')
        return None

def _save_final_cache(data: List[Dict[str, Any]], cfg: StreamingConfig) -> None:
    '''Save streaming data to cache.'''
    cache_path = _get_final_cache_path(cfg)
    try:
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        temp_path.replace(cache_path)
        logger.info(f'Saved streaming cache to {cache_path}')
    except Exception as e:
        logger.warning(f'Failed to save cache: {e}')

# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def create_streaming_generator(
    cfg: StreamingConfig, sampling_cfg: Optional[SamplingConfig] = None
) -> Iterator[Dict[str, Any]]:
    '''Create a generator that yields triplets for training, using cached data when available.'''

    # Identify worker process
    worker_info = None
    try:
        import torch

        worker_info = torch.utils.data.get_worker_info()
    except Exception:
        pass

    worker_id = f'Worker {worker_info.id}' if worker_info else 'Main'
    allow_cache_save = worker_info is None

    if sampling_cfg is None:
        sampling_cfg = SamplingConfig()

    triplet_rows = _load_final_cache(cfg)

    if triplet_rows is None:
        triplet_rows = _build_triplet_rows(cfg, sampling_cfg, worker_id)
        if not triplet_rows:
            return

        if allow_cache_save:
            _save_final_cache(triplet_rows, cfg)
        else:
            logger.info(
                f'{worker_id} Cache miss but worker context prevents saving; proceeding without cache'
            )
    else:
        if allow_cache_save:
            logger.info(
                f'{worker_id} Loaded streaming cache with {len(triplet_rows):,} triplet rows'
            )

    for row in triplet_rows:
        negatives = [
            {
                'negative_idx': neg['negative_idx'],
                'negative_code': neg['negative_code'],
                'relation_margin': neg['relation_margin'],
                'distance_margin': neg['distance_margin'],
            }
            for neg in row['negatives']
        ]

        yield {
            'anchor_idx': row['anchor_idx'],
            'anchor_code': row['anchor_code'],
            'positive_idx': row['positive_idx'],
            'positive_code': row['positive_code'],
            'positive_level': row.get('positive_level'),
            'stratum_id': row.get('stratum_id'),
            'stratum_wgt': row.get('stratum_wgt'),
            'negatives': negatives,
            'sampling_metadata': row.get('sampling_metadata'),
        }

# -------------------------------------------------------------------------------------------------
# Multi-epoch triplet builder
# -------------------------------------------------------------------------------------------------

def _get_multi_epoch_cache_path(cfg: StreamingConfig, n_epochs: int) -> Path:
    '''Get the cache file path for multi-epoch triplet cache.'''

    cache_dict = {
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'relations_parquet': str(cfg.relations_parquet),
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
        'n_epochs': n_epochs,
    }

    config_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    descriptions_path = Path(cfg.descriptions_parquet)
    cache_dir = descriptions_path.parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'multi_epoch_{cache_key}.pkl'


def _load_multi_epoch_cache(cfg: StreamingConfig, n_epochs: int) -> Optional[List[Dict[str, Any]]]:
    '''Load cached multi-epoch triplet data if available.'''
    cache_path = _get_multi_epoch_cache_path(cfg, n_epochs)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f'Loaded multi-epoch cache from {cache_path}')
        return data
    except Exception as e:
        logger.warning(f'Failed to load multi-epoch cache: {e}')
        return None


def _save_multi_epoch_cache(
    data: List[Dict[str, Any]], cfg: StreamingConfig, n_epochs: int
) -> None:
    '''Save multi-epoch triplet data to cache.'''
    cache_path = _get_multi_epoch_cache_path(cfg, n_epochs)
    try:
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        temp_path.replace(cache_path)
        logger.info(f'Saved multi-epoch cache to {cache_path}')
    except Exception as e:
        logger.warning(f'Failed to save multi-epoch cache: {e}')


def build_multi_epoch_triplets(
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    n_epochs: int = 100,
) -> List[Dict[str, Any]]:
    '''
    Build triplet rows for multiple epochs with different seeds per epoch.

    Each epoch gets a different random seed for negative sampling, providing
    diversity in training examples across epochs.

    Args:
        cfg: Streaming configuration
        sampling_cfg: Sampling configuration
        n_epochs: Number of epochs to pre-sample (default: 100)

    Returns:
        List of triplet rows for all epochs combined
    '''
    # Try to load from cache first
    cached = _load_multi_epoch_cache(cfg, n_epochs)
    if cached is not None:
        logger.info(f'Loaded {len(cached):,} triplet rows from multi-epoch cache')
        return cached

    logger.info(f'Building triplet rows for {n_epochs} epochs...')
    all_rows: List[Dict[str, Any]] = []
    base_seed = cfg.seed

    for epoch in range(n_epochs):
        # Create epoch-specific config with different seed
        epoch_cfg = cfg.model_copy()
        epoch_cfg.seed = base_seed + epoch

        epoch_rows = _build_triplet_rows(epoch_cfg, sampling_cfg, worker_id='Main')
        logger.info(f'  Epoch {epoch + 1}/{n_epochs}: {len(epoch_rows):,} triplet rows')
        all_rows.extend(epoch_rows)

    logger.info(f'Built {len(all_rows):,} total triplet rows for {n_epochs} epochs')

    # Save to cache
    _save_multi_epoch_cache(all_rows, cfg, n_epochs)

    return all_rows


# -------------------------------------------------------------------------------------------------
# Streaming dataset generator (legacy - kept for compatibility)
# -------------------------------------------------------------------------------------------------

def create_streaming_dataset(
    token_cache: Dict[int, Dict[str, Any]],
    cfg: StreamingConfig,
    sampling_cfg: Optional[SamplingConfig] = None,
) -> Iterator[Dict[str, Any]]:
    '''Create streaming dataset that yields per-positive triplets with tokenized embeddings.'''

    triplets_iterator = create_streaming_generator(cfg, sampling_cfg)

    def _extract_embedding(idx: int) -> Optional[Dict[str, Any]]:
        try:
            return {k: v for k, v in token_cache[idx].items() if k != 'code'}
        except KeyError:
            logger.warning(f'Missing token_cache for index {idx}, skipping sample')
            return None

    for triplet in triplets_iterator:
        anchor_idx = int(triplet['anchor_idx'])
        positive_idx = int(triplet['positive_idx'])

        anchor_embedding = _extract_embedding(anchor_idx)
        if anchor_embedding is None:
            continue

        positive_embedding = _extract_embedding(positive_idx)
        if positive_embedding is None:
            continue

        negative_entries = []
        for neg in triplet.get('negatives', []):
            neg_embedding = _extract_embedding(int(neg['negative_idx']))
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
            continue

        result: Dict[str, Any] = {
            'anchor_idx': anchor_idx,
            'anchor_code': triplet['anchor_code'],
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': triplet['positive_code'],
            'positive_level': triplet.get('positive_level', len(triplet['positive_code'])),
            'stratum_id': triplet.get('stratum_id', 0),
            'stratum_wgt': triplet.get('stratum_wgt', 1.0),
            'positive_embedding': positive_embedding,
            'negatives': negative_entries,
        }

        sampling_metadata = triplet.get('sampling_metadata')
        if sampling_metadata:
            result['sampling_metadata'] = sampling_metadata

        yield result


# -------------------------------------------------------------------------------------------------
# Repaired Stage-3: bundle-backed candidate pools
# -------------------------------------------------------------------------------------------------

NEGATIVE_ROLE_ID = SAMPLING_ROLE_TO_ID[SamplingRole.NEGATIVE]
STREAMING_CACHE_SCHEMA_VERSION = 'streaming-candidates-v1'
RAW_CANDIDATE_KEYS = (
    'negative_code_id',
    'negative_code',
    'negative_structural_distance',
    'sampling_role_id',
    'sampling_provenance_id',
)

class IndexDistanceLookup(Mapping):
    '''
    ``(anchor_code, candidate_code) -> structural distance`` view over a ``SupervisionIndex``.

    Lets the legacy sampling strategies read validated bundle structure without materializing a
    per-pair dictionary.
    '''

    def __init__(self, index: SupervisionIndex):
        self._index = index

    def __getitem__(self, key: Tuple[str, str]) -> float:
        anchor_code, candidate_code = key
        return float(
            self._index.structural_distance[
                self._index.code_to_id[anchor_code], self._index.code_to_id[candidate_code]
            ]
        )

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, tuple) and len(key) == 2 and key[0] in self._index.code_to_id
            and key[1] in self._index.code_to_id
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        codes = self._index.id_to_code
        return ((anchor, candidate) for anchor in codes for candidate in codes)

    def __len__(self) -> int:
        return len(self._index.id_to_code)**2

def build_candidate_pool(
    *,
    anchor_code_id: int,
    positive_code_id: int,
    raw_candidates: List[Dict[str, Any]],
    supervision_index: SupervisionIndex,
    n_candidates: int,
    final_k: int,
    epoch: int,
    seed: int,
) -> List[Dict[str, Any]]:
    '''
    Build one canonical candidate pool for an (anchor, positive) pair.

    The pool contains every explicit exclusion of the anchor (either direction) and unique
    ordinary codes: the raw candidates in a stable, epoch-dependent shuffle, backfilled from the
    remaining non-exclusion universe when needed. Because final selection admits exactly one
    exclusion, the pool keeps at least ``final_k - 1`` ordinary codes when any exclusion exists (or
    ``final_k`` when none does), and at least ``n_candidates - exclusions``. The anchor and positive
    codes never appear.

    Raises:
        ValueError: If ``final_k < 1`` or the universe cannot supply ``final_k`` selectable codes.
    '''

    if final_k < 1:
        raise ValueError('final negative count must be at least one')
    forbidden = {anchor_code_id, positive_code_id}
    exclusion_ids = tuple(
        code_id
        for code_id in supervision_index.exclusion_code_ids(anchor_code_id)
        if code_id not in forbidden
    )
    exclusion_set = set(exclusion_ids)

    def normalized_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {key: item[key] for key in RAW_CANDIDATE_KEYS if key in item}
        normalized['negative_code_id'] = int(item['negative_code_id'])
        normalized['negative_is_explicit_exclusion'] = normalized['negative_code_id'] in exclusion_set
        return normalized

    def backfill_candidate(code_id: int) -> Dict[str, Any]:
        return {
            'negative_code_id': code_id,
            'negative_code': supervision_index.id_to_code[code_id],
            'negative_structural_distance': float(
                supervision_index.structural_distance[anchor_code_id, code_id]
            ),
            'negative_is_explicit_exclusion': code_id in exclusion_set,
            'sampling_role_id': NEGATIVE_ROLE_ID,
            'sampling_provenance_id': int(SamplingProvenance.BACKFILL),
        }

    by_code: Dict[int, Dict[str, Any]] = {}
    for item in raw_candidates:
        code_id = int(item['negative_code_id'])
        if code_id not in forbidden:
            by_code.setdefault(code_id, normalized_candidate(item))
    for code_id in exclusion_ids:
        by_code.setdefault(code_id, backfill_candidate(code_id))

    exclusion_slots = min(len(exclusion_ids), 1)
    ordinary_target = max(n_candidates - len(exclusion_ids), final_k - exclusion_slots, 0)
    rng = np.random.default_rng((stable_hash(seed, anchor_code_id) + epoch) % (2**63))
    ordinary_ids = sorted(code_id for code_id in by_code if code_id not in exclusion_set)
    rng.shuffle(ordinary_ids)
    kept_ordinary = ordinary_ids[:ordinary_target]

    if len(kept_ordinary) < ordinary_target:
        universe = [
            code_id
            for code_id in range(len(supervision_index.id_to_code))
            if code_id not in forbidden and code_id not in exclusion_set
            and code_id not in by_code
        ]
        rng.shuffle(universe)
        for code_id in universe[:ordinary_target - len(kept_ordinary)]:
            by_code[code_id] = backfill_candidate(code_id)
            kept_ordinary.append(code_id)

    if len(kept_ordinary) + exclusion_slots < final_k:
        raise ValueError(
            f'anchor code ID {anchor_code_id} requires {final_k} selectable candidates; only '
            f'{len(kept_ordinary) + exclusion_slots} exist (one exclusion slot plus unique '
            'non-exclusion codes)'
        )
    return [by_code[code_id] for code_id in (*exclusion_ids, *kept_ordinary)]

def _validate_streaming_cache_envelope(
    envelope: Any,
    *,
    expected_contract: str,
    expected_bundle_id: str,
    expected_codebook_fingerprint: str,
    expected_source_fingerprints: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    '''
    Validate a repaired streaming-cache envelope and return its payload.

    Raises:
        ValueError: If the cache is unversioned, belongs to another bundle/contract/codebook, was
            derived from other source artifacts, or carries a malformed payload.
    '''

    if not isinstance(envelope, dict):
        raise ValueError('repaired streaming cache is not a versioned envelope')
    expected = {
        'contract_version': expected_contract,
        'bundle_id': expected_bundle_id,
        'codebook_fingerprint': expected_codebook_fingerprint,
        'cache_schema_version': STREAMING_CACHE_SCHEMA_VERSION,
    }
    if expected_source_fingerprints is not None:
        expected['source_fingerprints'] = expected_source_fingerprints
    for field, expected_value in expected.items():
        if field not in envelope:
            raise ValueError(f'repaired streaming cache lacks {field}')
        if envelope[field] != expected_value:
            raise ValueError(
                f'repaired streaming cache {field} mismatch: '
                f'expected {expected_value!r}, found {envelope[field]!r}'
            )
    payload = envelope.get('payload')
    if not isinstance(payload, list):
        raise ValueError('repaired streaming cache payload must be a list')
    return payload

def training_pair_partitions(bundle: ValidatedSupervisionBundle) -> Dict[int, Path]:
    '''Map anchor code ID -> its training-pair member file, from the manifest member list only.'''

    partitions: Dict[int, Path] = {}
    for path in bundle.member_paths('training_pairs'):
        directory = path.parent.name
        if not directory.startswith('anchor='):
            raise ValueError(f'unexpected training-pair member layout: {path}')
        partitions[int(directory.split('=', 1)[1])] = path
    return partitions

_RAW_CANDIDATE_COLUMNS = (
    'anchor_code_id',
    'positive_code_id',
    'negative_code_id',
    'negative_code',
    'negative_structural_distance',
)

def _raw_candidates_by_pair(frame: pl.DataFrame) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for anchor, positive, negative, code, distance in frame.select(_RAW_CANDIDATE_COLUMNS).rows():
        grouped.setdefault((int(anchor), int(positive)), []).append(
            {
                'negative_code_id': int(negative),
                'negative_code': code,
                'negative_structural_distance': float(distance),
                'sampling_role_id': NEGATIVE_ROLE_ID,
                'sampling_provenance_id': int(SamplingProvenance.GENERATED),
            }
        )
    return grouped

def load_bundle_candidates(
    bundle: ValidatedSupervisionBundle,
    required_pairs: Set[Tuple[int, int]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    '''Raw generated negatives for the required (anchor, positive) code-ID pairs.'''

    if not required_pairs:
        return {}
    partitions = training_pair_partitions(bundle)
    files = sorted(
        {str(partitions[anchor]) for anchor, _ in required_pairs if anchor in partitions}
    )
    if not files:
        return {}
    pairs = pl.DataFrame(
        sorted(required_pairs),
        schema={'anchor_code_id': pl.Int32, 'positive_code_id': pl.Int32},
        orient='row',
    )
    frame = pl.scan_parquet(files).select(_RAW_CANDIDATE_COLUMNS).with_columns(
        pl.col('anchor_code_id').cast(pl.Int32),
        pl.col('positive_code_id').cast(pl.Int32),
    ).join(pairs.lazy(), on=['anchor_code_id', 'positive_code_id'], how='semi').collect()
    return _raw_candidates_by_pair(frame)

def repaired_positive_sampler(
    cfg: StreamingConfig,
    bundle: ValidatedSupervisionBundle,
    index: SupervisionIndex,
) -> PositiveSampler:
    '''Positive sampler whose relations and code identity come only from the bundle.'''

    return create_positive_sampler(
        descriptions_parquet=cfg.descriptions_parquet,
        relations_parquet=str(bundle.artifact_path('relations')),
        max_per_stratum=4,
        seed=cfg.seed,
        code_to_idx=dict(index.code_to_id),
    )

def sampled_positive_pairs(
    sampler: PositiveSampler,
    index: SupervisionIndex,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
    '''
    Sample positives per anchor, dropping any positive that is an explicit exclusion of its anchor.

    Returns:
        ``(anchor_code_id, positive)`` pairs and the number of dropped exclusion positives.
    '''

    pairs: List[Tuple[int, Dict[str, Any]]] = []
    dropped = 0
    for anchor_code_id in sampler.anchors:
        exclusions = set(index.exclusion_code_ids(anchor_code_id))
        for positive in sampler.sample_positives(anchor_code_id):
            if int(positive['positive_idx']) in exclusions:
                dropped += 1
                continue
            pairs.append((anchor_code_id, positive))
    return pairs, dropped

def sample_raw_candidates(
    *,
    anchor_code: str,
    anchor_code_id: int,
    candidates: List[Dict[str, Any]],
    n_sample: int,
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    distance_lookup: IndexDistanceLookup,
    index: SupervisionIndex,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    '''
    Apply the configured raw sampling strategy without any exclusion weighting.

    Exclusion representation belongs to the candidate pool (every exclusion) and the selection
    quota (exactly one), never to a sampling weight.
    '''

    metadata: Optional[Dict[str, Any]] = None
    if sampling_cfg.strategy == 'sans_static':
        sampled, metadata = _sample_negatives_sans_static(
            anchor_code=anchor_code,
            candidate_negatives=candidates,
            n_negatives=n_sample,
            distance_lookup=distance_lookup,  # type: ignore[arg-type]
            sans_cfg=sampling_cfg.sans_static,
            seed=seed,
        )
    elif cfg.use_phase1_sampling:
        sampled = _sample_negatives_phase1(
            anchor_code=anchor_code,
            anchor_idx=anchor_code_id,
            candidate_negatives=candidates,
            n_negatives=n_sample,
            distance_lookup=distance_lookup,  # type: ignore[arg-type]
            excluded_map={},
            code_to_idx=index.code_to_id,
            alpha=cfg.phase1_alpha,
            exclusion_weight=None,
            seed=seed,
        )
    else:
        sampled = random.Random(seed).sample(candidates, min(n_sample, len(candidates)))
    return [{key: item[key] for key in RAW_CANDIDATE_KEYS} for item in sampled], metadata

def build_repaired_triplet_rows(
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    bundle: ValidatedSupervisionBundle,
    index: SupervisionIndex,
    *,
    sampling_epoch: int,
) -> List[Dict[str, Any]]:
    '''
    Raw (anchor, positive, raw candidates) rows for one pre-sampled epoch.

    Pairs whose positive is an explicit exclusion, or which have no generated raw candidates, are
    skipped. Candidate pools are built per item at dataset access time, not here.
    '''

    sampler = repaired_positive_sampler(cfg, bundle, index)
    pairs, dropped = sampled_positive_pairs(sampler, index)
    if dropped:
        logger.info(f'Dropped {dropped:,} sampled positives that are explicit exclusions')
    candidates_by_pair = load_bundle_candidates(
        bundle,
        {(anchor, int(positive['positive_idx'])) for anchor, positive in pairs},
    )
    distance_lookup = IndexDistanceLookup(index)
    rows: List[Dict[str, Any]] = []
    for anchor_code_id, positive in pairs:
        positive_code_id = int(positive['positive_idx'])
        candidates = candidates_by_pair.get((anchor_code_id, positive_code_id))
        if not candidates:
            continue
        anchor_code = index.id_to_code[anchor_code_id]
        raw, metadata = sample_raw_candidates(
            anchor_code=anchor_code,
            anchor_code_id=anchor_code_id,
            candidates=candidates,
            n_sample=cfg.n_negatives,
            cfg=cfg,
            sampling_cfg=sampling_cfg,
            distance_lookup=distance_lookup,
            index=index,
            seed=cfg.seed,
        )
        if not raw:
            continue
        row: Dict[str, Any] = {
            'anchor_code_id': anchor_code_id,
            'anchor_code': anchor_code,
            'positive_code_id': positive_code_id,
            'positive_code': index.id_to_code[positive_code_id],
            'positive_level': positive['positive_level'],
            'stratum_id': positive['stratum_id'],
            'stratum_wgt': positive['stratum_wgt'],
            'sampling_epoch': sampling_epoch,
            'raw_candidates': raw,
        }
        if metadata:
            row['sampling_metadata'] = metadata
        rows.append(row)
    logger.info(f'Built {len(rows):,} repaired triplet rows for sampling epoch {sampling_epoch}')
    return rows

def _repaired_source_fingerprints(bundle: ValidatedSupervisionBundle) -> Dict[str, str]:
    artifacts = bundle.manifest.artifacts
    return {
        name: aggregate_fingerprint(artifacts[name].files) for name in ('relations', 'training_pairs')
    }

def _get_repaired_multi_epoch_cache_path(
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    bundle: ValidatedSupervisionBundle,
    n_epochs: int,
) -> Path:
    '''Cache path keyed by bundle identity, source fingerprints, and every sampling parameter.'''

    manifest = bundle.manifest
    cache_dict = {
        'contract_version': manifest.contract_version,
        'bundle_id': manifest.bundle_id,
        'codebook_fingerprint': manifest.codebook_fingerprint,
        'cache_schema_version': STREAMING_CACHE_SCHEMA_VERSION,
        'source_fingerprints': _repaired_source_fingerprints(bundle),
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
        'use_phase1_sampling': cfg.use_phase1_sampling,
        'phase1_alpha': cfg.phase1_alpha,
        'sampling': sampling_cfg.model_dump(),
        'n_epochs': n_epochs,
    }
    config_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    cache_dir = Path(cfg.descriptions_parquet).parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'repaired_multi_epoch_{cache_key}.pkl'

def build_repaired_multi_epoch_rows(
    cfg: StreamingConfig,
    sampling_cfg: SamplingConfig,
    bundle: ValidatedSupervisionBundle,
    index: SupervisionIndex,
    n_epochs: int,
) -> List[Dict[str, Any]]:
    '''
    Raw repaired rows for ``n_epochs`` pre-sampled epochs, cached in a versioned envelope.

    A cached envelope is accepted only when it matches this bundle's contract, ID, codebook
    fingerprint, cache schema, and source-artifact fingerprints; anything else is fatal.
    '''

    manifest = bundle.manifest
    cache_path = _get_repaired_multi_epoch_cache_path(cfg, sampling_cfg, bundle, n_epochs)
    if cache_path.exists():
        with open(cache_path, 'rb') as stream:
            envelope = pickle.load(stream)
        rows = _validate_streaming_cache_envelope(
            envelope,
            expected_contract=manifest.contract_version,
            expected_bundle_id=manifest.bundle_id,
            expected_codebook_fingerprint=manifest.codebook_fingerprint,
            expected_source_fingerprints=_repaired_source_fingerprints(bundle),
        )
        logger.info(f'Loaded {len(rows):,} repaired rows from {cache_path}')
        return rows

    rows: List[Dict[str, Any]] = []
    for epoch in range(n_epochs):
        epoch_cfg = cfg.model_copy(update={'seed': cfg.seed + epoch})
        rows.extend(
            build_repaired_triplet_rows(
                epoch_cfg, sampling_cfg, bundle, index, sampling_epoch=epoch
            )
        )
    envelope = {
        'contract_version': manifest.contract_version,
        'bundle_id': manifest.bundle_id,
        'codebook_fingerprint': manifest.codebook_fingerprint,
        'cache_schema_version': STREAMING_CACHE_SCHEMA_VERSION,
        'source_fingerprints': _repaired_source_fingerprints(bundle),
        'payload': rows,
    }
    temp_path = cache_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as stream:
        pickle.dump(envelope, stream)
    temp_path.replace(cache_path)
    logger.info(f'Saved {len(rows):,} repaired rows to {cache_path}')
    return rows
