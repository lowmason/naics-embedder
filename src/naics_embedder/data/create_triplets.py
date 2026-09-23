'''
Training-pair projection from canonical supervision pair facts.

Positive/negative combinatorics reproduce the legacy generator: a positive is a canonical,
non-maximal, non-exclusion pair; a negative ``j`` for (anchor ``a``, positive ``p``) requires the
directed rows ``p -> j`` and ``a -> j``; cross-sector negatives are capped per (anchor, positive).
Semantics are explicit columns: structural values are never overloaded to carry exclusion meaning.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Iterator, List

import numpy as np
import polars as pl

from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE,
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
    LINEAL_ADJUSTED_DISTANCE_MARGIN,
    LINEAL_DISTANCE_DELTA,
    SamplingRole,
    SemanticSource,
    SemanticTarget,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

CROSS_SECTOR_NEGATIVE_CAP = 100
CROSS_SECTOR_CAP_SEED = 0
MAX_PAIRS_PER_BATCH = 4_000

# Legacy margin weights, preserved for graph-model compatibility. The structural margin special
# cases live in supervision.schema so the runtime eligibility rule shares them.
RELATION_MARGIN_WEIGHT = 0.3333
DISTANCE_MARGIN_WEIGHT = 0.6667

GENERATED_POSITIVE_PROVENANCE = 'generated_positive'
GENERATED_CANDIDATE_PROVENANCE = 'generated_candidate'

# -------------------------------------------------------------------------------------------------
# Directed anchor view
# -------------------------------------------------------------------------------------------------

_SHARED_PAIR_COLUMNS = (
    'structural_distance',
    'structural_relation_id',
    'structural_relation_name',
    'is_explicit_exclusion',
)

def _anchor_view(pair_facts: pl.DataFrame) -> pl.DataFrame:
    '''
    Directed (anchor, candidate) rows reproducing the legacy generator's orientations.

    Every canonical pair contributes its canonical direction. A same-level pair whose codes differ
    in their two-digit prefix also contributes the reversed direction, exactly as the legacy
    keep-filter emitted both orientations of such pairs. Structural values always come from the
    canonical fact; exclusion directions are mapped into the anchor's view.
    '''

    canonical = pair_facts.select(
        pl.col('code_i_id').alias('anchor_code_id'),
        pl.col('code_j_id').alias('candidate_code_id'),
        pl.col('code_i').alias('anchor_code'),
        pl.col('code_j').alias('candidate_code'),
        *_SHARED_PAIR_COLUMNS,
        pl.col('code_i_excludes_code_j').alias('anchor_excludes_candidate'),
        pl.col('code_j_excludes_code_i').alias('candidate_excludes_anchor'),
        pl.lit(False).alias('is_reversed'),
    )
    reversed_rows = pair_facts.filter(
        pl.col('code_i').str.len_chars().eq(pl.col('code_j').str.len_chars())
        & pl.col('code_i').str.slice(0, 2).ne(pl.col('code_j').str.slice(0, 2))
    ).select(
        pl.col('code_j_id').alias('anchor_code_id'),
        pl.col('code_i_id').alias('candidate_code_id'),
        pl.col('code_j').alias('anchor_code'),
        pl.col('code_i').alias('candidate_code'),
        *_SHARED_PAIR_COLUMNS,
        pl.col('code_j_excludes_code_i').alias('anchor_excludes_candidate'),
        pl.col('code_i_excludes_code_j').alias('candidate_excludes_anchor'),
        pl.lit(True).alias('is_reversed'),
    )
    return pl.concat([canonical, reversed_rows])

def _positive_pairs(anchor_view: pl.DataFrame, max_distance: float) -> pl.DataFrame:
    '''Canonical, non-maximal, non-exclusion pairs; reversed rows never become positives.'''

    return anchor_view.filter(
        ~pl.col('is_reversed'),
        pl.col('structural_distance').gt(0.0),
        pl.col('structural_distance').ne(max_distance),
        ~pl.col('is_explicit_exclusion'),
    ).select(
        pl.col('anchor_code_id'),
        pl.col('candidate_code_id').alias('positive_code_id'),
        pl.col('anchor_code'),
        pl.col('candidate_code').alias('positive_code'),
        pl.col('structural_distance').alias('positive_structural_distance'),
        pl.col('structural_relation_id').alias('positive_structural_relation_id'),
        pl.col('structural_relation_name').alias('positive_structural_relation_name'),
        pl.col('is_explicit_exclusion').alias('positive_is_explicit_exclusion'),
    )

def _negative_candidates(anchor_view: pl.DataFrame) -> pl.DataFrame:
    return anchor_view.select(
        pl.col('anchor_code_id'),
        pl.col('candidate_code_id').alias('negative_code_id'),
        pl.col('candidate_code').alias('negative_code'),
        pl.col('structural_distance').alias('negative_structural_distance'),
        pl.col('structural_relation_id').alias('negative_structural_relation_id'),
        pl.col('structural_relation_name').alias('negative_structural_relation_name'),
        pl.col('anchor_excludes_candidate').alias('anchor_excludes_negative'),
        pl.col('candidate_excludes_anchor').alias('negative_excludes_anchor'),
        pl.col('is_explicit_exclusion').alias('negative_is_explicit_exclusion'),
    )

# -------------------------------------------------------------------------------------------------
# Structural margins
# -------------------------------------------------------------------------------------------------

def _structural_margins(frame: pl.DataFrame) -> pl.DataFrame:
    '''
    Add relation/distance margins with the legacy special cases and keep ordered triplets.

    Cross-sector negatives receive fixed margins; equal distances and the -0.5 lineal adjustment
    receive fixed distance margins when the relation margin is positive. Triplets whose negative is
    not structurally farther than the positive are dropped.
    '''

    relation_delta = (
        pl.col('negative_structural_relation_id').cast(pl.Float64)
        - pl.col('positive_structural_relation_id').cast(pl.Float64)
    )
    distance_delta = (
        pl.col('negative_structural_distance').cast(pl.Float64)
        - pl.col('positive_structural_distance').cast(pl.Float64)
    )
    cross_sector = pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE)
    return frame.with_columns(
        relation_margin=pl.when(cross_sector).then(pl.lit(CROSS_SECTOR_RELATION_MARGIN)
                                                   ).otherwise(relation_delta),
        distance_margin=pl.when(relation_delta.gt(0) & distance_delta.eq(0.0)).then(
            pl.lit(EQUAL_DISTANCE_MARGIN)
        ).when(relation_delta.gt(0) & distance_delta.eq(LINEAL_DISTANCE_DELTA)).then(
            pl.lit(LINEAL_ADJUSTED_DISTANCE_MARGIN)
        ).when(cross_sector).then(pl.lit(CROSS_SECTOR_DISTANCE_MARGIN)).otherwise(distance_delta),
    ).filter(
        pl.col('relation_margin').gt(0),
        pl.col('distance_margin').gt(0),
    ).with_columns(
        margin=(
            pl.col('relation_margin').mul(RELATION_MARGIN_WEIGHT)
            + pl.col('distance_margin').mul(DISTANCE_MARGIN_WEIGHT)
        ).pow(-1)
    ).with_columns(pl.col('relation_margin', 'distance_margin', 'margin').cast(pl.Float32))

# -------------------------------------------------------------------------------------------------
# Deterministic cross-sector cap
# -------------------------------------------------------------------------------------------------

_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)

def _mix64(values: np.ndarray) -> np.ndarray:
    '''SplitMix64 finalizer over a uint64 array (wrapping arithmetic, platform independent).'''

    mixed = (values ^ (values >> np.uint64(30))) * _MIX_MULTIPLIER_1
    mixed = (mixed ^ (mixed >> np.uint64(27))) * _MIX_MULTIPLIER_2
    return mixed ^ (mixed >> np.uint64(31))

def _cap_keys(seed: int, *parts: np.ndarray) -> np.ndarray:
    '''Stable 64-bit ranking key per row from a seed and integer identity columns.'''

    key = _mix64(np.full(parts[0].shape, seed, dtype=np.uint64) + _GOLDEN_GAMMA)
    for part in parts:
        key = _mix64((key ^ part.astype(np.uint64)) + _GOLDEN_GAMMA)
    return key

def _cap_cross_sector(frame: pl.DataFrame, cap: int, seed: int) -> pl.DataFrame:
    '''
    Keep at most ``cap`` cross-sector, non-exclusion negatives per (anchor, positive).

    Rows are ranked by a stable hash of (seed, anchor, positive, negative), so the retained subset
    is reproducible across processes and platforms. Explicit exclusions are never capped.
    '''

    capped = (
        pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE)
        & ~pl.col('negative_is_explicit_exclusion')
    )
    eligible = frame.filter(capped)
    if eligible.is_empty():
        return frame
    keys = _cap_keys(
        seed,
        eligible.get_column('anchor_code_id').to_numpy(),
        eligible.get_column('positive_code_id').to_numpy(),
        eligible.get_column('negative_code_id').to_numpy(),
    )
    kept = eligible.with_columns(cap_key=pl.Series(keys, dtype=pl.UInt64)).sort(
        'anchor_code_id', 'positive_code_id', 'cap_key', 'negative_code_id'
    ).filter(
        pl.int_range(pl.len()).over('anchor_code_id', 'positive_code_id') < cap
    ).drop('cap_key')
    return pl.concat([frame.filter(~capped), kept])

# -------------------------------------------------------------------------------------------------
# Semantic projection
# -------------------------------------------------------------------------------------------------

def _semantic_negative_columns() -> List[pl.Expr]:
    return [
        pl.when(pl.col('negative_is_explicit_exclusion')).then(
            pl.lit(SemanticTarget.UNRELATED.value)
        ).otherwise(pl.lit(SemanticTarget.UNKNOWN.value)).alias('negative_semantic_target'),
        pl.when(pl.col('negative_is_explicit_exclusion')).then(
            pl.lit(SemanticSource.EXPLICIT_EXCLUSION.value)
        ).otherwise(pl.lit(SemanticSource.UNLABELED.value)).alias('negative_semantic_source'),
        pl.lit(SamplingRole.NEGATIVE.value).alias('negative_sampling_role'),
        pl.lit(GENERATED_CANDIDATE_PROVENANCE).alias('negative_sampling_provenance'),
    ]

def _project(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(SemanticTarget.RELATED.value).alias('positive_semantic_target'),
        pl.lit(SemanticSource.TRAINING_POSITIVE.value).alias('positive_semantic_source'),
        pl.lit(SamplingRole.POSITIVE.value).alias('positive_sampling_role'),
        pl.lit(GENERATED_POSITIVE_PROVENANCE).alias('positive_sampling_provenance'),
        *_semantic_negative_columns(),
    ).select(
        # Identity
        'anchor_code_id',
        'positive_code_id',
        'negative_code_id',
        'anchor_code',
        'positive_code',
        'negative_code',
        # Raw structure
        'positive_structural_distance',
        'positive_structural_relation_id',
        'positive_structural_relation_name',
        'negative_structural_distance',
        'negative_structural_relation_id',
        'negative_structural_relation_name',
        # Semantic supervision and sampling role/provenance
        'positive_semantic_target',
        'positive_semantic_source',
        'positive_sampling_role',
        'positive_sampling_provenance',
        'negative_semantic_target',
        'negative_semantic_source',
        'negative_sampling_role',
        'negative_sampling_provenance',
        # Exclusion provenance
        'positive_is_explicit_exclusion',
        'anchor_excludes_negative',
        'negative_excludes_anchor',
        'negative_is_explicit_exclusion',
        # Structural margins
        'relation_margin',
        'distance_margin',
        'margin',
        # Legacy compatibility for graph and legacy readers; never repaired Stage-3 authorities.
        # `unrelated` keeps its legacy structural meaning (cross-sector negative).
        pl.col('anchor_code_id').cast(pl.UInt32).alias('anchor_idx'),
        pl.col('positive_code_id').cast(pl.UInt32).alias('positive_idx'),
        pl.col('negative_code_id').cast(pl.UInt32).alias('negative_idx'),
        pl.col('positive_structural_distance').alias('positive_distance'),
        pl.col('negative_structural_distance').alias('negative_distance'),
        pl.col('positive_structural_relation_id').alias('positive_relation'),
        pl.col('negative_structural_relation_id').alias('negative_relation'),
        pl.col('negative_is_explicit_exclusion').alias('excluded'),
        pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE).alias('unrelated'),
    )

# -------------------------------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------------------------------

def _validate_training_pairs(training_pairs: pl.DataFrame) -> None:
    '''Fail closed on exclusion, semantic, or identity violations in training pairs.'''

    if training_pairs.filter(pl.col('positive_is_explicit_exclusion')).height:
        raise ValueError('direct positive cannot be an explicit exclusion')
    inconsistent = training_pairs.filter(
        pl.col('negative_is_explicit_exclusion').ne(
            pl.col('anchor_excludes_negative') | pl.col('negative_excludes_anchor')
        )
    )
    if inconsistent.height:
        raise ValueError('negative exclusion derivation is inconsistent')
    expected_target = pl.when(pl.col('negative_is_explicit_exclusion')).then(
        pl.lit(SemanticTarget.UNRELATED.value)
    ).otherwise(pl.lit(SemanticTarget.UNKNOWN.value))
    if training_pairs.filter(pl.col('negative_semantic_target').ne(expected_target)).height:
        raise ValueError('negative semantic target disagrees with exclusion provenance')
    if training_pairs.select(
        pl.any_horizontal(
            pl.col('anchor_code_id').is_null(),
            pl.col('positive_code_id').is_null(),
            pl.col('negative_code_id').is_null(),
        ).any()
    ).item():
        raise ValueError('training pair contains an unmapped code identity')
    if training_pairs.filter(
        pl.col('negative_code_id').eq(pl.col('anchor_code_id'))
        | pl.col('negative_code_id').eq(pl.col('positive_code_id'))
    ).height:
        raise ValueError('a training negative repeats its anchor or positive code')

# -------------------------------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------------------------------

def _triplets_for_positives(
    positives: pl.DataFrame,
    via: pl.DataFrame,
    negatives: pl.DataFrame,
    cross_sector_cap: int,
    cap_seed: int,
) -> pl.DataFrame:
    via_positive = via.filter(
        pl.col('positive_code_id').is_in(
            positives.get_column('positive_code_id').unique().implode()
        )
    )
    anchor_negatives = negatives.filter(
        pl.col('anchor_code_id').is_in(positives.get_column('anchor_code_id').unique().implode())
    )
    triplets = positives.join(via_positive, on='positive_code_id', how='inner').join(
        anchor_negatives, on=['anchor_code_id', 'negative_code_id'], how='inner'
    ).filter(
        pl.col('negative_code_id').ne(pl.col('positive_code_id')),
        pl.col('negative_code_id').ne(pl.col('anchor_code_id')),
    )
    triplets = _cap_cross_sector(_structural_margins(triplets), cross_sector_cap, cap_seed)
    training_pairs = _project(triplets).sort('anchor_code_id', 'positive_code_id', 'negative_code_id')
    _validate_training_pairs(training_pairs)
    return training_pairs

def iter_training_pair_batches(
    pair_facts: pl.DataFrame,
    *,
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
    max_pairs_per_batch: int = MAX_PAIRS_PER_BATCH,
) -> Iterator[pl.DataFrame]:
    '''
    Yield validated training pairs in ascending-anchor batches, each sorted by
    (anchor, positive, negative). Batching bounds memory: every (anchor, positive) can carry
    thousands of cross-sector candidates before the cap applies.
    '''

    if cross_sector_cap < 0 or cap_seed < 0:
        raise ValueError('cross-sector cap and cap seed must be nonnegative')
    anchor_view = _anchor_view(pair_facts)
    max_distance = pair_facts.get_column('structural_distance').max()
    positives = _positive_pairs(anchor_view, max_distance)
    via = anchor_view.select(
        pl.col('anchor_code_id').alias('positive_code_id'),
        pl.col('candidate_code_id').alias('negative_code_id'),
    )
    negatives = _negative_candidates(anchor_view)

    batch: List[int] = []
    batch_pairs = 0
    counts = positives.group_by('anchor_code_id').len().sort('anchor_code_id')
    for anchor_code_id, n_positives in counts.iter_rows():
        if batch and batch_pairs + n_positives > max_pairs_per_batch:
            yield _triplets_for_positives(
                positives.filter(pl.col('anchor_code_id').is_in(batch)),
                via,
                negatives,
                cross_sector_cap,
                cap_seed,
            )
            batch, batch_pairs = [], 0
        batch.append(anchor_code_id)
        batch_pairs += n_positives
    if batch:
        yield _triplets_for_positives(
            positives.filter(pl.col('anchor_code_id').is_in(batch)),
            via,
            negatives,
            cross_sector_cap,
            cap_seed,
        )

def build_training_pairs(
    pair_facts: pl.DataFrame,
    *,
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
) -> pl.DataFrame:
    '''
    Build every training pair from canonical pair facts.

    Args:
        pair_facts: Canonical pair facts with directional exclusion provenance.
        cross_sector_cap: Maximum cross-sector, non-exclusion negatives per (anchor, positive).
        cap_seed: Seed for the stable ranking that chooses capped negatives.

    Returns:
        Validated training pairs sorted by (anchor, positive, negative) code IDs.
    '''

    batches = list(
        iter_training_pair_batches(
            pair_facts,
            cross_sector_cap=cross_sector_cap,
            cap_seed=cap_seed,
        )
    )
    if batches:
        return pl.concat(batches)
    anchor_view = _anchor_view(pair_facts)
    return _triplets_for_positives(
        _positive_pairs(anchor_view, 0.0).head(0),
        anchor_view.select(
            pl.col('anchor_code_id').alias('positive_code_id'),
            pl.col('candidate_code_id').alias('negative_code_id'),
        ),
        _negative_candidates(anchor_view),
        cross_sector_cap,
        cap_seed,
    )
