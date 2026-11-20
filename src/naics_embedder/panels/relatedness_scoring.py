'''
Tie-aware scores for the relatedness judging panel (spec ``relatedness-artifact``, 4.4).

Scores
    Each function reads one anchor's 1-D arrays over its candidates. A group of candidates with
    exactly equal float64 similarity is tied. ``expected_dcg`` is the DCG@k averaged over every order
    within the tied groups, with the groups in descending similarity (McSherry and Najork, 2008). A
    group that covers the 1-based ranks s to e adds the mean gain of its members times the sum of
    1 / log2(p + 1) over the positions p from s to min(e, k). It adds nothing when s is above k.
    Gains are linear, never 2^g - 1. ``tie_aware_ndcg`` divides ``expected_dcg`` by ``ideal_dcg``,
    the DCG@k of the gains in descending order. ``tie_aware_auc`` is the share of (relevant,
    not relevant) candidate pairs that the similarity orders correctly, each tie counting one half.
    No result depends on the order of the candidates, bit for bit. A function refuses arrays of
    different length and a k below 1 with a ``ValueError``.

Summary
    ``score_anchors`` reads one part's (anchors, candidates) arrays into one row per anchor. An
    anchor is never one of its own candidates, so its own column is left out before anything else
    is read. ``summarize_relatedness`` reads those rows into the headline, the reports, the headline
    by sector and the counts (4.4). The headline is the mean NDCG@10 over the part's anchors, each
    anchor once. The reports never decide and never add to n: the mean NDCG@20, the mean AUC over the
    anchors that have one, and each sector's mean NDCG@10. An anchor's sector is the first level of
    its lineage, so 31-33, 44-45 and 48-49 are one sector each. Every mean uses ``math.fsum``, so no
    result depends on the order of the anchors.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from naics_embedder.utils.naics_hierarchy import code_lineage

# -------------------------------------------------------------------------------------------------
# Scores
# -------------------------------------------------------------------------------------------------

def _require_same_length(similarity: np.ndarray, other: np.ndarray, name: str) -> None:
    if similarity.shape != other.shape:
        raise ValueError(
            f'similarity and {name} must have the same length, got {similarity.shape} and '
            f'{other.shape}'
        )

def _require_k(k: int) -> None:
    if k < 1:
        raise ValueError(f'k must be at least 1, got {k}')

def _gain_arrays(similarity: np.ndarray, gains: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''Both arrays as float64, after the length and k checks.'''

    similarity = np.asarray(similarity, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    _require_same_length(similarity, gains, 'gains')
    _require_k(k)
    return similarity, gains

def _discounts(count: int) -> np.ndarray:
    '''1 / log2(p + 1) for the positions p = 1 to ``count``.'''

    return 1.0 / np.log2(np.arange(2, count + 2))

def expected_dcg(similarity: np.ndarray, gains: np.ndarray, k: int) -> float:
    '''
    The DCG@k of one anchor's candidates, averaged over every order within tied groups.

    Candidates whose similarities are exactly equal are tied. The groups are taken in descending
    similarity, and a group that covers the 1-based ranks s to e adds the mean gain of its members
    times the sum of 1 / log2(p + 1) over the positions p from s to min(e, k). A group that starts
    below rank k adds nothing. Gains are used as they are, not as 2^g - 1. The sum over a group's
    members does not depend on their order, so neither does the result.

    Args:
        similarity: Each candidate's similarity to the anchor, higher is nearer.
        gains: Each candidate's gain, in the same order as ``similarity``.
        k: The cut-off rank, at least 1. It may exceed the number of candidates.

    Returns:
        The expected DCG@k, 0.0 with no candidate.

    Raises:
        ValueError: When the arrays differ in length or k is below 1.
    '''

    similarity, gains = _gain_arrays(similarity, gains, k)
    # Descending similarity, and within a tie ascending gain, so a group's sum has one order
    order = np.lexsort((gains, -similarity))
    ranked_similarity = similarity[order]
    ranked_gains = gains[order]
    starts_group = np.ones(len(order), dtype=bool)
    starts_group[1:] = ranked_similarity[1:] != ranked_similarity[:-1]
    starts = np.flatnonzero(starts_group)
    sizes = np.diff(starts, append=len(order))
    mean_gains = np.add.reduceat(ranked_gains, starts) / sizes
    expected_gain_by_rank = np.repeat(mean_gains, sizes)[:k]
    return math.fsum(expected_gain_by_rank * _discounts(len(expected_gain_by_rank)))

def ideal_dcg(gains: np.ndarray, k: int) -> float:
    '''
    The DCG@k of the gains in descending order.

    Args:
        gains: Each candidate's gain.
        k: The cut-off rank, at least 1. With fewer candidates than k, every gain is summed.

    Returns:
        The ideal DCG@k, 0.0 with no candidate.

    Raises:
        ValueError: When k is below 1.
    '''

    _require_k(k)
    best = np.sort(gains)[::-1][:k]
    return math.fsum(best * _discounts(len(best)))

def tie_aware_ndcg(similarity: np.ndarray, gains: np.ndarray, k: int) -> float:
    '''
    The expected DCG@k over the ideal DCG@k of one anchor's candidates.

    Args:
        similarity: Each candidate's similarity to the anchor, higher is nearer.
        gains: Each candidate's gain, in the same order as ``similarity``.
        k: The cut-off rank, at least 1.

    Returns:
        ``expected_dcg(similarity, gains, k) / ideal_dcg(gains, k)``.

    Raises:
        ValueError: When the arrays differ in length, k is below 1, or no gain is positive, so the
            ideal DCG is 0.
    '''

    similarity, gains = _gain_arrays(similarity, gains, k)
    if not (gains > 0).any():
        raise ValueError('no gain is positive, so the ideal DCG is 0')
    return expected_dcg(similarity, gains, k) / ideal_dcg(gains, k)

def tie_aware_auc(similarity: np.ndarray, relevant: np.ndarray) -> Optional[float]:
    '''
    The share of (relevant, not relevant) candidate pairs that the similarity orders correctly.

    A pair is ordered correctly when the relevant candidate is more similar. A tie counts one half.
    The counts are whole numbers, so the result does not depend on the order of the candidates.

    Args:
        similarity: Each candidate's similarity to the anchor, higher is nearer.
        relevant: Which candidates are relevant, in the same order as ``similarity``. An entry that
            is true, or not zero, marks a relevant candidate.

    Returns:
        The share, or None when no candidate or every candidate is relevant, so there is no pair.

    Raises:
        ValueError: When the arrays differ in length.
    '''

    similarity = np.asarray(similarity, dtype=np.float64)
    relevant = np.asarray(relevant, dtype=bool)
    _require_same_length(similarity, relevant, 'relevant')
    n_relevant = int(relevant.sum())
    n_other = len(relevant) - n_relevant
    if n_relevant == 0 or n_other == 0:
        return None
    values, group = np.unique(similarity, return_inverse=True)
    relevant_in = np.bincount(group[relevant], minlength=len(values))
    other_in = np.bincount(group[~relevant], minlength=len(values))
    other_below = np.cumsum(other_in) - other_in
    wins = int((relevant_in * other_below).sum())
    ties = int((relevant_in * other_in).sum())
    return (2 * wins + ties) / (2 * n_relevant * n_other)

# -------------------------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------------------------

# The cut-off the headline reads, and the one a report reads beside it
HEADLINE_K = 10
REPORT_K = 20

# One row per anchor: its part, its relevant partners, its two NDCG and its AUC (None without one)
PER_ANCHOR_SCHEMA: Dict[str, pl.DataType] = {
    'anchor': pl.Utf8,
    'part': pl.Utf8,
    'n_relevant': pl.Int64,
    'ndcg_10': pl.Float64,
    'ndcg_20': pl.Float64,
    'auc': pl.Float64,
}

# The counts of a part (4.4), in the order a result lists them
COUNT_KEYS = (
    'n_industries',
    'n_anchors',
    'n_no_partner',
    'n_candidates',
    'n_relevant',
    'n_crossing',
)

_BY_SECTOR_SCHEMA: Dict[str, pl.DataType] = {
    'sector': pl.Utf8,
    'n_anchors': pl.Int64,
    'ndcg_10': pl.Float64,
}

def _require_shapes(
    similarity: np.ndarray,
    gains: np.ndarray,
    anchors: Sequence[str],
    candidates: Sequence[str],
) -> None:
    wanted = (len(anchors), len(candidates))
    if similarity.shape != wanted or gains.shape != wanted:
        raise ValueError(
            f'similarity and gains must both have shape {wanted}, one row for each of '
            f'{len(anchors)} anchors and one column for each of {len(candidates)} candidates, '
            f'got {similarity.shape} and {gains.shape}'
        )

def _require_finite(similarity: np.ndarray, anchors: Sequence[str]) -> None:
    '''Refuse the first anchor with a non-finite similarity anywhere in its row.'''

    bad = np.flatnonzero(~np.isfinite(similarity).all(axis=1))
    if len(bad):
        raise ValueError(f'the similarity of anchor {anchors[bad[0]]} is not finite')

def score_anchors(
    similarity: np.ndarray,
    gains: np.ndarray,
    anchors: Sequence[str],
    candidates: Sequence[str],
    part: str,
) -> pl.DataFrame:
    '''
    Score each anchor of a part: its relevant partners, NDCG@10, NDCG@20 and AUC.

    An anchor's own column, when it is a candidate, is left out before anything is read: the
    anchor is not one of its own candidates (4.4). A candidate with a positive gain is a relevant
    partner, and one with a gain of 0 has J = 0.

    Args:
        similarity: Each anchor's similarity to each candidate, (anchors, candidates), higher is
            nearer. The anchor's own column is read only to check that it is finite.
        gains: Each candidate's gain for each anchor, the same shape as ``similarity``.
        anchors: The anchors' six-digit codes, one for each row.
        candidates: The candidates' six-digit codes, one for each column.
        part: The part's name, carried into every row.

    Returns:
        One row per anchor, in the order given, with the columns and dtypes of
        ``PER_ANCHOR_SCHEMA``. ``n_relevant`` counts the anchor's positive gains and ``auc`` is
        null when every candidate has one.

    Raises:
        ValueError: When an array's shape is not (anchors, candidates). Then, once the shapes fit,
            when a similarity is not finite, the first anchor with one being named. Then, once all
            are finite, when an anchor has no positive gain among its candidates, the first being
            named.
    '''

    similarity = np.asarray(similarity, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    _require_shapes(similarity, gains, anchors, candidates)
    _require_finite(similarity, anchors)
    rows: Dict[str, list] = {name: [] for name in PER_ANCHOR_SCHEMA}
    for row, anchor in enumerate(anchors):
        keep = np.array([code != anchor for code in candidates], dtype=bool)
        row_similarity = similarity[row][keep]
        row_gains = gains[row][keep]
        relevant = row_gains > 0
        if not relevant.any():
            raise ValueError(f'anchor {anchor} has no candidate with a positive gain')
        rows['anchor'].append(anchor)
        rows['part'].append(part)
        rows['n_relevant'].append(int(relevant.sum()))
        rows['ndcg_10'].append(tie_aware_ndcg(row_similarity, row_gains, HEADLINE_K))
        rows['ndcg_20'].append(tie_aware_ndcg(row_similarity, row_gains, REPORT_K))
        rows['auc'].append(tie_aware_auc(row_similarity, relevant))
    return pl.DataFrame(rows, schema=PER_ANCHOR_SCHEMA)

@dataclass(frozen=True)
class RelatednessResult:
    '''
    A part's relatedness scores at three grains: per anchor, per sector and over the part.

    Every mean is correctly rounded and independent of row order, so the same scores always give
    the same values, bit for bit.

    Attributes:
        panel: The panel's name.
        part: The part's name.
        per_anchor: One row per anchor, in the order of the frame it was read from, in
            ``PER_ANCHOR_SCHEMA``. The decision's bootstrap resamples it.
        headline: ``n_anchors`` and ``ndcg_10``, the mean NDCG@10 over the anchors, each anchor
            once.
        reports: ``ndcg_20``, the mean NDCG@20 over the anchors, ``auc``, the mean AUC over the
            anchors that have one (None with none), and ``n_auc_anchors``, how many that is. A
            report never changes the headline and never adds to ``n_anchors``.
        by_sector: One row per sector, sorted by ``sector``: ``sector``, ``n_anchors`` and the
            sector's mean ``ndcg_10``. The combined sectors 31-33, 44-45 and 48-49 are one each.
        counts: The part's counts, keyed by ``COUNT_KEYS``.
        data_fingerprint: The part's data fingerprint, which names the table build the scores were
            read on. ``summarize_relatedness`` leaves it None, and the panel sets it.
    '''

    panel: str
    part: str
    per_anchor: pl.DataFrame
    headline: Dict[str, Union[int, float]]
    reports: Dict[str, Union[int, float, None]]
    by_sector: pl.DataFrame
    counts: Dict[str, int]
    data_fingerprint: Optional[str] = None

def _mean(values: Sequence[float]) -> Optional[float]:
    '''
    The correctly rounded mean of the values, whatever their order, or None when there are none.

    A float sum depends on its order. ``math.fsum`` has one result for the same values, so the same
    scores always give the same report, bit for bit.
    '''

    return math.fsum(values) / len(values) if values else None

def _sector_table(anchors: Sequence[str], ndcg_10: Sequence[float]) -> pl.DataFrame:
    '''The mean NDCG@10 of each sector, sorted by sector. A sector is the first level of a lineage.'''

    by_sector: Dict[str, List[float]] = {}
    for anchor, value in zip(anchors, ndcg_10):
        by_sector.setdefault(code_lineage(anchor)[0], []).append(value)
    sectors = sorted(by_sector)
    return pl.DataFrame(
        {
            'sector': sectors,
            'n_anchors': [len(by_sector[sector]) for sector in sectors],
            'ndcg_10': [_mean(by_sector[sector]) for sector in sectors],
        },
        schema=_BY_SECTOR_SCHEMA,
    )

def _require_counts(counts: Mapping[str, int], n_anchors: int) -> None:
    missing = sorted(set(COUNT_KEYS) - set(counts))
    unexpected = sorted(set(counts) - set(COUNT_KEYS))
    if missing or unexpected:
        raise ValueError(
            f'counts are not keyed by COUNT_KEYS: missing {missing}, unexpected {unexpected}'
        )
    if counts['n_anchors'] != n_anchors:
        raise ValueError(
            f"counts['n_anchors'] is {counts['n_anchors']}, but per_anchor has {n_anchors} rows"
        )

def summarize_relatedness(
    per_anchor: pl.DataFrame,
    *,
    panel: str,
    part: str,
    counts: Mapping[str, int],
) -> RelatednessResult:
    '''
    Read a part's per-anchor scores as the headline, the reports, the sectors and the counts.

    Args:
        per_anchor: ``score_anchors``'s frame for the part, one row per anchor.
        panel: The panel's name, carried into the result.
        part: The part's name, carried into the result.
        counts: The part's counts, keyed by ``COUNT_KEYS``. ``n_anchors`` is the frame's height.

    Returns:
        The part's ``RelatednessResult``, its ``data_fingerprint`` None.

    Raises:
        ValueError: When ``per_anchor`` is empty. Then, when ``counts`` is not keyed by
            ``COUNT_KEYS``, a key missing or unexpected. Then, when ``counts['n_anchors']`` differs
            from the frame's height.
    '''

    if per_anchor.is_empty():
        raise ValueError('per_anchor is empty: a part with no scored anchor has no headline')
    _require_counts(counts, per_anchor.height)
    anchors = per_anchor.get_column('anchor').to_list()
    ndcg_10 = per_anchor.get_column('ndcg_10').to_list()
    aucs = per_anchor.get_column('auc').drop_nulls().to_list()
    return RelatednessResult(
        panel=panel,
        part=part,
        per_anchor=per_anchor,
        headline={
            'n_anchors': per_anchor.height,
            'ndcg_10': _mean(ndcg_10),
        },
        reports={
            'ndcg_20': _mean(per_anchor.get_column('ndcg_20').to_list()),
            'auc': _mean(aucs),
            'n_auc_anchors': len(aucs),
        },
        by_sector=_sector_table(anchors, ndcg_10),
        counts={key: counts[key]
                for key in COUNT_KEYS},
    )
