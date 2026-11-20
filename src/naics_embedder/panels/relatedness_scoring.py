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
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from typing import Optional, Tuple

import numpy as np

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
