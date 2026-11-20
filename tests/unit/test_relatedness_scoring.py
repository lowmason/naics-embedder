'''
Tie-aware scores for the relatedness panel (plan 21, Task 3; spec relatedness-artifact, 4.4).

Each test reads one anchor's arrays. The oracle is brute force: the mean DCG over every order within
tied groups, written here from the definition and not from the scorer's formula. The tree row of
311211 on the base fixture is the real tied case: D* takes 5 values over its 13 candidates. The
closed forms below spell out each group's ranks and discounts by hand. No test reads a real table.

Task 4 adds the per-anchor frame and the result (plan 21; spec relatedness-artifact, 4.4). Its tests
read the same base fixture and hand-built frames, and the oracle for a row is Task 3's scorer on the
anchor's candidates with the anchor left out, itself pinned to brute force above.
'''

import dataclasses
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.relatedness import co_ownership_inputs
from naics_embedder.panels.relatedness_scoring import (
    COUNT_KEYS,
    HEADLINE_K,
    PER_ANCHOR_SCHEMA,
    REPORT_K,
    RelatednessResult,
    expected_dcg,
    ideal_dcg,
    score_anchors,
    summarize_relatedness,
    tie_aware_auc,
    tie_aware_ndcg,
)
from tests.fixtures.relatedness import RELATEDNESS_CODES, co_ownership_tables, tree_distance

ANCHOR = '311211'

# -------------------------------------------------------------------------------------------------
# Oracles
# -------------------------------------------------------------------------------------------------

def discount(position: int) -> float:
    '''The discount of a 1-based position.'''

    return 1.0 / math.log2(position + 1)

def discounts(first: int, last: int) -> float:
    '''The sum of the discounts of positions first to last, both included.'''

    return math.fsum(discount(position) for position in range(first, last + 1))

def brute_force_dcg(similarity: np.ndarray, gains: np.ndarray, k: int) -> float:
    '''The mean DCG@k over every order within tied groups, the groups in descending similarity.'''

    values = sorted(set(similarity.tolist()), reverse=True)
    groups = [
        [gain for tied, gain in zip(similarity.tolist(), gains.tolist()) if tied == value]
        for value in values
    ]
    totals = []
    for arrangement in itertools.product(*(itertools.permutations(group) for group in groups)):
        order = [gain for part in arrangement for gain in part]
        totals.append(math.fsum(gain * discount(p) for p, gain in enumerate(order[:k], start=1)))
    return math.fsum(totals) / len(totals)

def pairwise_auc(similarity: np.ndarray, relevant: np.ndarray) -> Optional[float]:
    '''The share of (relevant, other) pairs ordered correctly, each tie counted one half.'''

    wins = 0.0
    pairs = 0
    for sim_a, is_relevant in zip(similarity.tolist(), relevant.tolist()):
        if not is_relevant:
            continue
        for sim_b, other_is_relevant in zip(similarity.tolist(), relevant.tolist()):
            if other_is_relevant:
                continue
            pairs += 1
            wins += 1.0 if sim_a > sim_b else 0.5 if sim_a == sim_b else 0.0
    return wins / pairs if pairs else None

# -------------------------------------------------------------------------------------------------
# The tree row of 311211
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tree_row():
    '''The candidates of 311211, the tree arm's similarity to it and its gains.'''

    co_ownership = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    candidates = [code for code in co_ownership.candidates if code != ANCHOR]
    similarity = -tree_distance([ANCHOR, *candidates])[0, 1:]
    gains = np.array([co_ownership.gain(ANCHOR, code) for code in candidates])
    return candidates, similarity, gains

def test_the_311211_row_has_the_tied_groups_and_gains_the_brief_names(tree_row):
    candidates, similarity, gains = tree_row
    values, sizes = np.unique(similarity, return_counts=True)
    assert len(candidates) == 13
    assert values.tolist() == [-10.0, -8.0, -6.0, -4.0, -2.0]
    assert sizes.tolist() == [7, 2, 2, 1, 1]
    assert math.prod(math.factorial(size) for size in sizes.tolist()) == 2 * 2 * 5040
    by_code = dict(zip(candidates, gains.tolist()))
    assert by_code['311212'] == 1.0
    assert by_code['311221'] == 2 / 3
    assert by_code['111120'] == 1 / 3
    assert sum(1 for gain in gains.tolist() if gain > 0) == 3

def test_the_tree_row_of_311211_matches_brute_force(tree_row):
    _, similarity, gains = tree_row
    for k in (3, 10, 20):
        assert expected_dcg(similarity, gains,
                            k) == pytest.approx(brute_force_dcg(similarity, gains, k), rel=1e-12)

def test_the_tree_row_of_311211_matches_its_closed_form(tree_row):
    _, similarity, gains = tree_row
    # ranks 1, 2 are 311212 (1) and 311221 (2/3). Ranks 3-4 and 5-6 hold gain 0. Ranks 7-13 are
    # the seven codes at D* 10, whose mean gain is (1/3) / 7
    head = discount(1) + (2 / 3) * discount(2)
    assert expected_dcg(similarity, gains, 3) == pytest.approx(head, rel=1e-12)
    assert expected_dcg(similarity, gains,
                        10) == pytest.approx(head + (1 / 21) * discounts(7, 10), rel=1e-12)
    assert expected_dcg(similarity, gains,
                        20) == pytest.approx(head + (1 / 21) * discounts(7, 13), rel=1e-12)

def test_the_tree_row_of_311211_has_the_ndcg_of_its_closed_form(tree_row):
    _, similarity, gains = tree_row
    ideal = discount(1) + (2 / 3) * discount(2) + (1 / 3) * discount(3)
    head = discount(1) + (2 / 3) * discount(2)
    assert ideal_dcg(gains, 10) == pytest.approx(ideal, rel=1e-12)
    assert tie_aware_ndcg(similarity, gains, 10) == pytest.approx(
        (head + (1 / 21) * discounts(7, 10)) / ideal, rel=1e-12
    )
    assert tie_aware_ndcg(similarity, gains, 20) == pytest.approx(
        (head + (1 / 21) * discounts(7, 13)) / ideal, rel=1e-12
    )

def test_ndcg_does_not_depend_on_candidate_order(tree_row):
    candidates, similarity, gains = tree_row
    wanted = tie_aware_ndcg(similarity, gains, 10), tie_aware_ndcg(similarity, gains, 20)
    rng = np.random.default_rng(21)
    in_the_last_group = [index for index, tied in enumerate(similarity) if tied == -10.0]
    places = set()
    for _ in range(12):
        order = rng.permutation(len(candidates))
        assert (
            tie_aware_ndcg(similarity[order], gains[order], 10),
            tie_aware_ndcg(similarity[order], gains[order], 20),
        ) == wanted
        # where 111120, the one member of the last group with a gain, falls among its group
        group = [candidates[order[i]] for i in range(len(order)) if order[i] in in_the_last_group]
        places.add(group.index('111120'))
    assert len(places) > 1

def test_the_scores_are_bit_for_bit_the_same_in_any_candidate_order():
    # the tied group's gains give one float sum when 1.0 comes first (1.0 + 1e-16 is 1.0) and another
    # when the seven 1e-16 come first (1.0000000000000007)
    rng = np.random.default_rng(21)
    similarity = np.array([3.0] + [2.0] * 8 + [1.0, 1.0, 0.0])
    gains = np.array([0.7, 1.0] + [1e-16] * 7 + [0.3, 0.9, 0.0])
    wanted = expected_dcg(similarity, gains, 10), tie_aware_ndcg(similarity, gains, 10)
    for _ in range(20):
        order = rng.permutation(len(gains))
        assert (
            expected_dcg(similarity[order], gains[order], 10),
            tie_aware_ndcg(similarity[order], gains[order], 10),
        ) == wanted

# -------------------------------------------------------------------------------------------------
# Expected DCG
# -------------------------------------------------------------------------------------------------

def test_random_tie_structures_match_brute_force():
    rng = np.random.default_rng(21)
    for _ in range(25):
        similarity = rng.choice([0.0, 1.0, 2.0], size=7)
        gains = rng.choice([0.0, 0.25, 0.5, 1.0], size=7)
        k = int(rng.integers(1, 8))
        assert expected_dcg(similarity, gains,
                            k) == pytest.approx(brute_force_dcg(similarity, gains, k), rel=1e-12)
        in_order = np.arange(7, 0, -1, dtype=np.float64)
        assert ideal_dcg(gains, k) == pytest.approx(
            brute_force_dcg(in_order,
                            np.sort(gains)[::-1], k), rel=1e-12
        )

def test_without_ties_it_is_plain_dcg():
    sim = np.array([0.9, 0.5, 0.7])
    gains = np.array([1.0, 0.0, 0.5])
    assert expected_dcg(sim, gains, 10) == pytest.approx(1.0 + 0.5 / np.log2(3))

# The base fixture's rule, one row each: similarity, gains, k and the expected DCG@k worked by hand
CLOSED_FORMS = {
    'tie at the top, cut inside it': ([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1, 0.5 * discount(1)),
    'tie at the top, k covers it': ([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 2, 0.5 * discounts(1, 2)),
    'tie in the middle': (
        [3.0, 2.0, 2.0, 1.0],
        [0.5, 1.0, 0.0, 0.25],
        4,
        0.5 * discount(1) + 0.5 * discounts(2, 3) + 0.25 * discount(4),
    ),
    'tie straddling k': ([1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], 2, (1 / 3) * discounts(1, 2)),
    'tie group wholly below k': ([2.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], 1, 0.0),
    'one group covering all candidates': (
        [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], 2, 0.25 * discounts(1, 2)
    ),
    'one group covering all, k above the count': (
        [5.0, 5.0, 5.0], [0.9, 0.3, 0.0], 10, 0.4 * discounts(1, 3)
    ),
    'a single candidate': ([5.0], [0.5], 10, 0.5),
    'zero gains above a positive gain': ([3.0, 2.0, 1.0], [0.0, 0.0, 1.0], 3, discount(3)),
    'a gain below the tie, ranked after it': (
        [2.0, 2.0, 1.0], [0.0, 1.0, 0.5], 3, 0.5 * discounts(1, 2) + 0.5 * discount(3)
    ),
    'zero and negative zero are tied': ([0.0, -0.0, -1.0], [1.0, 0.0, 0.0], 1, 0.5),
    'no positive gain': ([2.0, 1.0], [0.0, 0.0], 2, 0.0),
    'no candidates': ([], [], 5, 0.0),
}

@pytest.mark.parametrize('case', CLOSED_FORMS)
def test_expected_dcg_matches_the_hand_worked_value(case):
    similarity, gains, k, wanted = CLOSED_FORMS[case]
    assert expected_dcg(np.array(similarity), np.array(gains),
                        k) == pytest.approx(wanted, rel=1e-12, abs=1e-15)

def test_two_similarities_one_ulp_apart_are_not_tied():
    low = 1.0
    high = float(np.nextafter(low, 2.0))
    assert high != low
    assert expected_dcg(np.array([low, high]), np.array([0.0, 1.0]), 1) == 1.0
    assert expected_dcg(np.array([high, low]), np.array([0.0, 1.0]), 1) == 0.0

def test_lists_and_float32_arrays_are_read_as_float64():
    assert expected_dcg([3, 2, 1], [1, 0, 1], 3) == pytest.approx(1.0 + discount(3), rel=1e-12)
    similarity = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    gains = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert expected_dcg(similarity, gains, 3) == expected_dcg(
        similarity.astype(np.float64), gains.astype(np.float64), 3
    )

def test_similarities_that_float64_cannot_tell_apart_are_tied():
    big = 2**53
    similarity = np.array([big + 1, big])
    assert expected_dcg(similarity, np.array([1.0, 0.0]), 1) == 0.5
    assert tie_aware_auc(similarity, np.array([True, False])) == 0.5

def test_a_gain_is_linear_never_two_to_the_gain_less_one():
    assert expected_dcg(np.array([2.0, 1.0]), np.array([0.5, 0.0]), 2) == pytest.approx(0.5)

# -------------------------------------------------------------------------------------------------
# Ideal DCG and NDCG
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'k, wanted', [
        (1, discount(1)),
        (2, discount(1) + 0.5 * discount(2)),
        (3, discount(1) + 0.5 * discount(2) + 0.25 * discount(3)),
        (4, discount(1) + 0.5 * discount(2) + 0.25 * discount(3)),
        (10, discount(1) + 0.5 * discount(2) + 0.25 * discount(3)),
    ]
)
def test_the_ideal_dcg_sorts_the_gains_and_cuts_at_k(k, wanted):
    assert ideal_dcg(np.array([0.25, 0.0, 1.0, 0.5]), k) == pytest.approx(wanted, rel=1e-12)

def test_the_ideal_dcg_of_no_candidates_is_zero():
    assert ideal_dcg(np.array([]), 3) == 0.0

def test_the_ideal_dcg_of_zero_gains_is_zero():
    assert ideal_dcg(np.array([0.0, 0.0]), 3) == 0.0

def test_a_perfect_ranking_scores_one_and_no_gain_is_refused():
    similarity = np.array([0.5, 0.9, 0.1, 0.7])
    gains = np.array([0.25, 1.0, 0.0, 0.5])
    for k in (1, 2, 3, 4, 10):
        assert tie_aware_ndcg(similarity, gains, k) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match='no gain is positive'):
        tie_aware_ndcg(similarity, np.zeros(4), 10)
    with pytest.raises(ValueError, match='no gain is positive'):
        tie_aware_ndcg(np.array([]), np.array([]), 10)

def test_a_reversed_ranking_scores_below_one():
    similarity = np.array([0.5, 0.9, 0.1, 0.7])
    gains = np.array([0.25, 1.0, 0.0, 0.5])
    # the order is gains 0, 0.25, 0.5 and 1
    reversed_dcg = 0.25 * discount(2) + 0.5 * discount(3) + discount(4)
    ideal = discount(1) + 0.5 * discount(2) + 0.25 * discount(3)
    assert tie_aware_ndcg(-similarity, gains, 10) == pytest.approx(reversed_dcg / ideal, rel=1e-12)

@pytest.mark.parametrize(
    'gains', [[0.0, 0.0], [0.0], [-0.5, -0.25], [-0.5, 0.0], []],
    ids=[
        'zeros',
        'one-zero',
        'negatives',
        'negative-and-zero',
        'empty',
    ]
)
def test_a_gain_set_without_a_positive_gain_is_refused(gains):
    with pytest.raises(ValueError, match='no gain is positive'):
        tie_aware_ndcg(np.arange(float(len(gains))), np.array(gains), 3)

def test_ndcg_is_the_expected_dcg_over_the_ideal_dcg():
    assert tie_aware_ndcg(np.array([1.0, 1.0, 0.0]), np.array([0.5, 0.0, 0.0]), 1) == 0.5
    assert tie_aware_ndcg(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1) == 0.5
    similarity = np.array([2.0, 1.0, 1.0])
    gains = np.array([0.25, 1.0, 0.0])
    ideal = discount(1) + 0.25 * discount(2)
    wanted = (0.25 * discount(1) + 0.5 * discounts(2, 3)) / ideal
    assert tie_aware_ndcg(similarity, gains, 3) == pytest.approx(wanted, rel=1e-12)

def test_one_positive_gain_is_enough_for_an_ndcg():
    assert tie_aware_ndcg(np.array([1.0, 0.0]), np.array([0.0, 0.25]),
                          2) == pytest.approx(discount(2), rel=1e-12)

# -------------------------------------------------------------------------------------------------
# Refusals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('k', [0, -1])
def test_a_k_below_one_is_refused(k):
    similarity = np.array([1.0, 0.0])
    gains = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match='k must be at least 1'):
        expected_dcg(similarity, gains, k)
    with pytest.raises(ValueError, match='k must be at least 1'):
        ideal_dcg(gains, k)
    with pytest.raises(ValueError, match='k must be at least 1'):
        tie_aware_ndcg(similarity, gains, k)

def test_a_k_of_one_is_accepted():
    similarity = np.array([1.0, 0.0])
    gains = np.array([1.0, 0.0])
    assert expected_dcg(similarity, gains, 1) == 1.0
    assert ideal_dcg(gains, 1) == 1.0
    assert tie_aware_ndcg(similarity, gains, 1) == 1.0

@pytest.mark.parametrize('lengths', [(3, 2), (2, 3), (3, 0)])
def test_arrays_of_different_length_are_refused(lengths):
    similarity = np.arange(float(lengths[0]))
    gains = np.ones(lengths[1])
    with pytest.raises(ValueError, match='same length'):
        expected_dcg(similarity, gains, 2)
    with pytest.raises(ValueError, match='same length'):
        tie_aware_ndcg(similarity, gains, 2)
    with pytest.raises(ValueError, match='same length'):
        tie_aware_auc(similarity, np.ones(lengths[1], dtype=bool))

# -------------------------------------------------------------------------------------------------
# AUC
# -------------------------------------------------------------------------------------------------

def test_auc_counts_ties_one_half():
    assert tie_aware_auc(np.array([1.0, 1.0, 0.0]), np.array([True, False, False])) == 0.75

def test_auc_is_none_without_a_relevant_or_an_unrelated_candidate():
    similarity = np.array([1.0, 0.5, 0.0])
    assert tie_aware_auc(similarity, np.array([True, True, True])) is None
    assert tie_aware_auc(similarity, np.array([False, False, False])) is None
    assert tie_aware_auc(np.array([1.0]), np.array([True])) is None
    assert tie_aware_auc(np.array([1.0]), np.array([False])) is None
    assert tie_aware_auc(np.array([]), np.array([], dtype=bool)) is None

def test_auc_is_one_for_a_perfect_ordering_and_zero_for_the_reverse():
    similarity = np.array([3.0, 2.0, 1.0, 0.0])
    relevant = np.array([True, True, False, False])
    assert tie_aware_auc(similarity, relevant) == 1.0
    assert tie_aware_auc(-similarity, relevant) == 0.0

def test_auc_is_one_half_when_every_candidate_is_tied():
    assert tie_aware_auc(np.full(5, 2.0), np.array([True, False, True, False, False])) == 0.5

def test_auc_compares_each_relevant_candidate_to_the_unrelated_ones_only():
    # pairs: 3 > 2, 3 > 0, 1 < 2, 1 > 0
    similarity = np.array([3.0, 1.0, 2.0, 0.0])
    assert tie_aware_auc(similarity, np.array([True, True, False, False])) == 0.75
    # two relevant candidates tied with each other make no pair
    assert tie_aware_auc(np.array([1.0, 1.0, 0.0]), np.array([True, True, False])) == 1.0

def test_auc_is_a_fraction_of_pairs_not_of_ties_or_wins_alone():
    # 2 relevant (2 and 1) and 3 others (3, 1 and 0): 6 pairs, 3 wins, 1 tie and 2 losses
    similarity = np.array([2.0, 1.0, 3.0, 1.0, 0.0])
    relevant = np.array([True, True, False, False, False])
    assert tie_aware_auc(similarity, relevant) == pytest.approx(3.5 / 6, rel=1e-12)

def test_auc_does_not_tie_two_similarities_one_ulp_apart():
    low = 1.0
    high = float(np.nextafter(low, 2.0))
    assert tie_aware_auc(np.array([high, low]), np.array([True, False])) == 1.0
    assert tie_aware_auc(np.array([low, high]), np.array([True, False])) == 0.0

def test_random_aucs_match_the_pairwise_count():
    rng = np.random.default_rng(21)
    seen: List[Optional[float]] = []
    for _ in range(60):
        size = int(rng.integers(0, 10))
        similarity = rng.choice([0.0, 1.0, 2.0, 3.0], size=size)
        relevant = rng.random(size) < 0.4
        wanted = pairwise_auc(similarity, relevant)
        assert tie_aware_auc(similarity, relevant) == wanted
        seen.append(wanted)
    assert None in seen
    assert any(value not in (None, 0.0, 1.0) for value in seen)

def test_the_auc_does_not_depend_on_candidate_order():
    rng = np.random.default_rng(21)
    similarity = np.array([2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0])
    relevant = np.array([False, True, False, True, False, False, True])
    wanted = tie_aware_auc(similarity, relevant)
    for _ in range(10):
        order = rng.permutation(len(similarity))
        assert tie_aware_auc(similarity[order], relevant[order]) == wanted

def test_lists_and_integer_flags_are_read_as_a_set_of_relevant_candidates():
    assert tie_aware_auc([1.0, 1.0, 0.0], [True, False, False]) == 0.75
    assert tie_aware_auc([1, 1, 0], [1, 0, 0]) == 0.75

# -------------------------------------------------------------------------------------------------
# Per-anchor scores
# -------------------------------------------------------------------------------------------------

# One anchor, 'a', over four other candidates, two of them tied: the rows its own column joins
OTHERS = ('b', 'c', 'd', 'e')
OTHER_SIMILARITY = [2.0, 3.0, 2.0, 2.0]
OTHER_GAINS = [0.5, 0.0, 0.25, 0.0]

@pytest.fixture(scope='module')
def base_panel():
    '''The base fixture's tree arm: each anchor with a partner, over every candidate.'''

    co_ownership = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    candidates = co_ownership.candidates
    anchors = tuple(code for code in candidates if co_ownership.partners(code))
    place = {code: index for index, code in enumerate(candidates)}
    similarity = -tree_distance(list(candidates))[[place[code] for code in anchors]]
    gains = np.array(
        [[co_ownership.gain(anchor, code) for code in candidates] for anchor in anchors]
    )
    return co_ownership, anchors, candidates, similarity, gains

def one_anchor(own_position: Optional[int], own_similarity: float = 9.0):
    '''
    Anchor 'a' over ``OTHERS`` with its own column inserted at a position, or left out for None.

    The own column's gain is 1.0, more than any other candidate's.
    '''

    candidates = list(OTHERS)
    similarity = list(OTHER_SIMILARITY)
    gains = list(OTHER_GAINS)
    if own_position is not None:
        candidates.insert(own_position, 'a')
        similarity.insert(own_position, own_similarity)
        gains.insert(own_position, 1.0)
    return tuple(candidates), np.array([similarity]), np.array([gains])

def expected_for_others() -> Dict[str, float]:
    '''What the anchor's row must be: the scores over ``OTHERS`` alone, by Task 3's scorer.'''

    similarity = np.array(OTHER_SIMILARITY)
    gains = np.array(OTHER_GAINS)
    return {
        'ndcg_10': tie_aware_ndcg(similarity, gains, 10),
        'ndcg_20': tie_aware_ndcg(similarity, gains, 20),
        'auc': tie_aware_auc(similarity, gains > 0),
    }

def test_the_constants_are_the_briefs():
    assert HEADLINE_K == 10
    assert REPORT_K == 20
    assert COUNT_KEYS == (
        'n_industries', 'n_anchors', 'n_no_partner', 'n_candidates', 'n_relevant', 'n_crossing'
    )
    assert PER_ANCHOR_SCHEMA == {
        'anchor': pl.Utf8,
        'part': pl.Utf8,
        'n_relevant': pl.Int64,
        'ndcg_10': pl.Float64,
        'ndcg_20': pl.Float64,
        'auc': pl.Float64,
    }
    assert list(PER_ANCHOR_SCHEMA) == ['anchor', 'part', 'n_relevant', 'ndcg_10', 'ndcg_20', 'auc']

def test_an_anchor_is_left_out_of_its_own_candidates():
    # 'a' among ('a', 'b', 'c'), its own similarity the highest and its own gain 1: over ('b', 'c')
    # alone the one relevant partner, b, ranks second, behind c
    similarity = np.array([[9.0, 1.0, 2.0]])
    gains = np.array([[1.0, 0.5, 0.0]])
    scores = score_anchors(similarity, gains, ('a', ), ('a', 'b', 'c'), 'validation')
    row = scores.row(0, named=True)
    assert row['ndcg_10'] == tie_aware_ndcg(np.array([1.0, 2.0]), np.array([0.5, 0.0]), 10)
    assert row['ndcg_10'] == pytest.approx(discount(2), rel=1e-12)
    assert row['ndcg_20'] == row['ndcg_10']
    assert row['n_relevant'] == 1
    assert row['auc'] == 0.0

@pytest.mark.parametrize('own_similarity', [9.0, 2.0, -9.0], ids=['highest', 'tied', 'lowest'])
@pytest.mark.parametrize(
    'own_position', [0, 1, 2, 3, 4, None],
    ids=['first', 'second', 'middle', 'fourth', 'last', 'absent']
)
def test_the_own_column_is_left_out_wherever_it_is(own_position, own_similarity):
    candidates, similarity, gains = one_anchor(own_position, own_similarity)
    row = score_anchors(similarity, gains, ('a', ), candidates, 'validation').row(0, named=True)
    wanted = expected_for_others()
    assert row['ndcg_10'] == wanted['ndcg_10']
    assert row['ndcg_20'] == wanted['ndcg_20']
    assert row['auc'] == wanted['auc'] == 0.25
    # the own column's gain of 1 is not a partner
    assert row['n_relevant'] == 2

def test_each_anchor_drops_its_own_column_not_a_neighbours():
    # row 0 is anchor 'c', whose own column is index 2, and row 1 is 'a', whose own column is index 0
    similarity = np.array([[1.0, 2.0, 9.0], [9.0, 3.0, 1.0]])
    gains = np.array([[0.5, 0.25, 1.0], [1.0, 0.0, 0.75]])
    scores = score_anchors(similarity, gains, ('c', 'a'), ('a', 'b', 'c'), 'validation')
    for_c, for_a = scores.rows(named=True)
    # c over (a, b): similarities 1 and 2, gains 0.5 and 0.25
    assert for_c['ndcg_10'] == tie_aware_ndcg(np.array([1.0, 2.0]), np.array([0.5, 0.25]), 10)
    assert for_c['n_relevant'] == 2
    assert for_c['auc'] is None
    # a over (b, c): similarities 3 and 1, gains 0 and 0.75
    assert for_a['ndcg_10'] == tie_aware_ndcg(np.array([3.0, 1.0]), np.array([0.0, 0.75]), 10)
    assert for_a['ndcg_10'] == pytest.approx(discount(2), rel=1e-12)
    assert for_a['n_relevant'] == 1
    assert for_a['auc'] == 0.0

def test_a_row_is_one_anchor_in_the_given_order_and_schema():
    candidates = ('311111', '311211', '445110', '621210')
    similarity = np.array([[0.0, 3.0, 2.0, 1.0], [3.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 3.0]])
    anchors = ('445110', '311111', '311211')
    gains = np.array([[0.5, 0.0, 0.0, 0.25], [0.0, 0.5, 0.0, 0.25], [0.5, 0.0, 0.0, 0.0]])
    scores = score_anchors(similarity, gains, anchors, candidates, 'sealed')
    assert dict(scores.schema) == PER_ANCHOR_SCHEMA
    assert scores.columns == list(PER_ANCHOR_SCHEMA)
    assert scores.get_column('anchor').to_list() == ['445110', '311111', '311211']
    assert scores.get_column('part').to_list() == ['sealed'] * 3
    assert scores.get_column('n_relevant').to_list() == [2, 2, 1]

def test_n_relevant_counts_the_positive_gains():
    candidates = ('b', 'c', 'd', 'e')
    similarity = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    gains = np.array([[0.5, 0.0, 0.25, 0.125], [0.0, 0.0, 0.0, 1.0]])
    scores = score_anchors(similarity, gains, ('x', 'y'), candidates, 'validation')
    assert scores.get_column('n_relevant').to_list() == [3, 1]

def test_ndcg_10_and_ndcg_20_read_their_own_cutoffs():
    # the one partner ranks 12th of 15 candidates: outside the top 10, inside the top 20
    candidates = ('a', *(f'c{index:02d}' for index in range(1, 16)))
    similarity = np.array([[99.0, *range(15, 0, -1)]], dtype=np.float64)
    gains = np.zeros((1, 16))
    gains[0, 12] = 1.0
    row = score_anchors(similarity, gains, ('a', ), candidates, 'validation').row(0, named=True)
    assert row['ndcg_10'] == 0.0
    assert row['ndcg_20'] == pytest.approx(discount(12), rel=1e-12)

def test_the_base_panel_rows_match_the_closed_forms_of_311211(base_panel):
    _, anchors, candidates, similarity, gains = base_panel
    scores = score_anchors(similarity, gains, anchors, candidates, 'validation')
    row = scores.filter(pl.col('anchor') == ANCHOR).row(0, named=True)
    head = discount(1) + (2 / 3) * discount(2)
    ideal = head + (1 / 3) * discount(3)
    assert row['n_relevant'] == 3
    assert row['ndcg_10'] == pytest.approx((head + (1 / 21) * discounts(7, 10)) / ideal, rel=1e-12)
    assert row['ndcg_20'] == pytest.approx((head + (1 / 21) * discounts(7, 13)) / ideal, rel=1e-12)
    # the same row, by brute force over every order within the tied groups
    others = [index for index, code in enumerate(candidates) if code != ANCHOR]
    own_row = list(anchors).index(ANCHOR)
    in_order = np.arange(len(others), 0, -1, dtype=np.float64)
    ideal_by_force = brute_force_dcg(in_order, np.sort(gains[own_row][others])[::-1], 10)
    assert row['ndcg_10'] == pytest.approx(
        brute_force_dcg(similarity[own_row][others], gains[own_row][others], 10) / ideal_by_force,
        rel=1e-12
    )

def test_every_base_panel_row_is_the_scorer_over_the_candidates_without_the_anchor(base_panel):
    co_ownership, anchors, candidates, similarity, gains = base_panel
    scores = score_anchors(similarity, gains, anchors, candidates, 'validation')
    assert scores.height == len(anchors) == 13
    for row, anchor in zip(scores.rows(named=True), anchors):
        index = list(anchors).index(anchor)
        others = [place for place, code in enumerate(candidates) if code != anchor]
        wanted_similarity = similarity[index][others]
        wanted_gains = gains[index][others]
        assert row['anchor'] == anchor
        assert row['n_relevant'] == len(co_ownership.partners(anchor))
        assert row['ndcg_10'] == tie_aware_ndcg(wanted_similarity, wanted_gains, 10)
        assert row['ndcg_20'] == tie_aware_ndcg(wanted_similarity, wanted_gains, 20)
        assert row['auc'] == pytest.approx(
            pairwise_auc(wanted_similarity, wanted_gains > 0), rel=1e-12
        )

# -------------------------------------------------------------------------------------------------
# Refusals of the per-anchor scores
# -------------------------------------------------------------------------------------------------

def three_anchors():
    '''Three anchors over four candidates, each with a partner outside its own column.'''

    anchors = ('311111', '311119', '311211')
    candidates = ('311111', '311119', '311211', '312111')
    similarity = np.arange(12.0).reshape(3, 4)
    gains = np.array([[0.0, 0.5, 0.0, 0.25], [0.5, 0.0, 0.0, 0.25], [0.5, 0.0, 0.0, 0.25]])
    return anchors, candidates, similarity, gains

@pytest.mark.parametrize(
    'similarity_shape, gains_shape, n_anchors, n_candidates', [
        ((2, 3), (3, 2), 2, 3),
        ((2, 3), (2, 4), 2, 3),
        ((2, 3), (1, 3), 2, 3),
        ((3, 3), (2, 3), 2, 3),
        ((2, 4), (2, 3), 2, 3),
        ((2, 3), (2, 3), 3, 3),
        ((2, 3), (2, 3), 2, 2),
        ((2, 3), (2, 3), 2, 4),
        ((6, ), (6, ), 2, 3),
    ],
    ids=[
        'transposed', 'wider-gains', 'shorter-gains', 'taller-similarity', 'wider-similarity',
        'anchor-short', 'candidate-short', 'candidate-long', 'one-dimensional'
    ]
)
def test_a_shape_that_does_not_fit_is_refused(
    similarity_shape, gains_shape, n_anchors, n_candidates
):
    anchors = tuple(f'a{index}' for index in range(n_anchors))
    candidates = tuple(f'c{index}' for index in range(n_candidates))
    with pytest.raises(ValueError, match='shape'):
        score_anchors(
            np.ones(similarity_shape), np.ones(gains_shape), anchors, candidates, 'validation'
        )

@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', '-inf'])
@pytest.mark.parametrize('column', [0, 1, 3], ids=['first', 'own', 'last'])
def test_a_non_finite_similarity_is_refused_naming_the_anchor(column, bad):
    anchors, candidates, similarity, gains = three_anchors()
    # the second anchor's row, 311119, whose own column is the second
    similarity[1, column] = bad
    with pytest.raises(ValueError, match='not finite') as caught:
        score_anchors(similarity, gains, anchors, candidates, 'validation')
    message = str(caught.value)
    assert '311119' in message
    assert '311111' not in message
    assert '311211' not in message

def test_the_first_anchor_with_a_non_finite_similarity_is_named():
    anchors, candidates, similarity, gains = three_anchors()
    similarity[1, 3] = math.nan
    similarity[2, 3] = math.nan
    with pytest.raises(ValueError, match='not finite') as caught:
        score_anchors(similarity, gains, anchors, candidates, 'validation')
    assert '311119' in str(caught.value)
    assert '311211' not in str(caught.value)

@pytest.mark.parametrize(
    'gain_row', [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -0.5, 0.0, -0.25]],
    ids=['all-zero', 'only-the-own-column', 'negative']
)
def test_an_anchor_with_no_positive_gain_is_refused_naming_the_anchor(gain_row):
    anchors, candidates, similarity, gains = three_anchors()
    # the second anchor, 311119, whose own column is the second
    gains[1] = gain_row
    with pytest.raises(ValueError, match='no candidate with a positive gain') as caught:
        score_anchors(similarity, gains, anchors, candidates, 'validation')
    message = str(caught.value)
    assert '311119' in message
    assert '311111' not in message
    assert '311211' not in message

def test_the_first_anchor_with_no_positive_gain_is_named():
    anchors, candidates, similarity, gains = three_anchors()
    gains[1] = 0.0
    gains[2] = 0.0
    with pytest.raises(ValueError, match='no candidate with a positive gain') as caught:
        score_anchors(similarity, gains, anchors, candidates, 'validation')
    assert '311119' in str(caught.value)
    assert '311211' not in str(caught.value)

def test_an_anchor_with_one_candidate_that_is_itself_is_refused():
    with pytest.raises(ValueError, match='no candidate with a positive gain'):
        score_anchors(np.array([[1.0]]), np.array([[1.0]]), ('a', ), ('a', ), 'validation')

def test_a_shape_mismatch_is_refused_before_a_non_finite_similarity():
    anchors, candidates, similarity, gains = three_anchors()
    similarity[0, 3] = math.nan
    with pytest.raises(ValueError, match='shape'):
        score_anchors(similarity, gains[:, :3], anchors, candidates, 'validation')

def test_a_non_finite_similarity_is_refused_before_an_anchor_with_no_gain():
    anchors, candidates, similarity, gains = three_anchors()
    gains[0] = 0.0
    similarity[2, 1] = math.inf
    with pytest.raises(ValueError, match='not finite') as caught:
        score_anchors(similarity, gains, anchors, candidates, 'validation')
    assert '311211' in str(caught.value)
    assert '311111' not in str(caught.value)

def test_lists_are_read_as_arrays():
    scores = score_anchors([[3, 1, 2]], [[0, 1, 0.5]], ['a'], ['a', 'b', 'c'], 'validation')
    assert scores.row(0, named=True) == {
        'anchor': 'a',
        'part': 'validation',
        'n_relevant': 2,
        'ndcg_10': tie_aware_ndcg(np.array([1.0, 2.0]), np.array([1.0, 0.5]), 10),
        'ndcg_20': tie_aware_ndcg(np.array([1.0, 2.0]), np.array([1.0, 0.5]), 20),
        'auc': None,
    }

def test_no_anchor_gives_an_empty_frame_in_the_schema():
    scores = score_anchors(np.zeros((0, 3)), np.zeros((0, 3)), (), ('a', 'b', 'c'), 'validation')
    assert scores.height == 0
    assert dict(scores.schema) == PER_ANCHOR_SCHEMA

# -------------------------------------------------------------------------------------------------
# The result
# -------------------------------------------------------------------------------------------------

def per_anchor_frame(
    rows: Sequence[Tuple[str, int, float, float, Optional[float]]],
    part: str = 'validation',
) -> pl.DataFrame:
    '''A per-anchor frame from (anchor, n_relevant, ndcg_10, ndcg_20, auc) rows.'''

    anchor, n_relevant, ndcg_10, ndcg_20, auc = (list(column) for column in zip(*rows))
    return pl.DataFrame(
        {
            'anchor': anchor,
            'part': [part] * len(rows),
            'n_relevant': n_relevant,
            'ndcg_10': ndcg_10,
            'ndcg_20': ndcg_20,
            'auc': auc,
        },
        schema=PER_ANCHOR_SCHEMA,
    )

def counts_for(n_anchors: int) -> Dict[str, int]:
    '''Counts for a part of ``n_anchors`` anchors, in the order of ``COUNT_KEYS``.'''

    return {
        'n_industries': 14,
        'n_anchors': n_anchors,
        'n_no_partner': 1,
        'n_candidates': 14,
        'n_relevant': 24,
        'n_crossing': 3,
    }

def summarize(frame: pl.DataFrame, **changes) -> RelatednessResult:
    settings = {
        'panel': 'relatedness_co_ownership',
        'part': 'validation',
        'counts': counts_for(frame.height),
        **changes,
    }
    return summarize_relatedness(frame, **settings)

def test_the_headline_is_the_mean_over_anchors_each_once():
    # 311211 and 321113 are one sector, 445110 another. The plain mean is 0.6. A mean weighted by
    # partners is 0.606 and a mean of the two sector means is 0.675
    frame = per_anchor_frame(
        [
            ('311211', 1, 0.3, 0.4, 0.5),
            ('321113', 50, 0.6, 0.7, 0.8),
            ('445110', 2, 0.9, 1.0, 0.25),
        ]
    )
    result = summarize(frame)
    assert result.headline == {'n_anchors': 3, 'ndcg_10': math.fsum([0.3, 0.6, 0.9]) / 3}
    # a report never changes it
    other = frame.with_columns(
        ndcg_20=pl.Series([0.0, 0.0, 0.0]), auc=pl.Series([None, 1.0, 0.0], dtype=pl.Float64)
    )
    assert summarize(other).headline == result.headline

def test_the_headline_of_scored_anchors_is_their_mean(base_panel):
    _, anchors, candidates, similarity, gains = base_panel
    scores = score_anchors(similarity, gains, anchors, candidates, 'validation')
    result = summarize(scores)
    assert result.headline['n_anchors'] == 13
    assert result.headline['ndcg_10'] == math.fsum(scores.get_column('ndcg_10').to_list()) / 13
    assert result.headline['ndcg_10'] == pytest.approx(
        sum(row['ndcg_10'] for row in scores.rows(named=True)) / 13, rel=1e-12
    )

def test_the_reports_are_the_means_of_ndcg_20_and_the_auc_of_the_anchors_that_have_one():
    frame = per_anchor_frame(
        [
            ('311211', 3, 0.3, 0.25, 0.5),
            ('321113', 3, 0.6, 0.5, None),
            ('445110', 3, 0.9, 1.0, 1.0),
        ]
    )
    reports = summarize(frame).reports
    assert reports == {
        'ndcg_20': math.fsum([0.25, 0.5, 1.0]) / 3,
        'auc': 0.75,
        'n_auc_anchors': 2,
    }

def test_an_anchor_with_no_unrelated_candidate_has_no_auc():
    # a is co-owned with every candidate. b and c have an unrelated candidate, d and e
    candidates = ('a', 'b', 'c', 'd', 'e')
    similarity = np.array(
        [
            [9.0, 4.0, 3.0, 2.0, 1.0],
            [4.0, 9.0, 1.0, 2.0, 3.0],
            [2.5, 1.0, 9.0, 2.0, 3.0],
        ]
    )
    gains = np.array(
        [
            [0.0, 1.0, 0.5, 0.25, 0.125],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    scores = score_anchors(similarity, gains, ('a', 'b', 'c'), candidates, 'validation')
    by_anchor = {row['anchor']: row for row in scores.rows(named=True)}
    assert by_anchor['a']['auc'] is None
    assert by_anchor['a']['n_relevant'] == 4
    # b: its partner a, at 4, beats the unrelated c, d and e, at 1, 2 and 3
    assert by_anchor['b']['auc'] == 1.0
    # c: its partner a, at 2.5, beats b and d, at 1 and 2, and loses to e, at 3
    assert by_anchor['c']['auc'] == pytest.approx(2 / 3, rel=1e-12)
    result = summarize(scores)
    assert result.reports['auc'] == math.fsum([1.0, by_anchor['c']['auc']]) / 2
    assert result.reports['n_auc_anchors'] == 2
    # the anchor without an AUC still counts in the headline
    assert result.headline['n_anchors'] == 3
    assert result.headline['ndcg_10'] == math.fsum(scores.get_column('ndcg_10').to_list()) / 3
    without_it = summarize(scores.filter(pl.col('anchor') != 'a'))
    assert without_it.headline['ndcg_10'] != result.headline['ndcg_10']

def test_the_auc_report_is_the_mean_over_the_other_anchors_not_zero_for_the_one_without():
    frame = per_anchor_frame(
        [
            ('311211', 1, 0.5, 0.5, None),
            ('321113', 1, 0.5, 0.5, 0.5),
            ('445110', 1, 0.5, 0.5, 1.0),
        ]
    )
    assert summarize(frame).reports['auc'] == 0.75
    assert summarize(frame).reports['n_auc_anchors'] == 2

def test_with_no_anchor_that_has_an_auc_the_report_is_none():
    frame = per_anchor_frame([('311211', 4, 0.5, 0.6, None), ('445110', 2, 0.7, 0.8, None)])
    result = summarize(frame)
    assert result.reports['auc'] is None
    assert result.reports['n_auc_anchors'] == 0
    assert result.reports['ndcg_20'] == math.fsum([0.6, 0.8]) / 2
    assert result.headline == {'n_anchors': 2, 'ndcg_10': math.fsum([0.5, 0.7]) / 2}

def test_sectors_join_31_to_33():
    frame = per_anchor_frame(
        [
            ('311211', 2, 0.5, 0.5, 0.5),
            ('321113', 2, 0.25, 0.25, 0.5),
            ('445110', 2, 0.75, 0.75, 0.5),
        ]
    )
    by_sector = summarize(frame).by_sector
    assert by_sector.rows() == [('31', 2, 0.375), ('44', 1, 0.75)]

def test_each_joined_sector_is_one_and_the_others_stand_alone():
    # unsorted, with 32, 33, 45 and 49 among them, and 11 and 62 as sectors of their own
    rows = [
        ('445110', 0.5),
        ('621210', 0.5),
        ('493110', 0.125),
        ('311211', 0.25),
        ('455110', 0.25),
        ('331110', 0.75),
        ('111120', 0.875),
        ('481111', 0.375),
        ('441110', 0.75),
        ('321113', 0.5),
    ]
    frame = per_anchor_frame([(code, 1, value, value, 0.5) for code, value in rows])
    by_sector = summarize(frame).by_sector
    assert by_sector.columns == ['sector', 'n_anchors', 'ndcg_10']
    assert dict(by_sector.schema) == {
        'sector': pl.Utf8,
        'n_anchors': pl.Int64,
        'ndcg_10': pl.Float64,
    }
    assert by_sector.rows() == [
        ('11', 1, 0.875),
        ('31', 3, math.fsum([0.25, 0.75, 0.5]) / 3),
        ('44', 3, math.fsum([0.5, 0.25, 0.75]) / 3),
        ('48', 2, 0.25),
        ('62', 1, 0.5),
    ]

def ill_conditioned_frame() -> pl.DataFrame:
    '''Nine anchors of one sector whose means sum to different floats in different orders.'''

    tiny = 1e-16
    # 1.0 + 1e-16 is 1.0, and the eight tiny values summed first make 1.0000000000000009
    ndcg_10 = [1.0] + [tiny] * 8
    ndcg_20 = [tiny] * 4 + [1.0] + [tiny] * 4
    auc = [1.0] + [tiny] * 7 + [None]
    codes = [f'31111{digit}' for digit in range(1, 10)]
    return per_anchor_frame(
        [(code, 1, a, b, c) for code, a, b, c in zip(codes, ndcg_10, ndcg_20, auc)]
    )

def test_every_mean_is_the_correctly_rounded_sum():
    frame = ill_conditioned_frame()
    result = summarize(frame)
    ndcg_10 = frame.get_column('ndcg_10').to_list()
    ndcg_20 = frame.get_column('ndcg_20').to_list()
    auc = frame.get_column('auc').drop_nulls().to_list()
    assert result.headline['ndcg_10'] == math.fsum(ndcg_10) / 9
    assert result.reports['ndcg_20'] == math.fsum(ndcg_20) / 9
    assert result.reports['auc'] == math.fsum(auc) / 8
    assert result.by_sector.row(0) == ('31', 9, math.fsum(ndcg_10) / 9)

def test_no_result_depends_on_the_order_of_the_anchors():
    frame = ill_conditioned_frame()
    wanted = summarize(frame)
    rng = np.random.default_rng(21)
    orders = [list(range(8, -1, -1))] + [list(rng.permutation(9)) for _ in range(12)]
    for order in orders:
        result = summarize(frame.select(pl.all().gather([int(place) for place in order])))
        assert result.headline == wanted.headline
        assert result.reports == wanted.reports
        assert result.by_sector.equals(wanted.by_sector)

def test_the_result_carries_its_inputs():
    frame = per_anchor_frame(
        [('445110', 2, 0.5, 0.5, 0.5), ('311211', 1, 0.25, 0.25, None)], part='sealed'
    )
    counts = counts_for(2)
    result = summarize_relatedness(
        frame, panel='relatedness_co_ownership', part='sealed', counts=counts
    )
    assert result.panel == 'relatedness_co_ownership'
    assert result.part == 'sealed'
    # in the order given, not sorted
    assert result.per_anchor.equals(frame)
    assert result.per_anchor.get_column('anchor').to_list() == ['445110', '311211']
    assert result.counts == counts
    assert list(result.counts) == list(COUNT_KEYS)
    assert result.data_fingerprint is None
    # a later change to the caller's counts leaves the result alone
    counts['n_relevant'] = 999
    assert result.counts['n_relevant'] == 24

def test_the_counts_come_back_in_the_order_of_count_keys():
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)])
    result = summarize(frame, counts=dict(reversed(list(counts_for(1).items()))))
    assert list(result.counts) == list(COUNT_KEYS)

def test_the_panel_the_part_and_the_counts_are_keywords():
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)])
    with pytest.raises(TypeError):
        summarize_relatedness(frame, 'relatedness_co_ownership', 'validation', counts_for(1))

def test_the_result_is_frozen():
    result = summarize(per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.part = 'sealed'  # type: ignore[misc]
    assert dataclasses.replace(result, data_fingerprint='abc').data_fingerprint == 'abc'

# -------------------------------------------------------------------------------------------------
# Refusals of the result
# -------------------------------------------------------------------------------------------------

def test_an_anchor_with_no_partner_and_an_empty_result_are_refused():
    # an industry with no partner is counted, not scored: it cannot be an anchor
    with pytest.raises(ValueError, match='no candidate with a positive gain'):
        score_anchors(
            np.array([[0.0, 1.0, 2.0]]), np.zeros((1, 3)), ('a', ), ('a', 'b', 'c'), 'validation'
        )
    empty = score_anchors(np.zeros((0, 3)), np.zeros((0, 3)), (), ('a', 'b', 'c'), 'validation')
    with pytest.raises(ValueError, match='empty'):
        summarize(empty, counts=counts_for(0))

@pytest.mark.parametrize('missing', COUNT_KEYS)
def test_counts_that_lack_a_key_are_refused(missing):
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)])
    counts = {key: value for key, value in counts_for(1).items() if key != missing}
    with pytest.raises(ValueError, match=rf"missing \['{missing}'\]"):
        summarize(frame, counts=counts)

def test_counts_with_an_extra_key_are_refused():
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)])
    with pytest.raises(ValueError, match=r"unexpected \['n_extra'\]"):
        summarize(frame, counts={**counts_for(1), 'n_extra': 1})

def test_counts_with_one_key_swapped_for_another_are_refused():
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5)])
    counts = counts_for(1)
    counts['n_other'] = counts.pop('n_crossing')
    with pytest.raises(ValueError, match=r"missing \['n_crossing'\].*unexpected \['n_other'\]"):
        summarize(frame, counts=counts)

@pytest.mark.parametrize('off_by', [-1, 1])
def test_counts_whose_n_anchors_is_not_the_height_are_refused(off_by):
    frame = per_anchor_frame([('445110', 2, 0.5, 0.5, 0.5), ('311211', 1, 0.25, 0.25, 0.5)])
    counts = counts_for(2 + off_by)
    with pytest.raises(ValueError, match='n_anchors'):
        summarize(frame, counts=counts)
    assert summarize(frame, counts=counts_for(2)).headline['n_anchors'] == 2
