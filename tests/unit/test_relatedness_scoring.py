'''
Tie-aware scores for the relatedness panel (plan 21, Task 3; spec relatedness-artifact, 4.4).

Each test reads one anchor's arrays. The oracle is brute force: the mean DCG over every order within
tied groups, written here from the definition and not from the scorer's formula. The tree row of
311211 on the base fixture is the real tied case: D* takes 5 values over its 13 candidates. The
closed forms below spell out each group's ranks and discounts by hand. No test reads a real table.
'''

import itertools
import math
from typing import List, Optional

import numpy as np
import pytest

from naics_embedder.panels.relatedness import co_ownership_inputs
from naics_embedder.panels.relatedness_scoring import (
    expected_dcg,
    ideal_dcg,
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
