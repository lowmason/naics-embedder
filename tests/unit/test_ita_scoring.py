'''
The ITA shortlist scorer (plan 17, Task 3; spec text-to-code-ita-panel, 4.3).

A pair ranks all candidates under the encoder's own distance and keeps the best rank over its
reported codes, with ties against the truth. Expected values are worked out by hand, never from
production code, except where a test compares the scorer with ``score_decoding`` on pairs of one
code. Most tests put the candidates on a line (the i-th at distance i + 1 from a query at the
origin) so each rank can be read off the order of the codes.
'''

import math
from typing import Dict, List, Sequence

import polars as pl
import pytest
import torch

from naics_embedder.panels.decoding import score_decoding
from naics_embedder.panels.ita_scoring import COARSE_LEVELS, score_shortlist
from tests.fixtures.ita import ITA_CANDIDATES

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

def _line(codes: Sequence[str]) -> Dict[str, torch.Tensor]:
    '''Candidates on a line, the i-th at position i + 1, for queries at the origin.'''

    return {
        'candidate_points': torch.tensor(
            [[float(i + 1)] for i in range(len(codes))], dtype=torch.float64
        ),
        'candidate_codes': list(codes),
    }

def _score_line(
    codes: Sequence[str],
    reported: Sequence[Sequence[str]],
    distance: str = 'euclidean',
) -> pl.DataFrame:
    '''Score pairs that all stand at the origin, over candidates on a line in the order given.'''

    return score_shortlist(
        torch.zeros(len(reported), 1, dtype=torch.float64),
        reported,
        **_line(codes),
        pair_ids=[f'p{index:02d}' for index in range(len(reported))],
        distance=distance,
    )

def _column(scores: pl.DataFrame, name: str) -> List:
    return scores.get_column(name).to_list()

# -------------------------------------------------------------------------------------------------
# Ranks
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('distance', ['euclidean', 'cosine'])
def test_single_code_pairs_rank_as_score_decoding_ranks_them(distance):
    generator = torch.Generator().manual_seed(7)
    candidates = list(ITA_CANDIDATES[:6])
    candidate_points = torch.randn(6, 4, generator=generator)
    query_points = torch.randn(20, 4, generator=generator)
    truth = [
        candidates[index] for index in torch.randint(0, 6, (20, ), generator=generator).tolist()
    ]

    expected = score_decoding(
        query_points, truth, candidate_points, candidates, distance=distance
    ).per_query
    scores = score_shortlist(
        query_points,
        [[code] for code in truth],
        candidate_points,
        candidates,
        [f'p{index:02d}' for index in range(20)],
        distance=distance,
    )

    assert max(_column(scores, 'rank')) > 1  # the runner-up branch of top1_code is compared
    assert _column(scores, 'rank') == _column(expected, 'rank')
    assert _column(scores, 'top1_code') == _column(expected, 'top1_code')
    assert _column(scores, 'reciprocal_rank') == _column(expected, 'reciprocal_rank')
    assert _column(scores, 'lca_level') == _column(expected, 'lca_level')

def test_ties_count_against_the_truth():
    scores = score_shortlist(
        torch.ones(3, 2),
        [['111110'], ['311111'], ['621210']],
        torch.ones(8, 2),
        list(ITA_CANDIDATES),
        ['p01', 'p02', 'p03'],
        distance='euclidean',
    )

    assert scores.get_column('rank').to_list() == [8, 8, 8]

def test_tied_reported_codes_do_not_count_against_each_other():
    # The query is at the origin: 111110 and 311111 are both at distance 1, all others farther
    points = {
        '111110': [1.0, 0.0],
        '311111': [0.0, 1.0],
    }
    scores = score_shortlist(
        torch.zeros(1, 2),
        [['311111', '111110']],
        torch.tensor([points.get(code, [5.0, 5.0]) for code in ITA_CANDIDATES]),
        list(ITA_CANDIDATES),
        ['p01'],
        distance='euclidean',
    )

    assert scores.get_column('rank').to_list() == [1]

def test_non_reported_codes_tied_with_the_best_reported_code_count_against_the_pair():
    # Two reported codes and two other codes all at distance 1, the rest farther: the other two
    # count against the pair, its reported codes do not
    tied = {code: [1.0, 0.0] for code in ('111110', '311111', '513210', '621210')}
    scores = score_shortlist(
        torch.zeros(1, 2),
        [['311111', '513210']],
        torch.tensor([tied.get(code, [5.0, 5.0]) for code in ITA_CANDIDATES]),
        list(ITA_CANDIDATES),
        ['p01'],
        distance='euclidean',
    )

    assert scores.get_column('rank').to_list() == [3]

def test_ties_count_against_the_truth_over_several_reported_codes():
    scores = score_shortlist(
        torch.ones(2, 2),
        [['311111', '513210'], ['111110', '111120', '211120']],
        torch.ones(8, 2),
        list(ITA_CANDIDATES),
        ['p01', 'p02'],
        distance='euclidean',
    )

    # Every candidate ties; the pair's own codes never count: 8 - 2 and 8 - 3 others, plus one
    assert scores.get_column('rank').to_list() == [7, 6]

def test_the_rank_is_the_best_over_reported_codes():
    # 321113 and 111110 are in front of 311111 (3rd), 513210 is 4th and 211120 5th: two non-reported
    # codes ahead of the best reported one
    scores = _score_line(
        ['321113', '111110', '311111', '513210', '211120', '111120', '236220', '621210'],
        [['211120', '311111']],
    )

    assert scores.select('rank', 'top1_code', 'lca_level').rows() == [(3, '321113', 2)]

def test_lca_level_is_the_highest_over_every_reported_code_not_only_the_best_ranked_one():
    # 111120 is the nearest code and is not reported. It shares its sector, three-digit and
    # four-digit code with the reported 111110 (LCA level 4) and nothing with the reported 311111,
    # the best-ranked code (LCA level 1)
    scores = _score_line(
        ['111120', '311111', '111110', '236220', '513210', '211120', '321113', '621210'],
        [['111110', '311111']],
    )

    assert scores.select('rank', 'top1_code', 'lca_level').rows() == [(2, '111120', 4)]

def test_the_rank_never_depends_on_how_a_pair_orders_or_repeats_its_codes():
    codes = ['321113', '111110', '311111', '513210', '211120', '111120', '236220', '621210']

    scores = _score_line(
        codes,
        [['211120', '311111'], ['311111', '211120'], ['311111', '211120', '311111']],
    )

    assert _column(scores, 'rank') == [3, 3, 3]
    assert _column(scores, 'codes') == [['211120', '311111']] * 3

# -------------------------------------------------------------------------------------------------
# The top-1 code and the LCA level
# -------------------------------------------------------------------------------------------------

def test_the_top1_code_is_the_nearest_reported_code_when_the_rank_is_one():
    scores = _score_line(
        ['311111', '211120', '111110', '236220', '513210', '111120', '321113', '621210'],
        [['211120', '311111']],
    )

    assert scores.select('rank', 'top1_code', 'lca_level').rows() == [(1, '311111', 6)]

def test_the_top1_code_is_a_non_reported_code_when_one_ties_with_the_best_reported_code():
    # 311111 is at distance 1 as the reported 111110 is: it ranks the pair second and is top-1
    points = {'111110': [1.0, 0.0], '311111': [0.0, 1.0]}
    scores = score_shortlist(
        torch.zeros(1, 2),
        [['111110']],
        torch.tensor([points.get(code, [5.0, 5.0]) for code in ITA_CANDIDATES]),
        list(ITA_CANDIDATES),
        ['p01'],
        distance='euclidean',
    )

    assert scores.select('rank', 'top1_code', 'lca_level').rows() == [(2, '311111', 1)]

def test_a_tie_for_top1_goes_to_the_code_first_in_the_candidate_order():
    # Two reported codes tie at distance 1 (311111 comes before 111110 here, and after it in sorted
    # order), and so do two codes that are not reported, ahead of the reported 513210 (621210 comes
    # before 111120 here, and after it in sorted order)
    reported_tie = score_shortlist(
        torch.zeros(1, 2),
        [['311111', '111110']],
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [5.0, 5.0]]),
        ['311111', '111110', '236220'],
        ['p01'],
        distance='euclidean',
    )
    other_tie = score_shortlist(
        torch.zeros(1, 2),
        [['513210']],
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]),
        ['621210', '111120', '513210'],
        ['p01'],
        distance='euclidean',
    )

    assert reported_tie.select('rank', 'top1_code').rows() == [(1, '311111')]
    assert other_tie.select('rank', 'top1_code').rows() == [(3, '621210')]

# -------------------------------------------------------------------------------------------------
# Coarser levels
# -------------------------------------------------------------------------------------------------

def test_coarser_levels_read_the_lineage_with_combined_sectors():
    # 321113 is nearest, then the reported 311111: 31 and 32 are one sector (level 2), 311 and 321
    # are not one code at level 3 or below
    scores = _score_line(
        ['321113', '311111', '111110', '236220', '513210', '211120', '111120', '621210'],
        [['311111']],
    )

    assert scores.select(
        'rank', 'rank_l2', 'rank_l3', 'rank_l4', 'rank_l5', 'hit_at_1_l2', 'hit_at_1_l3'
    ).rows() == [(2, 1, 2, 2, 2, True, False)]
    assert _column(scores, 'reciprocal_rank_l2') == [1.0]
    assert _column(scores, 'reciprocal_rank_l3') == [0.5]

@pytest.mark.parametrize(
    ('reported', 'near', 'shared_sector'),
    [
        ('311111', '332111', True),  # 31-33
        ('441110', '452210', True),  # 44-45
        ('481111', '493110', True),  # 48-49
        ('311111', '441110', False),
        ('441110', '481111', False),
        ('111110', '211120', False),
    ],
)
def test_the_three_combined_sectors_are_one_sector_each(reported, near, shared_sector):
    scores = _score_line([near, reported, '236220'], [[reported]])

    # The code in front is a sector match for rank_l2 only when the two share a sector
    assert scores.select('rank', 'rank_l2', 'rank_l3').rows() == [(2, 1 if shared_sector else 2, 2)]

def test_each_level_reads_one_step_further_down_the_lineage():
    # The reported 311111 is 6th. Each code in front departs from its lineage one level lower than
    # the one before: 111110 shares nothing, 321113 a sector (31-33), 311211 a three-digit code,
    # 311121 a four-digit, 311119 a five-digit one
    scores = _score_line(
        ['111110', '321113', '311211', '311121', '311119', '311111'],
        [['311111']],
    )

    assert scores.select('rank', 'rank_l5', 'rank_l4', 'rank_l3', 'rank_l2').rows() == [
        (6, 5, 4, 3, 2)
    ]
    assert scores.select(
        'reciprocal_rank_l5', 'reciprocal_rank_l4', 'reciprocal_rank_l3', 'reciprocal_rank_l2'
    ).rows() == [(0.2, 0.25, 1 / 3, 0.5)]
    assert scores.select('hit_at_5', 'hit_at_5_l5', 'hit_at_5_l4', 'hit_at_1_l2').rows() == [
        (False, True, True, False)
    ]

def test_a_level_starts_at_the_nearest_code_that_shares_an_ancestor_even_if_not_reported():
    # 321113 shares the sector of the reported 311111 and is in front of everything else; 111110
    # is nearer than 311111 and shares none, so it ranks behind the shared-sector match
    scores = _score_line(
        ['321113', '111110', '311111', '513210', '211120', '111120', '236220', '621210'],
        [['311111']],
    )

    assert scores.select('rank', 'rank_l2', 'rank_l3').rows() == [(3, 1, 3)]

def test_a_level_matches_the_ancestor_of_any_reported_code():
    # Both pairs report 111110 (sector 11) and 311111 (sector 31). The first stands on 321113, which
    # shares the sector of 311111 only. The second stands on 111120, which shares the sector of
    # 111110 only. Each is in front of every code that matches no reported code's sector
    codes = ['321113', '311111', '111110', '111120', '211120', '236220', '513210', '621210']
    positions = [0.0, 4.0, 6.0, 10.0, 20.0, 21.0, 22.0, 23.0]
    scores = score_shortlist(
        torch.tensor([[0.0], [10.0]], dtype=torch.float64),
        [['111110', '311111'], ['111110', '311111']],
        torch.tensor([[position] for position in positions], dtype=torch.float64),
        codes,
        ['p01', 'p02'],
        distance='euclidean',
    )

    assert scores.select('rank', 'rank_l2', 'rank_l3').rows() == [(2, 1, 2), (2, 1, 1)]

def test_ties_count_against_the_truth_at_every_level():
    scores = score_shortlist(
        torch.ones(1, 2),
        [['311111']],
        torch.ones(8, 2),
        list(ITA_CANDIDATES),
        ['p01'],
        distance='euclidean',
    )

    # 311111 and 321113 are one sector; 311 and 321 differ
    assert scores.select('rank', 'rank_l2', 'rank_l3', 'rank_l4', 'rank_l5').rows() == [
        (8, 7, 8, 8, 8)
    ]

def test_a_coarser_rank_is_never_behind_a_finer_one():
    generator = torch.Generator().manual_seed(11)
    query_points = torch.randn(40, 3, generator=generator)
    candidate_points = torch.randn(8, 3, generator=generator)
    reported = []
    for size in torch.randint(1, 4, (40, ), generator=generator).tolist():
        indexes = torch.randint(0, 8, (size, ), generator=generator).tolist()
        reported.append(sorted({ITA_CANDIDATES[index] for index in indexes}))

    scores = score_shortlist(
        query_points,
        reported,
        candidate_points,
        list(ITA_CANDIDATES),
        [f'p{index:02d}' for index in range(40)],
        distance='euclidean',
    )

    ranks = [_column(scores, name) for name in ('rank_l2', 'rank_l3', 'rank_l4', 'rank_l5', 'rank')]
    for coarser, finer in zip(ranks, ranks[1:]):
        assert all(a <= b for a, b in zip(coarser, finer))
    assert ranks[0] != ranks[-1]  # the levels are not all alike

# -------------------------------------------------------------------------------------------------
# Result shape
# -------------------------------------------------------------------------------------------------

EXPECTED_SCHEMA = {
    'pair_id': pl.Utf8,
    'codes': pl.List(pl.Utf8),
    'top1_code': pl.Utf8,
    'rank': pl.Int64,
    'reciprocal_rank': pl.Float64,
    'hit_at_1': pl.Boolean,
    'hit_at_5': pl.Boolean,
    'hit_at_10': pl.Boolean,
    'lca_level': pl.Int64,
    'rank_l2': pl.Int64,
    'reciprocal_rank_l2': pl.Float64,
    'hit_at_1_l2': pl.Boolean,
    'hit_at_5_l2': pl.Boolean,
    'hit_at_10_l2': pl.Boolean,
    'rank_l3': pl.Int64,
    'reciprocal_rank_l3': pl.Float64,
    'hit_at_1_l3': pl.Boolean,
    'hit_at_5_l3': pl.Boolean,
    'hit_at_10_l3': pl.Boolean,
    'rank_l4': pl.Int64,
    'reciprocal_rank_l4': pl.Float64,
    'hit_at_1_l4': pl.Boolean,
    'hit_at_5_l4': pl.Boolean,
    'hit_at_10_l4': pl.Boolean,
    'rank_l5': pl.Int64,
    'reciprocal_rank_l5': pl.Float64,
    'hit_at_1_l5': pl.Boolean,
    'hit_at_5_l5': pl.Boolean,
    'hit_at_10_l5': pl.Boolean,
}

def test_the_coarse_levels_are_two_to_five():
    assert COARSE_LEVELS == (2, 3, 4, 5)

def test_the_frame_has_one_row_per_pair_in_the_order_given_with_the_stated_columns():
    codes = ['321113', '111110', '311111', '513210', '211120', '111120', '236220', '621210']

    scores = score_shortlist(
        torch.zeros(3, 1, dtype=torch.float64),
        [['311111', '211120', '311111'], ['111110'], ['621210', '236220']],
        **_line(codes),
        pair_ids=['p09', 'p02', 'p05'],
        distance='euclidean',
    )

    assert dict(scores.schema) == EXPECTED_SCHEMA
    assert scores.columns == list(EXPECTED_SCHEMA)
    assert scores.select('pair_id', 'codes', 'rank', 'top1_code').rows() == [
        ('p09', ['211120', '311111'], 3, '321113'),
        ('p02', ['111110'], 2, '321113'),
        ('p05', ['236220', '621210'], 7, '321113'),
    ]

def test_hits_and_reciprocal_ranks_at_each_threshold_at_every_level():
    # Twelve candidates in twelve different sectors: no code in front shares an ancestor with the
    # reported one at any level, so every level reads the rank. The pairs' ranks are 1, 5, 6, 10, 11
    sectors = ['11', '21', '22', '23', '31', '42', '44', '51', '52', '53', '54', '56']
    codes = [f'{sector}1110' for sector in sectors]

    scores = _score_line(codes, [[codes[0]], [codes[4]], [codes[5]], [codes[9]], [codes[10]]])

    for suffix in ('', '_l2', '_l3', '_l4', '_l5'):
        assert _column(scores, f'rank{suffix}') == [1, 5, 6, 10, 11]
        assert _column(scores, f'reciprocal_rank{suffix}') == [1.0, 1 / 5, 1 / 6, 1 / 10, 1 / 11]
        assert _column(scores, f'hit_at_1{suffix}') == [True, False, False, False, False]
        assert _column(scores, f'hit_at_5{suffix}') == [True, True, False, False, False]
        assert _column(scores, f'hit_at_10{suffix}') == [True, True, True, True, False]

# -------------------------------------------------------------------------------------------------
# Precision and devices
# -------------------------------------------------------------------------------------------------

def test_distances_are_computed_in_float64_on_the_cpu():
    # In float32, sqrt(1 + 1e-10) is 1: the decoy at (1, 1e-5) would tie with the reported code
    # at (1, 0) and rank the pair second
    scores = score_shortlist(
        torch.zeros(1, 2, dtype=torch.float32),
        [['111110']],
        torch.tensor([[1.0, 0.0], [1.0, 1e-5]], dtype=torch.float32),
        ['111110', '311111'],
        ['p01'],
        distance='euclidean',
    )

    assert scores.get_column('rank').to_list() == [1]

def test_a_custom_distance_receives_float64_cpu_points():
    seen = []

    def spy(queries, candidates):
        seen.append((queries.dtype, queries.device.type, candidates.dtype, candidates.device.type))
        return torch.cdist(queries, candidates)

    score_shortlist(
        torch.zeros(1, 2, dtype=torch.float32),
        [['111110']],
        torch.ones(2, 2, dtype=torch.float32),
        ['111110', '311111'],
        ['p01'],
        distance=spy,
    )

    assert seen == [(torch.float64, 'cpu', torch.float64, 'cpu')]

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_points_on_mps_are_scored_in_float64_on_the_cpu():
    codes = ['321113', '111110', '311111', '513210']
    cpu = _score_line(codes, [['311111']])

    scores = score_shortlist(
        torch.zeros(1, 1, device='mps'),
        [['311111']],
        torch.tensor([[1.0], [2.0], [3.0], [4.0]], device='mps'),
        codes,
        ['p00'],
        distance='euclidean',
    )

    assert scores.equals(cpu)

# -------------------------------------------------------------------------------------------------
# Refusals
# -------------------------------------------------------------------------------------------------

CANDIDATES = ['111110', '111120', '111211', '211111']
CANDIDATE_POINTS = [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [5.0, 5.0]]

def _refusal_arguments(**overrides):
    '''Arguments of one valid pair, over four candidates, with the ones given replaced.'''

    return {
        'query_points': torch.zeros(1, 2),
        'reported_codes': [['111110']],
        'candidate_points': torch.tensor(CANDIDATE_POINTS),
        'candidate_codes': CANDIDATES,
        'pair_ids': ['p01'],
        'distance': 'euclidean',
        **overrides,
    }

def test_a_code_that_is_not_a_candidate_or_an_empty_code_list_is_refused():
    # An empty list of reported codes, on its own and in the second of two pairs
    with pytest.raises(ValueError, match='1 pairs report no code'):
        score_shortlist(**_refusal_arguments(reported_codes=[[]]))
    with pytest.raises(ValueError, match=r"report no code, e.g. \['p02'\]"):
        score_shortlist(
            **_refusal_arguments(
                query_points=torch.zeros(2, 2),
                reported_codes=[['111110'], []],
                pair_ids=['p01', 'p02'],
            )
        )
    # A code that is not a candidate, alone, beside a candidate and in the second of two pairs
    with pytest.raises(ValueError, match=r"1 reported codes are not candidates, e.g. \['112130'\]"):
        score_shortlist(**_refusal_arguments(reported_codes=[['112130']]))
    with pytest.raises(ValueError, match=r"1 reported codes are not candidates, e.g. \['112130'\]"):
        score_shortlist(**_refusal_arguments(reported_codes=[['111110', '112130']]))
    with pytest.raises(ValueError, match=r"1 reported codes are not candidates, e.g. \['112130'\]"):
        score_shortlist(
            **_refusal_arguments(
                query_points=torch.zeros(2, 2),
                reported_codes=[['111110'], ['112130']],
                pair_ids=['p01', 'p02'],
            )
        )

@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({
            'reported_codes': [[]]
        }, 'report no code'),
        ({
            'reported_codes': [['112130']]
        }, 'not candidates'),
        ({
            'reported_codes': [['111110', '112130']]
        }, 'not candidates'),
        ({
            'candidate_codes': ['111110', '111110', '111211', '211111']
        }, 'distinct'),
        ({
            'candidate_codes': ['111110', '111120', '111211']
        }, 'candidate points for'),
        ({
            'query_points': torch.zeros(1, 3)
        }, 'coordinates'),
        ({
            'query_points': torch.zeros(2, 2)
        }, 'must agree'),
        ({
            'reported_codes': [['111110'], ['111120']]
        }, 'must agree'),
        ({
            'pair_ids': ['p01', 'p02']
        }, 'must agree'),
        (
            {
                'query_points': torch.zeros(0, 2),
                'reported_codes': [],
                'pair_ids': []
            }, 'non-empty 2-D'
        ),
        ({
            'candidate_points': torch.zeros(0, 2),
            'candidate_codes': []
        }, 'non-empty 2-D'),
        ({
            'distance': 'manhattan'
        }, 'unknown distance'),
        ({
            'distance': lambda q, c: torch.full((q.shape[0], c.shape[0]), math.nan)
        }, 'non-finite'),
        ({
            'distance': lambda q, c: torch.full((q.shape[0], c.shape[0]), math.inf)
        }, 'non-finite'),
        (
            {
                'distance': lambda q, c: torch.zeros(q.shape[0], c.shape[0] + 1)
            }, 'distance returned shape'
        ),
    ],
)
def test_inconsistent_inputs_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        score_shortlist(**_refusal_arguments(**kwargs))
