'''
The ITA shortlist scorer (plan 17, Task 3; spec text-to-code-ita-panel, 4.3).

A pair ranks all candidates under the encoder's own distance and keeps the best rank over its
reported codes, with ties against the truth. Expected values are worked out by hand, never from
production code, except where a test compares the scorer with ``score_decoding`` on pairs of one
code. Most tests put the candidates on a line (the i-th at distance i + 1 from a query at the
origin) so each rank can be read off the order of the codes.

The summary's tests (plan 17, Task 4) score the fixture's validation pairs with the assigned stub,
or hand the summary a scorer-shaped frame whose ranks the test chose, and compare each report with
means worked out in the test.
'''

import math
from typing import Any, Dict, List, Sequence, Tuple

import polars as pl
import pytest
import torch

from naics_embedder.panels.decoding import score_decoding
from naics_embedder.panels.ita import PANEL_PAIR_SCHEMA, panel_pairs
from naics_embedder.panels.ita_scoring import (
    COARSE_LEVELS,
    SUMMARY_METRICS,
    ItaResult,
    score_shortlist,
    summarize_ita,
)
from tests.fixtures.ita import ITA_CANDIDATES, ITA_SETTINGS, AssignedStubEncoder, ita_tables

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

# -------------------------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------------------------

def _stub_scores(pairs: pl.DataFrame) -> pl.DataFrame:
    '''Score panel pairs with the assigned stub, by cosine, over every candidate.'''

    stub = AssignedStubEncoder()
    return score_shortlist(
        stub.encode_queries(pairs.get_column('query').to_list()),
        pairs.get_column('codes').to_list(),
        stub.encode_codes(ITA_CANDIDATES),
        ITA_CANDIDATES,
        pairs.get_column('pair_id').to_list(),
        distance='cosine',
    )

# The fixture's validation part: p01 to p05 and p07, p05 copied, p04 a conflict pair. p03 ranks
# 8th (the stub decodes it to 111110, not its 211120) and the others 1st
PANEL_PAIRS = panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt).filter(
    pl.col('part') == 'validation'
)
SCORES = _stub_scores(PANEL_PAIRS)
result = summarize_ita(SCORES, PANEL_PAIRS, 'cosine')

def _rank_columns(rank: int, suffix: str) -> Dict[str, Any]:
    '''One pair's rank, reciprocal rank and hits at a level, as the scorer names them.'''

    return {
        f'rank{suffix}': rank,
        f'reciprocal_rank{suffix}': 1 / rank,
        **{
            f'hit_at_{k}{suffix}': rank <= k
            for k in (1, 5, 10)
        },
    }

def _synthetic(specs: Sequence[Dict[str, Any]]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    A scorer-shaped frame and a panel pair frame for pairs given by hand, ids p00, p01, ...

    A spec names ``key`` (the description group) and may name ``codes`` (default 311111), ``rank``
    (default 1, the rank at the finest level and at each coarse one), ``levels`` (a rank for
    some of ``_l2`` to ``_l5``, over ``rank``), ``lca_level`` (default 6) and the flags ``copied``,
    ``conflict``, ``mojibake`` and ``company_fallback``.
    '''

    scores, pairs = [], []
    for index, spec in enumerate(specs):
        pair_id = f'p{index:02d}'
        codes = spec.get('codes', ['311111'])
        row = {
            'pair_id': pair_id,
            'codes': codes,
            'top1_code': codes[0],
            **_rank_columns(spec.get('rank', 1), ''),
            'lca_level': spec.get('lca_level', 6),
        }
        for level in COARSE_LEVELS:
            suffix = f'_l{level}'
            row.update(
                _rank_columns(spec.get('levels', {}).get(suffix, spec.get('rank', 1)), suffix)
            )
        scores.append(row)
        pairs.append(
            {
                'pair_id': pair_id,
                'description_key': spec['key'],
                'query': f'query {pair_id}',
                'codes': codes,
                'copied': spec.get('copied', False),
                'conflict': spec.get('conflict', False),
                'mojibake': spec.get('mojibake', False),
                'company_fallback': spec.get('company_fallback', False),
                'part': 'validation',
            }
        )
    return (
        pl.DataFrame(scores, schema=EXPECTED_SCHEMA),
        pl.DataFrame(pairs, schema=PANEL_PAIR_SCHEMA),
    )

def _summarize(specs: Sequence[Dict[str, Any]]) -> ItaResult:
    return summarize_ita(*_synthetic(specs), 'euclidean')

def test_the_summary_metrics_are_the_stated_names():
    assert SUMMARY_METRICS == (
        'mrr', 'top1', 'hit_at_1', 'hit_at_5', 'hit_at_10', 'lca_level',
        'mrr_l2', 'hit_at_1_l2', 'hit_at_5_l2', 'hit_at_10_l2',
        'mrr_l3', 'hit_at_1_l3', 'hit_at_5_l3', 'hit_at_10_l3',
        'mrr_l4', 'hit_at_1_l4', 'hit_at_5_l4', 'hit_at_10_l4',
        'mrr_l5', 'hit_at_1_l5', 'hit_at_5_l5', 'hit_at_10_l5',
    )  # yapf: disable

# -------------------------------------------------------------------------------------------------
# Summary: the plan's reports
# -------------------------------------------------------------------------------------------------

def test_the_headline_excludes_copied_pairs_and_counts_each_pair_once():
    assert result.headline['n_pairs'] == 5
    assert result.headline['mrr'] == pytest.approx((1 + 1 + 1 / 8 + 1 + 1) / 5)  # 0.825

def test_the_phrasing_report_counts_each_description_once():
    assert result.phrasing['n_groups'] == 4
    assert result.phrasing['mrr'] == pytest.approx((1 + 1 / 8 + 1 + 1) / 4)  # 0.78125

def test_copied_pairs_are_reported_apart():
    assert result.copied['n_pairs'] == 1 and result.copied['mrr'] == 1.0

def test_counts_report_conflicts_and_flags_without_changing_the_headline():
    assert result.counts == {
        'n_pairs': 6,
        'n_groups': 5,
        'n_copied': 1,
        'n_conflict': 1,
        'n_mojibake': 0,
        'n_company_fallback': 1,
    }

def test_per_code_counts_a_conflict_pair_under_each_code():
    # p04 reports 311111 and 111110, and the stub decodes it to 111110: rank 1. p01 and p02 report
    # 311111, p03 reports 211120 (rank 8) and p07 reports 236220. p05 reports 513210 but is copied
    per_code = result.per_code

    assert per_code.columns == ['code', 'n_pairs', *SUMMARY_METRICS]
    assert per_code.select('code', 'n_pairs', 'mrr', 'hit_at_1', 'hit_at_10').rows() == [
        ('111110', 1, 1.0, 1.0, 1.0),
        ('211120', 1, 1 / 8, 0.0, 1.0),
        ('236220', 1, 1.0, 1.0, 1.0),
        ('311111', 3, 1.0, 1.0, 1.0),
    ]

def test_a_part_with_no_headline_pair_is_refused():
    every_pair_copied = PANEL_PAIRS.with_columns(copied=pl.lit(True))

    with pytest.raises(ValueError, match='no headline pair'):
        summarize_ita(SCORES, every_pair_copied, 'cosine')

# -------------------------------------------------------------------------------------------------
# Summary: the reports in detail
# -------------------------------------------------------------------------------------------------

def test_a_part_with_no_pair_at_all_is_refused():
    with pytest.raises(ValueError, match='no headline pair'):
        summarize_ita(SCORES.clear(), PANEL_PAIRS.clear(), 'cosine')

def test_the_per_pair_frame_is_the_scorers_columns_with_the_description_and_copied_flag():
    per_pair = result.per_pair

    assert per_pair.columns == [*SCORES.columns, 'description_key', 'copied']
    assert per_pair.select(SCORES.columns).equals(SCORES)
    assert per_pair.select('pair_id', 'description_key', 'copied').rows() == [
        ('p01', 'we make dog food', False),
        ('p02', 'we make dog food', False),
        ('p03', 'oil and gas wells', False),
        ('p04', 'cat food plant', False),
        ('p05', 'software publisher', True),
        ('p07', 'general contractor', False),
    ]

def test_the_per_pair_frame_follows_the_scores_whatever_the_order_of_the_pairs():
    shuffled = PANEL_PAIRS.sort('pair_id', descending=True)

    per_pair = summarize_ita(SCORES, shuffled, 'cosine').per_pair

    assert per_pair.get_column('pair_id').to_list() == SCORES.get_column('pair_id').to_list()
    assert per_pair.get_column('description_key').to_list() == (
        result.per_pair.get_column('description_key').to_list()
    )
    assert per_pair.get_column('copied').to_list() == result.per_pair.get_column('copied').to_list()

def test_the_reports_name_the_distance_and_hold_every_metric_as_a_python_number():
    expected_keys = ['n_pairs', 'distance', *SUMMARY_METRICS]

    assert list(result.headline) == expected_keys
    assert list(result.copied) == expected_keys
    assert list(result.phrasing) == ['n_groups', *SUMMARY_METRICS]
    assert result.headline['distance'] == 'cosine' and result.copied['distance'] == 'cosine'
    for report in (result.headline, result.copied, result.phrasing):
        assert type(report.get('n_pairs', report.get('n_groups'))) is int
        assert all(type(report[name]) is float for name in SUMMARY_METRICS)
    assert all(type(count) is int for count in result.counts.values())

def test_the_distance_is_named_as_given():
    scores, pairs = _synthetic([{'key': 'a'}, {'key': 'b', 'copied': True}])

    summary = summarize_ita(scores, pairs, 'lorentz')

    assert summary.headline['distance'] == 'lorentz' and summary.copied['distance'] == 'lorentz'

def test_with_no_copied_pair_the_copied_report_has_none_for_every_metric():
    copied = _summarize([{'key': 'a'}, {'key': 'b', 'rank': 2}]).copied

    assert copied == {
        'n_pairs': 0,
        'distance': 'euclidean',
        **{
            name: None
            for name in SUMMARY_METRICS
        },
    }

def test_the_flags_change_no_report_but_the_counts():
    flagged = PANEL_PAIRS.with_columns(
        mojibake=pl.col('pair_id').is_in(['p01', 'p03']),
        company_fallback=pl.col('pair_id').is_in(['p02', 'p04']),
        conflict=pl.col('pair_id').is_in(['p01', 'p07']),
    )

    summary = summarize_ita(SCORES, flagged, 'cosine')

    assert summary.headline == result.headline
    assert summary.copied == result.copied
    assert summary.phrasing == result.phrasing
    assert summary.per_code.equals(result.per_code)
    assert summary.counts == {
        'n_pairs': 6,
        'n_groups': 5,
        'n_copied': 1,
        'n_conflict': 2,
        'n_mojibake': 2,
        'n_company_fallback': 2,
    }

def test_each_count_is_its_own_flag_over_every_pair_copied_ones_included():
    # Ten pairs in three groups: one copied (and mojibake), two conflict, three mojibake and four
    # company-fallback. No two counts are alike
    specs = [
        {
            'key': 'a',
            'copied': True,
            'mojibake': True
        },
        {
            'key': 'a',
            'conflict': True
        },
        {
            'key': 'a',
            'conflict': True
        },
        {
            'key': 'a',
            'mojibake': True
        },
        {
            'key': 'b',
            'mojibake': True
        },
        {
            'key': 'b',
            'company_fallback': True
        },
        {
            'key': 'b',
            'company_fallback': True
        },
        {
            'key': 'b',
            'company_fallback': True
        },
        {
            'key': 'c',
            'company_fallback': True
        },
        {
            'key': 'c'
        },
    ]

    assert _summarize(specs).counts == {
        'n_pairs': 10,
        'n_groups': 3,
        'n_copied': 1,
        'n_conflict': 2,
        'n_mojibake': 3,
        'n_company_fallback': 4,
    }

def test_a_group_that_is_all_copied_is_a_group_in_the_counts_but_not_in_the_phrasing_report():
    summary = _summarize([{'key': 'a'}, {'key': 'a', 'copied': True}, {'key': 'b', 'copied': True}])

    assert summary.counts['n_groups'] == 2
    assert summary.phrasing['n_groups'] == 1

def test_the_phrasing_report_averages_each_groups_non_copied_pairs_and_then_the_groups():
    # Group a holds three pairs at rank 1 and a copied one at rank 11, group b one pair at rank 2,
    # and group c only a copied pair. The groups' means are 1 and 1/2
    summary = _summarize(
        [
            {
                'key': 'a'
            },
            {
                'key': 'a'
            },
            {
                'key': 'a'
            },
            {
                'key': 'a',
                'copied': True,
                'rank': 11
            },
            {
                'key': 'b',
                'rank': 2
            },
            {
                'key': 'c',
                'copied': True,
                'rank': 11
            },
        ]
    )

    assert summary.phrasing['n_groups'] == 2
    assert summary.phrasing['mrr'] == pytest.approx((1 + 1 / 2) / 2)
    assert summary.phrasing['hit_at_1'] == pytest.approx((1 + 0) / 2)
    assert summary.headline['n_pairs'] == 4
    assert summary.headline['mrr'] == pytest.approx((3 + 1 / 2) / 4)  # pairs, not groups

def test_a_group_with_different_codes_keeps_every_pair():
    summary = _summarize(
        [
            {
                'key': 'a',
                'codes': ['311111'],
                'rank': 1
            },
            {
                'key': 'a',
                'codes': ['111110'],
                'rank': 3
            },
            {
                'key': 'b',
                'codes': ['236220'],
                'rank': 1
            },
        ]
    )

    assert summary.phrasing['mrr'] == pytest.approx(((1 + 1 / 3) / 2 + 1) / 2)
    assert summary.headline['n_pairs'] == 3

def test_per_code_means_each_metric_over_a_codes_non_copied_pairs():
    # 311111 is reported by p00 (rank 1), p01 (rank 3) and p02 (rank 12, which also reports
    # 111110); p03 reports 111110 and is copied, and p04 reports 236220 and is copied
    summary = _summarize(
        [
            {
                'key': 'a',
                'codes': ['311111'],
                'rank': 1,
                'lca_level': 6
            },
            {
                'key': 'b',
                'codes': ['311111'],
                'rank': 3,
                'lca_level': 4
            },
            {
                'key': 'c',
                'codes': ['111110', '311111'],
                'rank': 12,
                'lca_level': 1
            },
            {
                'key': 'd',
                'codes': ['111110'],
                'rank': 1,
                'copied': True
            },
            {
                'key': 'e',
                'codes': ['236220'],
                'rank': 1,
                'copied': True
            },
        ]
    )

    assert summary.per_code.select(
        'code', 'n_pairs', 'mrr', 'hit_at_1', 'hit_at_5', 'hit_at_10', 'lca_level'
    ).rows() == [
        ('111110', 1, pytest.approx(1 / 12), 0.0, 0.0, 0.0, 1.0),
        (
            '311111', 3, pytest.approx((1 + 1 / 3 + 1 / 12) / 3), pytest.approx(1 / 3),
            pytest.approx(2 / 3), pytest.approx(2 / 3), pytest.approx(11 / 3)
        ),
    ]

# A pair's rank at each level, finest first (the key ''), and its LCA level. Each level's ranks
# give it hits at 1, 5 and 10 that no other level has, so a metric read from the wrong level shows
PAIR_RANKS = [
    ({
        '': 2,
        '_l5': 1,
        '_l4': 1,
        '_l3': 1,
        '_l2': 1
    }, 6),
    ({
        '': 6,
        '_l5': 2,
        '_l4': 1,
        '_l3': 1,
        '_l2': 1
    }, 5),
    ({
        '': 11,
        '_l5': 6,
        '_l4': 2,
        '_l3': 1,
        '_l2': 1
    }, 4),
    ({
        '': 12,
        '_l5': 11,
        '_l4': 6,
        '_l3': 2,
        '_l2': 1
    }, 3),
    ({
        '': 13,
        '_l5': 12,
        '_l4': 11,
        '_l3': 6,
        '_l2': 2
    }, 2),
    ({
        '': 14,
        '_l5': 13,
        '_l4': 12,
        '_l3': 11,
        '_l2': 6
    }, 1),
]

def _expected_metrics(ranks: Dict[str, int], lca: int) -> Dict[str, float]:
    '''A pair's summary metrics, from its rank at each level and its LCA level.'''

    metrics = {'lca_level': float(lca), 'top1': float(ranks[''] == 1)}
    for suffix, rank in ranks.items():
        metrics[f'mrr{suffix}'] = 1 / rank
        for k in (1, 5, 10):
            metrics[f'hit_at_{k}{suffix}'] = float(rank <= k)
    return metrics

def _pair_specs(**extra) -> List[Dict[str, Any]]:
    '''PAIR_RANKS as pairs in groups of their own and on codes of their own, in code order.'''

    codes = ['111110', '211120', '236220', '311111', '513210', '621210']
    return [
        {
            'key': f'group {index}',
            'codes': [code],
            'rank': ranks[''],
            'levels': {
                suffix: rank
                for suffix, rank in ranks.items() if suffix
            },
            'lca_level': lca,
            **extra,
        } for index, (code, (ranks, lca)) in enumerate(zip(codes, PAIR_RANKS))
    ]

def _mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}

def test_the_oracle_names_every_summary_metric():
    assert set(_expected_metrics(*PAIR_RANKS[0])) == set(SUMMARY_METRICS)

def test_each_metric_reads_its_own_column_in_the_headline_and_the_phrasing_report():
    summary = _summarize(_pair_specs())

    expected = _mean_metrics([_expected_metrics(*pair) for pair in PAIR_RANKS])
    for name in SUMMARY_METRICS:
        assert summary.headline[name] == pytest.approx(expected[name]), name
        assert summary.phrasing[name] == pytest.approx(expected[name]), name

def test_each_metric_reads_its_own_column_in_the_copied_report():
    summary = _summarize([{'key': 'plain'}, *_pair_specs(copied=True)])

    expected = _mean_metrics([_expected_metrics(*pair) for pair in PAIR_RANKS])
    assert summary.copied['n_pairs'] == 6
    for name in SUMMARY_METRICS:
        assert summary.copied[name] == pytest.approx(expected[name]), name

def test_each_metric_reads_its_own_column_in_the_per_code_frame():
    summary = _summarize(_pair_specs())

    for row, pair in zip(summary.per_code.iter_rows(named=True), PAIR_RANKS):
        assert row['n_pairs'] == 1
        for name, value in _expected_metrics(*pair).items():
            assert row[name] == pytest.approx(value), (row['code'], name)

# -------------------------------------------------------------------------------------------------
# Summary: inputs the reports refuse
# -------------------------------------------------------------------------------------------------

def _renamed(scores: pl.DataFrame, old: str, new: str) -> pl.DataFrame:
    return scores.with_columns(
        pair_id=pl.when(pl.col('pair_id') == old).then(pl.lit(new)).otherwise(pl.col('pair_id'))
    )

@pytest.mark.parametrize(
    'scores, pairs',
    [
        pytest.param(SCORES, PANEL_PAIRS.head(5), id='extra_scored_pair'),
        pytest.param(SCORES.head(5), PANEL_PAIRS, id='unscored_panel_pair'),
        pytest.param(_renamed(SCORES, 'p01', 'p99'), PANEL_PAIRS, id='one_pair_id_renamed'),
        pytest.param(pl.concat([SCORES, SCORES.head(1)]), PANEL_PAIRS, id='pair_scored_twice'),
        pytest.param(
            SCORES,
            pl.concat([PANEL_PAIRS, PANEL_PAIRS.head(1)]),
            id='pair_listed_twice',
        ),
        pytest.param(
            pl.concat([SCORES, SCORES.head(1)]),
            pl.concat([PANEL_PAIRS, PANEL_PAIRS.head(1)]),
            id='pair_scored_and_listed_twice',
        ),
    ],
)
def test_scores_and_pairs_with_different_pair_ids_are_refused(scores, pairs):
    with pytest.raises(ValueError, match='pair ids'):
        summarize_ita(scores, pairs, 'cosine')

@pytest.mark.parametrize(
    'column', ['description_key', 'copied', 'conflict', 'mojibake', 'company_fallback']
)
def test_a_null_in_a_pair_column_the_reports_read_is_refused(column):
    pairs = PANEL_PAIRS.with_columns(
        pl.when(pl.col('pair_id') == 'p01').then(None).otherwise(pl.col(column)).alias(column)
    )

    with pytest.raises(ValueError, match=f'null.*{column}'):
        summarize_ita(SCORES, pairs, 'cosine')
