'''
Text-to-code decoding scores (Req 3): distances, ranks with pessimistic ties, partial credit.

Expected values are worked out by hand, never from production code.
'''

import math

import pytest
import torch

from naics_embedder.metrics.core import lorentz_distance_matrix
from naics_embedder.panels.decoding import (
    cosine_distances,
    euclidean_distances,
    lca_level,
    lorentz_distances,
    score_decoding,
)

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def _f64(rows):
    return torch.tensor(rows, dtype=torch.float64)

def test_euclidean_distances():
    distances = euclidean_distances(_f64([[0.0, 0.0]]), _f64([[3.0, 4.0], [0.0, 1.0]]))

    assert distances.tolist() == [[5.0, 1.0]]

def test_cosine_distances():
    distances = cosine_distances(_f64([[1.0, 0.0]]), _f64([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]]))

    assert distances.tolist() == [[0.0, 1.0, 2.0]]

def test_lorentz_distances_rederive_the_time_coordinate():
    origin = _f64([[1.0, 0.0, 0.0]])
    points = _f64([[math.cosh(1.0), math.sinh(1.0), 0.0], [math.cosh(2.0), 0.0, math.sinh(2.0)]])
    off_manifold = points.clone()
    off_manifold[:, 0] = 0.0

    torch.testing.assert_close(lorentz_distances(origin, points), _f64([[1.0, 2.0]]))
    torch.testing.assert_close(lorentz_distances(origin, off_manifold), _f64([[1.0, 2.0]]))

def test_lorentz_distances_agree_with_the_evaluation_matrix():
    generator = torch.Generator().manual_seed(0)
    space = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    points = torch.cat([torch.sqrt(1.0 + (space * space).sum(dim=1, keepdim=True)), space], dim=1)

    expected = lorentz_distance_matrix(points)[:2, 2:]

    torch.testing.assert_close(lorentz_distances(points[:2], points[2:]), expected)

# -------------------------------------------------------------------------------------------------
# Partial credit
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('code_a', 'code_b', 'level'),
    [
        ('111110', '111110', 6),
        ('111110', '111120', 4),
        ('111110', '111211', 3),
        ('311111', '332111', 2),  # 31-33 is one sector
        ('441110', '452210', 2),  # 44-45
        ('481111', '493110', 2),  # 48-49
        ('111110', '211120', 1),  # only the virtual root
    ],
)
def test_lca_level(code_a, code_b, level):
    assert lca_level(code_a, code_b) == level
    assert lca_level(code_b, code_a) == level

# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

CANDIDATES = ['111110', '111120', '111211', '211111']
CANDIDATE_POINTS = [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [5.0, 5.0]]

@pytest.fixture
def scored():
    return score_decoding(
        torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.0, 2.9], [0.6, 0.0]]),
        ['111110', '111120', '211111', '111110'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance='euclidean',
        query_ids=[10, 11, 12, 13],
    )

def test_per_query_ranks_top1_and_partial_credit(scored):
    # 0.1 from its code; 0.8 from its code but 0.2 from 111110; 5.42 from its code and nearer
    # every other candidate; 0.6 from its code but 0.4 from 111120
    assert scored.per_query.select('query_id', 'code', 'top1_code', 'rank', 'lca_level').rows() == [
        (10, '111110', '111110', 1, 6),
        (11, '111120', '111110', 2, 4),
        (12, '211111', '111211', 4, 1),
        (13, '111110', '111120', 2, 4),
    ]
    assert scored.per_query.get_column('hit_at_1').to_list() == [True, False, False, False]
    assert scored.per_query.get_column('hit_at_5').to_list() == [True, True, True, True]

def test_summary_reports_every_metric_query_weighted(scored):
    assert scored.summary == {
        'distance': 'euclidean',
        'n_queries': 4,
        'n_codes': 3,
        'n_candidates': 4,
        'top1': 0.25,
        'mrr': 0.5625,  # (1 + 1/2 + 1/4 + 1/2) / 4
        'hit_at_1': 0.25,
        'hit_at_5': 1.0,
        'hit_at_10': 1.0,
        'lca_level': 3.75,  # (6 + 4 + 1 + 4) / 4
    }

def test_per_code_means(scored):
    assert scored.per_code.select('code', 'n_queries', 'top1', 'mrr', 'lca_level').rows() == [
        ('111110', 2, 0.5, 0.75, 5.0),
        ('111120', 1, 0.0, 0.5, 4.0),
        ('211111', 1, 0.0, 0.25, 1.0),
    ]

def test_a_constant_encoder_ranks_every_truth_last():
    codes = [f'1111{index:02d}' for index in range(12)]

    result = score_decoding(
        torch.ones(3, 2), codes[:3], torch.ones(12, 2), codes, distance='euclidean'
    )

    assert result.per_query.get_column('rank').to_list() == [12, 12, 12]
    assert result.summary['top1'] == 0.0
    assert result.summary['mrr'] == pytest.approx(1 / 12)
    assert result.summary['hit_at_10'] == 0.0

def test_a_tie_with_the_truth_counts_against_it():
    result = score_decoding(
        torch.tensor([[0.5, 0.0]]),
        ['111120'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance='euclidean',
    )

    assert result.per_query.select('rank', 'top1_code').rows() == [(2, '111110')]

def test_a_custom_distance_is_pluggable():

    def manhattan(queries, candidates):
        return torch.cdist(queries, candidates, p=1.0)

    result = score_decoding(
        torch.tensor([[0.0, 2.5]]),
        ['111211'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance=manhattan,
    )

    assert result.summary['distance'] == 'manhattan'
    assert result.summary['top1'] == 1.0

@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({
            'query_codes': ['112130']
        }, 'not candidates'),
        ({
            'candidate_codes': ['111110', '111110', '111211', '211111']
        }, 'distinct'),
        ({
            'query_points': torch.zeros(1, 3)
        }, 'coordinates'),
        ({
            'query_points': torch.zeros(2, 2)
        }, 'must agree'),
        ({
            'query_points': torch.zeros(0, 2),
            'query_codes': []
        }, 'non-empty 2-D'),
        ({
            'distance': 'manhattan'
        }, 'unknown distance'),
        ({
            'distance': lambda q, c: torch.full((q.shape[0], c.shape[0]), math.nan)
        }, 'non-finite'),
    ],
)
def test_inconsistent_inputs_fail_closed(kwargs, message):
    arguments = {
        'query_points': torch.zeros(1, 2),
        'query_codes': ['111110'],
        'candidate_points': torch.tensor(CANDIDATE_POINTS),
        'candidate_codes': CANDIDATES,
        'distance': 'euclidean',
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        score_decoding(**arguments)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_points_on_mps_are_scored_in_float64_on_the_cpu(scored):
    result = score_decoding(
        torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.0, 2.9], [0.6, 0.0]], device='mps'),
        ['111110', '111120', '211111', '111110'],
        torch.tensor(CANDIDATE_POINTS, device='mps'),
        CANDIDATES,
        distance='euclidean',
        query_ids=[10, 11, 12, 13],
    )

    assert result.summary == scored.summary
