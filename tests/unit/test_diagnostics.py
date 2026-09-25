'''
Req 6's diagnostics, worked by hand on small trees and distance matrices.

The tree below has two sectors. 11111 has two six-digit children; 11121 has one, 111211, so
(11121, 111211) is its only unary pair (Req 9).
'''

import numpy as np
import polars as pl
import pytest

from naics_embedder.metrics.diagnostics import (
    DiagnosticsReport,
    Tree,
    average_precision,
    diagnostics_report,
    distance_pearson,
    ndcg_at,
    pairwise_distances,
    parent_retrieval,
    sector_separation,
    within_sector_rank_correlation,
)
from tests.fixtures.regressor_panel import CODEBOOK, coordinate_table

pytestmark = pytest.mark.unit

CODES = (
    '11', '111', '1111', '11111', '111111', '111112', '1112', '11121', '111211', '21', '211',
    '2111', '21111', '211111', '211112'
)

@pytest.fixture
def tree():
    return Tree.from_codes(CODES)

def _row(code):
    return CODES.index(code)

def _target(tree):
    return tree.depth[:, None] + tree.depth[None, :] - 2 * tree.lca_depth()

def test_d_star_runs_through_the_lowest_common_ancestor_and_a_virtual_root(tree):
    target = _target(tree)

    assert tree.depth[[_row('11'), _row('111211')]].tolist() == [1, 5]
    assert target[_row('111111'), _row('111112')] == 2  # siblings, through 11111
    assert target[_row('111111'), _row('1111')] == 2  # an ancestor two levels up
    assert target[_row('111111'), _row('111211')] == 6  # through 111
    assert target[_row('111111'), _row('21')] == 6  # through the virtual root: 5 + 1
    assert tree.parent[_row('111')] == _row('11')
    assert tree.unary_children().tolist() == [code == '111211' for code in CODES]

def test_a_code_whose_ancestor_is_missing_is_refused():
    with pytest.raises(ValueError, match='ancestor 1111 is not among the codes'):
        Tree.from_codes(['11', '111', '11111'])

def test_sector_separation_is_the_auc_of_cross_over_same_sector_distances():
    # Same-sector pairs at 1 and 2; cross-sector pairs at 2, 3, 4 and 5: of 8 comparisons, 7
    # are won and one (2 against 2) tied
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 2.0],
            [3.0, 5.0, 2.0, 0.0],
        ]
    )

    result = sector_separation(distances, np.array([0, 0, 1, 1]))

    assert result.auc == pytest.approx(7.5 / 8)
    assert (result.same_sector_pairs, result.cross_sector_pairs) == (2, 4)

def test_within_sector_rank_correlation_is_one_when_distance_orders_as_d_star(tree):
    target = _target(tree).astype(float)

    exact = within_sector_rank_correlation(target, target, tree.sector, CODES)
    reversed_ = within_sector_rank_correlation(-target, target, tree.sector, CODES)
    flat = within_sector_rank_correlation(np.ones_like(target), target, tree.sector, CODES)

    assert exact.mean_over_queries == pytest.approx(1.0)
    assert exact.by_sector == pytest.approx({'11': 1.0, '21': 1.0})
    assert (exact.queries, exact.undefined_queries) == (15, 0)
    assert reversed_.mean_over_sectors == pytest.approx(-1.0)
    assert (flat.mean_over_queries, flat.undefined_queries) == (None, 15)

def test_average_precision_breaks_ties_against_relevance():
    # Ranked: 0 (relevant), 2, 1 (relevant: tied with 2, so after it), 3 (relevant), 5, 4
    # (relevant: tied with 5)
    distances = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    relevant = np.array([True, True, False, True, True, False])

    assert average_precision(distances, relevant) == pytest.approx(
        (1 / 1 + 2 / 3 + 3 / 4 + 4 / 6) / 4
    )

def test_ndcg_uses_linear_integer_gains_and_breaks_ties_against_them():
    # Ranked gains: 1 then 4 (tied at distance 1), then 3 and 3, then 0
    distances = np.array([1.0, 2.0, 2.0, 3.0, 1.0])
    gains = np.array([4.0, 3.0, 3.0, 0.0, 1.0])

    dcg = 1 + 4 / np.log2(3) + 3 / np.log2(4)
    ideal = 4 + 3 / np.log2(3) + 3 / np.log2(4)
    assert ndcg_at(distances, gains, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at(distances, np.zeros(5), 3) is None

def test_the_pearson_statistic_correlates_distance_with_d_star_over_all_pairs(tree):
    target = _target(tree)

    result = distance_pearson(target.astype(float), target)

    assert result.value == pytest.approx(1.0)
    assert result.pairs == 15 * 14 // 2
    assert distance_pearson(np.ones(target.shape), target).value is None

def test_parent_retrieval_skips_the_unary_pairs_and_breaks_ties_against_the_parent(tree):
    # Under D*, a code's parent and children are all at 1: only a leaf finds its parent first
    result = parent_retrieval(_target(tree).astype(float), tree)

    assert (result.queries, result.unary_pairs_excluded) == (12, 1)
    # Leaves 111111, 111112, 211111 and 211112 rank their parent first
    assert result.at == pytest.approx({'1': 4 / 12, '5': 1.0})

def test_each_geometry_reads_its_own_distance():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0], [6.0, 8.0]])

    euclidean = pairwise_distances(points, 'euclidean')
    spherical = pairwise_distances(points[1:], 'spherical')
    hyperbolic = pairwise_distances(points, 'hyperbolic')
    curved = pairwise_distances(points, 'hyperbolic', curvature=4.0)

    assert euclidean[0, 1] == pytest.approx(5.0)
    assert spherical[0, 2] == pytest.approx(0.0, abs=1e-12)
    # A tangent vector at the origin maps to a point its length away, along its ray
    assert hyperbolic[0, 1] == pytest.approx(5.0)
    assert hyperbolic[1, 3] == pytest.approx(5.0)
    assert curved[0, 1] == pytest.approx(5.0)
    # Off the ray, distances grow faster than Euclidean ones, and faster with curvature
    assert euclidean[1, 2] < hyperbolic[1, 2] < curved[1, 2]

def _table(codes, seed=3, dimension=4):
    return coordinate_table(codes, dimension=dimension, seed=seed)

def test_the_report_holds_only_req_6s_statistics():
    assert set(DiagnosticsReport.model_fields) == {
        'codes',
        'geometry',
        'curvature',
        'sector_separation',
        'within_sector_rank_correlation',
        'map_over_ancestors',
        'ndcg',
        'distance_pearson',
        'parent_retrieval',
    }

def test_the_report_covers_every_code_of_the_tree():
    report = diagnostics_report(_table(CODEBOOK), 'hyperbolic', codebook_codes=CODEBOOK)

    n = len(CODEBOOK)
    assert report.codes == n
    assert report.curvature == 1.0
    pairs = report.sector_separation.same_sector_pairs + report.sector_separation.cross_sector_pairs
    assert pairs == report.distance_pearson.pairs == n * (n - 1) // 2
    assert report.within_sector_rank_correlation.queries == n
    assert report.map_over_ancestors.queries == n - 4  # every code but the four sectors
    assert set(report.ndcg) == {'@5', '@10', '@20'}
    assert report.ndcg['@10'].queries == n
    assert set(report.map_over_ancestors.by_level) == {'3', '4', '5', '6'}
    # 238110 is its five-digit parent's only child
    assert report.parent_retrieval.unary_pairs_excluded == 1
    assert report.parent_retrieval.queries == n - 4 - 1
    assert diagnostics_report(_table(CODEBOOK), 'euclidean').curvature is None

def test_the_report_refuses_a_table_that_is_not_the_codebook():
    with pytest.raises(ValueError, match='covers the codebook'):
        diagnostics_report(_table(CODEBOOK[1:]), 'euclidean', codebook_codes=CODEBOOK)
    lorentz = pl.DataFrame({'code': ['11', '111'], 'x0': [1.0, 2.0], 'x1': [0.0, 3.0**0.5]})
    with pytest.raises(ValueError, match='Lorentz points'):
        diagnostics_report(lorentz, 'hyperbolic')
