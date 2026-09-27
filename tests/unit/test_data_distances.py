'''
Unit tests for the structural distances: D* (Req 7) for every canonical pair of codes.
'''

import polars as pl
import pytest

from naics_embedder.data.compute_distances import compute_structural_distances
from naics_embedder.utils.config import DistancesConfig
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix

# -------------------------------------------------------------------------------------------------
# Structural-only distances
# -------------------------------------------------------------------------------------------------

def _distance(frame: pl.DataFrame, code_i: str, code_j: str) -> float:
    return frame.filter(pl.col('code_i').eq(code_i)
                        & pl.col('code_j').eq(code_j)).item(0, 'structural_distance')

@pytest.mark.unit
class TestStructuralDistances:
    '''Exclusion processing never touches structural distance, and every distance is D*.'''

    @pytest.fixture
    def distances(self, hierarchy_descriptions_parquet):
        return compute_structural_distances(
            hierarchy_descriptions_parquet,
            DistancesConfig(input_parquet=hierarchy_descriptions_parquet),
        )

    def test_output_is_structural_only(self, distances):
        assert distances.columns == [
            'idx_i',
            'idx_j',
            'code_i',
            'code_j',
            'structural_distance',
        ]

    def test_excluded_pairs_keep_tree_distances(self, distances):
        # '311111' excludes '321111' (merged 31-33 sector) and '441111' excludes '311211'.
        assert _distance(distances, '311111', '321111') == 8.0
        assert _distance(distances, '311211', '441111') == 10.0

    def test_no_structural_distance_is_a_sentinel(self, distances):
        values = distances.get_column('structural_distance')
        assert values.min() > 0.0
        # Two six-digit codes in different sectors are the farthest pair: 6 + 6 - 2
        assert values.max() == 10.0
        assert values.round(0).equals(values)

    def test_values_follow_the_tree(self, distances):
        # No half-step for a lineal pair, and a virtual root above the sectors
        assert _distance(distances, '31', '321') == 1.0
        assert _distance(distances, '311', '321') == 2.0
        assert _distance(distances, '3112', '31111') == 3.0
        assert _distance(distances, '31', '44') == 2.0

    def test_every_value_comes_from_the_one_tree_distance_function(self, distances):
        # Stage 4's diagnostics read D* from the same function, so the two agree on every pair
        codes = sorted(set(distances.get_column('code_i')) | set(distances.get_column('code_j')))
        position = {code: row for row, code in enumerate(codes)}
        matrix = tree_distance_matrix(codes)
        expected = [
            float(matrix[position[code_i], position[code_j]])
            for code_i, code_j in distances.select('code_i', 'code_j').rows()
        ]
        assert distances.get_column('structural_distance').to_list() == expected

    def test_every_unordered_pair_appears_once_in_canonical_orientation(self, distances):
        n_codes = 17
        assert distances.height == n_codes * (n_codes - 1) // 2
        oriented = distances.with_columns(
            level_i=pl.col('code_i').str.len_chars(),
            level_j=pl.col('code_j').str.len_chars(),
        )
        non_canonical = oriented.filter(
            pl.col('level_i').gt(pl.col('level_j'))
            | (pl.col('level_i').eq(pl.col('level_j')) & pl.col('code_i').ge(pl.col('code_j')))
        )
        assert non_canonical.height == 0
        # Same-level pairs across a merged-sector prefix are no longer emitted twice.
        assert distances.filter(pl.col('code_i').eq('321') & pl.col('code_j').eq('311')).height == 0
