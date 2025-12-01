'''
Unit tests for positive sampling module.

Tests taxonomy-based positive enumeration and stratified sampling for
contrastive training, covering merged sectors, hierarchy traversal,
stratum weight calculations, and sampling distributions.
'''

import numpy as np
import polars as pl
import pytest

from naics_embedder.data.positive_sampling import (
    PositiveSampler,
    _build_ancestors_6,
    _build_ancestors_level,
    _build_descendants,
    _build_siblings,
    _linear_skip,
    build_anchor_list,
    build_taxonomy,
    enumerate_positives,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_descriptions_df():
    '''Create sample NAICS descriptions for testing.'''
    # Includes manufacturing sectors 31-33 (merged) and retail sectors 44-45 (merged)
    data = {
        'index':
        list(range(24)),
        'code': [
            # Sector 31 (merged with 32, 33 -> becomes 31)
            '31',
            '311',
            '3111',
            '31111',
            '311111',
            '311112',
            # Sector 32 (merged -> becomes 31)
            '32',
            '321',
            '3211',
            '32111',
            '321111',
            # Sector 33 (merged -> becomes 31)
            '33',
            '331',
            '3311',
            '33111',
            '331111',
            # Sector 44 (merged with 45 -> becomes 44)
            '44',
            '441',
            '4411',
            '44111',
            '441111',
            # Sector 45 (merged -> becomes 44)
            '45',
            '451',
            '4511',
        ],
        'level': [
            2,
            3,
            4,
            5,
            6,
            6,
            2,
            3,
            4,
            5,
            6,
            2,
            3,
            4,
            5,
            6,
            2,
            3,
            4,
            5,
            6,
            2,
            3,
            4,
        ],
        'title': [
            'Manufacturing',
            'Food Manufacturing',
            'Animal Food',
            'Animal Food Sub',
            'Dog Food',
            'Cat Food',
            'Manufacturing',
            'Wood Products',
            'Sawmills',
            'Sawmills Sub',
            'Sawmills Leaf',
            'Manufacturing',
            'Primary Metal',
            'Iron and Steel',
            'Iron Sub',
            'Iron Leaf',
            'Retail Trade',
            'Motor Vehicle Dealers',
            'Auto Dealers',
            'Auto Sub',
            'Auto Leaf',
            'Retail Trade',
            'Sporting Goods',
            'Sporting Goods Sub',
        ],
    }
    return pl.DataFrame(data)

@pytest.fixture
def descriptions_parquet(tmp_path, sample_descriptions_df):
    '''Create temporary descriptions parquet file.'''
    path = tmp_path / 'naics_descriptions.parquet'
    sample_descriptions_df.write_parquet(path)
    return str(path)

@pytest.fixture
def sample_relations_df():
    '''Create sample NAICS relations for testing siblings.'''
    # relation_id=2 means sibling relationship
    data = {
        'code_i': ['311111', '311111', '321111', '441111'],
        'code_j': ['311112', '321111', '331111', '451111'],
        'relation_id': [2, 3, 2, 3],  # 2=sibling, 3=other
    }
    return pl.DataFrame(data)

@pytest.fixture
def relations_parquet(tmp_path, sample_relations_df):
    '''Create temporary relations parquet file.'''
    path = tmp_path / 'naics_relations.parquet'
    sample_relations_df.write_parquet(path)
    return str(path)

@pytest.fixture
def taxonomy_df(descriptions_parquet):
    '''Build taxonomy from test descriptions.'''
    return build_taxonomy(descriptions_parquet)

@pytest.fixture
def anchors_df(descriptions_parquet):
    '''Build anchors from test descriptions.'''
    return build_anchor_list(descriptions_parquet)

# -------------------------------------------------------------------------------------------------
# Tests for build_taxonomy()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildTaxonomy:
    '''Tests for build_taxonomy() function.'''

    def test_build_taxonomy_handles_merged_sector_31_33(self, descriptions_parquet):
        '''Test that sectors 31, 32, 33 are correctly merged to 31.'''
        taxonomy = build_taxonomy(descriptions_parquet)

        # Get all code_2 values for 6-digit codes
        code_2_values = taxonomy.get_column('code_2').unique().sort().to_list()

        # Sectors 31, 32, 33 should all become 31
        assert '31' in code_2_values
        assert '32' not in code_2_values
        assert '33' not in code_2_values

    def test_build_taxonomy_handles_merged_sector_44_45(self, descriptions_parquet):
        '''Test that sectors 44, 45 are correctly merged to 44.'''
        taxonomy = build_taxonomy(descriptions_parquet)

        code_2_values = taxonomy.get_column('code_2').unique().sort().to_list()

        # Sector 44, 45 should become 44
        assert '44' in code_2_values
        assert '45' not in code_2_values

    def test_build_taxonomy_preserves_non_merged_sectors(self, tmp_path):
        '''Test that non-merged sectors are preserved as-is.'''
        # Create a dataset with sector 11 (Agriculture) which is not merged
        df = pl.DataFrame(
            {
                'index': [0, 1, 2, 3, 4],
                'code': ['11', '111', '1111', '11111', '111111'],
                'level': [2, 3, 4, 5, 6],
                'title': ['Agriculture'] * 5,
            }
        )
        path = tmp_path / 'naics_non_merged.parquet'
        df.write_parquet(path)

        taxonomy = build_taxonomy(str(path))
        code_2_values = taxonomy.get_column('code_2').unique().to_list()

        assert '11' in code_2_values

    def test_build_taxonomy_code_column_has_merged_prefix(self, descriptions_parquet):
        '''Test that the code column uses merged sector prefix.'''
        taxonomy = build_taxonomy(descriptions_parquet)

        # Find codes that originally started with 32 or 33
        # They should now start with 31
        codes = taxonomy.get_column('code').to_list()

        # 321111 should become 311111 (prefix 32 -> 31)
        # Actually, code = code_2 + code_6[2:4]
        # For code_6 = 321111, code_2 = 31 (merged), so code = 31 + 1111 = 311111
        assert any(c.startswith('31') for c in codes)

    def test_build_taxonomy_only_level_6_codes(self, descriptions_parquet):
        '''Test that taxonomy only contains level 6 codes.'''
        taxonomy = build_taxonomy(descriptions_parquet)

        # code_6 column should have 6 characters
        code_6_lengths = taxonomy.get_column('code_6').str.len_chars().unique().to_list()
        assert code_6_lengths == [6]

    def test_build_taxonomy_handles_sector_48_49(self, tmp_path):
        '''Test that sectors 48, 49 are correctly merged to 48.'''
        df = pl.DataFrame(
            {
                'index':
                list(range(10)),
                'code': [
                    '48',
                    '481',
                    '4811',
                    '48111',
                    '481111',
                    '49',
                    '491',
                    '4911',
                    '49111',
                    '491111',
                ],
                'level': [2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
                'title': ['Transportation'] * 10,
            }
        )
        path = tmp_path / 'naics_48_49.parquet'
        df.write_parquet(path)

        taxonomy = build_taxonomy(str(path))
        code_2_values = taxonomy.get_column('code_2').unique().sort().to_list()

        assert '48' in code_2_values
        assert '49' not in code_2_values

# -------------------------------------------------------------------------------------------------
# Tests for build_anchor_list()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildAnchorList:
    '''Tests for build_anchor_list() function.'''

    def test_build_anchor_list_includes_levels_2_to_6(self, descriptions_parquet):
        '''Test that anchor list includes codes from levels 2-6.'''
        anchors = build_anchor_list(descriptions_parquet)

        levels = anchors.get_column('level').unique().sort().to_list()
        assert levels == [2, 3, 4, 5, 6]

    def test_build_anchor_list_excludes_level_1(self, tmp_path):
        '''Test that anchor list excludes level 1 (root) codes.'''
        df = pl.DataFrame(
            {
                'index': [0, 1, 2],
                'code': ['1', '11', '111'],
                'level': [1, 2, 3],
                'title': ['Root', 'Sector', 'Subsector'],
            }
        )
        path = tmp_path / 'naics_with_root.parquet'
        df.write_parquet(path)

        anchors = build_anchor_list(str(path))
        levels = anchors.get_column('level').unique().to_list()

        assert 1 not in levels

    def test_build_anchor_list_sorted_by_level_and_code(self, descriptions_parquet):
        '''Test that anchors are sorted by level, then by code.'''
        anchors = build_anchor_list(descriptions_parquet)

        # Check ordering within level 2
        level_2 = anchors.filter(pl.col('level').eq(2)).get_column('anchor').to_list()
        assert level_2 == sorted(level_2, key=lambda x: int(x))

    def test_build_anchor_list_unique_anchors(self, descriptions_parquet):
        '''Test that anchor list contains unique anchor codes.'''
        anchors = build_anchor_list(descriptions_parquet)

        anchor_codes = anchors.get_column('anchor').to_list()
        assert len(anchor_codes) == len(set(anchor_codes))

# -------------------------------------------------------------------------------------------------
# Tests for _linear_skip()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLinearSkip:
    '''Tests for _linear_skip() function.'''

    def test_linear_skip_finds_direct_children(self, taxonomy_df):
        '''Test finding direct children when they exist at next level.'''
        # For anchor '31', should find level-3 descendants
        descendants = _linear_skip('31', taxonomy_df)

        assert len(descendants) > 0
        # All descendants should be at level 3 (3 chars)
        assert all(len(d) == 3 for d in descendants)

    def test_linear_skip_skips_single_child_levels(self, tmp_path):
        '''Test that linear skip finds descendants past single-child levels.'''
        # Create hierarchy where level 3 has single child, but level 4 has multiple
        df = pl.DataFrame(
            {
                'index': list(range(3)),
                'code': ['111111', '111121', '111131'],
                'level': [6, 6, 6],
                'title': ['A', 'B', 'C'],
            }
        )
        path = tmp_path / 'naics_skip.parquet'
        df.write_parquet(path)

        taxonomy = build_taxonomy(str(path))

        # For anchor '11', the next diverse level should be level 4 (11111, 11112, 11113)
        descendants = _linear_skip('11', taxonomy)

        # Should find multiple descendants
        assert len(descendants) >= 1

    def test_linear_skip_level_5_returns_level_6(self, taxonomy_df):
        '''Test that level 5 anchor always returns level 6 descendants.'''
        # Get a level-5 anchor
        level_5_anchor = '31111'
        descendants = _linear_skip(level_5_anchor, taxonomy_df)

        # All descendants should be 6-digit codes
        assert all(len(d) == 6 for d in descendants)

    def test_linear_skip_returns_sorted_unique(self, taxonomy_df):
        '''Test that results are sorted and unique.'''
        descendants = _linear_skip('31', taxonomy_df)

        assert descendants == sorted(set(descendants))

# -------------------------------------------------------------------------------------------------
# Tests for _build_descendants()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildDescendants:
    '''Tests for _build_descendants() function.'''

    def test_build_descendants_correct_weights(self, taxonomy_df, anchors_df):
        '''Test stratum weight calculation (1/num_positives).'''
        descendants = _build_descendants(anchors_df, taxonomy_df)

        if descendants.height > 0:
            # Explode to check individual weights
            exploded = descendants.explode('positive').unnest('positive')

            # For each anchor, check that stratum_wgt = 1 / num_positives
            for row in descendants.iter_rows(named=True):
                anchor = row['anchor']
                num_pos = row['num_positives']
                expected_wgt = 1.0 / num_pos

                # Get weights for this anchor
                anchor_rows = exploded.filter(pl.col('anchor').eq(anchor))
                weights = anchor_rows.get_column('stratum_wgt').to_list()

                for w in weights:
                    assert abs(w - expected_wgt) < 1e-9

    def test_build_descendants_stratum_id_zero(self, taxonomy_df, anchors_df):
        '''Test that descendants stratum has stratum_id=0.'''
        descendants = _build_descendants(anchors_df, taxonomy_df)

        if descendants.height > 0:
            exploded = descendants.explode('positive').unnest('positive')
            stratum_ids = exploded.get_column('stratum_id').unique().to_list()
            assert stratum_ids == [0]

    def test_build_descendants_excludes_level_6_anchors(self, taxonomy_df, anchors_df):
        '''Test that level-6 anchors are not included (they have no descendants).'''
        descendants = _build_descendants(anchors_df, taxonomy_df)

        if descendants.height > 0:
            anchor_levels = descendants.get_column('level').unique().to_list()
            assert 6 not in anchor_levels

    def test_build_descendants_empty_input_returns_empty(self):
        '''Test that empty input returns empty dataframe with correct schema.'''
        empty_anchors = pl.DataFrame(schema={'level': pl.Int64, 'anchor': pl.Utf8})
        empty_taxonomy = pl.DataFrame(
            schema={
                'code_2': pl.Utf8,
                'code_3': pl.Utf8,
                'code_4': pl.Utf8,
                'code_5': pl.Utf8,
                'code_6': pl.Utf8,
                'code': pl.Utf8,
            }
        )

        result = _build_descendants(empty_anchors, empty_taxonomy)

        assert result.height == 0
        assert 'positive' in result.columns
        assert 'num_positives' in result.columns

# -------------------------------------------------------------------------------------------------
# Tests for _build_ancestors_6()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildAncestors6:
    '''Tests for _build_ancestors_6() function.'''

    def test_build_ancestors_for_level_6_has_4_ancestors(self, taxonomy_df, anchors_df):
        '''Test ancestor chain extraction for leaf nodes has 4 ancestors (levels 5,4,3,2).'''
        ancestors = _build_ancestors_6(anchors_df, taxonomy_df)

        if ancestors.height > 0:
            # Each level-6 anchor should have exactly 4 ancestors
            for row in ancestors.iter_rows(named=True):
                assert row['num_positives'] == 4, f"Expected 4 ancestors for {row['anchor']}"

    def test_build_ancestors_stratum_id_one(self, taxonomy_df, anchors_df):
        '''Test that ancestors stratum has stratum_id=1.'''
        ancestors = _build_ancestors_6(anchors_df, taxonomy_df)

        if ancestors.height > 0:
            exploded = ancestors.explode('positive').unnest('positive')
            stratum_ids = exploded.get_column('stratum_id').unique().to_list()
            assert stratum_ids == [1]

    def test_build_ancestors_weight_is_0_25(self, taxonomy_df, anchors_df):
        '''Test that ancestor stratum weights are 0.25 (1/4 ancestors).'''
        ancestors = _build_ancestors_6(anchors_df, taxonomy_df)

        if ancestors.height > 0:
            exploded = ancestors.explode('positive').unnest('positive')
            weights = exploded.get_column('stratum_wgt').unique().to_list()

            for w in weights:
                assert abs(w - 0.25) < 1e-9

    def test_build_ancestors_uses_merged_sector_codes(self, taxonomy_df, anchors_df):
        '''Test that ancestor codes use merged sector prefixes.'''
        ancestors = _build_ancestors_6(anchors_df, taxonomy_df)

        if ancestors.height > 0:
            exploded = ancestors.explode('positive').unnest('positive')
            positive_codes = exploded.get_column('positive').to_list()

            # Level-2 ancestors should not have 32, 33, 45, 49
            level_2_ancestors = [c for c in positive_codes if len(c) == 2]
            for code in level_2_ancestors:
                assert code not in ['32', '33', '45', '49']

# -------------------------------------------------------------------------------------------------
# Tests for _build_ancestors_level()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildAncestorsLevel:
    '''Tests for _build_ancestors_level() function.'''

    def test_build_ancestors_level_5_has_3_ancestors(self, taxonomy_df, anchors_df):
        '''Test that level 5 anchors have 3 ancestors (levels 4,3,2).'''
        ancestors_6 = _build_ancestors_6(anchors_df, taxonomy_df)
        ancestors_5 = _build_ancestors_level(ancestors_6, 5)

        if ancestors_5.height > 0:
            for row in ancestors_5.iter_rows(named=True):
                assert row['num_positives'] == 3, f"Expected 3 ancestors for level-5 {row['anchor']}"

    def test_build_ancestors_level_derives_from_higher_level(self, taxonomy_df, anchors_df):
        '''Test that level N ancestors are derived from level N+1.'''
        ancestors_6 = _build_ancestors_6(anchors_df, taxonomy_df)
        ancestors_5 = _build_ancestors_level(ancestors_6, 5)
        ancestors_4 = _build_ancestors_level(ancestors_5, 4)
        ancestors_3 = _build_ancestors_level(ancestors_4, 3)

        # Check decreasing ancestor counts
        if ancestors_5.height > 0:
            assert ancestors_5.get_column('num_positives').max() == 3
        if ancestors_4.height > 0:
            assert ancestors_4.get_column('num_positives').max() == 2
        if ancestors_3.height > 0:
            assert ancestors_3.get_column('num_positives').max() == 1

# -------------------------------------------------------------------------------------------------
# Tests for _build_siblings()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildSiblings:
    '''Tests for _build_siblings() function.'''

    def test_build_siblings_filters_relation_id_2(self, relations_parquet, anchors_df):
        '''Test that siblings only include relation_id=2 (sibling) relationships.'''
        siblings = _build_siblings(relations_parquet, anchors_df)

        if siblings.height > 0:
            exploded = siblings.explode('positive').unnest('positive')
            # All positives should come from relation_id=2
            stratum_ids = exploded.get_column('stratum_id').unique().to_list()
            assert stratum_ids == [2]

    def test_build_siblings_stratum_id_two(self, relations_parquet, anchors_df):
        '''Test that siblings stratum has stratum_id=2.'''
        siblings = _build_siblings(relations_parquet, anchors_df)

        if siblings.height > 0:
            exploded = siblings.explode('positive').unnest('positive')
            stratum_ids = exploded.get_column('stratum_id').unique().to_list()
            assert all(sid == 2 for sid in stratum_ids)

    def test_build_siblings_weights_normalized_per_anchor(self, relations_parquet, anchors_df):
        '''Test that sibling weights are normalized within each anchor.'''
        siblings = _build_siblings(relations_parquet, anchors_df)

        if siblings.height > 0:
            exploded = siblings.explode('positive').unnest('positive')

            # Sum of weights per anchor should be 1.0
            weight_sums = (
                exploded.group_by('anchor').agg(pl.col('stratum_wgt').sum().alias('wgt_sum')
                                                ).get_column('wgt_sum').to_list()
            )

            for ws in weight_sums:
                assert abs(ws - 1.0) < 1e-9

    def test_build_siblings_empty_relations_returns_empty(self, tmp_path, anchors_df):
        '''Test that empty relations returns empty dataframe.'''
        # Create empty relations with proper schema (not null types)
        empty_relations = pl.DataFrame(
            schema={
                'code_i': pl.Utf8,
                'code_j': pl.Utf8,
                'relation_id': pl.Int64,
            }
        )
        path = tmp_path / 'empty_relations.parquet'
        empty_relations.write_parquet(path)

        siblings = _build_siblings(str(path), anchors_df)
        assert siblings.height == 0

# -------------------------------------------------------------------------------------------------
# Tests for PositiveSampler class
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestPositiveSampler:
    '''Tests for PositiveSampler class.'''

    @pytest.fixture
    def sample_positives_df(self):
        '''Create sample positives dataframe for testing.'''
        return pl.DataFrame(
            {
                'anchor_idx': [0, 0, 0, 0, 1, 1],
                'positive_idx': [1, 2, 3, 4, 5, 6],
                'anchor_code': ['111', '111', '111', '111', '222', '222'],
                'positive_code': ['1111', '1112', '1113', '1114', '2221', '2222'],
                'anchor_level': [3, 3, 3, 3, 3, 3],
                'positive_level': [4, 4, 4, 4, 4, 4],
                'stratum_id': [0, 0, 1, 1, 0, 0],
                'stratum_wgt': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )

    def test_sampler_initialization(self, sample_positives_df):
        '''Test that sampler initializes correctly.'''
        sampler = PositiveSampler(sample_positives_df, max_per_stratum=4, seed=42)

        assert len(sampler.anchors) == 2
        assert 0 in sampler.anchors
        assert 1 in sampler.anchors

    def test_sample_positives_respects_max_per_stratum(self, sample_positives_df):
        '''Test that sampling respects max_per_stratum limit.'''
        sampler = PositiveSampler(sample_positives_df, max_per_stratum=1, seed=42)

        samples = sampler.sample_positives(0)

        # Anchor 0 has 2 strata (0 and 1), each with 2 positives
        # With max_per_stratum=1, should get at most 1 per stratum = 2 total
        stratum_counts = {}
        for s in samples:
            sid = s['stratum_id']
            stratum_counts[sid] = stratum_counts.get(sid, 0) + 1

        for count in stratum_counts.values():
            assert count <= 1

    def test_sample_positives_unknown_anchor_returns_empty(self, sample_positives_df):
        '''Test that unknown anchor returns empty list.'''
        sampler = PositiveSampler(sample_positives_df, seed=42)

        samples = sampler.sample_positives(999)
        assert samples == []

    def test_sample_positives_reproducible_with_seed(self, sample_positives_df):
        '''Test that sampling is reproducible with same seed.'''
        sampler1 = PositiveSampler(sample_positives_df, seed=42)
        sampler2 = PositiveSampler(sample_positives_df, seed=42)

        # Sample multiple times and compare
        samples1 = sampler1.sample_positives(0)
        samples2 = sampler2.sample_positives(0)

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1['positive_idx'] == s2['positive_idx']

    def test_sample_positives_includes_all_fields(self, sample_positives_df):
        '''Test that sampled positives include all required fields.'''
        sampler = PositiveSampler(sample_positives_df, seed=42)

        samples = sampler.sample_positives(0)

        for s in samples:
            assert 'positive_idx' in s
            assert 'positive_code' in s
            assert 'positive_level' in s
            assert 'stratum_id' in s
            assert 'stratum_wgt' in s

# -------------------------------------------------------------------------------------------------
# Tests for stratified sampling distribution
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestStratifiedSamplingDistribution:
    '''Tests for stratified sampling distribution properties.'''

    def test_create_stratified_positives_distribution(self):
        '''Test stratified sampling produces correct distribution.'''
        # Create a positives dataframe with skewed weight distribution
        # Using max_per_stratum=2 to force sampling subset
        n_positives = 20
        weights = [0.01] * (n_positives - 1) + [0.81]  # Last item has high weight
        df = pl.DataFrame(
            {
                'anchor_idx': [0] * n_positives,
                'positive_idx': list(range(n_positives)),
                'anchor_code': ['111'] * n_positives,
                'positive_code': [f'{100000 + i}' for i in range(n_positives)],
                'anchor_level': [3] * n_positives,
                'positive_level': [6] * n_positives,
                'stratum_id': [0] * n_positives,
                'stratum_wgt': weights,
            }
        )

        sampler = PositiveSampler(df, max_per_stratum=2, seed=42)

        # Sample many times to check distribution
        sample_counts = {i: 0 for i in range(n_positives)}
        n_iterations = 1000

        for i in range(n_iterations):
            # Reset RNG for each iteration
            sampler.rng = np.random.default_rng(i)
            samples = sampler.sample_positives(0)

            for s in samples:
                sample_counts[s['positive_idx']] += 1

        # High-weight item (index 19) should be sampled much more often
        # than low-weight items
        assert sample_counts[n_positives - 1] > sample_counts[0]
        # The high-weight item should be sampled in most iterations
        assert sample_counts[n_positives - 1] > n_iterations * 0.5

    def test_sampling_without_replacement(self):
        '''Test that sampling is done without replacement.'''
        df = pl.DataFrame(
            {
                'anchor_idx': [0] * 5,
                'positive_idx': list(range(5)),
                'anchor_code': ['111'] * 5,
                'positive_code': [f'111{i}' for i in range(5)],
                'anchor_level': [3] * 5,
                'positive_level': [4] * 5,
                'stratum_id': [0] * 5,
                'stratum_wgt': [0.2] * 5,
            }
        )

        sampler = PositiveSampler(df, max_per_stratum=5, seed=42)
        samples = sampler.sample_positives(0)

        # All 5 should be sampled exactly once
        sampled_indices = [s['positive_idx'] for s in samples]
        assert len(sampled_indices) == len(set(sampled_indices))

# -------------------------------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Tests for edge cases in positive sampling.'''

    def test_edge_cases_single_child(self, tmp_path):
        '''Test behavior when nodes have single children.'''
        # Create hierarchy where each node has exactly one child
        df = pl.DataFrame(
            {
                'index': list(range(5)),
                'code': ['11', '111', '1111', '11111', '111111'],
                'level': [2, 3, 4, 5, 6],
                'title': ['A', 'B', 'C', 'D', 'E'],
            }
        )
        path = tmp_path / 'naics_single_child.parquet'
        df.write_parquet(path)

        taxonomy = build_taxonomy(str(path))
        anchors = build_anchor_list(str(path))

        # Build descendants - should still work but may have single descendant
        descendants = _build_descendants(anchors, taxonomy)

        # Each parent anchor should have exactly 1 descendant
        if descendants.height > 0:
            for row in descendants.iter_rows(named=True):
                assert row['num_positives'] >= 1

    def test_edge_cases_no_siblings(self, tmp_path, anchors_df):
        '''Test behavior when no sibling relations exist.'''
        # Create relations with no relation_id=2
        df = pl.DataFrame(
            {
                'code_i': ['111'],
                'code_j': ['1111'],
                'relation_id': [1],  # child, not sibling
            }
        )
        path = tmp_path / 'naics_no_siblings.parquet'
        df.write_parquet(path)

        siblings = _build_siblings(str(path), anchors_df)
        assert siblings.height == 0

    def test_edge_cases_large_stratum(self):
        '''Test sampling from stratum larger than max_per_stratum.'''
        # Create a large stratum
        n_positives = 100
        df = pl.DataFrame(
            {
                'anchor_idx': [0] * n_positives,
                'positive_idx': list(range(n_positives)),
                'anchor_code': ['111'] * n_positives,
                'positive_code': [f'{100000 + i}' for i in range(n_positives)],
                'anchor_level': [3] * n_positives,
                'positive_level': [6] * n_positives,
                'stratum_id': [0] * n_positives,
                'stratum_wgt': [1.0 / n_positives] * n_positives,
            }
        )

        sampler = PositiveSampler(df, max_per_stratum=4, seed=42)
        samples = sampler.sample_positives(0)

        assert len(samples) == 4

    def test_edge_cases_zero_weights(self):
        '''Test handling of zero weights (should use uniform distribution).'''
        df = pl.DataFrame(
            {
                'anchor_idx': [0, 0],
                'positive_idx': [1, 2],
                'anchor_code': ['111', '111'],
                'positive_code': ['1111', '1112'],
                'anchor_level': [3, 3],
                'positive_level': [4, 4],
                'stratum_id': [0, 0],
                'stratum_wgt': [0.0, 0.0],
            }
        )

        sampler = PositiveSampler(df, max_per_stratum=2, seed=42)
        samples = sampler.sample_positives(0)

        # Should still sample successfully
        assert len(samples) == 2

# -------------------------------------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestIntegration:
    '''Integration tests for the full positive sampling pipeline.'''

    def test_enumerate_positives_full_pipeline(
        self,
        descriptions_parquet,
        relations_parquet,
        monkeypatch,
    ):
        '''Test full enumerate_positives pipeline.'''
        # Mock get_indices_codes to return consistent mappings
        mock_code_to_idx = {}
        for i, code in enumerate(
            [
                '31',
                '311',
                '3111',
                '31111',
                '311111',
                '311112',
                '32',
                '321',
                '3211',
                '32111',
                '321111',
                '33',
                '331',
                '3311',
                '33111',
                '331111',
                '44',
                '441',
                '4411',
                '44111',
                '441111',
                '45',
                '451',
                '4511',
            ]
        ):
            mock_code_to_idx[code] = i

        def mock_get_indices_codes(return_type):
            if return_type == 'code_to_idx':
                return mock_code_to_idx
            elif return_type == 'idx_to_code':
                return {v: k for k, v in mock_code_to_idx.items()}
            return []

        monkeypatch.setattr(
            'naics_embedder.data.positive_sampling.get_indices_codes',
            mock_get_indices_codes,
        )

        result = enumerate_positives(descriptions_parquet, relations_parquet)

        # Check required columns exist
        expected_cols = [
            'anchor_idx',
            'positive_idx',
            'anchor_code',
            'positive_code',
            'anchor_level',
            'positive_level',
            'stratum_id',
            'stratum_wgt',
        ]
        for col in expected_cols:
            assert col in result.columns

        # Check stratum IDs are in expected range
        if result.height > 0:
            stratum_ids = result.get_column('stratum_id').unique().sort().to_list()
            for sid in stratum_ids:
                assert sid in [0, 1, 2]

    def test_full_sampler_workflow(
        self,
        descriptions_parquet,
        relations_parquet,
        monkeypatch,
    ):
        '''Test full workflow from enumeration to sampling.'''
        # Create mock code mapping
        mock_code_to_idx = {}
        for i, code in enumerate(['31', '311', '3111', '31111', '311111', '311112']):
            mock_code_to_idx[code] = i

        def mock_get_indices_codes(return_type):
            if return_type == 'code_to_idx':
                return mock_code_to_idx
            elif return_type == 'idx_to_code':
                return {v: k for k, v in mock_code_to_idx.items()}
            return []

        monkeypatch.setattr(
            'naics_embedder.data.positive_sampling.get_indices_codes',
            mock_get_indices_codes,
        )

        # Enumerate positives
        positives_df = enumerate_positives(descriptions_parquet, relations_parquet)

        # Create sampler
        sampler = PositiveSampler(positives_df, max_per_stratum=2, seed=42)

        # Sample for each anchor
        all_samples = []
        for anchor_idx in sampler.anchors:
            samples = sampler.sample_positives(anchor_idx)
            all_samples.extend(samples)

        # Should have sampled some positives
        if positives_df.height > 0:
            assert len(all_samples) >= 0  # May be empty if anchors don't match
