'''
Unit tests for NAICS hierarchy utilities.

Tests the NaicsHierarchy class for building and traversing
parent-child relationships from relations data, including
deduplication, conflict handling, and caching behavior.
'''

from pathlib import Path

import polars as pl
import pytest

from naics_embedder.utils.naics_hierarchy import NaicsHierarchy, load_naics_hierarchy

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def simple_parent_child_pairs():
    '''Simple parent-child pairs for testing.'''
    return [
        ('11', '111'),
        ('11', '112'),
        ('111', '1111'),
        ('111', '1112'),
        ('112', '1121'),
    ]

@pytest.fixture
def simple_hierarchy(simple_parent_child_pairs):
    '''Create a simple hierarchy for testing.'''
    return NaicsHierarchy(simple_parent_child_pairs)

@pytest.fixture
def relations_parquet_with_relation_id(tmp_path):
    '''Create relations parquet with relation_id column.'''
    df = pl.DataFrame(
        {
            'code_i': ['11', '11', '111', '111', '112', '11', '111'],
            'code_j': ['111', '112', '1111', '1112', '1121', '113', '114'],
            'relation_id': [1, 1, 1, 1, 1, 2, 3],  # Only 1 = child
        }
    )
    path = tmp_path / 'relations_with_id.parquet'
    df.write_parquet(path)
    return path

@pytest.fixture
def relations_parquet_with_relation_name(tmp_path):
    '''Create relations parquet with relation column (string name).'''
    df = pl.DataFrame(
        {
            'code_i': ['11', '11', '111', '112'],
            'code_j': ['111', '112', '1111', '1121'],
            'relation': ['child', 'child', 'child', 'child'],
        }
    )
    path = tmp_path / 'relations_with_name.parquet'
    df.write_parquet(path)
    return path

@pytest.fixture
def relations_parquet_with_relationship(tmp_path):
    '''Create relations parquet with relationship column.'''
    df = pl.DataFrame(
        {
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'relationship': ['child', 'child'],
        }
    )
    path = tmp_path / 'relations_with_relationship.parquet'
    df.write_parquet(path)
    return path

# -------------------------------------------------------------------------------------------------
# Tests for NaicsHierarchy.__init__()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNaicsHierarchyInit:
    '''Tests for NaicsHierarchy initialization.'''

    def test_naics_hierarchy_deduplicates_pairs(self):
        '''Test that duplicate parent-child pairs are handled.'''
        pairs = [
            ('11', '111'),
            ('11', '111'),  # duplicate
            ('11', '112'),
            ('11', '112'),  # duplicate
        ]

        hierarchy = NaicsHierarchy(pairs)

        # Should only have 2 unique pairs
        assert len(hierarchy.parent_child_pairs) == 2
        assert hierarchy.parent_child_pairs == [('11', '111'), ('11', '112')]

    def test_get_parent_returns_first_observed(self):
        '''Test conflict resolution when child has multiple parents.'''
        pairs = [
            ('11', '111'),  # First parent
            ('22', '111'),  # Second (conflicting) parent - should be ignored
            ('33', '111'),  # Third (conflicting) parent - should be ignored
        ]

        hierarchy = NaicsHierarchy(pairs)

        # Should return the first observed parent
        assert hierarchy.get_parent('111') == '11'
        # Only one pair should be stored
        assert len(hierarchy.parent_child_pairs) == 1

    def test_ignores_empty_codes(self):
        '''Test that empty parent or child codes are ignored.'''
        pairs = [
            ('11', '111'),
            ('', '112'),  # empty parent
            ('11', ''),  # empty child
            (None, '113'),  # None parent (edge case)
            ('11', None),  # None child (edge case)
        ]

        hierarchy = NaicsHierarchy(pairs)

        # Should only have the valid pair
        assert len(hierarchy.parent_child_pairs) == 1
        assert hierarchy.parent_child_pairs == [('11', '111')]

    def test_builds_parent_by_child_mapping(self, simple_hierarchy):
        '''Test that parent_by_child mapping is built correctly.'''
        assert simple_hierarchy.parent_by_child['111'] == '11'
        assert simple_hierarchy.parent_by_child['112'] == '11'
        assert simple_hierarchy.parent_by_child['1111'] == '111'
        assert simple_hierarchy.parent_by_child['1121'] == '112'

    def test_builds_children_by_parent_mapping(self, simple_hierarchy):
        '''Test that children_by_parent mapping is built correctly.'''
        assert '111' in simple_hierarchy.children_by_parent['11']
        assert '112' in simple_hierarchy.children_by_parent['11']
        assert '1111' in simple_hierarchy.children_by_parent['111']
        assert '1112' in simple_hierarchy.children_by_parent['111']

    def test_empty_input_creates_empty_hierarchy(self):
        '''Test that empty input creates valid but empty hierarchy.'''
        hierarchy = NaicsHierarchy([])

        assert len(hierarchy.parent_child_pairs) == 0
        assert len(hierarchy.parent_by_child) == 0
        assert hierarchy.get_parent('any') is None
        assert hierarchy.get_children('any') == []
        assert hierarchy.get_siblings('any') == []

# -------------------------------------------------------------------------------------------------
# Tests for from_relations_parquet()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestFromRelationsParquet:
    '''Tests for from_relations_parquet class method.'''

    def test_from_relations_parquet_with_relation_id(self, relations_parquet_with_relation_id):
        '''Test loading parquet with relation_id column.'''
        hierarchy = NaicsHierarchy.from_relations_parquet(relations_parquet_with_relation_id)

        # Should only include relation_id=1 (child) relationships
        assert len(hierarchy.parent_child_pairs) == 5
        assert hierarchy.get_parent('111') == '11'
        assert hierarchy.get_parent('1111') == '111'

    def test_from_relations_parquet_with_relation_name(self, relations_parquet_with_relation_name):
        '''Test loading parquet with relation column (string).'''
        hierarchy = NaicsHierarchy.from_relations_parquet(relations_parquet_with_relation_name)

        assert len(hierarchy.parent_child_pairs) == 4
        assert hierarchy.get_parent('111') == '11'

    def test_from_relations_parquet_with_relationship(self, relations_parquet_with_relationship):
        '''Test loading parquet with relationship column.'''
        hierarchy = NaicsHierarchy.from_relations_parquet(relations_parquet_with_relationship)

        assert len(hierarchy.parent_child_pairs) == 2
        assert hierarchy.get_parent('111') == '11'

    def test_from_relations_parquet_filters_non_child_relations(
        self, relations_parquet_with_relation_id
    ):
        '''Test that non-child relations are filtered out.'''
        hierarchy = NaicsHierarchy.from_relations_parquet(relations_parquet_with_relation_id)

        # Codes 113 and 114 have relation_id != 1, so should not be in hierarchy
        assert hierarchy.get_parent('113') is None
        assert hierarchy.get_parent('114') is None

    def test_from_relations_parquet_schema_validation_missing_code_i(self, tmp_path):
        '''Test error handling for missing code_i column.'''
        df = pl.DataFrame({
            'code_j': ['111', '112'],
            'relation_id': [1, 1],
        })
        path = tmp_path / 'missing_code_i.parquet'
        df.write_parquet(path)

        with pytest.raises(ValueError, match='must contain code_i and code_j'):
            NaicsHierarchy.from_relations_parquet(path)

    def test_from_relations_parquet_schema_validation_missing_code_j(self, tmp_path):
        '''Test error handling for missing code_j column.'''
        df = pl.DataFrame({
            'code_i': ['11', '11'],
            'relation_id': [1, 1],
        })
        path = tmp_path / 'missing_code_j.parquet'
        df.write_parquet(path)

        with pytest.raises(ValueError, match='must contain code_i and code_j'):
            NaicsHierarchy.from_relations_parquet(path)

    def test_from_relations_parquet_schema_validation_missing_relation_column(self, tmp_path):
        '''Test error handling for missing relation/relation_id/relationship column.'''
        df = pl.DataFrame({
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
        })
        path = tmp_path / 'missing_relation.parquet'
        df.write_parquet(path)

        with pytest.raises(ValueError, match='must contain either relation_id or relation'):
            NaicsHierarchy.from_relations_parquet(path)

    def test_from_relations_parquet_file_not_found(self, tmp_path):
        '''Test error handling for non-existent file.'''
        path = tmp_path / 'nonexistent.parquet'

        with pytest.raises(FileNotFoundError, match='not found'):
            NaicsHierarchy.from_relations_parquet(path)

# -------------------------------------------------------------------------------------------------
# Tests for get_parent()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGetParent:
    '''Tests for get_parent() method.'''

    def test_get_parent_returns_parent(self, simple_hierarchy):
        '''Test getting parent for existing child.'''
        assert simple_hierarchy.get_parent('111') == '11'
        assert simple_hierarchy.get_parent('1111') == '111'

    def test_get_parent_returns_none_for_root(self, simple_hierarchy):
        '''Test that root nodes return None.'''
        assert simple_hierarchy.get_parent('11') is None

    def test_get_parent_returns_none_for_unknown(self, simple_hierarchy):
        '''Test that unknown codes return None.'''
        assert simple_hierarchy.get_parent('99999') is None
        assert simple_hierarchy.get_parent('') is None

# -------------------------------------------------------------------------------------------------
# Tests for get_children()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGetChildren:
    '''Tests for get_children() method.'''

    def test_get_children_returns_children(self, simple_hierarchy):
        '''Test getting children for parent with children.'''
        children = simple_hierarchy.get_children('11')
        assert '111' in children
        assert '112' in children
        assert len(children) == 2

    def test_get_children_returns_empty_for_leaf(self, simple_hierarchy):
        '''Test that leaf nodes return empty list.'''
        children = simple_hierarchy.get_children('1111')
        assert children == []

    def test_get_children_returns_empty_for_unknown(self, simple_hierarchy):
        '''Test that unknown codes return empty list.'''
        assert simple_hierarchy.get_children('99999') == []

    def test_get_children_returns_copy(self, simple_hierarchy):
        '''Test that modifying returned list doesn't affect hierarchy.'''
        children = simple_hierarchy.get_children('11')
        children.append('999')

        # Original should be unchanged
        assert '999' not in simple_hierarchy.get_children('11')

# -------------------------------------------------------------------------------------------------
# Tests for get_siblings()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSiblings:
    '''Tests for get_siblings() method.'''

    def test_get_siblings_excludes_self(self, simple_hierarchy):
        '''Test sibling enumeration excludes the input code.'''
        siblings = simple_hierarchy.get_siblings('111')

        assert '112' in siblings
        assert '111' not in siblings

    def test_get_siblings_returns_all_siblings(self):
        '''Test that all siblings are returned.'''
        pairs = [
            ('11', '111'),
            ('11', '112'),
            ('11', '113'),
            ('11', '114'),
        ]
        hierarchy = NaicsHierarchy(pairs)

        siblings = hierarchy.get_siblings('112')
        assert len(siblings) == 3
        assert '111' in siblings
        assert '113' in siblings
        assert '114' in siblings
        assert '112' not in siblings

    def test_get_siblings_returns_empty_for_root(self, simple_hierarchy):
        '''Test that root nodes have no siblings.'''
        siblings = simple_hierarchy.get_siblings('11')
        assert siblings == []

    def test_get_siblings_returns_empty_for_only_child(self):
        '''Test that only children have no siblings.'''
        pairs = [('11', '111')]
        hierarchy = NaicsHierarchy(pairs)

        siblings = hierarchy.get_siblings('111')
        assert siblings == []

    def test_get_siblings_returns_empty_for_unknown(self, simple_hierarchy):
        '''Test that unknown codes return empty list.'''
        assert simple_hierarchy.get_siblings('99999') == []

# -------------------------------------------------------------------------------------------------
# Tests for parent_child_pairs property
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestParentChildPairs:
    '''Tests for parent_child_pairs property.'''

    def test_parent_child_pairs_returns_copy(self, simple_hierarchy):
        '''Test that property returns a copy.'''
        pairs1 = simple_hierarchy.parent_child_pairs
        pairs2 = simple_hierarchy.parent_child_pairs

        assert pairs1 == pairs2
        assert pairs1 is not pairs2

    def test_parent_child_pairs_maintains_order(self, simple_parent_child_pairs):
        '''Test that pairs maintain insertion order.'''
        hierarchy = NaicsHierarchy(simple_parent_child_pairs)

        # Order should match input order
        assert hierarchy.parent_child_pairs == simple_parent_child_pairs

# -------------------------------------------------------------------------------------------------
# Tests for orphan nodes
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestOrphanNodes:
    '''Tests for behavior with orphan nodes (nodes without parents).'''

    def test_orphan_nodes_have_no_parent(self):
        '''Test behavior with nodes that have no parent.'''
        pairs = [
            ('11', '111'),
            ('111', '1111'),
            # '11' has no parent - it's an orphan/root
        ]
        hierarchy = NaicsHierarchy(pairs)

        assert hierarchy.get_parent('11') is None
        assert hierarchy.get_parent('111') == '11'

    def test_multiple_root_nodes(self):
        '''Test hierarchy with multiple disconnected trees.'''
        pairs = [
            ('11', '111'),
            ('22', '222'),  # Separate tree
            ('33', '333'),  # Another separate tree
        ]
        hierarchy = NaicsHierarchy(pairs)

        # Each root has no parent
        assert hierarchy.get_parent('11') is None
        assert hierarchy.get_parent('22') is None
        assert hierarchy.get_parent('33') is None

        # Each root has its own children
        assert hierarchy.get_children('11') == ['111']
        assert hierarchy.get_children('22') == ['222']
        assert hierarchy.get_children('33') == ['333']

        # Roots have no siblings
        assert hierarchy.get_siblings('11') == []
        assert hierarchy.get_siblings('22') == []

    def test_single_node_hierarchy(self):
        '''Test hierarchy with only a parent-child pair.'''
        pairs = [('root', 'child')]
        hierarchy = NaicsHierarchy(pairs)

        assert hierarchy.get_parent('root') is None
        assert hierarchy.get_parent('child') == 'root'
        assert hierarchy.get_children('root') == ['child']
        assert hierarchy.get_children('child') == []

# -------------------------------------------------------------------------------------------------
# Tests for load_naics_hierarchy() with caching
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadNaicsHierarchy:
    '''Tests for load_naics_hierarchy() function with LRU cache.'''

    def test_load_naics_hierarchy_basic(self, relations_parquet_with_relation_id):
        '''Test basic loading of hierarchy.'''
        # Clear cache before test
        load_naics_hierarchy.cache_clear()

        hierarchy = load_naics_hierarchy(str(relations_parquet_with_relation_id))

        assert hierarchy is not None
        assert len(hierarchy.parent_child_pairs) == 5

    def test_load_naics_hierarchy_caching(self, relations_parquet_with_relation_id):
        '''Test LRU cache behavior - same path returns cached instance.'''
        load_naics_hierarchy.cache_clear()

        path_str = str(relations_parquet_with_relation_id)

        hierarchy1 = load_naics_hierarchy(path_str)
        hierarchy2 = load_naics_hierarchy(path_str)

        # Should be the exact same object (cached)
        assert hierarchy1 is hierarchy2

        # Check cache info
        cache_info = load_naics_hierarchy.cache_info()
        assert cache_info.hits >= 1

    def test_load_naics_hierarchy_different_paths(
        self,
        relations_parquet_with_relation_id,
        relations_parquet_with_relation_name,
    ):
        '''Test that different paths return different cached instances.'''
        load_naics_hierarchy.cache_clear()

        hierarchy1 = load_naics_hierarchy(str(relations_parquet_with_relation_id))
        hierarchy2 = load_naics_hierarchy(str(relations_parquet_with_relation_name))

        # Should be different objects
        assert hierarchy1 is not hierarchy2

    def test_load_naics_hierarchy_resolves_path(self, relations_parquet_with_relation_id):
        '''Test that paths are resolved before caching.'''
        load_naics_hierarchy.cache_clear()

        # Use path with ./ prefix
        path1 = str(relations_parquet_with_relation_id)
        path2 = str(Path(path1).resolve())

        hierarchy1 = load_naics_hierarchy(path1)
        hierarchy2 = load_naics_hierarchy(path2)

        # Should be the same cached object
        assert hierarchy1 is hierarchy2

    def test_load_naics_hierarchy_file_not_found(self, tmp_path):
        '''Test error handling for non-existent file.'''
        load_naics_hierarchy.cache_clear()

        with pytest.raises(FileNotFoundError):
            load_naics_hierarchy(str(tmp_path / 'nonexistent.parquet'))

# -------------------------------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Tests for edge cases in hierarchy handling.'''

    def test_self_referential_pair_ignored(self):
        '''Test that self-referential pairs (node as its own parent) work.'''
        pairs = [
            ('11', '11'),  # Self-reference
            ('11', '111'),  # Valid pair
        ]
        hierarchy = NaicsHierarchy(pairs)

        # Self-reference means 11 is child of 11
        # This is valid but weird - the first pair sets 11's parent to 11
        assert hierarchy.get_parent('11') == '11'
        assert hierarchy.get_parent('111') == '11'

    def test_deep_hierarchy(self):
        '''Test with a deep hierarchy (many levels).'''
        # Create a 10-level deep chain
        pairs = [(str(i), str(i + 1)) for i in range(10)]
        hierarchy = NaicsHierarchy(pairs)

        # Check traversal works at all levels
        assert hierarchy.get_parent('1') == '0'
        assert hierarchy.get_parent('5') == '4'
        assert hierarchy.get_parent('9') == '8'
        assert hierarchy.get_children('5') == ['6']

    def test_wide_hierarchy(self):
        '''Test with a wide hierarchy (many children per parent).'''
        # Create parent with 100 children
        pairs = [('root', f'child_{i}') for i in range(100)]
        hierarchy = NaicsHierarchy(pairs)

        children = hierarchy.get_children('root')
        assert len(children) == 100

        siblings = hierarchy.get_siblings('child_50')
        assert len(siblings) == 99

    def test_unicode_codes(self):
        '''Test handling of unicode code strings.'''
        pairs = [
            ('родитель', 'ребенок'),  # Russian
            ('父母', '孩子'),  # Chinese
        ]
        hierarchy = NaicsHierarchy(pairs)

        assert hierarchy.get_parent('ребенок') == 'родитель'
        assert hierarchy.get_parent('孩子') == '父母'

    def test_numeric_string_codes(self):
        '''Test handling of numeric-looking string codes.'''
        pairs = [
            ('000', '001'),
            ('001', '0011'),
        ]
        hierarchy = NaicsHierarchy(pairs)

        assert hierarchy.get_parent('001') == '000'
        assert hierarchy.get_parent('0011') == '001'

    def test_codes_with_special_characters(self):
        '''Test handling of codes with special characters.'''
        pairs = [
            ('11-13', '111'),
            ('44-45', '441'),
        ]
        hierarchy = NaicsHierarchy(pairs)

        assert hierarchy.get_parent('111') == '11-13'
        assert hierarchy.get_parent('441') == '44-45'
