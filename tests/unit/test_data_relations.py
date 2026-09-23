import polars as pl
import pytest

from naics_embedder.data import compute_relations

@pytest.mark.unit
def test_get_relations_handles_child_sibling_and_cousin():
    depths = {'A': 0, 'B': 1, 'C': 2, 'D': 1, 'E': 2}
    ancestors = {
        'A': ['A'],
        'B': ['A', 'B'],
        'C': ['A', 'B', 'C'],
        'D': ['A', 'D'],
        'E': ['A', 'D', 'E'],
    }

    assert compute_relations._get_relations('A', 'B', depths, ancestors) == 'child'
    assert compute_relations._get_relations('B', 'C', depths, ancestors) == 'child'
    assert compute_relations._get_relations('B', 'D', depths, ancestors) == 'sibling'
    assert compute_relations._get_relations('C', 'E', depths, ancestors) == 'cousin'

@pytest.mark.unit
def test_get_relations_handles_extended_family_names():
    depths = {
        'A': 0,
        'B': 1,
        'C': 1,
        'D': 2,
        'E': 3,
        'F': 4,
        'G': 2,
        'H': 3,
        'I': 4,
    }
    ancestors = {
        'A': ['A'],
        'B': ['A', 'B'],
        'C': ['A', 'C'],
        'D': ['A', 'C', 'D'],
        'E': ['A', 'C', 'D', 'E'],
        'F': ['A', 'C', 'D', 'E', 'F'],
        'G': ['A', 'B', 'G'],
        'H': ['A', 'B', 'G', 'H'],
        'I': ['A', 'B', 'G', 'H', 'I'],
    }

    grand_niece = compute_relations._get_relations('B', 'F', depths, ancestors)
    assert grand_niece == 'grand-grand-nephew/niece'

    removed = compute_relations._get_relations('D', 'I', depths, ancestors)
    assert removed == 'cousin_2_times_removed'

RELATION_IDS = {
    'child': 1,
    'sibling': 2,
    'grandchild': 3,
    'great-grandchild': 4,
    'nephew/niece': 5,
    'great-great-grandchild': 6,
    'cousin': 7,
    'grand-nephew/niece': 8,
    'grand-grand-nephew/niece': 9,
    'cousin_1_times_removed': 10,
    'second_cousin': 11,
    'cousin_2_times_removed': 12,
    'second_cousin_1_times_removed': 13,
    'third_cousin': 14,
    'cross_sector': 99,
}

def _relation(frame: pl.DataFrame, code_i: str, code_j: str) -> tuple[int, str]:
    row = frame.filter(pl.col('code_i').eq(code_i) & pl.col('code_j').eq(code_j)).row(
        0, named=True
    )
    return row['structural_relation_id'], row['structural_relation_name']

@pytest.fixture
def structural_relations(hierarchy_descriptions_parquet):
    return compute_relations.compute_structural_relations(
        hierarchy_descriptions_parquet, RELATION_IDS
    )

@pytest.mark.unit
def test_structural_relations_are_structural_only(structural_relations):
    assert structural_relations.columns == [
        'idx_i',
        'idx_j',
        'code_i',
        'code_j',
        'structural_relation_id',
        'structural_relation_name',
    ]

@pytest.mark.unit
def test_excluded_pairs_keep_tree_relations(structural_relations):
    # '311111' excludes '321111' (merged 31-33 sector) and '441111' excludes '311211'.
    assert _relation(structural_relations, '311111', '321111') == (14, 'third_cousin')
    assert _relation(structural_relations, '311211', '441111') == (99, 'cross_sector')

@pytest.mark.unit
def test_no_structural_relation_is_an_exclusion_sentinel(structural_relations):
    assert not structural_relations.get_column('structural_relation_id').eq(0).any()
    assert not structural_relations.get_column('structural_relation_name').eq('excluded').any()
    assert set(structural_relations.get_column('structural_relation_name')) <= set(RELATION_IDS)

@pytest.mark.unit
def test_cross_sector_is_an_explicit_structural_value(structural_relations):
    assert _relation(structural_relations, '31', '44') == (99, 'cross_sector')
    assert _relation(structural_relations, '31', '321') == (1, 'child')
    assert _relation(structural_relations, '311', '321') == (2, 'sibling')
    assert _relation(structural_relations, '3112', '31111') == (5, 'nephew/niece')
    assert structural_relations.filter(
        pl.col('code_i').eq('321') & pl.col('code_j').eq('311')
    ).height == 0

@pytest.mark.unit
def test_unmapped_tree_relation_is_fatal(hierarchy_descriptions_parquet):
    incomplete = {name: rid for name, rid in RELATION_IDS.items() if name != 'nephew/niece'}

    with pytest.raises(ValueError, match='nephew/niece'):
        compute_relations.compute_structural_relations(hierarchy_descriptions_parquet, incomplete)
