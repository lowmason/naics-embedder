import polars as pl
import pytest

from naics_embedder.data.create_triplets import (
    _structural_margins,
    _validate_training_pairs,
    build_training_pairs,
)


def test_training_pairs_keep_semantics_separate_from_structure(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    excluded = pairs.filter(pl.col('negative_is_explicit_exclusion')).row(0, named=True)
    ordinary = pairs.filter(~pl.col('negative_is_explicit_exclusion')).row(0, named=True)

    assert excluded['negative_semantic_target'] == 'unrelated'
    assert excluded['negative_semantic_source'] == 'explicit_exclusion'
    assert excluded['negative_structural_distance'] > 0.0
    assert excluded['negative_sampling_role'] == 'negative'
    assert ordinary['negative_semantic_target'] == 'unknown'
    assert ordinary['negative_semantic_source'] == 'unlabeled'


def test_explicit_exclusion_cannot_be_a_direct_positive(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(positive_is_explicit_exclusion=pl.lit(True))

    with pytest.raises(ValueError, match='direct positive.*explicit exclusion'):
        _validate_training_pairs(bad)


# -------------------------------------------------------------------------------------------------
# Legacy combinatorics
# -------------------------------------------------------------------------------------------------

def _triples(pairs: pl.DataFrame) -> list[tuple[int, int, int]]:
    return pairs.select('anchor_code_id', 'positive_code_id', 'negative_code_id').rows()


def test_fixture_triples_match_the_legacy_combinatorics(pair_facts_fixture):
    # Positives are canonical, non-maximal, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is an
    # exclusion. A negative j needs rows positive -> j and anchor -> j.
    pairs = build_training_pairs(pair_facts_fixture)

    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (1, 2, 3), (1, 2, 4)]
    assert not pairs.get_column('positive_is_explicit_exclusion').any()


@pytest.fixture
def cross_prefix_pair_facts() -> pl.DataFrame:
    # Codes 0 '111111', 1 '111112', 2 '222221', 3 '222222'; '111111' excludes '222221'.
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 1, 1, 2],
            'code_j_id': [1, 2, 3, 2, 3, 3],
            'code_i': ['111111', '111111', '111111', '111112', '111112', '222221'],
            'code_j': ['111112', '222221', '222222', '222221', '222222', '222222'],
            'structural_distance': [2.0, 99.0, 99.0, 99.0, 99.0, 2.0],
            'structural_relation_id': [2, 99, 99, 99, 99, 2],
            'structural_relation_name': [
                'sibling',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'sibling',
            ],
            'code_i_excludes_code_j': [False, True, False, False, False, False],
            'code_j_excludes_code_i': [False, False, False, False, False, False],
            'is_explicit_exclusion': [False, True, False, False, False, False],
        },
        schema_overrides={
            'code_i_id': pl.Int32,
            'code_j_id': pl.Int32,
            'structural_distance': pl.Float32,
            'structural_relation_id': pl.Int16,
        },
    )


def test_reversed_cross_prefix_rows_seed_negatives_for_later_anchors(cross_prefix_pair_facts):
    # Anchor 2 ('222221') only reaches codes 0 and 1 through reversed same-level rows, exactly as
    # the legacy keep-filter admitted both orientations of cross-prefix pairs.
    pairs = build_training_pairs(cross_prefix_pair_facts)

    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3), (2, 3, 0), (2, 3, 1)]


def test_reversed_rows_map_exclusion_directions_into_the_anchor_view(cross_prefix_pair_facts):
    pairs = build_training_pairs(cross_prefix_pair_facts)
    forward = pairs.filter(
        pl.col('anchor_code_id').eq(0) & pl.col('negative_code_id').eq(2)
    ).row(0, named=True)
    reverse = pairs.filter(
        pl.col('anchor_code_id').eq(2) & pl.col('negative_code_id').eq(0)
    ).row(0, named=True)

    assert (forward['anchor_excludes_negative'], forward['negative_excludes_anchor']) == (
        True,
        False,
    )
    assert (reverse['anchor_excludes_negative'], reverse['negative_excludes_anchor']) == (
        False,
        True,
    )
    assert reverse['negative_is_explicit_exclusion'] is True
    assert reverse['negative_semantic_target'] == 'unrelated'
    assert reverse['negative_structural_distance'] == 99.0


# -------------------------------------------------------------------------------------------------
# Deterministic cross-sector cap
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def wide_cross_sector_pair_facts() -> pl.DataFrame:
    # Codes 0 '111111' and 1 '111112' are siblings; codes 2-6 are cross-sector. '111111' excludes
    # '444444' (code 4). Every other pair is cross-sector.
    codes = ['111111', '111112', '222222', '333333', '444444', '555555', '666666']
    rows = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            siblings = (i, j) == (0, 1)
            rows.append(
                {
                    'code_i_id': i,
                    'code_j_id': j,
                    'code_i': codes[i],
                    'code_j': codes[j],
                    'structural_distance': 2.0 if siblings else 99.0,
                    'structural_relation_id': 2 if siblings else 99,
                    'structural_relation_name': 'sibling' if siblings else 'cross_sector',
                    'code_i_excludes_code_j': (i, j) == (0, 4),
                    'code_j_excludes_code_i': False,
                    'is_explicit_exclusion': (i, j) == (0, 4),
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides={
            'code_i_id': pl.Int32,
            'code_j_id': pl.Int32,
            'structural_distance': pl.Float32,
            'structural_relation_id': pl.Int16,
        },
    )


def test_cross_sector_cap_keeps_a_deterministic_subset_and_exempts_exclusions(
    wide_cross_sector_pair_facts,
):
    uncapped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=100)
    capped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)
    again = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)

    assert _triples(uncapped) == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 1, 6)]
    assert capped.equals(again)
    assert set(_triples(capped)) <= set(_triples(uncapped))
    assert capped.filter(pl.col('negative_is_explicit_exclusion')).height == 1
    assert capped.filter(~pl.col('negative_is_explicit_exclusion')).height == 2


# -------------------------------------------------------------------------------------------------
# Structural margins (legacy consumers such as HGCN read these values)
# -------------------------------------------------------------------------------------------------

def test_structural_margins_preserve_the_legacy_special_cases():
    frame = pl.DataFrame(
        {
            'positive_structural_relation_id': [7, 2, 1, 1, 7],
            'positive_structural_distance': [4.0, 2.0, 0.5, 0.5, 4.0],
            'negative_structural_relation_id': [8, 3, 99, 2, 2],
            'negative_structural_distance': [4.0, 1.5, 99.0, 2.0, 2.0],
        },
        schema_overrides={
            'positive_structural_relation_id': pl.Int16,
            'negative_structural_relation_id': pl.Int16,
            'positive_structural_distance': pl.Float32,
            'negative_structural_distance': pl.Float32,
        },
    )

    margins = _structural_margins(frame)

    # Rows: equal distance with a farther relation, a -0.5 lineal adjustment, a cross-sector
    # negative, an ordinary farther negative; the structurally closer negative is dropped.
    assert margins.get_column('relation_margin').to_list() == pytest.approx([1.0, 1.0, 15.0, 1.0])
    assert margins.get_column('distance_margin').to_list() == pytest.approx(
        [0.3333, 0.6667, 10.0, 1.5]
    )
    assert margins.get_column('margin').to_list() == pytest.approx(
        [
            1.0 / (1.0 * 0.3333 + 0.3333 * 0.6667),
            1.0 / (1.0 * 0.3333 + 0.6667 * 0.6667),
            1.0 / (15.0 * 0.3333 + 10.0 * 0.6667),
            1.0 / (1.0 * 0.3333 + 1.5 * 0.6667),
        ],
        rel=1e-5,
    )


# -------------------------------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------------------------------

def test_inconsistent_exclusion_derivation_is_fatal(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(negative_is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        _validate_training_pairs(bad)


def test_semantic_target_must_follow_exclusion_provenance(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(negative_semantic_target=pl.lit('unknown'))

    with pytest.raises(ValueError, match='semantic'):
        _validate_training_pairs(bad)
