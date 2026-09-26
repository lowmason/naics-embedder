import polars as pl
import pytest

from naics_embedder.data.create_triplets import (
    _anchor_view,
    _positive_pairs,
    _structural_margins,
    _validate_training_pairs,
    build_training_pairs,
)

def test_no_training_negative_is_an_explicit_exclusion(pair_facts_fixture):
    # '111111' excludes '111113' and '222222' excludes '111112': neither pair is ever a negative
    pairs = build_training_pairs(pair_facts_fixture)

    assert not pairs.get_column('negative_is_explicit_exclusion').any()
    assert pairs.get_column('negative_semantic_target').unique().to_list() == ['unknown']
    assert pairs.get_column('negative_semantic_source').unique().to_list() == ['unlabeled']
    assert pairs.get_column('negative_sampling_role').unique().to_list() == ['negative']

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
    # Positives are canonical, within-sector, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is
    # an exclusion. A negative j needs rows positive -> j and anchor -> j, and is never an
    # exclusion of its anchor: 2 is anchor 0's, and 3 excludes anchor 1.
    pairs = build_training_pairs(pair_facts_fixture)

    assert _triples(pairs) == [(0, 1, 3), (0, 1, 4), (1, 2, 4)]
    assert not pairs.get_column('positive_is_explicit_exclusion').any()

def test_a_unary_pair_is_never_a_positive(pair_facts_fixture):
    # Flag (0, 1): its two triples go, and the third stays
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    assert _triples(build_training_pairs(flagged)) == [(1, 2, 4)]

@pytest.fixture
def cross_prefix_pair_facts() -> pl.DataFrame:
    # Codes 0 '111111', 1 '111112', 2 '222221', 3 '222222'; '111111' excludes '222221'.
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 1, 1, 2],
            'code_j_id': [1, 2, 3, 2, 3, 3],
            'code_i': ['111111', '111111', '111111', '111112', '111112', '222221'],
            'code_j': ['111112', '222221', '222222', '222221', '222222', '222222'],
            'structural_distance': [2.0, 10.0, 10.0, 10.0, 10.0, 2.0],
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
            'unary_pair': [False] * 6,
        },
        schema_overrides={
            'code_i_id': pl.Int32,
            'code_j_id': pl.Int32,
            'structural_distance': pl.Float32,
            'structural_relation_id': pl.Int16,
        },
    )

def test_reversed_cross_prefix_rows_seed_negatives_for_later_anchors(cross_prefix_pair_facts):
    # Anchor 2 ('222221') reaches code 1 only through a reversed same-level row, exactly as the
    # legacy keep-filter admitted both orientations of cross-prefix pairs. Code 0 excludes it, so
    # neither is ever the other's negative.
    pairs = build_training_pairs(cross_prefix_pair_facts)

    assert _triples(pairs) == [(0, 1, 3), (2, 3, 1)]

def test_reversed_rows_map_exclusion_directions_into_the_anchor_view(cross_prefix_pair_facts):
    # The directions still reach the anchor view, which is how the generator finds the exclusions
    # it must drop
    view = _anchor_view(cross_prefix_pair_facts)
    forward = view.filter(pl.col('anchor_code_id').eq(0)
                          & pl.col('candidate_code_id').eq(2)).row(0, named=True)
    reverse = view.filter(pl.col('anchor_code_id').eq(2)
                          & pl.col('candidate_code_id').eq(0)).row(0, named=True)

    assert (forward['anchor_excludes_candidate'], forward['candidate_excludes_anchor']) == (
        True,
        False,
    )
    assert (reverse['anchor_excludes_candidate'], reverse['candidate_excludes_anchor']) == (
        False,
        True,
    )
    assert reverse['is_explicit_exclusion'] is True
    assert reverse['structural_distance'] == 10.0

@pytest.fixture
def labelled_cross_sector_pair_facts() -> pl.DataFrame:
    # Codes 0 '111111' and 1 '111112' are siblings; 2 '22' is another sector and 3 '222222' its
    # six-digit descendant. The cross-sector pairs carry D* (6 or 10), not a sentinel, so only
    # their relation label, 99, marks them.
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 1, 1, 2],
            'code_j_id': [1, 2, 3, 2, 3, 3],
            'code_i': ['111111', '111111', '111111', '111112', '111112', '22'],
            'code_j': ['111112', '22', '222222', '22', '222222', '222222'],
            'structural_distance': [2.0, 6.0, 10.0, 6.0, 10.0, 4.0],
            'structural_relation_id': [2, 99, 99, 99, 99, 6],
            'structural_relation_name': [
                'sibling',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'great-great-grandchild',
            ],
            'code_i_excludes_code_j': [False] * 6,
            'code_j_excludes_code_i': [False] * 6,
            'is_explicit_exclusion': [False] * 6,
            'unary_pair': [False] * 6,
        },
        schema_overrides={
            'code_i_id': pl.Int32,
            'code_j_id': pl.Int32,
            'structural_distance': pl.Float32,
            'structural_relation_id': pl.Int16,
        },
    )

def test_cross_sector_pairs_are_read_from_their_relation_label(labelled_cross_sector_pair_facts):
    positives = _positive_pairs(_anchor_view(labelled_cross_sector_pair_facts))
    pairs = build_training_pairs(labelled_cross_sector_pair_facts)
    capped = build_training_pairs(labelled_cross_sector_pair_facts, cross_sector_cap=1)

    # A cross-sector pair is never a positive, whatever its distance
    assert positives.select('anchor_code_id', 'positive_code_id').rows() == [(0, 1), (2, 3)]
    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3)]
    assert pairs.get_column('relation_margin').to_list() == [15.0, 15.0]
    # The distance margin is the D* difference: '22' is 6 from the anchor, '222222' 10, and the
    # positive 2
    assert pairs.get_column('distance_margin').to_list() == [4.0, 8.0]
    assert pairs.get_column('unrelated').to_list() == [True, True]
    assert capped.height == 1

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
                    'structural_distance': 2.0 if siblings else 10.0,
                    'structural_relation_id': 2 if siblings else 99,
                    'structural_relation_name': 'sibling' if siblings else 'cross_sector',
                    'code_i_excludes_code_j': (i, j) == (0, 4),
                    'code_j_excludes_code_i': False,
                    'is_explicit_exclusion': (i, j) == (0, 4),
                    'unary_pair': False,
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

def test_cross_sector_cap_keeps_a_deterministic_subset(wide_cross_sector_pair_facts):
    uncapped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=100)
    capped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)
    again = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)

    # '444444' (code 4) is anchor 0's exclusion, so it is never a negative, capped or not
    assert _triples(uncapped) == [(0, 1, 2), (0, 1, 3), (0, 1, 5), (0, 1, 6)]
    assert capped.equals(again)
    assert set(_triples(capped)) <= set(_triples(uncapped))
    assert capped.height == 2

# -------------------------------------------------------------------------------------------------
# Structural margins (legacy consumers such as HGCN read these values)
# -------------------------------------------------------------------------------------------------

def test_structural_margins_read_d_star_differences():
    frame = pl.DataFrame(
        {
            'positive_structural_relation_id': [7, 1, 1, 2, 7, 7],
            'positive_structural_distance': [4.0, 1.0, 1.0, 2.0, 4.0, 8.0],
            'negative_structural_relation_id': [8, 3, 99, 5, 2, 99],
            'negative_structural_distance': [4.0, 2.0, 6.0, 3.0, 2.0, 6.0],
        },
        schema_overrides={
            'positive_structural_relation_id': pl.Int16,
            'negative_structural_relation_id': pl.Int16,
            'positive_structural_distance': pl.Float32,
            'negative_structural_distance': pl.Float32,
        },
    )

    margins = _structural_margins(frame)

    # Rows: an equal D* with a farther relation (the one fixed distance margin), a lineal negative
    # one step past a lineal positive, a cross-sector negative, an ordinary farther negative. The
    # structurally closer negative is dropped, and so is a cross-sector negative that is closer in
    # D* than the positive.
    assert margins.get_column('relation_margin').to_list() == pytest.approx([1.0, 2.0, 15.0, 3.0])
    assert margins.get_column('distance_margin').to_list() == pytest.approx([0.3333, 1.0, 5.0, 1.0])
    assert margins.get_column('margin').to_list() == pytest.approx(
        [
            1.0 / (1.0 * 0.3333 + 0.3333 * 0.6667),
            1.0 / (2.0 * 0.3333 + 1.0 * 0.6667),
            1.0 / (15.0 * 0.3333 + 5.0 * 0.6667),
            1.0 / (3.0 * 0.3333 + 1.0 * 0.6667),
        ],
        rel=1e-5,
    )

# -------------------------------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------------------------------

def test_inconsistent_exclusion_derivation_is_fatal(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(anchor_excludes_negative=pl.lit(True))

    with pytest.raises(ValueError, match='exclusion derivation'):
        _validate_training_pairs(bad)

def test_semantic_target_must_follow_exclusion_provenance(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(negative_semantic_target=pl.lit('unrelated'))

    with pytest.raises(ValueError, match='semantic'):
        _validate_training_pairs(bad)

def test_an_exclusion_negative_is_fatal(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(
        anchor_excludes_negative=pl.lit(True),
        negative_is_explicit_exclusion=pl.lit(True),
        negative_semantic_target=pl.lit('unrelated'),
    )

    with pytest.raises(ValueError, match='training negative cannot be an explicit exclusion'):
        _validate_training_pairs(bad)
