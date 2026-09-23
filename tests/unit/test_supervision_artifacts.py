import polars as pl
import pytest

from naics_embedder.data.create_triplets import build_training_pairs
from naics_embedder.data.supervision_bundle import (
    build_codebook,
    build_pair_facts,
    codebook_fingerprint,
    distance_matrix_from_pair_facts,
    relation_matrix_from_pair_facts,
)


def test_exclusion_provenance_does_not_mutate_structure(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)

    facts = build_pair_facts(distances, relations, descriptions, codebook)
    row = facts.filter(
        pl.col('code_i').eq('111111') & pl.col('code_j').eq('111113')
    ).row(0, named=True)

    assert row['structural_distance'] == 2.0
    assert row['structural_relation_id'] == 2
    assert row['structural_relation_name'] == 'sibling'
    assert row['code_i_excludes_code_j'] is True
    assert row['code_j_excludes_code_i'] is False
    assert row['is_explicit_exclusion'] is True


def test_reverse_direction_survives_canonical_orientation(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)

    facts = build_pair_facts(distances, relations, descriptions, codebook)
    row = facts.filter(
        pl.col('code_i').eq('111112') & pl.col('code_j').eq('222222')
    ).row(0, named=True)

    assert row['code_i_excludes_code_j'] is False
    assert row['code_j_excludes_code_i'] is True
    assert row['is_explicit_exclusion'] is True


def test_matrices_reconcile_with_pair_facts_and_codebook_order(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)
    facts = build_pair_facts(distances, relations, descriptions, codebook)

    distance_matrix = distance_matrix_from_pair_facts(facts, codebook)
    relation_matrix = relation_matrix_from_pair_facts(facts, codebook)

    assert distance_matrix.row(0)[1] == 0.5
    assert distance_matrix.row(1)[0] == 0.5
    assert relation_matrix.row(0)[1] == 1
    assert relation_matrix.row(1)[0] == 1
    assert codebook_fingerprint(codebook) == codebook_fingerprint(codebook.clone())


def test_training_pairs_expose_identity_and_supervision_columns_deterministically(
    pair_facts_fixture,
):
    pair_facts = pair_facts_fixture
    training_pairs = build_training_pairs(pair_facts)

    expected_identity_columns = {
        'anchor_code_id',
        'positive_code_id',
        'negative_code_id',
        'anchor_code',
        'positive_code',
        'negative_code',
    }
    expected_supervision_columns = {
        'positive_structural_distance',
        'negative_structural_distance',
        'positive_structural_relation_id',
        'negative_structural_relation_id',
        'positive_semantic_target',
        'negative_semantic_target',
        'positive_semantic_source',
        'negative_semantic_source',
        'anchor_excludes_negative',
        'negative_excludes_anchor',
        'negative_is_explicit_exclusion',
    }
    assert expected_identity_columns <= set(training_pairs.columns)
    assert expected_supervision_columns <= set(training_pairs.columns)
    assert training_pairs.equals(build_training_pairs(pair_facts.clone()))


def test_pair_rows_keep_the_generator_canonical_orientation(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    facts = build_pair_facts(
        distances,
        relations,
        descriptions_fixture,
        build_codebook(descriptions_fixture),
    )
    assert facts.select(
        pl.col('code_i_id').lt(pl.col('code_j_id')).all()
    ).item()


# -------------------------------------------------------------------------------------------------
# Production-shaped orientation
#
# Real code IDs follow lexicographic code order, so the canonical shallower-first orientation can
# carry descending IDs ('3112' is shallower than '31111' but sorts after it). These frames are
# hand-written: IDs 0-3 are '311', '3111', '31111', '3112'.
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def depth_first_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    descriptions = pl.DataFrame(
        {
            'index': [0, 1, 2, 3],
            'code': ['311', '3111', '31111', '3112'],
            'excluded_codes': [None, None, ['3112'], None],
        },
        schema_overrides={'excluded_codes': pl.List(pl.Utf8)},
    )
    pairs = {
        'idx_i': [0, 0, 0, 1, 1, 3],
        'idx_j': [1, 2, 3, 2, 3, 2],
        'code_i': ['311', '311', '311', '3111', '3111', '3112'],
        'code_j': ['3111', '31111', '3112', '31111', '3112', '31111'],
    }
    distances = pl.DataFrame(pairs | {'structural_distance': [0.5, 1.5, 0.5, 0.5, 2.0, 3.0]})
    relations = pl.DataFrame(
        pairs
        | {
            'structural_relation_id': [1, 3, 1, 1, 2, 5],
            'structural_relation_name': [
                'child',
                'grandchild',
                'child',
                'child',
                'sibling',
                'nephew/niece',
            ],
        }
    )
    return descriptions, distances, relations


def test_pair_facts_accept_shallower_first_rows_with_descending_ids(depth_first_frames):
    descriptions, distances, relations = depth_first_frames

    facts = build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))
    row = facts.filter(
        pl.col('code_i').eq('3112') & pl.col('code_j').eq('31111')
    ).row(0, named=True)

    assert (row['code_i_id'], row['code_j_id']) == (3, 2)
    assert row['structural_distance'] == 3.0
    assert row['structural_relation_name'] == 'nephew/niece'
    assert row['code_i_excludes_code_j'] is False
    assert row['code_j_excludes_code_i'] is True
    assert row['is_explicit_exclusion'] is True
    assert facts.height == 6


def test_pair_facts_reject_a_deeper_first_row(depth_first_frames):
    descriptions, distances, relations = depth_first_frames

    def reorient(frame: pl.DataFrame) -> pl.DataFrame:
        swapped = pl.col('code_i').eq('3112')
        return frame.with_columns(
            idx_i=pl.when(swapped).then(pl.col('idx_j')).otherwise(pl.col('idx_i')),
            idx_j=pl.when(swapped).then(pl.col('idx_i')).otherwise(pl.col('idx_j')),
            code_i=pl.when(swapped).then(pl.col('code_j')).otherwise(pl.col('code_i')),
            code_j=pl.when(swapped).then(pl.col('code_i')).otherwise(pl.col('code_j')),
        )

    with pytest.raises(ValueError, match='canonical orientation'):
        build_pair_facts(
            reorient(distances),
            reorient(relations),
            descriptions,
            build_codebook(descriptions),
        )


def test_pair_facts_reject_a_reversed_duplicate_pair(depth_first_frames):
    descriptions, distances, relations = depth_first_frames
    reversed_key = {'idx_i': [3], 'idx_j': [1], 'code_i': ['3112'], 'code_j': ['3111']}
    distances = pl.concat(
        [distances, pl.DataFrame(reversed_key | {'structural_distance': [99.0]})]
    )
    relations = pl.concat(
        [
            relations,
            pl.DataFrame(
                reversed_key
                | {
                    'structural_relation_id': [99],
                    'structural_relation_name': ['cross_sector'],
                }
            ),
        ]
    )

    with pytest.raises(ValueError, match='duplicate unordered'):
        build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))


def test_pair_facts_reject_an_exclusion_that_cannot_attach(depth_first_frames):
    descriptions, distances, relations = depth_first_frames
    self_exclusion = descriptions.with_columns(
        excluded_codes=pl.Series([['311'], None, ['3112'], None], dtype=pl.List(pl.Utf8))
    )

    with pytest.raises(ValueError, match='could not be attached'):
        build_pair_facts(distances, relations, self_exclusion, build_codebook(self_exclusion))


def test_pair_facts_reject_ids_that_disagree_with_the_codebook(depth_first_frames):
    descriptions, distances, relations = depth_first_frames
    mislabeled = descriptions.with_columns(
        code=pl.when(pl.col('index').eq(0)).then(pl.lit('399')).otherwise(pl.col('code'))
    )

    with pytest.raises(ValueError, match='codebook'):
        build_pair_facts(distances, relations, mislabeled, build_codebook(mislabeled))
