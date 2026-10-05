import json
import re
import uuid

import polars as pl
import pyarrow.parquet as pq
import pytest

from naics_embedder.data.create_triplets import build_training_pairs
from naics_embedder.data.redirections import activity_phrase
from naics_embedder.data.supervision_bundle import (
    build_codebook,
    build_pair_facts,
    codebook_fingerprint,
    distance_matrix_from_pair_facts,
    generate_supervision_bundle,
    input_window_record,
    relation_matrix_from_pair_facts,
)
from naics_embedder.panels.index_roles import verify_role_leakage
from naics_embedder.supervision.artifacts import (
    METADATA_BUNDLE,
    REDIRECTIONS_SCHEMA,
    REQUIRED_VALIDATION_RESULTS,
    load_validated_bundle,
    sha256_file,
    validate_redirection_table,
    validate_training_pairs_members,
)
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ChannelOverflow,
    InputWindowRecord,
)
from naics_embedder.utils.config import SupervisionBuildConfig
from tests.fixtures.supervision import FIVE_CODE_INPUT_WINDOW

def test_exclusion_provenance_does_not_mutate_structure(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)

    facts = build_pair_facts(distances, relations, descriptions, codebook)
    # yapf: disable
    row = facts.filter(
        pl.col('code_i').eq('111111') & pl.col('code_j').eq('111113')
    ).row(0, named=True)
    # yapf: enable

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
    # yapf: disable
    row = facts.filter(
        pl.col('code_i').eq('111112') & pl.col('code_j').eq('222222')
    ).row(0, named=True)
    # yapf: enable

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

    assert distance_matrix.row(0)[1] == 2.0
    assert distance_matrix.row(1)[0] == 2.0
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
    assert facts.select(pl.col('code_i_id').lt(pl.col('code_j_id')).all()).item()

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
    distances = pl.DataFrame(pairs | {'structural_distance': [1.0, 2.0, 1.0, 1.0, 2.0, 3.0]})
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
    # yapf: disable
    row = facts.filter(
        pl.col('code_i').eq('3112') & pl.col('code_j').eq('31111')
    ).row(0, named=True)
    # yapf: enable

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
    distances = pl.concat([distances, pl.DataFrame(reversed_key | {'structural_distance': [2.0]})])
    relations = pl.concat(
        [
            relations,
            pl.DataFrame(
                reversed_key
                | {
                    'structural_relation_id': [2],
                    'structural_relation_name': ['sibling'],
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

# -------------------------------------------------------------------------------------------------
# D* (Req 7): every stored distance is checked
# -------------------------------------------------------------------------------------------------

def _set_pair(frame: pl.DataFrame, code_i: str, code_j: str, column: str, value) -> pl.DataFrame:
    chosen = pl.col('code_i').eq(code_i) & pl.col('code_j').eq(code_j)
    return frame.with_columns(
        pl.when(chosen).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )

@pytest.mark.parametrize(
    ('code_i', 'code_j', 'value', 'message'),
    [
        # D* has no half-step for a lineal pair
        ('311', '3111', 0.5, 'no half-step'),
        # 3112 and 31111 are 3 apart; at 4 they would be farther than through 311 (1 + 2)
        ('3112', '31111', 4.0, 'triangle inequality'),
        # A parent and its child are 1 apart; 2 keeps the triangle inequality but is not D*
        ('311', '3111', 2.0, 'differs from D\\*'),
    ],
)
def test_pair_facts_reject_a_distance_that_is_not_d_star(
    depth_first_frames, code_i, code_j, value, message
):
    descriptions, distances, relations = depth_first_frames
    distances = _set_pair(distances, code_i, code_j, 'structural_distance', value)

    with pytest.raises(ValueError, match=message):
        build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))

def test_pair_facts_reject_a_cross_sector_label_inside_a_sector(depth_first_frames):
    descriptions, distances, relations = depth_first_frames
    relations = _set_pair(relations, '3111', '3112', 'structural_relation_id', 99)

    with pytest.raises(ValueError, match='cross_sector relation label'):
        build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))

@pytest.mark.parametrize(
    ('value', 'message'),
    [(99.0, 'retired cross-sector constant 99'), (9.0, 'cross-sector distances must equal')],
)
def test_pair_facts_reject_a_cross_sector_distance_off_the_formula(
    descriptions_fixture, structural_frames_fixture, value, message
):
    # '111111' and '222222' meet only at the virtual root: 6 + 6 - 2 = 10
    distances, relations = structural_frames_fixture
    distances = _set_pair(distances, '111111', '222222', 'structural_distance', value)

    with pytest.raises(ValueError, match=message):
        build_pair_facts(
            distances, relations, descriptions_fixture, build_codebook(descriptions_fixture)
        )

# -------------------------------------------------------------------------------------------------
# Immutable bundle publication
# -------------------------------------------------------------------------------------------------

def test_bundle_writes_manifest_last_with_matching_parquet_metadata(build_bundle):
    manifest_path = build_bundle()

    manifest = json.loads(manifest_path.read_text())
    codebook_path = manifest_path.parent / manifest['artifacts']['codebook']['path']
    metadata = pq.read_metadata(codebook_path).metadata

    assert manifest_path.name == 'manifest.json'
    assert manifest['contract_version'] == CONTRACT_VERSION
    assert metadata[b'naics_embedder.contract_version'].decode() == CONTRACT_VERSION
    assert metadata[b'naics_embedder.bundle_id'].decode() == 'bundle-a'
    assert metadata[b'naics_embedder.schema_version'].decode() == 'codebook-v1'

def test_bundle_never_overwrites_an_existing_generation(build_bundle):
    build_bundle()

    with pytest.raises(FileExistsError, match='bundle-a'):
        build_bundle()

def test_failed_validation_publishes_no_manifest(tmp_path, build_bundle, pair_facts_fixture):
    inconsistent = pair_facts_fixture.with_columns(is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        build_bundle(bundle_id='broken', pair_facts=inconsistent)

    assert not (tmp_path / 'broken' / 'manifest.json').exists()

def test_two_generated_bundles_have_equal_logical_frames_but_distinct_ids(build_bundle):
    manifests = [build_bundle(bundle_id=bundle_id) for bundle_id in ('bundle-a', 'bundle-b')]
    loaded = [json.loads(path.read_text()) for path in manifests]
    frames = [
        pl.read_parquet(path.parent / manifest['artifacts']['pair_facts']['path'])
        for path, manifest in zip(manifests, loaded)
    ]

    assert loaded[0]['bundle_id'] != loaded[1]['bundle_id']
    assert frames[0].equals(frames[1])

def test_bundle_records_every_artifact_member_with_hash_and_contract_metadata(
    tmp_path, build_bundle
):
    manifest_path = build_bundle()
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest['artifacts']

    assert set(artifacts) == {
        'codebook',
        'pair_facts',
        'distances',
        'distance_matrix',
        'relations',
        'relation_matrix',
        'training_pairs',
        'difficulty_thresholds',
        'index_roles',
        'redirections',
    }
    assert manifest['codebook_order'] == ['111111', '111112', '111113', '222222', '333333']
    assert artifacts['pair_facts']['row_count'] == 10
    assert artifacts['pair_facts']['exclusion_count'] == 2
    # Two of the five triples ran through an exclusion pair, which is never a negative
    assert artifacts['training_pairs']['row_count'] == 3
    assert artifacts['training_pairs']['exclusion_count'] == 0
    assert manifest['validation_results']['no_exclusion_negatives'] is True
    assert all(manifest['validation_results'].values())
    for check in (
        'distance_is_d_star',
        'cross_sector_distance_formula',
        'cross_sector_relation_label',
        'distance_triangle_inequality',
    ):
        assert manifest['validation_results'][check] is True
    for record in artifacts.values():
        assert sum(member['row_count'] for member in record['files']) == record['row_count']
        for member in record['files']:
            path = manifest_path.parent / member['path']
            assert sha256_file(path) == member['sha256']
            if path.suffix == '.parquet':
                metadata = pq.read_metadata(path).metadata
                assert metadata[b'naics_embedder.bundle_id'] == b'bundle-a'
                assert metadata[b'naics_embedder.schema_version'].decode() == (
                    record['schema_version']
                )
    assert not list(tmp_path.glob('.*staging*'))

def test_failed_generation_leaves_no_staging_directory(tmp_path, build_bundle, pair_facts_fixture):
    with pytest.raises(ValueError):
        build_bundle(
            bundle_id='broken',
            pair_facts=pair_facts_fixture.with_columns(structural_distance=pl.lit(0.0)),
        )

    assert list(tmp_path.iterdir()) == []

def test_production_bundle_uses_a_uuid_and_the_descriptions_file_hash(
    hierarchy_manifest, hierarchy_descriptions_parquet
):
    manifest = json.loads(hierarchy_manifest.read_text())

    assert str(uuid.UUID(manifest['bundle_id'])) == manifest['bundle_id']
    assert hierarchy_manifest.parent.name == manifest['bundle_id']
    assert manifest['description_fingerprint'] == sha256_file(hierarchy_descriptions_parquet)
    assert manifest['structural_relation_ids']['cross_sector'] == 99
    assert manifest['artifacts']['pair_facts']['row_count'] == 17 * 16 // 2
    assert manifest['artifacts']['pair_facts']['exclusion_count'] == 2

def test_the_unary_pairs_are_flagged_and_never_generated_positives(hierarchy_manifest):
    bundle = load_validated_bundle(hierarchy_manifest)
    facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
    pairs = pl.read_parquet(list(bundle.member_paths('training_pairs')))

    unary = facts.filter(pl.col('unary_pair')).select('code_i', 'code_j').rows()
    positives = set(pairs.select('anchor_code', 'positive_code').unique().rows())
    # Each five-digit code in the hierarchy has one six-digit child
    assert unary == [
        ('31111', '311111'),
        ('31121', '311211'),
        ('32111', '321111'),
        ('44111', '441111'),
    ]
    assert not positives & set(unary)
    assert bundle.manifest.artifacts['pair_facts'].schema_version == 'pair-facts-v2'
    assert bundle.manifest.validation_results['unary_pairs_flagged'] is True
    assert bundle.manifest.validation_results['no_unary_positives'] is True

def test_pair_facts_refuse_a_unary_flag_the_codes_do_not_support(build_bundle, pair_facts_fixture):
    # '111111' and '111112' are six-digit siblings, not a five-digit code and its only child
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='unary_pair is wrong on 1 pairs, e.g. 111111/111112'):
        build_bundle(pair_facts=flagged)

def test_training_pairs_refuse_a_unary_positive(tmp_path, pair_facts_fixture):
    # Rows generated while (0, 1) was unflagged keep it as a positive; the flagged facts refuse them
    path = tmp_path / 'training_pairs.parquet'
    build_training_pairs(pair_facts_fixture).write_parquet(path)
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='a training positive is a unary pair'):
        validate_training_pairs_members([path], flagged, n_codes=5)

# -------------------------------------------------------------------------------------------------
# Fail-closed bundle loading
# -------------------------------------------------------------------------------------------------

def _rewrite_member(manifest_path, logical_name, transform, member_index=0):
    '''Rewrite one member (keeping its contract metadata) and re-hash it in the manifest.'''

    manifest = json.loads(manifest_path.read_text())
    member = manifest['artifacts'][logical_name]['files'][member_index]
    path = manifest_path.parent / member['path']
    table = pq.read_table(path)
    frame = transform(pl.from_arrow(table))
    pq.write_table(frame.to_arrow().replace_schema_metadata(table.schema.metadata), path)
    member['sha256'] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2))

def test_loader_accepts_a_generated_bundle(generated_bundle):
    bundle = load_validated_bundle(generated_bundle, expected_contract=CONTRACT_VERSION)

    assert bundle.manifest.bundle_id == 'bundle-a'
    assert bundle.manifest_path == generated_bundle.resolve()
    assert bundle.artifact_path('pair_facts') == generated_bundle.parent.resolve() / (
        'naics_pair_facts.parquet'
    )
    with pytest.raises(ValueError, match='no .*missing_artifact'):
        bundle.artifact_path('missing_artifact')

def test_loader_rejects_another_contract_version(generated_bundle):
    with pytest.raises(ValueError, match='expected supervision contract stage3-supervision-v1'):
        load_validated_bundle(generated_bundle, expected_contract='stage3-supervision-v1')

def test_loader_rejects_bytes_that_do_not_match_the_manifest_hash(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    path = generated_bundle.parent / manifest['artifacts']['codebook']['files'][0]['path']
    path.write_bytes(path.read_bytes() + b'tampered')

    with pytest.raises(ValueError, match='codebook hash mismatch'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_missing_member(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    member = manifest['artifacts']['training_pairs']['files'][0]
    (generated_bundle.parent / member['path']).unlink()

    with pytest.raises(ValueError, match='training_pairs artifact missing'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_mixed_bundle_metadata(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    member = manifest['artifacts']['pair_facts']['files'][0]
    pair_path = generated_bundle.parent / member['path']
    table = pq.read_table(pair_path)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_BUNDLE] = b'bundle-b'
    pq.write_table(table.replace_schema_metadata(metadata), pair_path)
    member['sha256'] = sha256_file(pair_path)
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*bundle-b'):
        load_validated_bundle(
            generated_bundle,
            expected_contract='stage3-supervision-v2',
        )

def test_loader_rejects_rehashed_inconsistent_pair_facts(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(is_explicit_exclusion=pl.lit(False)),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*exclusion derivation'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_structural_sentinel(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(
            structural_distance=pl.when(pl.col('is_explicit_exclusion')).then(
                pl.lit(0.0, dtype=pl.Float32)
            ).otherwise(pl.col('structural_distance'))
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*structural distance zero'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_legacy_cross_sector_distance(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(
            structural_distance=pl.when(pl.col('structural_relation_id').eq(99)).then(
                pl.lit(99.0, dtype=pl.Float32)
            ).otherwise(pl.col('structural_distance'))
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*retired cross-sector constant'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_matrix_that_drifts_from_pair_facts(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'distance_matrix',
        lambda frame: frame.with_columns(pl.col(frame.columns[1]) + 1.0),
    )

    with pytest.raises(ValueError, match='distance_matrix.*bundle-a.*reconcile'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_excluded_direct_positive(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(positive_is_explicit_exclusion=pl.lit(True)),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*direct positive'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_training_exclusions_that_disagree_with_pair_facts(generated_bundle):
    # Anchor 0's negatives become code 2, its exclusion, with every exclusion flag still false
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            negative_code_id=pl.when(pl.col('anchor_code_id').eq(0)).then(
                pl.lit(2).cast(frame.schema['negative_code_id'])
            ).otherwise(pl.col('negative_code_id'))
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_unary_flag(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(
            unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*unary_pair is wrong'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_exclusion_negative(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            anchor_excludes_negative=pl.lit(True),
            negative_is_explicit_exclusion=pl.lit(True),
            negative_semantic_target=pl.lit('unrelated'),
            negative_semantic_source=pl.lit('explicit_exclusion'),
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*explicit exclusions of'):
        load_validated_bundle(generated_bundle)

# -------------------------------------------------------------------------------------------------
# The index-roles and redirections members
# -------------------------------------------------------------------------------------------------

def test_bundle_carries_the_index_roles_after_checking_them(generated_bundle, index_roles_fixture):
    manifest = json.loads(generated_bundle.read_text())
    record = manifest['artifacts']['index_roles']

    assert record['path'] == 'naics_index_roles.parquet'
    assert record['schema_version'] == 'index-roles-v1'
    assert record['row_count'] == 6
    for check in (
        'index_roles_one_role_per_entry',
        'index_roles_examples_channel',
        'index_roles_no_leakage',
    ):
        assert manifest['validation_results'][check] is True
    bundle = load_validated_bundle(generated_bundle)
    assert pl.read_parquet(bundle.artifact_path('index_roles')).equals(index_roles_fixture)

def test_bundle_carries_the_redirection_table_after_checking_it(
    generated_bundle, redirections_fixture
):
    manifest = json.loads(generated_bundle.read_text())
    record = manifest['artifacts']['redirections']

    assert record['path'] == 'naics_redirections.parquet'
    assert record['schema_version'] == 'redirections-v1'
    assert record['row_count'] == 2
    for check in (
        'redirections_well_formed',
        'redirections_match_exclusion_channel',
        'redirections_match_pair_facts',
    ):
        assert manifest['validation_results'][check] is True
    bundle = load_validated_bundle(generated_bundle)
    assert pl.read_parquet(bundle.artifact_path('redirections')).equals(redirections_fixture)

def test_bundle_refuses_an_examples_channel_holding_queries(
    tmp_path, build_bundle, text_descriptions_fixture
):
    stale = text_descriptions_fixture.with_columns(
        examples=pl.when(pl.col('code') == '111111').then(
            pl.lit('Soybean farming; Edamame farming')
        ).otherwise('examples')
    )

    with pytest.raises(ValueError, match='examples channel other than'):
        build_bundle(bundle_id='stale', descriptions=stale)
    assert list(tmp_path.iterdir()) == []

def test_bundle_refuses_a_held_out_query_matching_training_text(build_bundle, index_roles_fixture):
    # Entry 1 (validation) becomes another code's title
    leaky = index_roles_fixture.with_columns(
        text=pl.when(pl.col('entry_id') == 1).then(pl.lit('Industry 222222')).otherwise('text')
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        build_bundle(bundle_id='leaky', index_roles=leaky)

def test_the_leakage_check_reads_the_activity_phrases(monkeypatch, build_bundle):
    # Stage 7 trains on the activity phrases as queries, so no held-out query may match one
    seen = []

    def spy(descriptions, role_rows, *args, extra_texts=(), **kwargs):
        seen.append(list(extra_texts))
        return verify_role_leakage(
            descriptions, role_rows, *args, extra_texts=extra_texts, **kwargs
        )

    monkeypatch.setattr('naics_embedder.data.supervision_bundle.verify_role_leakage', spy)
    build_bundle()

    assert seen == [['Growing peanuts', 'Canola crushing']]

def test_bundle_refuses_an_exclusion_channel_the_table_does_not_build(
    tmp_path, build_bundle, text_descriptions_fixture
):
    # Code 111111's channel repeats its one cross-reference, which must appear once
    doubled = text_descriptions_fixture.with_columns(
        excluded=pl.when(pl.col('code') == '111111').then(
            pl.concat_str('excluded', pl.lit(' '), 'excluded')
        ).otherwise('excluded')
    )

    with pytest.raises(ValueError, match='exclusion channel other than the redirection table'):
        build_bundle(bundle_id='doubled', descriptions=doubled)
    assert list(tmp_path.iterdir()) == []

# A well-formed table over the codes '11111', '111111', '111112' and '222222': a cross-reference,
# a cross-reference naming its code's parent, and a withheld "Excluded" paragraph
WELL_FORMED_REDIRECTIONS = [
    (
        0,
        'cross_reference',
        '111111',
        'Growing peanuts--are classified in Industry 111112.',
        'Growing peanuts',
        ['111112'],
        [],
        False,
    ),
    (
        1,
        'cross_reference',
        '111112',
        'Mixed farming--are classified in Industry 11111.',
        'Mixed farming',
        ['11111'],
        ['11111'],
        False,
    ),
    (
        2,
        'description',
        '222222',
        'Farm supplies are classified in Industry 111111.',
        None,
        ['111111'],
        [],
        True,
    ),
]
REDIRECTION_CODES = ['11111', '111111', '111112', '222222']

def _redirections(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=REDIRECTIONS_SCHEMA, orient='row')

def test_redirection_table_accepts_a_well_formed_table():
    validate_redirection_table(_redirections(WELL_FORMED_REDIRECTIONS), REDIRECTION_CODES)

@pytest.mark.parametrize(
    ('row', 'column', 'value', 'message'),
    [
        (0, 'reference_id', 5, 'reference IDs must run from zero'),
        (0, 'source', 'index', 'unknown redirection sources'),
        (0, 'named_codes', ['999999'], 'names a code outside the codebook'),
        (0, 'named_codes', ['111111'], 'names its own code'),
        (1, 'lineal_codes', [], 'lineal_codes must be'),
        (2, 'activity', 'Farm supplies', 'activity phrase'),
    ],
)
def test_redirection_table_refuses_a_malformed_row(row, column, value, message):
    rows = [list(values) for values in WELL_FORMED_REDIRECTIONS]
    rows[row][list(REDIRECTIONS_SCHEMA).index(column)] = value

    with pytest.raises(ValueError, match=message):
        validate_redirection_table(_redirections(rows), REDIRECTION_CODES)

def test_redirection_table_refuses_other_columns():
    table = _redirections(WELL_FORMED_REDIRECTIONS).drop('withheld')

    with pytest.raises(ValueError, match='redirection columns must be'):
        validate_redirection_table(table, REDIRECTION_CODES)

def test_loader_rejects_an_index_entry_with_two_roles(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'index_roles',
        lambda frame: frame.with_columns(entry_id=pl.lit(0, pl.Int64)),
    )

    with pytest.raises(ValueError, match='index_roles .*more than one role'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_redirection_naming_another_code(generated_bundle):
    # Row 1 now sends canola crushing to '333333', which '222222' does not exclude
    _rewrite_member(
        generated_bundle,
        'redirections',
        lambda frame: frame.with_columns(
            named_codes=pl.Series([['111113'], ['333333']], dtype=pl.List(pl.Utf8))
        ),
    )

    with pytest.raises(
        ValueError, match='redirections .*bundle-a.*1 named pairs are not exclusions'
    ):
        load_validated_bundle(generated_bundle)

@pytest.mark.parametrize('phrase', ['Growing peanut', None], ids=['another-phrase', 'no-phrase'])
def test_loader_rejects_a_rehashed_redirection_whose_phrase_its_text_does_not_give(
    generated_bundle, phrase
):
    # Row 0's text, 'Growing peanuts--are classified in Industry 111113.', gives 'Growing peanuts'
    _rewrite_member(
        generated_bundle,
        'redirections',
        lambda frame: frame.with_columns(
            activity=pl.when(pl.col('reference_id') == 0).then(pl.lit(phrase, pl.Utf8)).otherwise(
                'activity'
            )
        ),
    )
    refusal = (
        f"redirection 0 (111111): its activity phrase is {phrase!r}, but its text gives "
        "'Growing peanuts'"
    )

    with pytest.raises(ValueError, match=re.escape(f'redirections (bundle-a): {refusal}')):
        load_validated_bundle(generated_bundle)

def test_bundle_refuses_a_redirection_whose_phrase_its_text_does_not_give(
    tmp_path, build_bundle, redirections_fixture
):
    # Row 1's text, 'Canola crushing--are classified in Industry 111112.', gives 'Canola crushing'
    misread = redirections_fixture.with_columns(
        activity=pl.when(pl.col('reference_id') == 1).then(pl.lit('Canola')).otherwise('activity')
    )
    refusal = "redirection 1 (222222): its activity phrase is 'Canola', but its text gives"

    with pytest.raises(ValueError, match=re.escape(refusal)):
        build_bundle(bundle_id='misread', redirections=misread)
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('manifest', ['generated_bundle', 'hierarchy_manifest'])
def test_the_fixture_bundles_carry_the_phrases_their_texts_give(request, manifest):
    bundle = load_validated_bundle(request.getfixturevalue(manifest))
    redirections = pl.read_parquet(bundle.artifact_path('redirections'))

    for row in redirections.filter(~pl.col('withheld')).iter_rows(named=True):
        redirects = row['source'] == 'cross_reference' and bool(row['named_codes'])
        assert row['activity'] == (activity_phrase(row['text']) if redirects else None)

@pytest.mark.parametrize('member', ['index_roles', 'redirections'])
def test_loader_requires_both_members(generated_bundle, member):
    manifest = json.loads(generated_bundle.read_text())
    del manifest['artifacts'][member]
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match=rf"lacks required artifacts: \['{member}'\]"):
        load_validated_bundle(generated_bundle)

def test_a_build_records_exactly_the_required_validation_results(generated_bundle):
    recorded = json.loads(generated_bundle.read_text())['validation_results']

    assert set(recorded) == set(REQUIRED_VALIDATION_RESULTS)
    # The member checks are among them, Req 3's leakage check included
    assert {
        'index_roles_one_role_per_entry',
        'index_roles_examples_channel',
        'index_roles_no_leakage',
        'redirections_well_formed',
        'redirections_match_exclusion_channel',
        'redirections_match_pair_facts',
    } <= set(REQUIRED_VALIDATION_RESULTS)

def test_loader_rejects_a_manifest_missing_a_required_validation_result(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    del manifest['validation_results']['index_roles_no_leakage']
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(
        ValueError, match=r"lacks required validation results: \['index_roles_no_leakage'\]"
    ):
        load_validated_bundle(generated_bundle)

def test_production_bundle_takes_its_members_from_its_config(
    tmp_path, hierarchy_descriptions, hierarchy_redirections, count_words
):
    roles = pl.DataFrame(
        [
            (0, '311111', 'Dog food manufacturing', 'examples'),
            (1, '311111', 'Cat food manufacturing', 'validation'),
            (2, '441111', 'New car dealers', 'examples'),
        ],
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )
    examples = {'311111': 'Dog food manufacturing', '441111': 'New car dealers'}
    descriptions = hierarchy_descriptions.with_columns(
        examples=pl.col('code').replace_strict(examples, default=None)
    )
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    roles_path = tmp_path / 'naics_index_roles.parquet'
    redirections_path = tmp_path / 'naics_redirections.parquet'
    descriptions.write_parquet(descriptions_path)
    roles.write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    cfg = SupervisionBuildConfig(
        descriptions_parquet=str(descriptions_path),
        index_roles_parquet=str(roles_path),
        redirections_parquet=str(redirections_path),
        output_root=str(tmp_path / 'bundles'),
    )

    manifest = json.loads(generate_supervision_bundle(cfg, count_tokens=count_words).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
    assert manifest['artifacts']['redirections']['row_count'] == 2
    parameters = manifest['generation_parameters']
    assert parameters['index_roles_parquet'] == str(roles_path.resolve())
    assert parameters['redirections_parquet'] == str(redirections_path.resolve())

# -------------------------------------------------------------------------------------------------
# The input-window record
# -------------------------------------------------------------------------------------------------

def _five_code_record(examples: int = 3, **changes) -> InputWindowRecord:
    '''The five-code record, counting ``examples`` present examples texts, with fields changed.'''

    channels = dict(FIVE_CODE_INPUT_WINDOW['channels'])
    channels['examples'] = {'present': examples, 'over': 0, 'share': 0.0}
    fields = {**FIVE_CODE_INPUT_WINDOW, 'channels': channels, **changes}
    return InputWindowRecord.model_validate(fields)

def test_the_input_window_record_counts_each_channels_texts_beyond_the_window(
    text_descriptions_fixture, count_words
):
    # Under the word count, 127 words make 129 tokens, one beyond the window; 126 words fit it
    texts = {'111111': ' '.join(['farming'] * 127), '111112': ' '.join(['farming'] * 126)}
    descriptions = text_descriptions_fixture.with_columns(
        description=pl.col('code').replace_strict(texts, default=pl.col('description'))
    )

    record = input_window_record(
        descriptions, 'sentence-transformers/all-MiniLM-L6-v2', count_words
    )

    assert record.window == 128
    assert record.channels['description'] == ChannelOverflow(present=5, over=1, share=0.2)
    assert record.channels['examples'] == ChannelOverflow(present=3, over=0, share=0.0)
    assert record.channels['excluded'] == ChannelOverflow(present=2, over=0, share=0.0)

def test_a_bundle_records_its_input_window(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())

    assert manifest['input_window'] == FIVE_CODE_INPUT_WINDOW
    assert load_validated_bundle(generated_bundle).manifest.input_window == _five_code_record()

@pytest.mark.parametrize(
    ('record', 'message'),
    [
        (_five_code_record(window=256), 'all-MiniLM-L6-v2 is 128 tokens, not 256'),
        (_five_code_record(channels={}), 'must cover the channels'),
        (_five_code_record(examples=4), 'counts 4 examples texts, but the descriptions hold 3'),
    ],
)
def test_bundle_refuses_an_input_window_record_that_does_not_fit(
    tmp_path, build_bundle, record, message
):
    with pytest.raises(ValueError, match=message):
        build_bundle(input_window=record)
    assert list(tmp_path.iterdir()) == []

def test_production_bundle_counts_tokens_with_the_backbones_cached_tokenizer(
    monkeypatch, hierarchy_build_config
):
    import transformers

    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        # Every text is 130 tokens, beyond the window
        return lambda texts, truncation: {'input_ids': [[0] * 130 for _ in texts]}

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', from_pretrained)

    manifest = json.loads(generate_supervision_bundle(hierarchy_build_config).read_text())

    assert calls == [('sentence-transformers/all-MiniLM-L6-v2', {'local_files_only': True})]
    assert manifest['input_window'] == {
        'backbone': 'sentence-transformers/all-MiniLM-L6-v2',
        'window': 128,
        'channels': {
            'title': {
                'present': 17,
                'over': 17,
                'share': 1.0
            },
            'description': {
                'present': 17,
                'over': 17,
                'share': 1.0
            },
            'examples': {
                'present': 0,
                'over': 0,
                'share': 0.0
            },
            'excluded': {
                'present': 2,
                'over': 2,
                'share': 1.0
            },
        },
    }

def test_loader_names_the_contract_of_a_manifest_it_cannot_parse(generated_bundle):
    # A v1 manifest predates the input-window record, so it does not parse as a v2 one
    manifest = json.loads(generated_bundle.read_text())
    manifest['contract_version'] = 'stage3-supervision-v1'
    del manifest['input_window']
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(
        ValueError,
        match='expected supervision contract stage3-supervision-v2, found stage3-supervision-v1',
    ):
        load_validated_bundle(generated_bundle)
