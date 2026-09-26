import json
import uuid

import polars as pl
import pyarrow.parquet as pq
import pytest

from naics_embedder.data.create_triplets import build_training_pairs
from naics_embedder.data.supervision_bundle import (
    build_codebook,
    build_pair_facts,
    codebook_fingerprint,
    distance_matrix_from_pair_facts,
    generate_supervision_bundle,
    generate_supervision_bundle_from_frames,
    relation_matrix_from_pair_facts,
)
from naics_embedder.supervision.artifacts import (
    load_validated_bundle,
    sha256_file,
    validate_training_pairs_members,
)
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.utils.config import SupervisionBuildConfig

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

def test_bundle_writes_manifest_last_with_matching_parquet_metadata(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifest_path = generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )

    manifest = json.loads(manifest_path.read_text())
    codebook_path = manifest_path.parent / manifest['artifacts']['codebook']['path']
    metadata = pq.read_metadata(codebook_path).metadata

    assert manifest_path.name == 'manifest.json'
    assert manifest['contract_version'] == CONTRACT_VERSION
    assert metadata[b'naics_embedder.contract_version'].decode() == CONTRACT_VERSION
    assert metadata[b'naics_embedder.bundle_id'].decode() == 'bundle-a'
    assert metadata[b'naics_embedder.schema_version'].decode() == 'codebook-v1'

def test_bundle_never_overwrites_an_existing_generation(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    kwargs = {
        'output_root': tmp_path,
        'bundle_id': 'bundle-a',
        'generator_revision': 'revision-a',
        'naics_vintage': 2022,
        'descriptions': descriptions_fixture,
        'pair_facts': pair_facts_fixture,
    }
    generate_supervision_bundle_from_frames(**kwargs)

    with pytest.raises(FileExistsError, match='bundle-a'):
        generate_supervision_bundle_from_frames(**kwargs)

def test_failed_validation_publishes_no_manifest(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    inconsistent = pair_facts_fixture.with_columns(is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='broken',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=inconsistent,
        )

    assert not (tmp_path / 'broken' / 'manifest.json').exists()

def test_two_generated_bundles_have_equal_logical_frames_but_distinct_ids(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifests = [
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id=bundle_id,
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=pair_facts_fixture,
        ) for bundle_id in ('bundle-a', 'bundle-b')
    ]
    loaded = [json.loads(path.read_text()) for path in manifests]
    frames = [
        pl.read_parquet(path.parent / manifest['artifacts']['pair_facts']['path'])
        for path, manifest in zip(manifests, loaded)
    ]

    assert loaded[0]['bundle_id'] != loaded[1]['bundle_id']
    assert frames[0].equals(frames[1])

def test_bundle_records_every_artifact_member_with_hash_and_contract_metadata(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifest_path = generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
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

def test_failed_generation_leaves_no_staging_directory(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    with pytest.raises(ValueError):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='broken',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=pair_facts_fixture.with_columns(structural_distance=pl.lit(0.0)),
        )

    assert list(tmp_path.iterdir()) == []

def test_production_bundle_uses_a_uuid_and_the_descriptions_file_hash(
    tmp_path, hierarchy_descriptions_parquet
):
    cfg = SupervisionBuildConfig(
        descriptions_parquet=hierarchy_descriptions_parquet,
        output_root=str(tmp_path / 'bundles'),
    )

    manifest_path = generate_supervision_bundle(cfg)
    manifest = json.loads(manifest_path.read_text())

    assert str(uuid.UUID(manifest['bundle_id'])) == manifest['bundle_id']
    assert manifest_path.parent.name == manifest['bundle_id']
    assert manifest['description_fingerprint'] == sha256_file(hierarchy_descriptions_parquet)
    assert manifest['structural_relation_ids']['cross_sector'] == 99
    assert manifest['artifacts']['pair_facts']['row_count'] == 17 * 16 // 2
    assert manifest['artifacts']['pair_facts']['exclusion_count'] == 2

def test_the_unary_pairs_are_flagged_and_never_generated_positives(
    tmp_path, hierarchy_descriptions_parquet
):
    manifest_path = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    bundle = load_validated_bundle(manifest_path)
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

def test_pair_facts_refuse_a_unary_flag_the_codes_do_not_support(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    # '111111' and '111112' are six-digit siblings, not a five-digit code and its only child
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='unary_pair is wrong on 1 pairs, e.g. 111111/111112'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='bundle-a',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=flagged,
        )

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
# The optional index-roles member
# -------------------------------------------------------------------------------------------------

def test_bundle_carries_the_index_roles_after_checking_them(
    generated_bundle_with_roles, index_roles_fixture
):
    manifest = json.loads(generated_bundle_with_roles.read_text())
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
    bundle = load_validated_bundle(generated_bundle_with_roles)
    assert pl.read_parquet(bundle.artifact_path('index_roles')).equals(index_roles_fixture)

def test_bundle_refuses_an_examples_channel_holding_queries(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    stale = text_descriptions_fixture.with_columns(
        examples=pl.when(pl.col('code') == '111111').then(
            pl.lit('Soybean farming; Edamame farming')
        ).otherwise('examples')
    )

    with pytest.raises(ValueError, match='examples channel other than'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='stale',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=stale,
            pair_facts=pair_facts_fixture,
            index_roles=index_roles_fixture,
        )
    assert list(tmp_path.iterdir()) == []

def test_bundle_refuses_a_held_out_query_matching_training_text(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    # Entry 1 (validation) becomes another code's title
    leaky = index_roles_fixture.with_columns(
        text=pl.when(pl.col('entry_id') == 1).then(pl.lit('Industry 222222')).otherwise('text')
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='leaky',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=text_descriptions_fixture,
            pair_facts=pair_facts_fixture,
            index_roles=leaky,
        )

def test_loader_rejects_an_index_entry_with_two_roles(generated_bundle_with_roles):
    _rewrite_member(
        generated_bundle_with_roles,
        'index_roles',
        lambda frame: frame.with_columns(entry_id=pl.lit(0, pl.Int64)),
    )

    with pytest.raises(ValueError, match='index_roles .*more than one role'):
        load_validated_bundle(generated_bundle_with_roles)

def test_production_bundle_takes_the_index_roles_from_its_config(tmp_path, hierarchy_descriptions):
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
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.lit(None, pl.Utf8),
    )
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions.write_parquet(descriptions_path)
    roles.write_parquet(roles_path)
    cfg = SupervisionBuildConfig(
        descriptions_parquet=str(descriptions_path),
        index_roles_parquet=str(roles_path),
        output_root=str(tmp_path / 'bundles'),
    )

    manifest = json.loads(generate_supervision_bundle(cfg).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
    assert manifest['generation_parameters']['index_roles_parquet'] == str(roles_path.resolve())
