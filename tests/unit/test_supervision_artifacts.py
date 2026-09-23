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
from naics_embedder.supervision.artifacts import load_validated_bundle, sha256_file
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
        )
        for bundle_id in ('bundle-a', 'bundle-b')
    ]
    loaded = [json.loads(path.read_text()) for path in manifests]
    frames = [
        pl.read_parquet(
            path.parent / manifest['artifacts']['pair_facts']['path']
        )
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
    assert artifacts['training_pairs']['row_count'] == 5
    assert artifacts['training_pairs']['exclusion_count'] == 2
    assert all(manifest['validation_results'].values())
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
    with pytest.raises(ValueError, match='expected supervision contract stage3-supervision-v2'):
        load_validated_bundle(generated_bundle, expected_contract='stage3-supervision-v2')


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
            structural_distance=pl.when(pl.col('is_explicit_exclusion'))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(pl.col('structural_distance'))
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*structural distance zero'):
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
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            anchor_excludes_negative=pl.lit(False),
            negative_excludes_anchor=pl.lit(False),
            negative_is_explicit_exclusion=pl.lit(False),
            negative_semantic_target=pl.lit('unknown'),
            negative_semantic_source=pl.lit('unlabeled'),
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)
