from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    ArtifactRecord,
    SemanticSource,
    SemanticTarget,
    SupervisionManifest,
)

def _manifest() -> SupervisionManifest:
    artifact_file = ArtifactFile(path='naics_codebook.parquet', sha256='a' * 64, row_count=3)
    return SupervisionManifest(
        contract_version=CONTRACT_VERSION,
        bundle_id='bundle-123',
        generated_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
        generator_revision='abc123',
        naics_vintage=2022,
        codebook_order=('111111', '111112', '111113'),
        codebook_fingerprint='b' * 64,
        description_fingerprint='c' * 64,
        exclusion_fingerprint='d' * 64,
        generation_parameters={'seed': 42},
        structural_relation_ids={'child': 1, 'cross_sector': 99},
        artifacts={
            'codebook': ArtifactRecord(
                path='naics_codebook.parquet',
                schema_version='codebook-v1',
                row_count=3,
                exclusion_count=0,
                files=(artifact_file,),
            )
        },
        validation_results={'codebook_unique': True},
    )


def test_manifest_round_trip_preserves_contract_identity(tmp_path):
    manifest = _manifest()
    path = tmp_path / 'manifest.json'
    path.write_text(manifest.model_dump_json(indent=2))

    restored = SupervisionManifest.model_validate_json(path.read_text())

    assert restored.contract_version == 'stage3-supervision-v1'
    assert restored.codebook_order == ('111111', '111112', '111113')
    assert restored.artifacts['codebook'].files[0].sha256 == 'a' * 64
    assert SemanticTarget.UNRELATED.value == 'unrelated'
    assert SemanticSource.EXPLICIT_EXCLUSION.value == 'explicit_exclusion'


def test_manifest_rejects_parent_traversal():
    manifest = _manifest().model_dump()
    manifest['artifacts']['codebook']['path'] = '../outside.parquet'

    with pytest.raises(ValidationError, match='relative bundle path'):
        SupervisionManifest.model_validate(manifest)
