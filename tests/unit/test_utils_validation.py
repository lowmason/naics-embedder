import polars as pl
import pytest
import torch

from naics_embedder.utils.config import Config, SupervisionBuildConfig, TokenizationConfig
from naics_embedder.utils.validation import (
    ValidationError,
    require_valid_config,
    require_valid_supervision_bundle,
    validate_data_paths,
    validate_descriptions_schema,
    validate_distances_schema,
    validate_tokenization_cache,
    validate_training_config,
)

def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('test')
    return str(path)

@pytest.mark.unit
def test_validate_data_paths_all_present(tmp_path):
    cfg = Config()
    streaming = cfg.data_loader.streaming
    streaming.descriptions_parquet = _touch(tmp_path / 'descriptions.parquet')
    streaming.distances_parquet = _touch(tmp_path / 'distances.parquet')
    streaming.distance_matrix_parquet = _touch(tmp_path / 'distance_matrix.parquet')
    streaming.relations_parquet = _touch(tmp_path / 'relations.parquet')
    triplets_dir = tmp_path / 'triplets'
    triplets_dir.mkdir()
    (triplets_dir / 'batch.parquet').write_text('rows')
    streaming.triplets_parquet = str(triplets_dir)

    result = validate_data_paths(cfg)
    assert result.valid

@pytest.mark.unit
def test_validate_data_paths_missing_file(tmp_path):
    cfg = Config()
    streaming = cfg.data_loader.streaming
    streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')

    result = validate_data_paths(cfg)
    assert result.valid is False
    assert any('Descriptions file not found' in err for err in result.errors)

@pytest.mark.unit
def test_validate_descriptions_schema_success(tmp_path):
    path = tmp_path / 'descriptions.parquet'
    pl.DataFrame(
        {
            'index': [0],
            'code': ['11'],
            'level': [2],
            'title': ['Manufacturing'],
            'description': ['Test'],
        }
    ).write_parquet(path)

    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(path)
    result = validate_descriptions_schema(cfg)
    assert result.valid

@pytest.mark.unit
def test_validate_distances_schema_missing_column(tmp_path):
    path = tmp_path / 'distances.parquet'
    pl.DataFrame({'idx_i': [0], 'idx_j': [1]}).write_parquet(path)
    cfg = Config()
    cfg.data_loader.streaming.distances_parquet = str(path)

    result = validate_distances_schema(cfg)
    assert result.valid is False
    assert any('missing columns' in err for err in result.errors)

@pytest.mark.unit
def test_validate_tokenization_cache_missing_returns_warning(tmp_path):
    cfg = Config()
    token_cfg = TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.tokenization.max_length,
        output_path=str(tmp_path / 'cache.pt'),
    )

    result = validate_tokenization_cache(cfg, tokenization_cfg=token_cfg)
    assert result.valid
    assert result.warnings

@pytest.mark.unit
def test_validate_tokenization_cache_structure_error(tmp_path):
    cache_path = tmp_path / 'cache.pt'
    torch.save({0: {'code': '11', 'title': {}, 'description': {}}}, cache_path)

    cfg = Config()
    token_cfg = TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.tokenization.max_length,
        output_path=str(cache_path),
    )

    result = validate_tokenization_cache(cfg, tokenization_cfg=token_cfg)
    assert result.valid is False
    assert any('wrong structure' in err for err in result.errors)

@pytest.mark.unit
def test_validate_training_config_reports_errors(tmp_path):
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')

    result = validate_training_config(cfg)
    assert result.valid is False
    assert result.errors

@pytest.mark.unit
def test_require_valid_config_raises_validation_error(tmp_path):
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')
    with pytest.raises(ValidationError):
        require_valid_config(cfg)

# -------------------------------------------------------------------------------------------------
# Repaired Stage-3 supervision gate
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def production_bundle(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle

    return generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )

def _repaired_cfg(manifest_path, descriptions_path) -> Config:
    cfg = Config()
    cfg.supervision.manifest_path = str(manifest_path) if manifest_path else None
    cfg.data_loader.streaming.descriptions_parquet = str(descriptions_path)
    return cfg

@pytest.mark.unit
def test_supervision_gate_requires_a_manifest_in_repaired_mode(hierarchy_descriptions_parquet):
    cfg = _repaired_cfg(None, hierarchy_descriptions_parquet)

    with pytest.raises(ValidationError, match='supervision.manifest_path') as excinfo:
        require_valid_supervision_bundle(cfg)

    assert any('naics-embedder data supervision' in step for step in excinfo.value.remediation)

@pytest.mark.unit
def test_supervision_gate_skips_explicit_legacy_containment():
    cfg = Config.model_validate({'supervision': {'mode': 'legacy_containment'}})

    assert require_valid_supervision_bundle(cfg) is None

@pytest.mark.unit
def test_supervision_gate_returns_the_validated_bundle(
    production_bundle, hierarchy_descriptions_parquet
):
    cfg = _repaired_cfg(production_bundle, hierarchy_descriptions_parquet)

    bundle = require_valid_supervision_bundle(cfg)

    assert bundle is not None
    assert bundle.manifest_path == production_bundle.resolve()

@pytest.mark.unit
def test_supervision_gate_rejects_other_descriptions(
    tmp_path, production_bundle, hierarchy_descriptions
):
    other = tmp_path / 'other_descriptions.parquet'
    hierarchy_descriptions.head(5).write_parquet(other)
    cfg = _repaired_cfg(production_bundle, other)

    with pytest.raises(ValidationError, match='does not match the supervision bundle'):
        require_valid_supervision_bundle(cfg)

@pytest.mark.unit
def test_supervision_gate_rejects_a_tampered_bundle(
    production_bundle, hierarchy_descriptions_parquet
):
    (production_bundle.parent / 'naics_codebook.parquet').write_bytes(b'tampered')
    cfg = _repaired_cfg(production_bundle, hierarchy_descriptions_parquet)

    with pytest.raises(ValueError):
        require_valid_supervision_bundle(cfg)

@pytest.mark.unit
def test_repaired_data_paths_do_not_require_legacy_artifacts(tmp_path):
    # Structural facts and training pairs come from the bundle, which the supervision gate
    # validates; legacy long-form paths are only read in legacy containment.
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = _touch(tmp_path / 'descriptions.parquet')
    cfg.data_loader.streaming.distances_parquet = str(tmp_path / 'missing_distances.parquet')
    cfg.data_loader.streaming.triplets_parquet = str(tmp_path / 'missing_triplets')

    assert validate_data_paths(cfg).valid

    cfg.supervision.mode = 'legacy_containment'
    result = validate_data_paths(cfg)
    assert result.valid is False
    assert any('Distances file not found' in err for err in result.errors)
