'''
Unit tests for configuration management.

Tests Pydantic config models, YAML loading, and validation.
'''

from pathlib import Path
from typing import Any, List, Type, get_args, get_origin

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
    DirConfig,
    DistancesConfig,
    DownloadConfig,
    GraphConfig,
    OutcomePanelConfig,
    RegressorBranchRecord,
    RegressorPanelConfig,
    SamplingConfig,
    SansStaticConfig,
    StructuralPreferenceConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    load_config,
)
from tests.fixtures.regressor_panel import BRANCH_RECORD

# -------------------------------------------------------------------------------------------------
# DirConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDirConfig:
    '''Test suite for directory configuration.'''

    def test_default_values(self):
        '''Test that DirConfig has correct default values.'''

        config = DirConfig()

        assert config.checkpoint_dir == './checkpoints'
        assert config.conf_dir == './conf'
        assert config.data_dir == './data'
        assert config.docs_dir == './docs'
        assert config.log_dir == './logs'
        assert config.output_dir == './outputs'

    def test_custom_values(self):
        '''Test that custom values override defaults.'''

        config = DirConfig(
            checkpoint_dir='/custom/checkpoints',
            data_dir='/custom/data',
        )

        assert config.checkpoint_dir == '/custom/checkpoints'
        assert config.data_dir == '/custom/data'
        # Other fields should still have defaults
        assert config.conf_dir == './conf'

    def test_serialization(self):
        '''Test that config can be serialized to dict.'''

        config = DirConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert 'checkpoint_dir' in config_dict
        assert 'data_dir' in config_dict

# -------------------------------------------------------------------------------------------------
# DownloadConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDownloadConfig:
    '''Test suite for download configuration.'''

    def test_default_output_parquet(self) -> None:
        '''Test default output parquet path.'''

        config = DownloadConfig()

        assert config.output_parquet == './data/naics_descriptions.parquet'

    def test_custom_output_parquet(self):
        '''Test custom output parquet path.'''

        config = DownloadConfig(output_parquet='/custom/path/data.parquet')

        assert config.output_parquet == '/custom/path/data.parquet'

    def test_validation(self):
        '''Test that invalid configuration raises validation error.'''

        # output_parquet should be a string, not a number
        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='12345')

    def test_output_parquet_extension_validation(self):
        '''Test that output_parquet must be a parquet file.'''

        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='./data/output.csv')
        with pytest.raises(ValidationError):
            DownloadConfig(index_roles_parquet='./data/roles.csv')

    def test_yaml_matches_defaults(self):
        '''The shipped YAML pins the index file and names the committed role table.'''

        cfg = load_config(DownloadConfig, 'data/download.yaml')

        assert cfg == DownloadConfig()
        assert cfg.index_sha256 == (
            '6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63'
        )
        assert cfg.index_roles_csv == './conf/data/index_roles.csv'
        assert cfg.source_dir is None

    def test_index_sha256_must_be_a_hex_digest(self):
        with pytest.raises(ValidationError):
            DownloadConfig(index_sha256='not-a-digest')

# -------------------------------------------------------------------------------------------------
# DistancesConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDistancesConfig:
    '''Test suite for distances configuration.'''

    def test_required_fields(self):
        '''Test that config can be created with required fields.'''

        config = DistancesConfig(
            input_parquet='./data/input.parquet',
            distances_parquet='./data/distances.parquet',
            distance_matrix_parquet='./data/matrix.parquet',
        )

        assert config.input_parquet == './data/input.parquet'
        assert config.distances_parquet == './data/distances.parquet'
        assert config.distance_matrix_parquet == './data/matrix.parquet'

    def test_missing_required_field_uses_defaults(self):
        '''Test that missing fields fall back to defaults.'''

        config = DistancesConfig()

        assert config.input_parquet == './data/naics_descriptions.parquet'
        assert config.distances_parquet == './data/naics_distances.parquet'
        assert config.distance_matrix_parquet == './data/naics_distance_matrix.parquet'

# -------------------------------------------------------------------------------------------------
# Config Loading Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadConfig:
    '''Test suite for config loading function.'''

    def test_load_from_valid_yaml(self, tmp_path):
        '''Test loading config from valid YAML file.'''

        # Create temporary YAML file
        yaml_path = tmp_path / 'conf' / 'test_config.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            'checkpoint_dir': '/test/checkpoints',
            'data_dir': '/test/data',
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(DirConfig, yaml_path)

        assert config.checkpoint_dir == '/test/checkpoints'
        assert config.data_dir == '/test/data'

    def test_load_with_conf_prefix(self, tmp_path, monkeypatch):
        '''Test that conf/ prefix is added if not present.'''

        yaml_path = tmp_path / 'conf' / 'test.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/test'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        # Load without conf/ prefix
        config = load_config(DirConfig, 'test.yaml')

        assert config.checkpoint_dir == '/test'

    def test_load_missing_file_uses_defaults(self, tmp_path):
        '''Test that missing file falls back to default values.'''

        # Try to load non-existent file
        config = load_config(DirConfig, 'nonexistent.yaml')

        # Should use default values
        assert config.checkpoint_dir == './checkpoints'
        assert config.data_dir == './data'

    def test_load_empty_yaml(self, tmp_path, monkeypatch):
        '''Test loading empty YAML file uses defaults.'''

        yaml_path = tmp_path / 'conf' / 'empty.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty file
        yaml_path.touch()

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'empty.yaml')

        # Should use defaults
        assert config.checkpoint_dir == './checkpoints'

    def test_load_partial_yaml(self, tmp_path, monkeypatch):
        '''Test loading YAML with partial fields.'''

        yaml_path = tmp_path / 'conf' / 'partial.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/custom/checkpoints'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'partial.yaml')

        # Custom field
        assert config.checkpoint_dir == '/custom/checkpoints'
        # Default fields
        assert config.data_dir == './data'
        assert config.log_dir == './logs'

    def test_load_custom_relative_path(self, tmp_path, monkeypatch):
        '''Test loading config from a relative path outside conf/.'''

        yaml_path = tmp_path / 'custom' / 'dir' / 'custom.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {'checkpoint_dir': '/custom/path'}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        config = load_config(DirConfig, 'custom/dir/custom.yaml')

        assert config.checkpoint_dir == '/custom/path'

# -------------------------------------------------------------------------------------------------
# Validation Tests
# -------------------------------------------------------------------------------------------------

def _error_locs_and_types(excinfo: pytest.ExceptionInfo[ValidationError]) -> list:
    '''(loc, type) for every error in a raised ValidationError.'''

    return [(error['loc'], error['type']) for error in excinfo.value.errors()]

@pytest.mark.unit
class TestConfigValidation:
    '''Test suite for configuration validation.'''

    def test_invalid_type_raises_error(self):
        '''Test that invalid field types raise validation errors.'''

        with pytest.raises(ValidationError):
            DirConfig(checkpoint_dir=12345)  # type: ignore[arg-type]  # Should be string

    def test_extra_fields_rejected(self):
        '''A key a section does not define raises instead of being silently dropped.'''

        with pytest.raises(ValidationError) as excinfo:
            DirConfig(checkpoint_dir='./checkpoints', extra_field='value')

        assert _error_locs_and_types(excinfo) == [(('extra_field', ), 'extra_forbidden')]

    def test_config_rejects_an_unknown_top_level_key(self):
        with pytest.raises(ValidationError) as excinfo:
            Config(bogus_key=1)

        assert _error_locs_and_types(excinfo) == [(('bogus_key', ), 'extra_forbidden')]

    def test_config_rejects_an_unknown_nested_key(self):
        with pytest.raises(ValidationError) as excinfo:
            Config.model_validate({'training': {'learnig_rate': 1e-4}})

        assert _error_locs_and_types(excinfo) == [(('training', 'learnig_rate'), 'extra_forbidden')]

    @pytest.mark.parametrize(
        ('override', 'loc'),
        [
            ('trainig.learning_rate', ('trainig', )),
            ('training.learnig_rate', ('training', 'learnig_rate')),
        ],
    )
    def test_override_rejects_a_misspelled_path(self, override, loc):
        '''A typo'd CLI override fails instead of training on the default value.'''

        with pytest.raises(ValidationError) as excinfo:
            Config().override({override: 1e-4})

        assert _error_locs_and_types(excinfo) == [(loc, 'extra_forbidden')]

    def test_from_yaml_rejects_the_graph_config(self):
        '''conf/graph.yaml loaded by mistake raises instead of keeping only its seed.'''

        with pytest.raises(ValidationError) as excinfo:
            Config.from_yaml('conf/graph.yaml')

        assert {error['type'] for error in excinfo.value.errors()} == {'extra_forbidden'}

    def test_field_validation(self):
        '''Test that field validators work correctly.'''

        # This test depends on whether any validators are defined
        config = DirConfig(checkpoint_dir='  ./checkpoints  ')

        # Should accept the value (may strip whitespace depending on validators)
        assert isinstance(config.checkpoint_dir, str)

def _models_in(annotation: Any) -> List[Type[BaseModel]]:
    '''The Pydantic models an annotation names, unwrapping Optional, List, Dict and the like.'''

    # Check get_origin first: on Python 3.10, isinstance(list[Model], type) is True, so an
    # isinstance-first check stops at the alias and never finds the Model inside it.
    if get_origin(annotation) is None:
        is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        return [annotation] if is_model else []
    return [model for arg in get_args(annotation) for model in _models_in(arg)]

def _models_reachable_from(root: Type[BaseModel]) -> List[Type[BaseModel]]:
    '''root plus every Pydantic model nested under its fields, at any depth.'''

    found: List[Type[BaseModel]] = []
    pending = [root]
    while pending:
        model = pending.pop()
        if model not in found:
            found.append(model)
            for field in model.model_fields.values():
                pending.extend(_models_in(field.annotation))
    return found

@pytest.mark.unit
def test_every_section_under_config_forbids_unknown_keys():
    '''A section added later cannot quietly bring back silent key-dropping.'''

    permissive = [
        model.__name__ for model in _models_reachable_from(Config)
        if model.model_config.get('extra') != 'forbid'
    ]

    assert permissive == []

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigIntegration:
    '''Integration tests for configuration system.'''

    def test_load_multiple_configs(self, tmp_path, monkeypatch):
        '''Test loading multiple different config types.'''

        # Create directory
        conf_dir = tmp_path / 'conf'
        conf_dir.mkdir()

        # DirConfig
        dir_yaml = conf_dir / 'dir.yaml'
        with open(dir_yaml, 'w') as f:
            yaml.dump({'checkpoint_dir': '/test/checkpoints'}, f)

        # DownloadConfig
        download_yaml = conf_dir / 'download.yaml'
        with open(download_yaml, 'w') as f:
            yaml.dump({'output_parquet': '/test/output.parquet'}, f)

        monkeypatch.chdir(tmp_path)

        # Load both
        dir_config = load_config(DirConfig, 'dir.yaml')
        download_config = load_config(DownloadConfig, 'download.yaml')

        assert dir_config.checkpoint_dir == '/test/checkpoints'
        assert download_config.output_parquet == '/test/output.parquet'

    def test_config_serialization_roundtrip(self):
        '''Test that config can be serialized and deserialized.'''

        original = DirConfig(checkpoint_dir='/test/checkpoints', data_dir='/test/data')

        # Serialize
        config_dict = original.model_dump()

        # Deserialize
        restored = DirConfig(**config_dict)

        assert restored.checkpoint_dir == original.checkpoint_dir
        assert restored.data_dir == original.data_dir

    def test_config_json_export(self):
        '''Test that config can be exported to JSON.'''

        config = DirConfig(checkpoint_dir='/test')

        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert '/test' in json_str
        assert 'checkpoint_dir' in json_str

# -------------------------------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigErrorHandling:
    '''Test suite for configuration error handling.'''

    def test_malformed_yaml(self, tmp_path, monkeypatch):
        '''Test handling of malformed YAML file.'''

        yaml_path = tmp_path / 'conf' / 'malformed.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write malformed YAML
        with open(yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [')

        monkeypatch.chdir(tmp_path)

        # Should handle gracefully (either raise or use defaults)
        with pytest.raises(Exception):
            load_config(DirConfig, 'malformed.yaml')

    def test_yaml_with_invalid_structure(self, tmp_path, monkeypatch):
        '''Test YAML with structure that doesn't match config model.'''

        yaml_path = tmp_path / 'conf' / 'invalid_structure.yaml'
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML with wrong types
        config_data = {'checkpoint_dir': ['this', 'should', 'be', 'string']}

        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        monkeypatch.chdir(tmp_path)

        # Should raise validation error
        with pytest.raises(ValidationError):
            load_config(DirConfig, 'invalid_structure.yaml')

@pytest.mark.unit
class TestSamplingConfig:

    def test_sampling_defaults(self):
        cfg = SamplingConfig()

        assert cfg.strategy == 'sadc'
        assert cfg.sans_static.near_distance_threshold == 4.0

    def test_invalid_bucket_weights_raise(self):
        with pytest.raises(ValidationError):
            SamplingConfig(
                sans_static=SansStaticConfig(
                    near_bucket_weight=0.0,
                    far_bucket_weight=0.0,
                )
            )

@pytest.mark.unit
class TestSupervisionBuildConfig:
    '''The supervision bundle build configuration.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(SupervisionBuildConfig, 'data/supervision.yaml')

        # The shipped build carries the index roles; the default (for fixtures) does not
        assert cfg == SupervisionBuildConfig(index_roles_parquet='./data/naics_index_roles.parquet')
        assert cfg.contract_version == 'stage3-supervision-v1'
        assert cfg.relation_id['cross_sector'] == 99
        assert cfg.output_root == './data/supervision/stage3-supervision-v1'

    def test_rejects_other_contract_versions(self):
        with pytest.raises(ValidationError):
            SupervisionBuildConfig(contract_version='legacy')

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            SupervisionBuildConfig(rank_order_weight=0.35)

@pytest.mark.unit
class TestOutcomePanelConfig:
    '''How index-entry roles are drawn (roadmap D4), and the selection log.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')

        assert cfg == OutcomePanelConfig()
        assert cfg.fractions == {
            'examples': 0.30,
            'training': 0.35,
            'validation': 0.20,
            'test': 0.15,
        }
        assert cfg.seed == 20260924
        assert cfg.selection_log == './logs/selection_log.jsonl'

    @pytest.mark.parametrize(
        'fractions',
        [
            {
                'examples': 0.30,
                'training': 0.35,
                'validation': 0.35
            },
            {
                'examples': 0.30,
                'training': 0.35,
                'validation': 0.20,
                'test': 0.10
            },
            {
                'examples': 0.60,
                'training': 0.35,
                'validation': 0.20,
                'test': -0.15
            },
        ],
    )
    def test_fractions_name_every_role_and_sum_to_one(self, fractions):
        with pytest.raises(ValidationError):
            OutcomePanelConfig(fractions=fractions)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            OutcomePanelConfig(test_fraction=0.15)

@pytest.mark.unit
class TestRegressorPanelConfig:
    '''The regressor panel (roadmap Stage 3): QCEW pins, the held-out draw, fitting, the record.'''

    def test_yaml_matches_defaults_but_for_the_branch_record(self):
        cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')

        assert cfg.model_copy(update={'branch_record': None}) == RegressorPanelConfig()
        assert cfg.branch_record is not None
        assert sorted(cfg.qcew_sha256) == [f'{year}_US000_annual.csv' for year in range(2022, 2026)]
        assert cfg.alphas == sorted(set(cfg.alphas))
        assert (cfg.seed, cfg.heldout_fraction, cfg.fold_seed) == (20260924, 0.2, 20260924)
        assert (cfg.folds, cfg.repeats, cfg.inner_folds, cfg.min_groups) == (5, 5, 5, 10)
        assert cfg.heldout_groups_csv == './conf/data/regressor_heldout_groups.csv'
        assert cfg.selection_log == './logs/selection_log.jsonl'

    def test_the_text_only_comparator_reads_like_the_arm(self):
        # D9: the arm's own backbone, at the arm's tokenization length
        arm = yaml.safe_load(Path('conf/config.yaml').read_text())
        text_only = RegressorPanelConfig().text_only

        assert text_only.backbone == arm['model']['base_model_name']
        assert text_only.max_length == arm['data_loader']['tokenization']['max_length']

    @pytest.mark.parametrize('fraction', [0.0, 1.0])
    def test_the_held_out_fraction_lies_strictly_between_zero_and_one(self, fraction):
        with pytest.raises(ValidationError):
            RegressorPanelConfig(heldout_fraction=fraction)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            RegressorPanelConfig(heldout_share=0.2)
        with pytest.raises(ValidationError):
            RegressorBranchRecord(**BRANCH_RECORD, rule='plan 3')

    def test_the_branch_record_has_no_defaults(self):
        with pytest.raises(ValidationError):
            RegressorBranchRecord(branch='A')
        with pytest.raises(ValidationError):
            RegressorBranchRecord(**{**BRANCH_RECORD, 'branch': 'D'})

# -------------------------------------------------------------------------------------------------
# Repaired Stage-3 runtime configuration
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def valid_config_dict():
    return yaml.safe_load(Path('conf/config.yaml').read_text())

def test_base_config_parses_as_repaired_pre_generation(valid_config_dict):
    cfg = Config.model_validate(valid_config_dict)

    assert cfg.supervision.mode == 'repaired'
    assert cfg.supervision.manifest_path is None
    assert cfg.supervision.contract_version == 'stage3-supervision-v1'
    assert cfg.loss.structural_preference == StructuralPreferenceConfig()
    assert cfg.loss.rank_order_weight is None
    assert cfg.data_loader.streaming.phase1_exclusion_weight is None

def test_repaired_config_rejects_legacy_rank_key(valid_config_dict):
    valid_config_dict['supervision'] = {
        'mode': 'repaired',
        'manifest_path': '/tmp/bundle/manifest.json',
    }
    valid_config_dict['loss']['rank_order_weight'] = 0.35

    with pytest.raises(
        ValidationError,
        match='rank_order_weight.*structural_preference',
    ):
        Config.model_validate(valid_config_dict)

def test_repaired_config_rejects_high_exclusion_weight(valid_config_dict):
    valid_config_dict['supervision'] = {
        'mode': 'repaired',
        'manifest_path': '/tmp/bundle/manifest.json',
    }
    valid_config_dict['data_loader']['streaming']['phase1_exclusion_weight'] = 100.0

    with pytest.raises(
        ValidationError,
        match='phase1_exclusion_weight.*one-slot exclusion quota',
    ):
        Config.model_validate(valid_config_dict)

def test_overrides_cannot_reintroduce_legacy_keys_in_repaired_mode():
    with pytest.raises(ValidationError, match='rank_order_weight'):
        Config().override({'loss.rank_order_weight': 0.35})

def test_legacy_containment_is_the_only_mode_accepting_legacy_keys(valid_config_dict):
    valid_config_dict['supervision'] = {'mode': 'legacy_containment'}
    valid_config_dict['loss']['rank_order_weight'] = 0.35
    valid_config_dict['data_loader']['streaming']['phase1_exclusion_weight'] = 100.0

    cfg = Config.model_validate(valid_config_dict)

    assert cfg.supervision.mode == 'legacy_containment'
    assert cfg.loss.rank_order_weight == 0.35

@pytest.mark.parametrize(
    'supervision',
    [
        {
            'mode': 'legacy'
        },
        {
            'contract_version': 'stage3-supervision-v0'
        },
        {
            'manifest': '/tmp/bundle/manifest.json'
        },
    ],
)
def test_supervision_runtime_config_rejects_unknown_values(supervision):
    with pytest.raises(ValidationError):
        SupervisionRuntimeConfig(**supervision)

def test_checkpoint_load_modes_are_explicit():
    assert [mode.value for mode in CheckpointLoadMode] == ['exact', 'weights_only']

@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('temperature', 0.0, 'temperature'),
        ('margin', -0.1, 'margin'),
        ('tie_tolerance', -0.1, 'tie_tolerance'),
    ],
)
def test_structural_preference_config_bounds(field, value, message):
    data = {
        'weight': 0.35,
        'margin': 0.1,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        StructuralPreferenceConfig(**data)

# -------------------------------------------------------------------------------------------------
# GraphConfig Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGraphConfig:
    '''The HGCN configuration read from conf/graph.yaml.'''

    def test_from_yaml_rejects_a_misspelled_key(self, tmp_path):
        yaml_path = tmp_path / 'graph.yaml'
        yaml_path.write_text(yaml.dump({'supervision_manifest': '/tmp/bundle/manifest.json'}))

        with pytest.raises(ValidationError) as excinfo:
            GraphConfig.from_yaml(str(yaml_path))

        assert [(error['loc'], error['type']) for error in excinfo.value.errors()] == [
            (('supervision_manifest', ), 'extra_forbidden')
        ]

    def test_from_yaml_rejects_the_text_model_config(self):
        with pytest.raises(ValidationError) as excinfo:
            GraphConfig.from_yaml('conf/config.yaml')

        assert {error['type'] for error in excinfo.value.errors()} == {'extra_forbidden'}

    def test_checked_in_graph_yaml_sets_only_graph_config_fields(self):
        cfg = GraphConfig.from_yaml('conf/graph.yaml')

        assert cfg.model_fields_set == set(yaml.safe_load(Path('conf/graph.yaml').read_text()))
