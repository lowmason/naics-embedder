'''
Unit tests for configuration management.

Tests Pydantic config models, YAML loading, and validation.
'''

from pathlib import Path
from typing import Any, List, Type, get_args, get_origin

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from naics_embedder.text_model.dataloader.datamodule import DEFAULT_QUERIES_PER_STEP
from naics_embedder.text_model.loss import LogitScale
from naics_embedder.utils import config as config_module
from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
    DataLoaderConfig,
    DecisionConfig,
    DirConfig,
    DistancesConfig,
    DownloadConfig,
    GraphConfig,
    OutcomePanelConfig,
    RegressorBranchRecord,
    RegressorPanelConfig,
    StreamingConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    TextOnlyConfig,
    TokenizationConfig,
    TrainerConfig,
    load_config,
    parse_override_value,
)
from naics_embedder.utils.training import parse_config_overrides
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
        with pytest.raises(ValidationError):
            DownloadConfig(redirections_parquet='./data/redirections.csv')

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
class TestSupervisionBuildConfig:
    '''The supervision bundle build configuration.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(SupervisionBuildConfig, 'data/supervision.yaml')

        assert cfg == SupervisionBuildConfig()
        assert cfg.index_roles_parquet == './data/naics_index_roles.parquet'
        assert cfg.redirections_parquet == './data/naics_redirections.parquet'
        assert cfg.contract_version == 'stage3-supervision-v2'
        assert cfg.relation_id['cross_sector'] == 99
        assert cfg.output_root == './data/supervision/stage3-supervision-v2'

    def test_rejects_other_contract_versions(self):
        with pytest.raises(ValidationError):
            SupervisionBuildConfig(contract_version='legacy')

    def test_backbone_is_the_training_backbone(self, valid_config_dict):
        # The manifest records the window of the backbone that training reads
        assert SupervisionBuildConfig().backbone == valid_config_dict['model']['base_model_name']

    def test_rejects_a_backbone_without_a_recorded_window(self):
        with pytest.raises(ValidationError, match='no trained input window is recorded'):
            SupervisionBuildConfig(backbone='bert-base-uncased')

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
        assert text_only.max_length == arm['data_loader']['streaming']['max_length']

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

@pytest.mark.unit
class TestDecisionConfig:
    '''Req 5's decision procedure (roadmap Stage 4): the paired bootstrap and the seed floor.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(DecisionConfig, 'data/decision.yaml')

        assert cfg == DecisionConfig()
        assert (cfg.replicates, cfg.bootstrap_seed, cfg.min_seeds) == (10000, 20260924, 5)

    def test_the_seed_floor_is_req_5s(self):
        # Req 5: "Each arm runs at least 5 seeds"
        with pytest.raises(ValidationError):
            DecisionConfig(min_seeds=4)

    def test_rejects_too_few_replicates_and_unknown_keys(self):
        with pytest.raises(ValidationError):
            DecisionConfig(replicates=999)
        with pytest.raises(ValidationError):
            DecisionConfig(seeds=5)

# -------------------------------------------------------------------------------------------------
# Repaired Stage-3 runtime configuration
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def valid_config_dict():
    return yaml.safe_load(Path('conf/config.yaml').read_text())

def test_base_config_parses_as_repaired_pre_generation(valid_config_dict):
    cfg = Config.model_validate(valid_config_dict)

    assert cfg.supervision.manifest_path is None
    assert cfg.supervision.contract_version == 'stage3-supervision-v2'

def test_the_model_fuses_by_masked_mean_at_dimension_16(valid_config_dict):
    cfg = Config.model_validate(valid_config_dict)

    assert (cfg.model.fusion, cfg.model.dimension) == ('masked_mean', 16)

@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('model.fusion', 'attention'),
        ('model.fusion', 'moe'),
        ('model.dimension', 8),
        ('model.dimension', 32),
    ],
)
def test_every_fusion_and_dimension_in_its_set_is_accepted(key, value):
    cfg = Config().override({key: value})

    assert getattr(cfg.model, key.split('.')[1]) == value

@pytest.mark.parametrize(('key', 'value'), [('model.fusion', 'concat'), ('model.dimension', 12)])
def test_a_fusion_or_dimension_outside_its_set_is_refused(key, value):
    with pytest.raises(ValidationError) as excinfo:
        Config().override({key: value})

    # A Literal refusal: before the keys are declared, the same override fails as extra_forbidden
    assert _error_locs_and_types(excinfo) == [(('model', key.split('.')[1]), 'literal_error')]

def test_the_geometry_is_hyperbolic_by_default_and_as_shipped(valid_config_dict):
    '''Req 5's reference configuration is hyperbolic, and the YAML states the key itself.'''

    assert Config().model.geometry == 'hyperbolic'
    assert valid_config_dict['model']['geometry'] == 'hyperbolic'
    assert Config.model_validate(valid_config_dict).model.geometry == 'hyperbolic'

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical', 'hyperbolic'])
def test_every_geometry_arm_is_accepted(geometry):
    assert Config().override({'model.geometry': geometry}).model.geometry == geometry

def test_a_geometry_outside_the_three_arms_is_refused():
    with pytest.raises(ValidationError) as excinfo:
        Config().override({'model.geometry': 'poincare'})

    assert _error_locs_and_types(excinfo) == [(('model', 'geometry'), 'literal_error')]

def test_the_radius_bound_is_8_by_default_and_as_shipped(valid_config_dict):
    # Spec 4.2 and 4.5: R = 8, so a six-digit code at its target r = 5 keeps dr/dν ≈ 0.61
    assert Config().model.radius_bound == 8.0
    assert Config.model_validate(valid_config_dict).model.radius_bound == 8.0
    assert valid_config_dict['model']['radius_bound'] == 8

@pytest.mark.parametrize(
    ('value', 'error'),
    [(0.0, 'greater_than'), (-1.0, 'greater_than'), (float('inf'), 'finite_number')],
)
def test_a_radius_bound_that_is_not_positive_and_finite_is_refused(value, error):
    '''Spec section 5: a radius bound at or below 0 is refused; so is an infinite one.'''

    with pytest.raises(ValidationError) as excinfo:
        Config().override({'model.radius_bound': value})

    # Pinned on the type: before the key is declared, the same override fails as extra_forbidden
    assert _error_locs_and_types(excinfo) == [(('model', 'radius_bound'), error)]

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

def test_supervision_mode_is_a_removed_key(valid_config_dict):
    '''D2: training is always repaired, so the mode key is refused, even at its old default.'''

    valid_config_dict['supervision']['mode'] = 'repaired'

    with pytest.raises(ValidationError) as excinfo:
        Config.model_validate(valid_config_dict)

    assert _error_locs_and_types(excinfo) == [(('supervision', 'mode'), 'extra_forbidden')]

def test_exact_resume_is_the_only_checkpoint_load_mode():
    # D2 deleted the weights-only migration; --checkpoint-load-mode keeps its name and default
    assert [mode.value for mode in CheckpointLoadMode] == ['exact']

# -------------------------------------------------------------------------------------------------
# The old objective's removed keys (spec 4.5; P22)
# -------------------------------------------------------------------------------------------------

# Spec 4.5's removed keys, plus P22's two (loss.rank_order_weight and data_loader.num_workers), each
# at its old default, so a value the old configuration accepted is refused too. A removed section
# is set empty.
REMOVED_KEYS = {
    'supervision.mode': 'repaired',
    'curriculum': {},
    'sampling': {},
    'false_negatives': {},
    'data_loader.batch_size': 32,
    'data_loader.num_workers': 4,
    'data_loader.val_split': 0.05,
    'data_loader.n_epochs': 100,
    'data_loader.streaming.seed': 42,
    'data_loader.streaming.n_negatives': 24,
    'data_loader.streaming.use_phase1_sampling': True,
    'data_loader.streaming.phase1_alpha': 1.5,
    'data_loader.streaming.phase1_exclusion_weight': None,
    'data_loader.streaming.use_on_the_fly_sampling': False,
    'data_loader.streaming.n_candidates': 48,
    'data_loader.streaming.n_negatives_phase1': 24,
    'data_loader.streaming.phase1_easy_start': 0.7,
    'data_loader.streaming.phase1_easy_end': 0.2,
    'data_loader.streaming.phase1_semi_start': 0.2,
    'data_loader.streaming.phase1_semi_end': 0.4,
    'data_loader.streaming.distances_parquet': './data/naics_distances.parquet',
    'data_loader.streaming.distance_matrix_parquet': './data/naics_distance_matrix.parquet',
    'data_loader.streaming.relations_parquet': './data/naics_relations.parquet',
    'data_loader.streaming.triplets_parquet': './data/naics_training_pairs',
    'model.eval_sample_size': 500,
    'model.eval_every_n_epochs': 1,
    'model.parent_eval_top_k': 1,
    'model.child_eval_top_k': 5,
    'loss.temperature': 0.07,
    'loss.curvature': 1.0,
    'loss.base_margin': 0.5,
    'loss.hierarchy_weight': 0.1,
    'loss.structural_preference': {},
    'loss.rank_order_weight': None,
    'loss.radius_reg_weight': 0.01,
    'loss.level_radius_weight': 0.05,
    'training.warmup_steps': 500,
    'training.use_warmup_cosine': False,
}

@pytest.mark.parametrize(('key', 'value'), list(REMOVED_KEYS.items()), ids=list(REMOVED_KEYS))
def test_a_removed_key_is_refused_as_extra(valid_config_dict, key, value):
    '''Spec 4.5: the shipped YAML with a removed key, or an override that sets one, is refused as an
    unknown key, so no run reads a setting of the old objective.'''

    loc = tuple(key.split('.'))
    section = valid_config_dict
    for part in loc[:-1]:
        section = section[part]
    section[loc[-1]] = value

    with pytest.raises(ValidationError) as from_yaml:
        Config.model_validate(valid_config_dict)
    with pytest.raises(ValidationError) as from_override:
        Config().override({key: value})

    assert _error_locs_and_types(from_yaml) == [(loc, 'extra_forbidden')]
    assert _error_locs_and_types(from_override) == [(loc, 'extra_forbidden')]

def test_the_text_streaming_config_holds_only_the_text_fields():
    '''P22: data_loader.streaming is TextStreamingConfig; StreamingConfig, with the sampling fields
    HGCN's cache reads, belongs to graph_model alone.'''

    text_streaming = config_module.TextStreamingConfig

    assert list(text_streaming.model_fields) == [
        'descriptions_parquet', 'tokenizer_name', 'max_length'
    ]
    assert type(Config().data_loader.streaming) is text_streaming
    assert StreamingConfig not in _models_reachable_from(Config)
    # StreamingConfig keeps the fields hgcn_datamodule._streaming_cfg_from_loader sets
    assert set(StreamingConfig.model_fields) >= {
        'descriptions_parquet', 'relations_parquet', 'triplets_parquet', 'n_negatives', 'seed'
    }

# -------------------------------------------------------------------------------------------------
# StreamingConfig's ratio checks, kept unchanged for HGCN (P22)
# -------------------------------------------------------------------------------------------------

class TestStreamingConfigValidation:
    '''Tests for config validation.'''

    @pytest.mark.unit
    def test_valid_config_ratios(self):
        '''Valid ratio configs should pass validation.'''
        cfg = StreamingConfig(
            phase1_easy_start=0.60,
            phase1_easy_end=0.30,
            phase1_semi_start=0.30,
            phase1_semi_end=0.40,
        )
        # Should not raise
        assert cfg.phase1_easy_start == 0.60

    @pytest.mark.unit
    def test_invalid_start_ratios_sum(self):
        '''Start ratios summing > 1.0 should fail validation.'''
        with pytest.raises(ValueError, match='phase1_easy_start.*phase1_semi_start.*<= 1.0'):
            StreamingConfig(
                phase1_easy_start=0.70,
                phase1_semi_start=0.40,  # Sum = 1.1 > 1.0
            )

    @pytest.mark.unit
    def test_invalid_end_ratios_sum(self):
        '''End ratios summing > 1.0 should fail validation.'''
        with pytest.raises(ValueError, match='phase1_easy_end.*phase1_semi_end.*<= 1.0'):
            StreamingConfig(
                phase1_easy_end=0.50,
                phase1_semi_end=0.60,  # Sum = 1.1 > 1.0
            )

    @pytest.mark.unit
    def test_n_negatives_phase1_cannot_exceed_candidates(self):
        '''n_negatives_phase1 > n_candidates should fail validation.'''
        with pytest.raises(ValueError, match='n_negatives_phase1.*n_candidates'):
            StreamingConfig(
                n_candidates=24,
                n_negatives_phase1=48,  # More than candidates
            )

# -------------------------------------------------------------------------------------------------
# Trainer settings (spec 4.2 and 4.5)
# -------------------------------------------------------------------------------------------------

def test_the_trainer_precision_is_bf16_mixed_by_default_and_as_shipped(valid_config_dict):
    # R9: the reference trains at bf16-mixed on CUDA; off CUDA the trainer runs 32-true
    assert TrainerConfig().precision == 'bf16-mixed'
    assert Config.model_validate(valid_config_dict).training.trainer.precision == 'bf16-mixed'

@pytest.mark.parametrize('precision', ['32', '16', '16-mixed', 'bf16', 'bf16-mixed'])
def test_the_precision_validator_still_accepts_its_values(precision):
    assert TrainerConfig(precision=precision).precision == precision

@pytest.mark.parametrize('precision', ['bf16-true', '16-true'])
def test_no_run_can_select_a_true_half_precision(precision):
    with pytest.raises(ValidationError, match='precision must be one of'):
        TrainerConfig(precision=precision)

@pytest.mark.parametrize('devices', [2, 8])
def test_more_than_one_device_is_refused(devices):
    '''Spec 4.5 and section 5: training runs on one device, because its code cache is per
    process.'''

    with pytest.raises(ValidationError, match=f'devices must be 1, not {devices}') as excinfo:
        Config().override({'training.trainer.devices': devices})

    assert _error_locs_and_types(excinfo) == [(('training', 'trainer', 'devices'), 'value_error')]

# -------------------------------------------------------------------------------------------------
# The objective's added and changed keys (spec 4.5 and section 5; P5, P22)
# -------------------------------------------------------------------------------------------------

INF = float('inf')
NAN = float('nan')

# Spec 4.5's added keys and the keys it changes, at the defaults the spec states (R7, 4.1-4.5)
SPEC_DEFAULTS = {
    'experiment_name': 'reference',
    'data_loader.queries_per_step': 128,
    'loss.code_code_weight': 1.0,
    'loss.radial_weight': 1.0,
    'loss.target_temperature': 1.0,
    'loss.radial_step': 1.0,
    'loss.logit_scale_init': 1.0,
    'loss.logit_scale_range': [0.01, 100.0],
    'training.learning_rate': 1e-4,
    'training.weight_decay': 0.01,
    'training.warmup_epochs': 1,
    'training.lr_plateau_factor': 0.5,
    'training.lr_plateau_patience': 2,
    'training.early_stopping_patience': 5,
    'training.trainer.max_epochs': 40,
    'training.trainer.accumulate_grad_batches': 1,
    'training.trainer.precision': 'bf16-mixed',
}

def _dotted(source: Any, key: str) -> Any:
    '''The value at a dotted key, in a Config or in the YAML's dict.'''

    for part in key.split('.'):
        source = source[part] if isinstance(source, dict) else getattr(source, part)
    return source

@pytest.mark.parametrize(('key', 'value'), list(SPEC_DEFAULTS.items()), ids=list(SPEC_DEFAULTS))
def test_an_added_or_changed_key_has_the_specs_default_in_the_model_and_the_yaml(
    valid_config_dict, key, value
):
    '''P5: the Pydantic default is the spec's, and the shipped YAML states the same value.'''

    assert _dotted(Config(), key) == value
    # The YAML sets the key itself: a missing key would validate to the default
    _dotted(valid_config_dict, key)
    assert _dotted(Config.model_validate(valid_config_dict), key) == value

def test_the_configured_queries_per_step_default_is_the_datamodules():
    assert DataLoaderConfig().queries_per_step == DEFAULT_QUERIES_PER_STEP

@pytest.mark.parametrize(
    ('key', 'value', 'error'),
    [
        ('data_loader.queries_per_step', 0, 'greater_than_equal'),
        ('data_loader.queries_per_step', -1, 'greater_than_equal'),
        ('data_loader.queries_per_step', 1.5, 'int_from_float'),
        ('loss.code_code_weight', -0.1, 'greater_than_equal'),
        ('loss.code_code_weight', INF, 'finite_number'),
        ('loss.radial_weight', -1.0, 'greater_than_equal'),
        ('loss.radial_weight', NAN, 'finite_number'),
        ('loss.target_temperature', 0.0, 'greater_than'),
        ('loss.target_temperature', -1.0, 'greater_than'),
        ('loss.target_temperature', INF, 'finite_number'),
        ('loss.radial_step', 0.0, 'greater_than'),
        ('loss.radial_step', -0.5, 'greater_than'),
        ('loss.radial_step', INF, 'finite_number'),
        ('loss.logit_scale_init', INF, 'finite_number'),
        ('loss.logit_scale_init', NAN, 'finite_number'),
        ('loss.logit_scale_range', [0.01], 'too_short'),
        ('loss.logit_scale_range', [0.01, 1.0, 100.0], 'too_long'),
        ('training.warmup_epochs', -1, 'greater_than_equal'),
        ('training.warmup_epochs', 0.5, 'int_from_float'),
        ('training.lr_plateau_factor', 0.0, 'greater_than'),
        ('training.lr_plateau_factor', -0.5, 'greater_than'),
        ('training.lr_plateau_factor', 1.0, 'less_than'),
        ('training.lr_plateau_factor', 1.5, 'less_than'),
        ('training.lr_plateau_patience', -1, 'greater_than_equal'),
        ('training.early_stopping_patience', 0, 'greater_than_equal'),
        ('training.early_stopping_patience', -1, 'greater_than_equal'),
    ],
)
def test_an_added_key_outside_its_range_is_refused(key, value, error):
    '''Spec section 5 and P22: queries_per_step below 1, a weight below 0, a temperature or step at
    or below 0, a non-finite value, an epoch count below 0, a plateau factor outside (0, 1), an
    early-stopping patience below 1.'''

    with pytest.raises(ValidationError) as excinfo:
        Config().override({key: value})

    # Pinned on the type: before the key is declared, the same override fails as extra_forbidden
    assert _error_locs_and_types(excinfo) == [(tuple(key.split('.')), error)]

@pytest.mark.parametrize(
    'bounds',
    [[1.0, 1.0], [100.0, 0.01], [0.0, 100.0], [-1.0, 100.0], [0.01, INF], [NAN, 100.0]],
    ids=['one-point', 'reversed', 'zero', 'negative', 'infinite', 'nan'],
)
def test_a_logit_scale_range_that_is_empty_or_not_positive_is_refused(bounds):
    '''Spec section 5: the range must satisfy 0 < low < high, with finite ends.'''

    with pytest.raises(ValidationError, match='logit_scale_range') as excinfo:
        Config().override({'loss.logit_scale_range': bounds})

    assert _error_locs_and_types(excinfo) == [(('loss', 'logit_scale_range'), 'value_error')]

@pytest.mark.parametrize('init', [0.001, 1000.0], ids=['below', 'above'])
def test_a_logit_scale_init_outside_its_range_is_refused(init):
    '''P22: both scales start at logit_scale_init, which must lie inside logit_scale_range.'''

    with pytest.raises(ValidationError, match='logit_scale_init') as excinfo:
        Config().override({'loss.logit_scale_init': init})

    assert _error_locs_and_types(excinfo) == [(('loss', ), 'value_error')]

@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('data_loader.queries_per_step', 1),
        ('loss.code_code_weight', 0.0),
        ('loss.radial_weight', 0.0),
        ('loss.logit_scale_init', 0.01),
        ('loss.logit_scale_init', 100.0),
        ('training.warmup_epochs', 0),
        ('training.lr_plateau_patience', 0),
        ('training.early_stopping_patience', 1),
    ],
)
def test_an_added_key_at_the_edge_of_its_range_is_accepted(key, value):
    assert _dotted(Config().override({key: value}), key) == value

@pytest.mark.parametrize(
    ('init', 'low', 'high'),
    [
        (1.0, 0.01, 100.0),
        (0.01, 0.01, 100.0),
        (100.0, 0.01, 100.0),
        (0.5, 0.25, 2.0),
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 0.5),
        (1.0, 0.0, 100.0),
        (1.0, -1.0, 100.0),
        (0.001, 0.01, 100.0),
        (1000.0, 0.01, 100.0),
        (INF, 0.01, 100.0),
        (1.0, 0.01, INF),
        (NAN, 0.01, 100.0),
        (1.0, NAN, 100.0),
    ],
)
def test_the_logit_scale_keys_refuse_exactly_what_the_logit_scale_refuses(init, low, high):
    '''P22: a config that validates never fails when the model builds its logit scales, and the
    config refuses nothing they accept.'''

    try:
        LogitScale(init, low, high)
    except ValueError:
        builds = False
    else:
        builds = True
    try:
        Config().override({'loss.logit_scale_init': init, 'loss.logit_scale_range': [low, high]})
    except ValidationError:
        validates = False
    else:
        validates = True

    assert validates is builds

@pytest.mark.parametrize(
    ('text', 'value'),
    [
        ('[0.05, 50]', [0.05, 50]),
        ('[0.01, 100.0]', [0.01, 100.0]),
        ('[1, 100]', [1, 100]),
        ('1e-4', 1e-4),
        ('0.5', 0.5),
        ('128', 128),
        ('true', True),
        ('bf16-mixed', 'bf16-mixed'),
        ('reference', 'reference'),
        ('[not, a, list', '[not, a, list'),
    ],
)
def test_an_override_parses_a_list_with_decimals_and_every_other_value_as_before(text, value):
    '''A list with a decimal point was a string, which the logit-scale range refused.'''

    parsed = parse_override_value(text)

    assert parsed == value
    assert type(parsed) is type(value)

def test_the_logit_scale_range_can_be_overridden_on_the_command_line():
    overrides, invalid = parse_config_overrides(['loss.logit_scale_range=[0.05, 50]'])

    assert invalid == []
    assert Config().override(overrides).loss.logit_scale_range == [0.05, 50.0]

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

# -------------------------------------------------------------------------------------------------
# The backbone's trained input window (Req 9)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_every_tokenizing_config_defaults_to_the_trained_window():
    assert TokenizationConfig().max_length == 128
    assert StreamingConfig().max_length == 128
    assert config_module.TextStreamingConfig().max_length == 128
    assert TextOnlyConfig().max_length == 128

@pytest.mark.unit
@pytest.mark.parametrize(
    'config_class',
    ['TokenizationConfig', 'StreamingConfig', 'TextStreamingConfig', 'TextOnlyConfig'],
)
def test_a_max_length_beyond_the_trained_window_is_refused(config_class):
    with pytest.raises(ValidationError, match='trained input window'):
        getattr(config_module, config_class)(max_length=512)
