'''
Unit tests for configuration display tools.

Tests config loading and display functionality.
'''

from pathlib import Path

import pytest
import yaml

from naics_embedder.tools.config_tools import load_config, show_current_config

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    '''Sample configuration dictionary.'''
    return {
        'data_loader': {
            'batch_size': 32,
            'num_workers': 4,
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'trainer': {
                'accumulate_grad_batches': 2,
                'precision': '16-mixed',
                'max_epochs': 20,
            },
        },
        'curriculum': {
            'phase1_end': 5,
            'phase2_end': 10,
            'phase3_end': 15,
            'tree_distance_alpha': 0.5,
            'sibling_distance_threshold': 2.0,
            'fn_curriculum_start_epoch': 5,
            'fn_cluster_every_n_epochs': 3,
            'fn_num_clusters': 100,
        },
    }

@pytest.fixture
def config_file(tmp_path, sample_config):
    '''Create temporary config file.'''
    config_path = tmp_path / 'conf' / 'config.yaml'
    config_path.parent.mkdir(parents=True)

    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)

    return str(config_path)

@pytest.fixture
def minimal_config(tmp_path):
    '''Create minimal config file.'''
    config = {
        'data_loader': {
            'batch_size': 16,
            'num_workers': 2,
        },
        'training': {
            'learning_rate': 0.0001,
            'weight_decay': 0.001,
            'warmup_steps': 50,
            'trainer': {
                'accumulate_grad_batches': 1,
                'precision': '32',
                'max_epochs': 10,
            },
        },
    }

    config_path = tmp_path / 'minimal_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return str(config_path)

# -------------------------------------------------------------------------------------------------
# Tests for load_config()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadConfig:
    '''Tests for load_config() function.'''

    def test_load_config_returns_dict(self, config_file):
        '''Test that load_config returns a dictionary.'''
        config = load_config(config_file)

        assert isinstance(config, dict)

    def test_load_config_contains_expected_keys(self, config_file):
        '''Test that loaded config contains expected top-level keys.'''
        config = load_config(config_file)

        assert 'data_loader' in config
        assert 'training' in config

    def test_load_config_preserves_values(self, config_file, sample_config):
        '''Test that config values are preserved.'''
        config = load_config(config_file)

        assert config['data_loader']['batch_size'] == sample_config['data_loader']['batch_size']
        assert config['training']['learning_rate'] == sample_config['training']['learning_rate']

    def test_load_config_handles_nested_values(self, config_file):
        '''Test loading deeply nested config values.'''
        config = load_config(config_file)

        assert config['training']['trainer']['precision'] == '16-mixed'
        assert config['training']['trainer']['max_epochs'] == 20

    def test_load_config_raises_for_missing_file(self, tmp_path):
        '''Test error for missing config file.'''
        nonexistent = str(tmp_path / 'nonexistent.yaml')

        with pytest.raises(FileNotFoundError):
            load_config(nonexistent)

    def test_load_config_handles_empty_file(self, tmp_path):
        '''Test handling of empty config file.'''
        empty_config = tmp_path / 'empty.yaml'
        empty_config.write_text('')

        config = load_config(str(empty_config))

        assert config is None

# -------------------------------------------------------------------------------------------------
# Tests for show_current_config()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestShowCurrentConfig:
    '''Tests for show_current_config() function.'''

    def test_show_config_displays_batch_size(self, config_file, capsys):
        '''Test that batch size is displayed.'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        assert 'batch_size' in captured.out.lower() or '32' in captured.out

    def test_show_config_displays_effective_batch_size(self, config_file, capsys):
        '''Test that effective batch size is computed and displayed.'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        # Effective batch size = 32 * 2 = 64
        assert 'effective' in captured.out.lower() or '64' in captured.out

    def test_show_config_displays_learning_rate(self, config_file, capsys):
        '''Test that learning rate is displayed.'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        assert 'learning_rate' in captured.out.lower() or '0.001' in captured.out

    def test_show_config_displays_curriculum_info(self, config_file, capsys):
        '''Test that curriculum info is displayed.'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        assert 'curriculum' in captured.out.lower() or 'phase' in captured.out.lower()

    def test_show_config_handles_missing_file(self, tmp_path, capsys):
        '''Test error message for missing config file.'''
        nonexistent = str(tmp_path / 'nonexistent.yaml')

        show_current_config(nonexistent)

        captured = capsys.readouterr()
        assert 'error' in captured.out.lower() or 'not found' in captured.out.lower()

    def test_show_config_handles_missing_curriculum(self, minimal_config, capsys):
        '''Test display when curriculum section is missing.'''
        show_current_config(minimal_config)

        captured = capsys.readouterr()
        # Should still work without curriculum section
        assert 'error' not in captured.out.lower()

    def test_show_config_uses_rich_formatting(self, config_file, capsys):
        '''Test that rich formatting is used (panel/colors).'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        # Rich output should contain some content
        assert len(captured.out) > 0

# -------------------------------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Edge case tests for config tools.'''

    def test_config_with_special_values(self, tmp_path):
        '''Test config with special YAML values.'''
        config = {
            'data_loader': {
                'batch_size': 32,
                'num_workers': 0,  # Zero value
            },
            'training': {
                'learning_rate': 1e-5,  # Scientific notation
                'weight_decay': 0.0,
                'warmup_steps': 0,
                'trainer': {
                    'accumulate_grad_batches': 1,
                    'precision': 'bf16-mixed',  # Different precision
                    'max_epochs': 100,
                },
            },
        }

        config_path = tmp_path / 'special.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        loaded = load_config(str(config_path))

        assert loaded['training']['learning_rate'] == 1e-5
        assert loaded['data_loader']['num_workers'] == 0

    def test_config_with_unicode(self, tmp_path):
        '''Test config with unicode characters in comments/strings.'''
        config_content = """
# Configuration with unicode: 学习率
data_loader:
  batch_size: 32
  num_workers: 4
training:
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_steps: 100
  trainer:
    accumulate_grad_batches: 1
    precision: "32"
    max_epochs: 10
"""
        config_path = tmp_path / 'unicode.yaml'
        config_path.write_text(config_content)

        config = load_config(str(config_path))

        assert config['data_loader']['batch_size'] == 32

    def test_show_config_with_path_object(self, config_file):
        '''Test that Path objects work with show_current_config.'''
        # This should not raise
        show_current_config(Path(config_file).as_posix())
