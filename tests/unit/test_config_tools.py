'''
Unit tests for configuration display tools.

``tools config`` shows the configuration a run would use: the file validated as ``Config`` over the
defaults, so it reads only keys that exist and refuses a removed one (spec 4.5).
'''

import re
from pathlib import Path

import click
import pytest
import yaml
from typer.testing import CliRunner

from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.tools.config_tools import load_config, show_current_config
from naics_embedder.utils import training as utils_training
from naics_embedder.utils.config import Config

SHIPPED_CONFIG = 'conf/config.yaml'

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    '''A partial configuration: what it sets overrides the defaults, which fill the rest.'''
    return {
        'experiment_name': 'display',
        'data_loader': {
            'queries_per_step': 64,
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.02,
            'warmup_epochs': 2,
            'trainer': {
                'accumulate_grad_batches': 2,
                'precision': '16-mixed',
                'max_epochs': 20,
            },
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

        assert config['data_loader']['queries_per_step'] == 64
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

    @pytest.mark.parametrize('which', ['partial', 'shipped'])
    def test_show_config_displays_every_run_setting(self, which, config_file, capsys):
        '''P21's settings, each at the value the validated configuration gives it.'''
        path = config_file if which == 'partial' else SHIPPED_CONFIG

        assert show_current_config(path) is True

        output = capsys.readouterr().out
        cfg = Config.from_yaml(path)
        trainer = cfg.training.trainer
        settings = utils_training.run_settings(
            cfg, accelerator=trainer.accelerator, precision=trainer.precision
        )
        for name, value in settings.items():
            if name != 'precision':
                assert f'{name}: {value}' in output, name
        # A run trains at the configured precision on CUDA only (P31)
        elsewhere = utils_training.FULL_PRECISION
        assert f'precision: {trainer.precision} on CUDA, {elsewhere} elsewhere' in output
        assert f'experiment_name: {cfg.experiment_name}' in output

    def test_show_config_displays_learning_rate(self, config_file, capsys):
        '''Test that learning rate is displayed.'''
        show_current_config(config_file)

        captured = capsys.readouterr()
        assert 'learning_rate' in captured.out.lower() or '0.001' in captured.out

    def test_show_config_refuses_a_removed_key(self, tmp_path, capsys):
        '''A configuration that sets a key of the old objective is refused, not shown (spec 4.5).'''
        old = tmp_path / 'old.yaml'
        old.write_text(yaml.dump({'data_loader': {'batch_size': 16}}))

        assert show_current_config(str(old)) is False

        output = capsys.readouterr().out
        assert 'not a valid configuration' in output
        assert 'data_loader.batch_size' in output
        assert 'learning_rate' not in output

    def test_show_config_handles_missing_file(self, tmp_path, capsys):
        '''Test error message for missing config file.'''
        nonexistent = str(tmp_path / 'nonexistent.yaml')

        assert show_current_config(nonexistent) is False

        captured = capsys.readouterr()
        assert 'error' in captured.out.lower() or 'not found' in captured.out.lower()

    def test_show_config_prints_a_missing_path_as_written(self, tmp_path, capsys):
        '''A missing file's path is printed as text, never read as Rich markup.'''
        missing = tmp_path / 'missing[/x][red].yaml'

        assert show_current_config(str(missing)) is False

        output = click.unstyle(capsys.readouterr().out).replace('\n', '')
        assert 'Config file not found' in output
        assert str(missing) in output

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
                'queries_per_step': 1,
            },
            'training': {
                'learning_rate': 1e-5,  # Scientific notation
                'weight_decay': 0.0,  # Zero value
                'warmup_epochs': 0,
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
        assert loaded['training']['weight_decay'] == 0.0

    def test_config_with_unicode(self, tmp_path):
        '''Test config with unicode characters in comments/strings.'''
        config_content = """
# Configuration with unicode: 学习率
data_loader:
  queries_per_step: 32
training:
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_epochs: 1
  trainer:
    accumulate_grad_batches: 1
    precision: "32"
    max_epochs: 10
"""
        config_path = tmp_path / 'unicode.yaml'
        config_path.write_text(config_content)

        config = load_config(str(config_path))

        assert config['data_loader']['queries_per_step'] == 32

    def test_show_config_with_path_object(self, config_file):
        '''Test that Path objects work with show_current_config.'''
        assert show_current_config(Path(config_file)) is True

# -------------------------------------------------------------------------------------------------
# The tools config command
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def runner(monkeypatch):
    '''A CLI runner. The command's log file is not opened, so a test writes only under tmp_path.'''
    monkeypatch.setattr(tools_cli, 'configure_logging', lambda *_, **__: None)
    return CliRunner()

def _plain(result) -> str:
    '''The command's output unstyled and unwrapped: CI styles Typer's help and wraps at 80.'''
    return click.unstyle(result.output).replace('\n', '')

@pytest.mark.unit
class TestToolsConfigCommand:
    '''``tools config`` exits 1 when it prints an error instead of the configuration.'''

    def test_a_removed_key_exits_1_with_the_error(self, runner, tmp_path):
        old = tmp_path / 'old.yaml'
        old.write_text(yaml.dump({'data_loader': {'batch_size': 16}}))

        result = runner.invoke(tools_cli.app, ['config', '--config', str(old)])

        assert result.exit_code == 1
        assert 'not a valid configuration' in _plain(result)
        assert 'data_loader.batch_size' in _plain(result)

    def test_a_missing_file_exits_1_with_the_error(self, runner, tmp_path):
        missing = tmp_path / 'nonexistent.yaml'

        result = runner.invoke(tools_cli.app, ['config', '--config', str(missing)])

        assert result.exit_code == 1
        assert 'Config file not found' in _plain(result)
        assert str(missing) in _plain(result)

    def test_the_shipped_config_is_shown_and_exits_0(self, runner):
        result = runner.invoke(tools_cli.app, ['config', '--config', SHIPPED_CONFIG])

        assert result.exit_code == 0
        assert 'Current Training Configuration' in _plain(result)
        assert 'Run settings:' in _plain(result)
        # User-facing text cites no plan decision number
        assert re.search(r'\bP\d+\b', _plain(result)) is None

    def test_the_help_cites_no_plan_decision(self, runner):
        result = runner.invoke(tools_cli.app, ['config', '--help'])

        assert result.exit_code == 0
        assert 'Display the training configuration a run would use.' in _plain(result)
        assert re.search(r'\bP\d+\b', _plain(result)) is None
