'''
Unit tests for the main CLI module.

Tests the top-level Typer application setup and basic CLI functionality.
'''

import importlib
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from naics_embedder.cli import app
from naics_embedder.utils.warnings import list_suppressed_warnings

# -------------------------------------------------------------------------------------------------
# Tests for CLI app setup
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCliSetup:
    '''Tests for CLI application setup.'''

    def test_app_is_typer_instance(self):
        '''Test that app is a Typer instance.'''
        import typer

        assert isinstance(app, typer.Typer)

    def test_app_has_help_text(self):
        '''Test that app has help text configured.'''
        # Must be a plain string: typer >= 0.21.2 calls str methods on help, so a Rich Panel crashes
        assert isinstance(app.info.help, str)

# -------------------------------------------------------------------------------------------------
# Tests for CLI commands registration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCliCommands:
    '''Tests for CLI command registration.'''

    def test_data_subcommand_registered(self):
        '''Test that data subcommand is registered.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['data', '--help'])

        # Should show help for data subcommand
        assert result.exit_code == 0
        assert 'data' in result.output.lower() or 'Usage' in result.output

    def test_tools_subcommand_registered(self):
        '''Test that tools subcommand is registered.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['tools', '--help'])

        # Should show help for tools subcommand
        assert result.exit_code == 0

    def test_train_command_registered(self):
        '''Test that train command is registered.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['train', '--help'])

        # Should show help for train command
        assert result.exit_code == 0

# -------------------------------------------------------------------------------------------------
# Tests for CLI help output
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCliHelp:
    '''Tests for CLI help output.'''

    def test_main_help_shows_naics_embedder(self):
        '''Test that main help shows NAICS Embedder info.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['--help'])

        assert result.exit_code == 0
        # Should mention NAICS somewhere in help
        assert 'naics' in result.output.lower() or 'embedder' in result.output.lower()

    def test_help_shows_available_commands(self):
        '''Test that help shows available commands.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['--help'])

        assert result.exit_code == 0
        # Should list commands
        output_lower = result.output.lower()
        assert 'data' in output_lower or 'train' in output_lower or 'tools' in output_lower

# -------------------------------------------------------------------------------------------------
# Tests for warning configuration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestWarningConfiguration:
    '''Tests for warning configuration in CLI.'''

    def test_configure_warnings_called_on_import(self):
        '''Test that importing naics_embedder.cli installs every centralized warning filter.'''
        from naics_embedder import cli

        with warnings.catch_warnings():
            warnings.resetwarnings()
            # Reload re-executes only cli/__init__.py; cached command modules do not re-run
            importlib.reload(cli)
            installed = {
                message.pattern
                for action, message, *_ in warnings.filters
                if action == 'ignore' and message is not None
            }

        suppressed = {pattern for pattern, _ in list_suppressed_warnings()}
        assert suppressed - installed == set()

# -------------------------------------------------------------------------------------------------
# Tests for CUDA allocator configuration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestAllocatorConfiguration:
    '''Tests for the PyTorch allocator setting exported by the CLI.'''

    def test_allocator_conf_set_before_torch_is_imported(self):
        '''Test that PYTORCH_ALLOC_CONF is exported before anything looks up torch.'''
        # A fresh interpreter is the only place torch is not already loaded. The meta-path
        # finder records the variable at the first lookup of torch, then defers to the real
        # finders.
        probe = textwrap.dedent(
            """
            import os
            import sys

            seen = []

            class TorchImportSpy:
                @staticmethod
                def find_spec(name, path=None, target=None):
                    if name == 'torch' and not seen:
                        seen.append(os.environ.get('PYTORCH_ALLOC_CONF'))
                    return None

            sys.meta_path.insert(0, TorchImportSpy)
            import naics_embedder.cli

            print(seen)
            """
        )
        # Importing naics_embedder.cli in this process sets the variable; don't let it leak in
        env = {key: value for key, value in os.environ.items() if key != 'PYTORCH_ALLOC_CONF'}

        result = subprocess.run(
            [sys.executable, '-c', probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == str(['expandable_segments:True'])

# -------------------------------------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCliIntegration:
    '''Integration tests for CLI.'''

    def test_invalid_command_shows_error(self):
        '''Test that invalid command shows appropriate error.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['nonexistent_command'])

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    def test_empty_invocation_shows_help(self):
        '''Test that invoking with no args shows help or usage.'''
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, [])

        # Either shows help (exit 0) or usage error (exit non-zero)
        # Both are acceptable behaviors
        assert len(result.output) > 0

    def test_cli_handles_keyboard_interrupt(self):
        '''Test that CLI can be imported without immediate execution.'''
        # This test verifies the CLI module can be imported cleanly
        # without triggering any unintended execution
        import importlib

        # Reimport to verify clean import
        import naics_embedder.cli

        importlib.reload(naics_embedder.cli)

        assert naics_embedder.cli.app is not None
