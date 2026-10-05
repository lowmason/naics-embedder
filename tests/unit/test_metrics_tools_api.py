'''The summary visualization wrapper and CLI, using temporary files only.'''

import json

import pytest
from typer.testing import CliRunner

from naics_embedder.cli.commands import tools
from naics_embedder.tools.metrics_tools import HAS_MATPLOTLIB, HAS_VISUALIZE, visualize_metrics
from tests.fixtures.epoch_summary import summary_rows

@pytest.fixture
def sample_summary(tmp_path):
    path = tmp_path / 'epoch_summary.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in summary_rows()))
    return path

@pytest.mark.unit
class TestVisualizeMetrics:

    def test_visualize_metrics_returns_dict(self, sample_summary, tmp_path):
        result = visualize_metrics(summary=sample_summary, output_dir=tmp_path / 'plots')
        assert isinstance(result, dict)
        assert result['metrics'] == summary_rows()
        assert result['output_file'].exists()

    def test_visualize_metrics_extracts_correct_count(self, sample_summary):
        result = visualize_metrics(summary=sample_summary)
        assert result['num_epochs'] == 3

    def test_visualize_metrics_tables_no_structural_statistic(self, sample_summary):
        result = visualize_metrics(summary=sample_summary)
        assert all(
            not any('spearman' in key or 'cophenetic' in key for key in row)
            for row in result['metrics']
        )

    def test_visualize_metrics_raises_for_missing_log(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            visualize_metrics(summary=tmp_path / 'missing.jsonl')

    def test_visualize_metrics_raises_for_no_metrics(self, tmp_path):
        path = tmp_path / 'empty.jsonl'
        path.write_text('')
        with pytest.raises(ValueError, match='No metrics'):
            visualize_metrics(summary=path)

    def test_visualize_metrics_creates_output_file(self, sample_summary, tmp_path):
        result = visualize_metrics(summary=sample_summary, output_dir=tmp_path / 'plots')
        assert result['output_file'] == tmp_path / 'plots' / 'epoch_metrics.png'
        assert result['output_file'].stat().st_size > 0

@pytest.mark.unit
class TestEdgeCases:

    def test_visualize_uses_defaults(self, sample_summary):
        result = visualize_metrics(summary=sample_summary)
        assert result['output_file'
                      ] == sample_summary.parent / 'visualizations' / 'epoch_metrics.png'

    def test_has_flags_are_booleans(self):
        assert isinstance(HAS_MATPLOTLIB, bool)
        assert isinstance(HAS_VISUALIZE, bool)

def test_cli_visualize_reads_a_summary_and_writes_its_plot(sample_summary, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'figures'
    result = CliRunner().invoke(
        tools.app, ['visualize', '--summary',
                    str(sample_summary), '--output-dir',
                    str(directory)]
    )
    assert result.exit_code == 0, result.output
    assert (directory / 'epoch_metrics.png').stat().st_size > 0
    assert not list(tmp_path.rglob('*selection_log*'))

def test_cli_visualize_missing_summary_exits_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        tools.app, ['visualize', '--summary',
                    str(tmp_path / 'missing.jsonl')]
    )
    assert result.exit_code == 1
    assert 'Error' in result.output
