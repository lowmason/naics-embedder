'''
Unit tests for metrics_tools API module.

Tests the high-level API for visualizing metrics.
'''

import pytest

from naics_embedder.tools.metrics_tools import (
    HAS_MATPLOTLIB,
    HAS_VISUALIZE,
    visualize_metrics,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_log_file(tmp_path):
    '''Create sample log file for testing.'''
    log_content = """
2024-01-15 10:00:00 - INFO - Using curriculum stage: test_stage
2024-01-15 10:01:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:01:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
2024-01-15 10:01:02 - INFO - Hierarchy preservation: cophenetic=0.5 (500 pairs)
2024-01-15 10:01:03 - INFO - Norm CV: 0.8
2024-01-15 10:01:04 - INFO - Distance CV: 0.7
2024-01-15 10:01:05 - INFO - Collapse: False

2024-01-15 10:30:00 - INFO - Running evaluation metrics (epoch 1)
2024-01-15 10:30:01 - INFO - Hyperbolic radius: 4.0 ± 0.2
2024-01-15 10:30:02 - INFO - Hierarchy preservation: cophenetic=0.6 (500 pairs)
"""
    log_file = tmp_path / 'logs' / 'train_sequential.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text(log_content)
    return tmp_path

# -------------------------------------------------------------------------------------------------
# Tests for visualize_metrics()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestVisualizeMetrics:
    '''Tests for visualize_metrics() function.'''

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_returns_dict(self, sample_log_file):
        '''Test that visualize_metrics returns a dictionary.'''
        result = visualize_metrics(
            stage='test_stage',
            log_file=sample_log_file / 'logs' / 'train_sequential.log',
            output_dir=sample_log_file / 'output',
            project_root=sample_log_file,
        )

        assert isinstance(result, dict)
        assert 'metrics' in result
        assert 'stage' in result
        assert 'num_epochs' in result

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_extracts_correct_count(self, sample_log_file):
        '''Test that correct number of epochs is extracted.'''
        result = visualize_metrics(
            stage='test_stage',
            log_file=sample_log_file / 'logs' / 'train_sequential.log',
            output_dir=sample_log_file / 'output',
            project_root=sample_log_file,
        )

        assert result['num_epochs'] == 2

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_tables_no_structural_statistic(self, sample_log_file, capsys):
        '''Req 6: the summary table carries no cophenetic or Spearman column.'''
        visualize_metrics(
            stage='test_stage',
            log_file=sample_log_file / 'logs' / 'train_sequential.log',
            output_dir=sample_log_file / 'output',
            project_root=sample_log_file,
        )

        output = capsys.readouterr().out
        assert 'METRICS SUMMARY TABLE' in output
        assert 'Cophenetic' not in output
        assert 'Spearman' not in output

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_raises_for_missing_log(self, tmp_path):
        '''Test error for missing log file.'''
        with pytest.raises(FileNotFoundError):
            visualize_metrics(
                log_file=tmp_path / 'nonexistent.log',
                project_root=tmp_path,
            )

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_raises_for_no_metrics(self, tmp_path):
        '''Test error when no metrics found for stage.'''
        log_file = tmp_path / 'empty.log'
        log_file.write_text('No metrics here')

        with pytest.raises(ValueError, match='No metrics found'):
            visualize_metrics(
                stage='nonexistent_stage',
                log_file=log_file,
                project_root=tmp_path,
            )

    @pytest.mark.skipif(
        not HAS_VISUALIZE or not HAS_MATPLOTLIB, reason='dependencies not available'
    )
    def test_visualize_metrics_creates_output_file(self, sample_log_file):
        '''Test that output file is created.'''
        output_dir = sample_log_file / 'output'

        result = visualize_metrics(
            stage='test_stage',
            log_file=sample_log_file / 'logs' / 'train_sequential.log',
            output_dir=output_dir,
            project_root=sample_log_file,
        )

        if result['output_file'] is not None:
            assert result['output_file'].exists()

# -------------------------------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Edge case tests for metrics tools API.'''

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_uses_defaults(self, sample_log_file, monkeypatch):
        '''Test that default paths are used correctly.'''
        monkeypatch.chdir(sample_log_file)

        # This should use default paths relative to project_root
        result = visualize_metrics(
            stage='test_stage',
            project_root=sample_log_file,
        )

        assert result is not None

    def test_has_flags_are_booleans(self):
        '''Test that HAS_* flags are booleans.'''
        assert isinstance(HAS_MATPLOTLIB, bool)
        assert isinstance(HAS_VISUALIZE, bool)
