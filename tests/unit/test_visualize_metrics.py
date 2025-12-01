'''
Unit tests for metrics visualization tools.

Tests log file parsing, metric extraction, and visualization generation.
'''

import pytest

from naics_embedder.tools._visualize_metrics import (
    HAS_MATPLOTLIB,
    create_visualizations,
    parse_log_file,
    print_analysis,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_log_content():
    '''Sample log file content with training metrics.'''
    return """
2024-01-15 10:00:00 - INFO - Using curriculum stage: 02_text
2024-01-15 10:01:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:01:01 - INFO - Hyperbolic radius: 2.5432 ± 0.1234
2024-01-15 10:01:02 - INFO - Hierarchy preservation: cophenetic=0.4521 (500 pairs)
2024-01-15 10:01:03 - INFO - Norm CV: 0.8765
2024-01-15 10:01:04 - INFO - Distance CV: 0.7654
2024-01-15 10:01:05 - INFO - Collapse: False

2024-01-15 10:30:00 - INFO - Running evaluation metrics (epoch 1)
2024-01-15 10:30:01 - INFO - Hyperbolic radius: 4.2345 ± 0.2345
2024-01-15 10:30:02 - INFO - Hierarchy preservation: cophenetic=0.5678 (500 pairs)
2024-01-15 10:30:03 - INFO - Norm CV: 0.7890
2024-01-15 10:30:04 - INFO - Distance CV: 0.6789
2024-01-15 10:30:05 - INFO - Collapse: False

2024-01-15 11:00:00 - INFO - Running evaluation metrics (epoch 2)
2024-01-15 11:00:01 - INFO - Hyperbolic radius: 8.7654 ± 0.3456
2024-01-15 11:00:02 - INFO - Hierarchy preservation: cophenetic=0.6543 (500 pairs)
2024-01-15 11:00:03 - INFO - Norm CV: 0.6543
2024-01-15 11:00:04 - INFO - Distance CV: 0.5432
2024-01-15 11:00:05 - INFO - Collapse: False
"""

@pytest.fixture
def sample_log_with_stages():
    '''Sample log with multiple stages.'''
    return """
2024-01-15 10:00:00 - INFO - Using curriculum stage: 01_text
2024-01-15 10:01:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:01:01 - INFO - Hyperbolic radius: 1.0000 ± 0.1000
2024-01-15 10:01:02 - INFO - Hierarchy preservation: cophenetic=0.2000 (500 pairs)

2024-01-15 12:00:00 - INFO - Using curriculum stage: 02_text
2024-01-15 12:01:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 12:01:01 - INFO - Hyperbolic radius: 2.0000 ± 0.2000
2024-01-15 12:01:02 - INFO - Hierarchy preservation: cophenetic=0.4000 (500 pairs)

2024-01-15 14:00:00 - INFO - Using curriculum stage: 03_text
2024-01-15 14:01:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 14:01:01 - INFO - Hyperbolic radius: 3.0000 ± 0.3000
2024-01-15 14:01:02 - INFO - Hierarchy preservation: cophenetic=0.6000 (500 pairs)
"""

@pytest.fixture
def sample_log_file(tmp_path, sample_log_content):
    '''Create temporary log file.'''
    log_file = tmp_path / 'train.log'
    log_file.write_text(sample_log_content)
    return log_file

@pytest.fixture
def multi_stage_log_file(tmp_path, sample_log_with_stages):
    '''Create temporary log file with multiple stages.'''
    log_file = tmp_path / 'train_multi.log'
    log_file.write_text(sample_log_with_stages)
    return log_file

# -------------------------------------------------------------------------------------------------
# Tests for parse_log_file()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestParseLogFile:
    '''Tests for parse_log_file() function.'''

    def test_parse_extracts_epoch_numbers(self, sample_log_file):
        '''Test that epoch numbers are extracted correctly.'''
        metrics = parse_log_file(sample_log_file)

        epochs = [m['epoch'] for m in metrics]
        assert epochs == [0, 1, 2]

    def test_parse_extracts_hyperbolic_radius(self, sample_log_file):
        '''Test extraction of hyperbolic radius mean and std.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['radius_mean'] == pytest.approx(2.5432)
        assert metrics[0]['radius_std'] == pytest.approx(0.1234)
        assert metrics[2]['radius_mean'] == pytest.approx(8.7654)
        assert metrics[2]['radius_std'] == pytest.approx(0.3456)

    def test_parse_extracts_cophenetic_correlation(self, sample_log_file):
        '''Test extraction of cophenetic correlation and pair count.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['cophenetic'] == pytest.approx(0.4521)
        assert metrics[0]['n_pairs'] == 500
        assert metrics[1]['cophenetic'] == pytest.approx(0.5678)

    def test_parse_extracts_norm_cv(self, sample_log_file):
        '''Test extraction of Norm CV.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['norm_cv'] == pytest.approx(0.8765)
        assert metrics[1]['norm_cv'] == pytest.approx(0.7890)

    def test_parse_extracts_distance_cv(self, sample_log_file):
        '''Test extraction of Distance CV.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['dist_cv'] == pytest.approx(0.7654)
        assert metrics[2]['dist_cv'] == pytest.approx(0.5432)

    def test_parse_extracts_collapse_flag(self, sample_log_file):
        '''Test extraction of collapse flag.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['collapse'] is False
        assert metrics[1]['collapse'] is False

    def test_parse_extracts_timestamp(self, sample_log_file):
        '''Test extraction of timestamp.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['timestamp'] == '2024-01-15 10:01:00'
        assert metrics[1]['timestamp'] == '2024-01-15 10:30:00'

    def test_parse_filters_by_stage(self, multi_stage_log_file):
        '''Test filtering by stage name.'''
        metrics_01 = parse_log_file(multi_stage_log_file, stage='01_text')
        metrics_02 = parse_log_file(multi_stage_log_file, stage='02_text')
        metrics_03 = parse_log_file(multi_stage_log_file, stage='03_text')

        # Each stage should have 1 epoch
        assert len(metrics_01) == 1
        assert len(metrics_02) == 1
        assert len(metrics_03) == 1

        # Values should match stage-specific data
        assert metrics_01[0]['radius_mean'] == pytest.approx(1.0)
        assert metrics_02[0]['radius_mean'] == pytest.approx(2.0)
        assert metrics_03[0]['radius_mean'] == pytest.approx(3.0)

    def test_parse_without_stage_filter(self, sample_log_file):
        '''Test parsing without stage filter returns all metrics.'''
        metrics = parse_log_file(sample_log_file, stage=None)

        assert len(metrics) == 3

    def test_parse_empty_log_returns_empty_list(self, tmp_path):
        '''Test parsing empty log file.'''
        empty_log = tmp_path / 'empty.log'
        empty_log.write_text('')

        metrics = parse_log_file(empty_log)
        assert metrics == []

    def test_parse_sorted_by_epoch(self, tmp_path):
        '''Test that metrics are sorted by epoch.'''
        # Create log with out-of-order epochs
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 2)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 3.0 ± 0.3

2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 1.0 ± 0.1

2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 1)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.0 ± 0.2
"""
        log_file = tmp_path / 'unordered.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        epochs = [m['epoch'] for m in metrics]
        assert epochs == [0, 1, 2]

    def test_parse_handles_missing_metrics(self, tmp_path):
        '''Test handling of log with incomplete metrics.'''
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
"""
        log_file = tmp_path / 'incomplete.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        assert len(metrics) == 1
        assert metrics[0]['radius_mean'] == pytest.approx(2.5)
        assert 'cophenetic' not in metrics[0]
        assert 'norm_cv' not in metrics[0]

    def test_parse_collapse_true(self, tmp_path):
        '''Test parsing collapse flag when True.'''
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
2024-01-15 10:00:02 - INFO - Collapse: True
"""
        log_file = tmp_path / 'collapsed.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        assert metrics[0]['collapse'] is True

# -------------------------------------------------------------------------------------------------
# Tests for create_visualizations()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateVisualizations:
    '''Tests for create_visualizations() function.'''

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_creates_output_file(self, sample_log_file, tmp_path):
        '''Test that visualization file is created.'''
        metrics = parse_log_file(sample_log_file)
        output_dir = tmp_path / 'output'

        create_visualizations(metrics, output_dir, 'test_stage')

        output_file = output_dir / 'test_stage_metrics.png'
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_creates_output_dir(self, sample_log_file, tmp_path):
        '''Test that output directory is created if it doesn't exist.'''
        metrics = parse_log_file(sample_log_file)
        output_dir = tmp_path / 'nested' / 'output'

        create_visualizations(metrics, output_dir, 'test_stage')

        assert output_dir.exists()

    def test_create_visualizations_handles_empty_metrics(self, tmp_path, capsys):
        '''Test handling of empty metrics list.'''
        create_visualizations([], tmp_path, 'test_stage')

        captured = capsys.readouterr()
        assert 'No metrics found' in captured.out

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_handles_single_epoch(self, tmp_path):
        '''Test visualization with single epoch.'''
        metrics = [
            {
                'epoch': 0,
                'timestamp': '2024-01-15 10:00:00',
                'radius_mean': 2.5,
                'radius_std': 0.1,
                'cophenetic': 0.5,
                'n_pairs': 500,
            }
        ]

        create_visualizations(metrics, tmp_path, 'single')

        output_file = tmp_path / 'single_metrics.png'
        assert output_file.exists()

# -------------------------------------------------------------------------------------------------
# Tests for print_analysis()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestPrintAnalysis:
    '''Tests for print_analysis() function.'''

    def test_print_analysis_empty_metrics(self, capsys):
        '''Test handling of empty metrics.'''
        print_analysis([], 'test_stage')

        captured = capsys.readouterr()
        assert 'No metrics to analyze' in captured.out

    def test_print_analysis_shows_stage(self, sample_log_file, capsys):
        '''Test that stage name is shown in output.'''
        metrics = parse_log_file(sample_log_file)

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'TEST_STAGE' in captured.out

    def test_print_analysis_shows_radius_info(self, sample_log_file, capsys):
        '''Test that radius analysis is shown.'''
        metrics = parse_log_file(sample_log_file)

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'HYPERBOLIC RADIUS' in captured.out
        assert 'Initial' in captured.out
        assert 'Latest' in captured.out

    def test_print_analysis_shows_hierarchy_preservation(self, sample_log_file, capsys):
        '''Test that hierarchy preservation analysis is shown.'''
        metrics = parse_log_file(sample_log_file)

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'HIERARCHY PRESERVATION' in captured.out

    def test_print_analysis_warns_large_radius(self, tmp_path, capsys):
        '''Test warning for large radius.'''
        metrics = [{
            'epoch': 0,
            'radius_mean': 25.0,
            'radius_std': 1.0,
        }]

        print_analysis(metrics, 'test')

        captured = capsys.readouterr()
        assert 'WARNING' in captured.out or 'large' in captured.out.lower()

    def test_print_analysis_shows_recommendations(self, sample_log_file, capsys):
        '''Test that recommendations are shown.'''
        metrics = parse_log_file(sample_log_file)

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'RECOMMENDATIONS' in captured.out

# -------------------------------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Edge case tests for visualization tools.'''

    def test_parse_handles_special_characters_in_path(self, tmp_path):
        '''Test handling paths with special characters.'''
        log_dir = tmp_path / 'dir with spaces'
        log_dir.mkdir()
        log_file = log_dir / 'train.log'
        log_file.write_text(
            """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
"""
        )

        metrics = parse_log_file(log_file)
        assert len(metrics) == 1

    def test_parse_handles_very_large_values(self, tmp_path):
        '''Test parsing very large metric values.'''
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 999999.999 ± 99999.999
"""
        log_file = tmp_path / 'large_values.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        assert metrics[0]['radius_mean'] == pytest.approx(999999.999)

    def test_parse_handles_negative_cophenetic(self, tmp_path):
        '''Test parsing negative cophenetic values.'''
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
2024-01-15 10:00:02 - INFO - Hierarchy preservation: cophenetic=-0.1234 (500 pairs)
"""
        log_file = tmp_path / 'negative_cophenetic.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        assert metrics[0]['cophenetic'] == pytest.approx(-0.1234)
