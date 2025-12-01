'''
Unit tests for hierarchy investigation tools.

Tests correlation analysis, distance matrix analysis, and report generation.
'''

import pytest
import torch

# Check for optional dependencies
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from naics_embedder.tools._investigate_hierarchy import (
    analyze_correlation_issues,
    analyze_ground_truth_distances,
    check_evaluation_sample_size,
    main,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_distance_matrix(tmp_path):
    '''Create sample distance matrix parquet.'''
    if not HAS_POLARS:
        pytest.skip('polars not available')

    n = 10
    # Create symmetric distance matrix with zeros on diagonal
    distances = torch.rand(n, n)
    distances = (distances + distances.T) / 2  # Make symmetric
    distances.fill_diagonal_(0)

    df = pl.DataFrame(distances.numpy())
    path = tmp_path / 'distance_matrix.parquet'
    df.write_parquet(path)
    return path

@pytest.fixture
def sample_config_file(tmp_path):
    '''Create sample config file.'''
    if not HAS_YAML:
        pytest.skip('yaml not available')

    config = {
        'model': {
            'eval_sample_size': 500,
            'embedding_dim': 384,
        },
        'training': {
            'learning_rate': 0.001,
        },
    }

    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path

@pytest.fixture
def small_eval_config(tmp_path):
    '''Create config with small evaluation sample size.'''
    if not HAS_YAML:
        pytest.skip('yaml not available')

    config = {
        'model': {
            'eval_sample_size': 50,  # Small
        }
    }

    path = tmp_path / 'small_eval_config.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path

# -------------------------------------------------------------------------------------------------
# Tests for analyze_ground_truth_distances()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzeGroundTruthDistances:
    '''Tests for analyze_ground_truth_distances() function.'''

    @pytest.mark.skipif(not HAS_POLARS, reason='polars not available')
    def test_analyze_prints_statistics(self, sample_distance_matrix, capsys):
        '''Test that statistics are printed.'''
        analyze_ground_truth_distances(sample_distance_matrix)

        captured = capsys.readouterr()
        assert 'GROUND TRUTH DISTANCE MATRIX ANALYSIS' in captured.out
        assert 'Mean:' in captured.out
        assert 'Std:' in captured.out
        assert 'Min:' in captured.out
        assert 'Max:' in captured.out

    @pytest.mark.skipif(not HAS_POLARS, reason='polars not available')
    def test_analyze_returns_distances(self, sample_distance_matrix):
        '''Test that distances tensor is returned.'''
        distances = analyze_ground_truth_distances(sample_distance_matrix)

        assert distances is not None
        assert isinstance(distances, torch.Tensor)
        assert distances.shape[0] == distances.shape[1]  # Square matrix

    def test_analyze_missing_file(self, tmp_path, capsys):
        '''Test handling of missing distance matrix file.'''
        nonexistent = tmp_path / 'nonexistent.parquet'

        result = analyze_ground_truth_distances(nonexistent)

        assert result is None
        captured = capsys.readouterr()
        assert 'not found' in captured.out

    @pytest.mark.skipif(not HAS_POLARS, reason='polars not available')
    def test_analyze_prints_percentiles(self, sample_distance_matrix, capsys):
        '''Test that distance percentiles are printed.'''
        analyze_ground_truth_distances(sample_distance_matrix)

        captured = capsys.readouterr()
        assert 'Distance Distribution' in captured.out
        assert 'percentile' in captured.out

    @pytest.mark.skipif(not HAS_POLARS, reason='polars not available')
    def test_analyze_detects_zero_pairs(self, tmp_path, capsys):
        '''Test detection of zero-distance pairs.'''
        n = 5
        # Create matrix with some zero off-diagonal values
        distances = torch.zeros(n, n)
        distances[0, 1] = 1.0  # Only one non-zero pair
        distances[1, 0] = 1.0

        df = pl.DataFrame(distances.numpy())
        path = tmp_path / 'zeros.parquet'
        df.write_parquet(path)

        analyze_ground_truth_distances(path)

        captured = capsys.readouterr()
        # Should detect zero pairs (excluding diagonal)
        assert 'zero' in captured.out.lower()

# -------------------------------------------------------------------------------------------------
# Tests for check_evaluation_sample_size()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckEvaluationSampleSize:
    '''Tests for check_evaluation_sample_size() function.'''

    @pytest.mark.skipif(not HAS_YAML, reason='yaml not available')
    def test_check_returns_sample_size(self, sample_config_file):
        '''Test that sample size is returned.'''
        result = check_evaluation_sample_size(sample_config_file)

        assert result == 500

    @pytest.mark.skipif(not HAS_YAML, reason='yaml not available')
    def test_check_prints_info(self, sample_config_file, capsys):
        '''Test that evaluation info is printed.'''
        check_evaluation_sample_size(sample_config_file)

        captured = capsys.readouterr()
        assert 'EVALUATION CONFIGURATION' in captured.out
        assert 'Evaluation Sample Size' in captured.out

    @pytest.mark.skipif(not HAS_YAML, reason='yaml not available')
    def test_check_warns_small_sample(self, small_eval_config, capsys):
        '''Test warning for small sample size.'''
        check_evaluation_sample_size(small_eval_config)

        captured = capsys.readouterr()
        assert 'WARNING' in captured.out or 'small' in captured.out.lower()

    @pytest.mark.skipif(not HAS_YAML, reason='yaml not available')
    def test_check_uses_default_when_missing(self, tmp_path):
        '''Test default value when eval_sample_size is not in config.'''
        config = {'model': {'other_setting': 123}}
        path = tmp_path / 'no_eval_size.yaml'
        with open(path, 'w') as f:
            yaml.dump(config, f)

        result = check_evaluation_sample_size(path)

        # Should use default of 500
        assert result == 500

# -------------------------------------------------------------------------------------------------
# Tests for analyze_correlation_issues()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestAnalyzeCorrelationIssues:
    '''Tests for analyze_correlation_issues() function.'''

    def test_analyze_prints_potential_issues(self, capsys):
        '''Test that potential issues are printed.'''
        analyze_correlation_issues()

        captured = capsys.readouterr()
        assert 'POTENTIAL ISSUES' in captured.out

    def test_analyze_covers_key_topics(self, capsys):
        '''Test that key correlation topics are covered.'''
        analyze_correlation_issues()

        captured = capsys.readouterr()
        # Should mention various causes
        assert 'TRAINING' in captured.out.upper() or 'training' in captured.out
        assert 'SAMPLE' in captured.out.upper() or 'sample' in captured.out

# -------------------------------------------------------------------------------------------------
# Tests for main() function
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestMain:
    '''Tests for main() entry point.'''

    @pytest.mark.skipif(not HAS_POLARS or not HAS_YAML, reason='dependencies not available')
    def test_main_runs_without_error(self, sample_distance_matrix, sample_config_file, capsys):
        '''Test that main runs without error with valid inputs.'''
        project_root = sample_distance_matrix.parent

        # Create distance matrix at expected location
        (project_root / 'data').mkdir(exist_ok=True)
        sample_distance_matrix.rename(project_root / 'data' / 'naics_distance_matrix.parquet')

        # Create config at expected location
        (project_root / 'conf').mkdir(exist_ok=True)
        sample_config_file.rename(project_root / 'conf' / 'config.yaml')

        main(project_root)

        captured = capsys.readouterr()
        assert 'RECOMMENDATIONS' in captured.out

    def test_main_handles_missing_files(self, tmp_path, capsys):
        '''Test that main handles missing files gracefully.'''
        main(tmp_path)

        captured = capsys.readouterr()
        # Should still print recommendations even if files are missing
        assert 'RECOMMENDATIONS' in captured.out

    def test_main_uses_cwd_by_default(self, monkeypatch, tmp_path, capsys):
        '''Test that main uses current directory by default.'''
        monkeypatch.chdir(tmp_path)

        main()

        # Should run without error
        captured = capsys.readouterr()
        assert len(captured.out) > 0

# -------------------------------------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestIntegration:
    '''Integration tests for hierarchy investigation.'''

    @pytest.mark.skipif(not HAS_POLARS, reason='polars not available')
    def test_full_analysis_workflow(self, sample_distance_matrix, capsys):
        '''Test full analysis workflow.'''
        # Analyze distances
        analyze_ground_truth_distances(sample_distance_matrix)

        # Print correlation issues
        analyze_correlation_issues()

        captured = capsys.readouterr()

        # Should have comprehensive output
        assert 'GROUND TRUTH' in captured.out
        assert 'POTENTIAL ISSUES' in captured.out
