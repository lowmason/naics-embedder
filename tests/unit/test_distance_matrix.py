import polars as pl
import pytest
import torch

from naics_embedder.metrics import HierarchyMetrics, StructuralMetricInputError
from naics_embedder.utils.distance_matrix import load_distance_submatrix

pytestmark = pytest.mark.unit

@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('position', [(0, 1), (1, 0)])
def test_file_backed_nonfinite_values_reach_metric_boundary(
    tmp_path, structural_distance_matrices, value, position
):
    prediction, target = structural_distance_matrices
    target[position] = value
    codes = ['n0', 'n1', 'n2', 'n3']
    path = tmp_path / 'distances.parquet'
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(path)
    loaded = load_distance_submatrix(path, codes)
    assert not torch.isfinite(loaded[position])
    metric = HierarchyMetrics()
    metric.device = 'cpu'
    with pytest.raises(StructuralMetricInputError, match='finite'):
        metric.spearman_correlation(prediction, loaded, min_distance=1000.0)

def test_loader_preserves_order_and_diagonal_sentinels(tmp_path, structural_distance_matrices):
    prediction, target = structural_distance_matrices
    target.fill_diagonal_(float('nan'))
    codes = ['n0', 'n1', 'n2', 'n3']
    path = tmp_path / 'distances.parquet'
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(path)
    order = [2, 0, 3, 1]
    loaded = load_distance_submatrix(path, [codes[i] for i in order])
    torch.testing.assert_close(loaded, target[order][:, order].float(), equal_nan=True)
    metric = HierarchyMetrics()
    metric.device = 'cpu'
    result = metric.spearman_correlation(prediction[order][:, order], loaded)
    assert result['correlation'].item() == pytest.approx(0.87831006565368, abs=1e-7)
    assert result['n_pairs'] == result['n_total'] == 6
