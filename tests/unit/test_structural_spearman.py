import importlib
import itertools
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from naics_embedder import metrics as public_metrics
from naics_embedder.metrics import HierarchyMetrics

pytestmark = pytest.mark.unit

@pytest.fixture
def metric() -> HierarchyMetrics:
    result = HierarchyMetrics()
    result.device = 'cpu'
    return result

def _matrix(pairs: Sequence[float | int], dtype: torch.dtype = torch.float64) -> torch.Tensor:
    a, b, c, d, e, f = pairs
    return torch.tensor([[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]], dtype=dtype)

def test_motivating_ties(metric, structural_distance_matrices):
    prediction, target = structural_distance_matrices
    result = metric.spearman_correlation(prediction, target)
    assert result['correlation'].item() == pytest.approx(0.87831006565368, abs=1e-7)
    assert result['n_total'] == result['n_pairs'] == 6
    assert result['definition'] == 'structural-spearman-v1'
    assert result['status'] == 'defined'
    assert result['reason'] is None
    assert public_metrics.STRUCTURAL_SPEARMAN_DEFINITION == 'structural-spearman-v1'
    assert public_metrics.STRUCTURAL_SPEARMAN_KEY == 'structural_spearman_v1'
    assert issubclass(public_metrics.StructuralMetricInputError, ValueError)

@pytest.mark.parametrize('order', list(itertools.permutations(range(4))))
def test_node_permutations_and_repeated_calls(metric, structural_distance_matrices, order):
    prediction, target = structural_distance_matrices
    expected = metric.spearman_correlation(prediction, target)['correlation']
    indices = torch.tensor(order)
    for _ in range(3):
        result = metric.spearman_correlation(
            prediction[indices][:, indices], target[indices][:, indices]
        )
        assert torch.equal(result['correlation'], expected)
        assert result['n_pairs'] == result['n_total'] == 6

@pytest.mark.parametrize(
    'predicted,target',
    [
        ([1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 1, 2]),
        ([6, 5, 4, 3, 2, 1], [1, 1, 1, 2, 2, 2]),
        ([1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]),
        ([1, 2, 3, 4, 5, 6], [1, 1 + 1e-8, 1 + 2e-8, 2, 2, 2]),
    ],
)
def test_matches_direct_scipy_without_near_tie_grouping(metric, predicted, target):
    expected = float(spearmanr(np.array(predicted, dtype=np.float64), target).statistic)
    actual = metric.spearman_correlation(_matrix(predicted), _matrix(target))
    assert actual['correlation'].item() == pytest.approx(expected, abs=1e-7)

@pytest.mark.parametrize('diagonal', [0.0, -100.0, 100.0, float('nan'), float('inf')])
def test_diagonal_is_ignored(metric, structural_distance_matrices, diagonal):
    prediction, target = structural_distance_matrices
    prediction.fill_diagonal_(diagonal)
    target.fill_diagonal_(diagonal)
    result = metric.spearman_correlation(prediction, target, min_distance=-200.0)
    assert result['correlation'].item() == pytest.approx(0.87831006565368, abs=1e-7)
    assert result['n_total'] == result['n_pairs'] == 6

def test_filter_counts_and_average_before_threshold(metric):
    prediction = _matrix([1, 2, 3, 4, 5, 6])
    target = _matrix([1, 2, 3, 0, 0, 0])
    target[0, 1] -= 1e-8
    target[1, 0] += 1e-8
    result = metric.spearman_correlation(prediction, target, min_distance=1.0)
    assert result['n_total'] == 6
    assert result['n_pairs'] == 3
    assert result['correlation'].item() == pytest.approx(1.0)

def test_near_symmetry_is_averaged_permutation_invariant_and_nonmutating(metric):
    prediction = _matrix([1, 1, 2, 3, 4, 5])
    target = _matrix([1, 2, 3, 4, 5, 6])
    prediction[0, 1] -= 1e-8
    prediction[1, 0] += 1e-8
    original = prediction.clone()
    expected = float(spearmanr([1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]).statistic)
    for order in itertools.permutations(range(4)):
        indices = torch.tensor(order)
        result = metric.spearman_correlation(
            prediction[indices][:, indices], target[indices][:, indices]
        )
        assert result['correlation'].item() == pytest.approx(expected, abs=1e-7)
    assert torch.equal(prediction, original)

@pytest.mark.parametrize(
    'dtype',
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_representable_values_agree_across_dtypes(metric, structural_distance_matrices, dtype):
    prediction, target = structural_distance_matrices
    expected = metric.spearman_correlation(prediction, target)['correlation']
    actual = metric.spearman_correlation(prediction.to(dtype), target.to(dtype))
    assert torch.equal(actual['correlation'], expected)
    assert actual['correlation'].dtype == torch.float32

@pytest.mark.parametrize(
    'dtype,atol',
    [(torch.float64, 1e-9), (torch.float32, 1e-7), (torch.float16, 1e-3), (torch.bfloat16, 1e-3)],
)
@pytest.mark.parametrize('side', [0, 1])
def test_each_input_uses_its_source_absolute_tolerance(
    metric, structural_distance_matrices, dtype, atol, side
):
    matrices = list(structural_distance_matrices)
    accepted = matrices[side].to(dtype)
    accepted[0, 1] = 0.0
    accepted[1, 0] = atol / 2.0
    matrices[side] = accepted
    assert metric.spearman_correlation(*matrices)['status'] == 'defined'
    accepted[1, 0] = atol * 4.0
    with pytest.raises(public_metrics.StructuralMetricInputError, match='symmetric'):
        metric.spearman_correlation(*matrices)

@pytest.mark.parametrize(
    'dtype,delta',
    [(torch.float64, 5e-6), (torch.float32, 5e-4), (torch.float16, 0.0625)],
)
def test_relative_tolerance_is_not_replaced_by_absolute_only(
    metric, structural_distance_matrices, dtype, delta
):
    prediction, target = structural_distance_matrices
    prediction = prediction.to(dtype)
    prediction[0, 1] = 100.0
    prediction[1, 0] = 100.0 + delta
    assert metric.spearman_correlation(prediction, target)['status'] == 'defined'

def test_integer_symmetry_is_checked_before_float64_conversion(metric):
    prediction = _matrix([2**53, 2, 3, 4, 5, 6], dtype=torch.int64)
    prediction[1, 0] = 2**53 + 1
    with pytest.raises(public_metrics.StructuralMetricInputError, match='symmetric'):
        metric.spearman_correlation(prediction, _matrix([1, 1, 1, 2, 2, 2]))

@pytest.mark.parametrize('transpose', [False, True])
def test_relative_tolerance_acceptance_does_not_depend_on_orientation(
    metric, structural_distance_matrices, transpose
):
    prediction, target = structural_distance_matrices
    tolerance = 1e-7 + 1e-9
    prediction[1, 0] = 1.0 + tolerance * (1.0 + 0.5e-7)
    if transpose:
        prediction = prediction.T
    with pytest.raises(public_metrics.StructuralMetricInputError, match='symmetric'):
        metric.spearman_correlation(prediction, target)

@pytest.mark.parametrize(
    'prediction,target',
    [
        (torch.ones(3, 3), torch.ones(4, 4)),
        (torch.ones(3, 4), torch.ones(3, 4)),
        (torch.ones(6), torch.ones(6)),
        (torch.ones(2, 2, 2), torch.ones(2, 2, 2)),
        (torch.ones(4, 4, dtype=torch.bool), torch.ones(4, 4)),
        (torch.ones(4, 4), torch.ones(4, 4, dtype=torch.bool)),
        (torch.ones(4, 4, dtype=torch.complex64), torch.ones(4, 4)),
        (torch.ones(4, 4), torch.ones(4, 4, dtype=torch.complex64)),
    ],
)
def test_invalid_shapes_and_dtypes(metric, prediction, target):
    with pytest.raises(public_metrics.StructuralMetricInputError):
        metric.spearman_correlation(prediction, target)

@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_threshold_is_invalid(metric, structural_distance_matrices, threshold):
    with pytest.raises(public_metrics.StructuralMetricInputError, match='min_distance'):
        metric.spearman_correlation(*structural_distance_matrices, min_distance=threshold)

@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('position', [(0, 1), (1, 0)])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_off_diagonal_cannot_be_filtered_out(
    metric, structural_distance_matrices, side, position, value
):
    matrices = list(structural_distance_matrices)
    matrices[side][position] = value
    with pytest.raises(public_metrics.StructuralMetricInputError, match='finite'):
        metric.spearman_correlation(*matrices, min_distance=1000.0)

@pytest.mark.parametrize('size', [0, 1, 2])
def test_small_square_inputs_are_valid_but_undefined(metric, size):
    matrix = torch.ones(size, size)
    result = metric.spearman_correlation(matrix, matrix)
    assert result['status'] == 'undefined'
    assert result['reason'] == 'fewer_than_two_observations'
    assert result['n_pairs'] == result['n_total'] == size * (size - 1) // 2
    assert torch.isnan(result['correlation'])

@pytest.mark.parametrize(
    'prediction,target,threshold,reason,n_pairs',
    [
        ([1] * 6, [1] * 6, 2.0, 'fewer_than_two_observations', 0),
        ([1] * 6, [0, 0, 0, 0, 0, 1], 1.0, 'fewer_than_two_observations', 1),
        ([1] * 6, [1] * 6, 0.1, 'constant_prediction_and_target', 6),
        ([1] * 6, [1, 1, 1, 2, 2, 2], 0.1, 'constant_prediction', 6),
        ([1, 2, 3, 4, 5, 6], [1] * 6, 0.1, 'constant_target', 6),
        ([1, 2, 3, 4, 5, 6], [1, 1, 1, 2, 2, 2], 2.0, 'constant_target', 3),
    ],
)
def test_undefined_reason_precedence(metric, prediction, target, threshold, reason, n_pairs):
    result = metric.spearman_correlation(_matrix(prediction), _matrix(target), threshold)
    assert result['status'] == 'undefined'
    assert result['reason'] == reason
    assert result['n_total'] == 6
    assert result['n_pairs'] == n_pairs
    assert result['correlation'].ndim == 0
    assert torch.isnan(result['correlation'])

@pytest.mark.parametrize('threshold', [0.1, 2.0])
def test_known_undefined_cases_do_not_call_scipy(monkeypatch, metric, threshold):
    implementation = importlib.import_module('naics_embedder.metrics.structural_spearman')

    def unexpected_call(*_args):
        pytest.fail('SciPy must not be called for a known undefined case')

    monkeypatch.setattr(implementation, 'spearmanr', unexpected_call)
    result = metric.spearman_correlation(torch.ones(4, 4), torch.ones(4, 4), threshold)
    assert result['status'] == 'undefined'
    assert torch.isnan(result['correlation'])

def test_extreme_finite_means_do_not_overflow_or_erase_subnormals(metric):
    smallest = np.nextafter(0.0, 1.0)
    prediction = _matrix([1e308, 1e308, smallest, smallest, 3, 4])
    prediction[2, 1] = smallest * 2.0
    target = _matrix([6, 5, 1, 2, 3, 4])
    expected_pairs = [1e308, 1e308, smallest, smallest * 2.0, 3, 4]
    result = metric.spearman_correlation(prediction, target)
    expected = float(spearmanr(expected_pairs, [6, 5, 1, 2, 3, 4]).statistic)
    assert result['correlation'].item() == pytest.approx(expected, abs=1e-7)

def test_scipy_receives_float64_vectors_and_result_has_no_grad(
    monkeypatch, metric, structural_distance_matrices
):
    implementation = importlib.import_module('naics_embedder.metrics.structural_spearman')
    observed = []

    def checked_spearman(prediction, target):
        observed.append((prediction.dtype, target.dtype, prediction.shape, target.shape))
        return spearmanr(prediction, target)

    monkeypatch.setattr(implementation, 'spearmanr', checked_spearman)
    prediction, target = structural_distance_matrices
    result = metric.spearman_correlation(
        prediction.float().requires_grad_(),
        target.float().requires_grad_()
    )
    assert observed == [(np.dtype('float64'), np.dtype('float64'), (6, ), (6, ))]
    scalar = result['correlation']
    assert scalar.dtype == torch.float32
    assert scalar.device == torch.device('cpu')
    assert scalar.ndim == 0
    assert scalar.requires_grad is False
    assert scalar.grad_fn is None

@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_unexpected_scipy_failure_is_not_an_undefined_result(
    monkeypatch, metric, structural_distance_matrices, value
):
    implementation = importlib.import_module('naics_embedder.metrics.structural_spearman')
    monkeypatch.setattr(
        implementation,
        'spearmanr',
        lambda *_: SimpleNamespace(statistic=value),
    )
    with pytest.raises(RuntimeError, match='structural-spearman-v1.*n_pairs=6'):
        metric.spearman_correlation(*structural_distance_matrices)

@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_available_devices_match_cpu(metric, structural_distance_matrices, device):
    available = (
        torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f'{device} is unavailable')
    prediction, target = [value.float() for value in structural_distance_matrices]
    expected = metric.spearman_correlation(prediction, target)['correlation']
    metric.device = device
    actual = metric.spearman_correlation(prediction.to(device), target.to(device))['correlation']
    assert actual.device.type == device
    assert torch.equal(actual.cpu(), expected)
