# Structural Spearman Metric Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace order-sensitive ordinal-rank reporting with one validated, tie-correct structural
Spearman contract across evaluation, training artifacts, and Stage-4 verification.

**Architecture:** A focused metric component validates complete square distance matrices,
canonicalizes unordered pairs on CPU, and delegates average-rank correlation to SciPy.
`HierarchyMetrics.spearman_correlation()` adapts that result to its tensor API; each caller owns
its existing logging/serialization boundary and publishes only versioned fields.

**Tech Stack:** Python 3.10+, PyTorch 2.4+, SciPy 1.13+, NumPy, Polars 1.9+, PyTorch Lightning
2.4+, Typer 0.12+, pytest 8.3+, Ruff, YAPF, MkDocs. All are existing dependencies.

## Global Constraints

The following requirements are copied from the
[approved spec](../structural-spearman-metric-integrity.md); its sections remain authoritative.

- The symbolic definition identifier is: `structural-spearman-v1`.
- External numeric fields use: `structural_spearman_v1`.
- `HierarchyMetrics.spearman_correlation()` remains the public Python entry point for source compatibility.
- Supported source dtypes are real floating and integer tensor dtypes.
- The diagonal is outside the observation population.
- A non-finite value in either orientation of an off-diagonal pair is invalid.
- `min_distance` remains in the public signature for compatibility. It must be finite.
- Apply the existing target filter after extraction: `canonical_target >= min_distance`.
- Invalid values are checked before filtering so a threshold cannot hide malformed data.
- Only exactly equal canonical `float64` observations form a tie; the symmetry tolerance validates mirrored entries but does not merge merely near-equal observations into one rank group.
- The public compatibility wrapper returns correlation as a detached `torch.float32` scalar on `HierarchyMetrics.device`.
- Malformed input is fatal at every public caller.
- Undefined correlation is non-fatal and is reported explicitly.
- Historical files are not rewritten.
- New code does not dual-write a legacy key, and no automatic numerical conversion is possible.
- The existing cophenetic, NDCG, and local retrieval checks remain the complete pass/fail gate.
- No Spearman threshold or CLI threshold option is added.
- No dependency migration is needed.
- Stage-4 verification remains explicitly fixed at `1.0`.
- HGCN full evaluation remains explicitly fixed at `1.0`.
- Text-stage comparison configurations explicitly retain `loss.curvature: 1.0`.
- The implementation neither repairs nor generalizes non-unit-curvature evaluation.
- Do not add a wall-clock pytest assertion.

Use the spec's exact source-dtype symmetry tolerances:

| Source dtype | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| `float64` | `1e-7` | `1e-9` |
| `float32` | `1e-5` | `1e-7` |
| `float16`, `bfloat16` | `1e-3` | `1e-3` |
| Integer | exact | exact |

Keep curvature formulas/gradients, HGCN design and objectives, QCEW evaluation, Stage-3 supervision,
other metric definitions, and unrelated visualization changes out of this plan. Preserve any user
edits, especially `conf/config.yaml`; this plan does not edit that file.

---

## Execution Baseline

The planning worktree starts from `da513da`, whose only change over `9fa0140` is the approved spec.
The spec was recovered from that Git commit, not by reading another checkout. Line ranges below
refer to this baseline; method names are the stable edit anchors.

This is one cross-cutting metric repair, not five independent features. Task 1 establishes the
shared contract; the remaining tasks integrate it without independently extracting or ranking
pairs. Execute in order because documentation and shared fixtures accumulate between tasks.

Use the current app-managed worktree if executing here. If executing elsewhere, first ensure the
saved plan and its source spec are present on that execution branch. Do not create another branch
inside an existing app-managed worktree.

Before implementation, inspect the worktree and capture the actual lint baseline:

```bash
git status --short --branch
uv run ruff check src/ tests/
```

The deferred backlog describes an older lint baseline; do not assume those failures still exist.
Record the current output, do not auto-fix unrelated files, and compare the final run against it.
Use `uv run` for project commands. If a selected command fails because dependencies are absent,
restore them with `uv sync --locked`, then rerun that command.

## File Structure and Ownership

| File | Responsibility | Task |
| --- | --- | --- |
| `src/naics_embedder/metrics/structural_spearman.py` (new) | Validation, canonical pairs, undefined classification, SciPy calculation, result types/constants | 1 |
| `src/naics_embedder/metrics/core.py:413-467,553-570` | Compatibility wrapper; remove obsolete ordinal ranking | 1 |
| `src/naics_embedder/metrics/__init__.py` | Public definition/key/error exports | 1 |
| `tests/conftest.py` | Two small shared matrix/embedding fixtures | 1 |
| `tests/unit/test_structural_spearman.py` (new) | Mathematical/input/device contract | 1 |
| `src/naics_embedder/utils/distance_matrix.py:54-61` | Preserve malformed file-backed values for the metric boundary | 2 |
| `src/naics_embedder/metrics/runner.py:82-91` | Versioned result and validation before other hierarchy metrics | 2 |
| `tests/unit/test_evaluation.py` | Runner contract and existing expectation migration | 2 |
| `tests/unit/test_distance_matrix.py` (new) | File-backed invalid-value preservation | 2 |
| `README.md:139-160`, `docs/overview.md:453-467` | Definition and historical migration documentation | 2 |
| `src/naics_embedder/text_model/mixins/validation.py` | Fatal input errors; Lightning and JSON reporting | 3 |
| `tests/unit/test_text_validation_metrics.py` (new) | Real validation hooks and JSON output without loading an encoder | 3 |
| `docs/text_training.md` | Text scalar/count/metadata contract and fixed-curvature comparisons | 3 |
| `src/naics_embedder/graph_model/hgcn.py` | Full-evaluation result, numeric-only logging, typed history metadata | 4 |
| `tests/unit/test_hgcn_metrics.py` | Full evaluation, hook/history/export regressions | 4 |
| `docs/hgcn_training.md:50-71` | HGCN metrics and `val_` history fields | 4 |
| `src/naics_embedder/tools/embeddings_verification.py` | Corrected pre/post/delta and metadata, unchanged gate | 5 |
| `src/naics_embedder/cli/commands/tools.py:221-329` | `N/A` rendering, report-only documentation | 5 |
| `tests/unit/test_embeddings_verification.py`, `tests/unit/test_cli_commands.py` | Verifier, CLI, malformed input, and gate regressions | 5 |
| `docs/hgcn_training.md:73-95`, `README.md:148-160` | Verifier output/migration and report-only policy | 5 |

Do not add a serialization framework, a second rank implementation, a new dependency, or a
standalone benchmark module. Existing API pages derive from source docstrings.

## Test Strategy: Invariants Before Test Code

| Invariant | Cheapest check | Fixture / marker |
| --- | --- | --- |
| Tied ranks yield `0.87831006565368`, not `1.0` | Public wrapper plus fixed numeric oracle | Four nodes; `unit` |
| Exactly `N(N-1)/2` candidates; diagonal never participates | Counts, filtering, diagonal sentinels | Handwritten matrices; `unit` |
| Joint node permutations and repeated calls preserve results | All 24 permutations, exact float32 equality | Four-node motivating case; `unit` |
| Mirrored roundoff is averaged, not independently ranked or selected | Independent SciPy oracle and threshold boundary | Near-symmetric float64 matrices; `unit` |
| Each source dtype uses its own tolerance; integer symmetry is exact | Parameterized validation, including integers above `2**53` | CPU matrices; `unit` |
| Near-equal observations are not collapsed into ties | Direct SciPy comparison | Exactly symmetric, near-equal targets; `unit` |
| Invalid observations cannot be hidden by filtering | Both orientations, both inputs, high threshold | `NaN`/infinity/asymmetry; `unit` |
| Undefined reasons have the specified precedence | Small populations and constant vectors | Empty/one/two-node and filtered matrices; `unit` |
| One CPU-float64 algorithm feeds a detached float32 scalar | SciPy input spy, dtype parity, autograd check | CPU dtypes; `unit` |
| Available accelerators agree with CPU | Conditional CUDA/MPS parity | Float32 inputs; `unit`, `gpu` |
| Every caller emits versioned keys and propagates input errors | Real public boundary with tiny distance inputs | Runner and validation harnesses; `unit` |
| Undefined values become JSON `null`, not numeric zero or `NaN` | Strict JSON decoding and captured Lightning logs | `tmp_path` artifacts; `unit` |
| Verifier reports Spearman without adding a release condition | Negative/undefined Spearman with passing original checks | Tiny Parquet inputs and real metric; `unit` |
| CLI displays undefined values/deltas as `N/A` and fails on invalid inputs | `CliRunner` | Existing CLI fixture; `unit` |

Mock neural inference, device selection, and the distance-producing boundary when needed; do not
mock `HierarchyMetrics.spearman_correlation()` in caller correctness tests. The mathematical
suite separately tests extraction/calculation against SciPy. No test downloads data or models.
Only accelerator availability causes skips; do not catch an available device's test failures.

Reuse the existing registered `unit` and `gpu` markers; no marker or workflow change is needed.
The current CI test command, on Python 3.10 and 3.12, is:

```bash
uv run pytest tests/ -v --cov=src/naics_embedder --cov-report=xml:coverage.xml --cov-report=term
```

---

### Task 1: Establish the Canonical Structural Spearman Contract

**Files:**
- Create: `src/naics_embedder/metrics/structural_spearman.py`
- Modify: `src/naics_embedder/metrics/core.py:13-20,413-467,553-570`
- Modify: `src/naics_embedder/metrics/__init__.py`
- Modify: `tests/conftest.py`
- Create: `tests/unit/test_structural_spearman.py`

**Interfaces:**
- Consumes: existing `HierarchyMetrics.device`; no earlier task.
- Produces: `STRUCTURAL_SPEARMAN_DEFINITION: str`, `STRUCTURAL_SPEARMAN_KEY: str`,
  `StructuralMetricInputError(ValueError)`.
- Produces:
  `compute_structural_spearman(embedding_distances: torch.Tensor, tree_distances: torch.Tensor, min_distance: float = 0.1) -> StructuralSpearmanComputation`.
- Produces:
  `HierarchyMetrics.spearman_correlation(embedding_distances: torch.Tensor, tree_distances: torch.Tensor, min_distance: float = 0.1) -> StructuralSpearmanResult`.
- `StructuralSpearmanComputation` holds a Python-float correlation; `StructuralSpearmanResult`
  has the same six fields but a scalar-tensor correlation. Metadata is `n_pairs: int`,
  `n_total: int`, `definition: str`, `status: Literal['defined', 'undefined']`,
  `reason: str | None`.
- Produces shared pytest fixtures
  `structural_distance_matrices -> tuple[torch.Tensor, torch.Tensor]` and
  `structural_lorentz_embeddings -> torch.Tensor`.

- [ ] **Step 1: Add the small fixtures and mathematical regressions**

Append these fixtures to `tests/conftest.py`. These are synthetic matrices, not asserted NAICS
relationships. Keep existing fixtures unchanged.

```python
@pytest.fixture
def structural_distance_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]],
        dtype=torch.float64,
    )
    return prediction, target

@pytest.fixture
def structural_lorentz_embeddings() -> torch.Tensor:
    spatial = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 0.3], [0.2, 0.4]])
    time = torch.sqrt(1.0 + spatial.square().sum(dim=1, keepdim=True))
    return torch.cat([time, spatial], dim=1)
```

Create `tests/unit/test_structural_spearman.py` with the following complete tests. Public exports
are accessed through `public_metrics` inside tests so the motivating regression can run before
the new component exists.

```python
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

def _matrix(
    pairs: Sequence[float | int], dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    a, b, c, d, e, f = pairs
    return torch.tensor(
        [[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]], dtype=dtype
    )

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
        torch.float64, torch.float32, torch.float16, torch.bfloat16,
        torch.uint8, torch.uint16, torch.uint32, torch.uint64,
        torch.int8, torch.int16, torch.int32, torch.int64,
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
    [(torch.float64, 1e-9), (torch.float32, 1e-7),
     (torch.float16, 1e-3), (torch.bfloat16, 1e-3)],
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
        prediction.float().requires_grad_(), target.float().requires_grad_()
    )
    assert observed == [(np.dtype('float64'), np.dtype('float64'), (6,), (6,))]
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
        implementation, 'spearmanr',
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
```

- [ ] **Step 2: Demonstrate the original numerical failure**

```bash
uv run pytest tests/unit/test_structural_spearman.py::test_motivating_ties -q
```

Expected: FAIL because the current implementation returns `1.0`, not approximately
`0.87831006565368`. Then run the file once to expose the missing metadata and input guards:

```bash
uv run pytest tests/unit/test_structural_spearman.py -q
```

Expected: failures in the new contract tests; do not weaken the assertions.

- [ ] **Step 3: Implement validation, canonicalization, and the SciPy boundary**

Create `src/naics_embedder/metrics/structural_spearman.py`:

```python
'''Canonical, evaluation-only structural Spearman correlation.'''

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr

# -------------------------------------------------------------------------------------------------
# Contract
# -------------------------------------------------------------------------------------------------

STRUCTURAL_SPEARMAN_DEFINITION = 'structural-spearman-v1'
STRUCTURAL_SPEARMAN_KEY = 'structural_spearman_v1'
_MIN_OBSERVATIONS = 2
_FLOAT64_HALF_MAX = np.finfo(np.float64).max / 2.0
_SYMMETRY_TOLERANCES = {
    torch.float64: (1e-7, 1e-9),
    torch.float32: (1e-5, 1e-7),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-3, 1e-3),
}
_INTEGER_DTYPES = {
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    torch.int8, torch.int16, torch.int32, torch.int64,
}

class StructuralMetricInputError(ValueError):
    '''Malformed distance matrices or filtering threshold.'''

@dataclass(frozen=True)
class StructuralSpearmanComputation:
    correlation: float
    n_pairs: int
    n_total: int
    status: Literal['defined', 'undefined']
    reason: str | None
    definition: str = STRUCTURAL_SPEARMAN_DEFINITION

class StructuralSpearmanResult(TypedDict):
    correlation: torch.Tensor
    n_pairs: int
    n_total: int
    definition: str
    status: Literal['defined', 'undefined']
    reason: str | None

# -------------------------------------------------------------------------------------------------
# Canonical observations
# -------------------------------------------------------------------------------------------------

def _canonical_pairs(
    matrix: torch.Tensor,
    name: str,
    indices: tuple[np.ndarray, np.ndarray],
) -> NDArray[np.float64]:
    source_dtype = matrix.dtype
    cpu = matrix.detach().cpu()
    is_integer = source_dtype in _INTEGER_DTYPES
    values = cpu.numpy() if is_integer else cpu.to(torch.float64).numpy()
    rows, columns = indices
    upper = values[rows, columns]
    lower = values[columns, rows]

    if not (np.isfinite(upper).all() and np.isfinite(lower).all()):
        raise StructuralMetricInputError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: {name} off-diagonal values must be finite'
        )

    if is_integer:
        symmetric = np.array_equal(upper, lower)
    else:
        rtol, atol = _SYMMETRY_TOLERANCES[source_dtype]
        symmetric = (
            np.allclose(upper, lower, rtol=rtol, atol=atol)
            and np.allclose(lower, upper, rtol=rtol, atol=atol)
        )
    if not symmetric:
        raise StructuralMetricInputError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: {name} must be symmetric for {source_dtype}'
        )

    upper = upper.astype(np.float64, copy=False)
    lower = lower.astype(np.float64, copy=False)
    large = (np.abs(upper) > _FLOAT64_HALF_MAX) | (np.abs(lower) > _FLOAT64_HALF_MAX)
    canonical = np.empty_like(upper)
    # Preserve subnormal means without overflowing the sum of large finite operands.
    canonical[~large] = (upper[~large] + lower[~large]) / 2.0
    canonical[large] = upper[large] / 2.0 + lower[large] / 2.0
    return canonical

# -------------------------------------------------------------------------------------------------
# Calculation
# -------------------------------------------------------------------------------------------------

def compute_structural_spearman(
    embedding_distances: torch.Tensor,
    tree_distances: torch.Tensor,
    min_distance: float = 0.1,
) -> StructuralSpearmanComputation:
    '''Validate square distances and correlate canonical non-self unordered pairs.

    Diagonal entries are ignored, including non-finite sentinels. Mirrored entries
    must be finite and symmetric within their source dtype's tolerance. Their
    float64 arithmetic means are filtered by target >= min_distance. Exact ties
    receive SciPy average ranks; no tolerance-based rank grouping is performed.

    Raises:
        StructuralMetricInputError: Invalid shape, dtype, threshold, or observations.
        RuntimeError: SciPy returns a non-finite correlation for defined inputs.
    '''
    for name, matrix in (
        ('embedding_distances', embedding_distances),
        ('tree_distances', tree_distances),
    ):
        if not isinstance(matrix, torch.Tensor):
            raise StructuralMetricInputError(
                f'{STRUCTURAL_SPEARMAN_DEFINITION}: {name} must be a torch.Tensor'
            )
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise StructuralMetricInputError(
                f'{STRUCTURAL_SPEARMAN_DEFINITION}: {name} must be a square 2D matrix'
            )
        if matrix.dtype not in _SYMMETRY_TOLERANCES and matrix.dtype not in _INTEGER_DTYPES:
            raise StructuralMetricInputError(
                f'{STRUCTURAL_SPEARMAN_DEFINITION}: unsupported {name} dtype {matrix.dtype}'
            )
    if embedding_distances.shape != tree_distances.shape:
        raise StructuralMetricInputError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: distance matrices must have the same shape'
        )
    if not math.isfinite(min_distance):
        raise StructuralMetricInputError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: min_distance must be finite'
        )

    indices = np.triu_indices(embedding_distances.shape[0], k=1)
    prediction = _canonical_pairs(embedding_distances, 'embedding_distances', indices)
    target = _canonical_pairs(tree_distances, 'tree_distances', indices)
    n_total = int(target.size)
    selected = target >= min_distance
    prediction = prediction[selected]
    target = target[selected]
    n_pairs = int(target.size)

    reason = None
    if n_pairs < _MIN_OBSERVATIONS:
        reason = 'fewer_than_two_observations'
    else:
        constant_prediction = bool(np.all(prediction == prediction[0]))
        constant_target = bool(np.all(target == target[0]))
        if constant_prediction and constant_target:
            reason = 'constant_prediction_and_target'
        elif constant_prediction:
            reason = 'constant_prediction'
        elif constant_target:
            reason = 'constant_target'
    if reason is not None:
        return StructuralSpearmanComputation(
            float('nan'), n_pairs, n_total, 'undefined', reason
        )

    correlation = float(spearmanr(prediction, target).statistic)
    if not math.isfinite(correlation):
        raise RuntimeError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: non-finite SciPy correlation for n_pairs={n_pairs}'
        )
    return StructuralSpearmanComputation(correlation, n_pairs, n_total, 'defined', None)
```

The two `allclose` directions are intentional: together they match comparing the entire matrix
with its transpose, rather than accepting an order-dependent relative-tolerance boundary.
Integer equality precedes float64 promotion so a one-unit mismatch above `2**53` is not hidden.
The arithmetic mean is commutative even when a permutation swaps a pair's orientation.

- [ ] **Step 4: Delegate the compatibility method and publish the contract**

Add this import to `src/naics_embedder/metrics/core.py`:

```python
from naics_embedder.metrics.structural_spearman import (
    StructuralSpearmanResult,
    compute_structural_spearman,
)
```

Replace the complete `HierarchyMetrics.spearman_correlation` method with:

```python
    def spearman_correlation(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        min_distance: float = 0.1,
    ) -> StructuralSpearmanResult:
        '''Compute structural-spearman-v1 over unique unordered non-self pairs.

        Inputs must be same-shaped square, real numeric tensors. The calculation
        validates both off-diagonal orientations, averages mirrored values on CPU
        in float64, filters canonical targets >= min_distance, and uses SciPy's
        average ranks for exact ties. Diagonal values are ignored.

        Returns:
            A detached float32 scalar on self.device, pair counts, definition,
            status, and reason. Undefined correlations are NaN with an explicit
            reason; malformed input raises StructuralMetricInputError.
        '''
        result = compute_structural_spearman(
            embedding_distances, tree_distances, min_distance=min_distance
        )
        return {
            'correlation': torch.tensor(
                result.correlation, dtype=torch.float32, device=self.device
            ),
            'n_pairs': result.n_pairs,
            'n_total': result.n_total,
            'definition': result.definition,
            'status': result.status,
            'reason': result.reason,
        }
```

Delete the entire `_rank_tensor` method at baseline lines 553-570. Do not alter
`_pearson_correlation`, which is still used by cophenetic correlation.

Add the following import to `src/naics_embedder/metrics/__init__.py` and these three exact
strings to its existing `__all__` list:

```python
from .structural_spearman import (
    STRUCTURAL_SPEARMAN_DEFINITION,
    STRUCTURAL_SPEARMAN_KEY,
    StructuralMetricInputError,
)
```

```python
    'STRUCTURAL_SPEARMAN_DEFINITION',
    'STRUCTURAL_SPEARMAN_KEY',
    'StructuralMetricInputError',
```

- [ ] **Step 5: Verify the complete focused contract**

Run formatting and the focused files:

```bash
./scripts/format_code.sh src/naics_embedder/metrics/structural_spearman.py \
  src/naics_embedder/metrics/core.py src/naics_embedder/metrics/__init__.py \
  tests/conftest.py tests/unit/test_structural_spearman.py
uv run pytest tests/unit/test_structural_spearman.py tests/unit/test_evaluation.py -q
```

Expected: PASS, with CUDA/MPS tests skipped only when unavailable. Existing runner key
expectations still pass at this task because the runner key is migrated in Task 2.

- [ ] **Step 6: Commit the canonical metric**

```bash
git add src/naics_embedder/metrics/structural_spearman.py \
  src/naics_embedder/metrics/core.py src/naics_embedder/metrics/__init__.py \
  tests/conftest.py tests/unit/test_structural_spearman.py
git commit -m "fix(metrics): define canonical tied-rank structural Spearman" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Integrate the Runner and Preserve File-Backed Invalid Observations

**Files:**
- Modify: `src/naics_embedder/metrics/runner.py:82-91`
- Modify: `src/naics_embedder/utils/distance_matrix.py:54-61`
- Modify: `tests/unit/test_evaluation.py:565-574,593-613`
- Create: `tests/unit/test_distance_matrix.py`
- Modify: `README.md:139-147`, `docs/overview.md:453-467`

**Interfaces:**
- Consumes: Task 1's public wrapper, `STRUCTURAL_SPEARMAN_KEY`,
  `StructuralMetricInputError`, and `structural_distance_matrices` fixture.
- Produces:
  `NAICSEvaluationRunner.evaluate(embeddings: torch.Tensor, tree_distances: Optional[torch.Tensor] = None, ground_truth_relevance: Optional[torch.Tensor] = None, k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]`
  with top-level
  `structural_spearman_v1: StructuralSpearmanResult`; no `spearman_correlation` result key.
- Preserves:
  `load_distance_submatrix(distance_matrix_path: Union[str, Path], node_codes: Sequence[str]) -> torch.Tensor`;
  file-backed non-finite values now reach the shared validation boundary unchanged.

- [ ] **Step 1: Add runner and Parquet-boundary regressions**

Add these imports to `tests/unit/test_evaluation.py`:

```python
from unittest.mock import MagicMock

from naics_embedder.metrics import StructuralMetricInputError
```

Append these tests to that file. Device overrides keep the tests independent of the host's
preferred accelerator; the runner's model is not used by `evaluate`.

```python
@pytest.mark.unit
def test_runner_emits_complete_versioned_spearman(
    monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    prediction, target = structural_distance_matrices
    runner = NAICSEvaluationRunner(MagicMock())
    runner.embedding_stats.device = 'cpu'
    runner.hierarchy_metrics.device = 'cpu'
    monkeypatch.setattr(
        runner.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    result = runner.evaluate(structural_lorentz_embeddings, tree_distances=target)
    assert 'spearman_correlation' not in result
    record = result['structural_spearman_v1']
    assert set(record) == {'correlation', 'n_pairs', 'n_total', 'definition', 'status', 'reason'}
    assert record['correlation'].item() == pytest.approx(0.87831006565368, abs=1e-7)
    assert record['definition'] == 'structural-spearman-v1'
    assert record['n_pairs'] == record['n_total'] == 6
    assert record['status'] == 'defined'
    assert record['reason'] is None

@pytest.mark.unit
def test_runner_preserves_undefined_metadata(
    monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    prediction, target = structural_distance_matrices
    target.fill_(1.0)
    runner = NAICSEvaluationRunner(MagicMock())
    runner.embedding_stats.device = 'cpu'
    runner.hierarchy_metrics.device = 'cpu'
    monkeypatch.setattr(
        runner.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    result = runner.evaluate(structural_lorentz_embeddings, tree_distances=target)
    record = result['structural_spearman_v1']
    assert record['status'] == 'undefined'
    assert record['reason'] == 'constant_target'
    assert torch.isnan(record['correlation'])
    assert record['n_pairs'] == record['n_total'] == 6
    assert 'spearman_correlation' not in result

@pytest.mark.unit
def test_runner_propagates_malformed_inputs(
    monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    prediction, target = structural_distance_matrices
    target[1, 0] = float('nan')
    runner = NAICSEvaluationRunner(MagicMock())
    runner.embedding_stats.device = 'cpu'
    runner.hierarchy_metrics.device = 'cpu'
    monkeypatch.setattr(
        runner.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    with pytest.raises(StructuralMetricInputError, match='tree_distances'):
        runner.evaluate(structural_lorentz_embeddings, tree_distances=target)
```

In each of the two existing runner tests named in **Files**, replace the old Spearman-key
assertion with these exact assertions:

```python
        assert 'structural_spearman_v1' in results
        assert 'spearman_correlation' not in results
```

Create `tests/unit/test_distance_matrix.py`:

```python
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
```

- [ ] **Step 2: Run the boundary tests red**

```bash
uv run pytest tests/unit/test_evaluation.py tests/unit/test_distance_matrix.py -q
```

Expected: new key assertions fail; the loader loses `NaN` observations and diagonal sentinels.
Infinity-only preservation tests can already pass; that does not invalidate the `NaN` regression.

- [ ] **Step 3: Migrate the runner and remove only loader sanitization**

Add this import to `src/naics_embedder/metrics/runner.py`:

```python
from naics_embedder.metrics.structural_spearman import STRUCTURAL_SPEARMAN_KEY
```

Replace the runner's entire `if tree_distances is not None:` block with:

```python
        if tree_distances is not None:
            logger.info('Evaluating hierarchy preservation...')
            results[STRUCTURAL_SPEARMAN_KEY] = self.hierarchy_metrics.spearman_correlation(
                emb_distances, tree_distances
            )
            results['cophenetic_correlation'] = self.hierarchy_metrics.cophenetic_correlation(
                emb_distances, tree_distances
            )
            results['distortion'] = self.hierarchy_metrics.distortion(emb_distances, tree_distances)
```

Replace the tail of `load_distance_submatrix`, beginning with `index_array =`, with:

```python
    index_array = np.array([code_to_idx[code] for code in node_codes], dtype=np.int64)
    matrix_np = df.to_numpy()
    subset = matrix_np[np.ix_(index_array, index_array)]
    return torch.from_numpy(subset).float()
```

Remove its now-unused `import logging` and `logger = logging.getLogger(__name__)` lines.
Append this sentence to the loader's existing return documentation:

```text
        Non-finite entries are preserved for validation at the metric boundary.
```

Do not change column parsing, code ordering, missing-code errors, or float32 conversion.

- [ ] **Step 4: Document the metric definition and historical incompatibility**

In `README.md` under **5.4 Validation Metrics**, replace the Spearman bullet with:

```markdown
- Structural Spearman v1 (`structural_spearman_v1`) and unique-pair counts
```

After that section's metric list, insert:

```markdown
`structural-spearman-v1` validates square symmetric distance matrices, averages each mirrored
pair in CPU float64, and uses only the strict upper triangle (`i < j`), excluding the diagonal.
After filtering canonical target distances at `min_distance` (default `0.1`), SciPy computes
Spearman correlation with average ranks for exact ties. Counts distinguish all unordered pairs
from those retained by the filter.

Malformed off-diagonal values, shapes, dtypes, thresholds, or asymmetry raise
`StructuralMetricInputError`. Valid but undefined correlations have an explicit status/reason
and serialize as JSON `null`; they are not logged as numeric zero or `NaN` in Lightning.

Historical unversioned fields (`spearman`, `spearman_correlation`, `val/spearman_correlation`,
and `val_spearman_correlation`) are `legacy-ordinal-rank-v0`: order-sensitive ordinal-rank
results, not valid tied-rank Spearman coefficients. They are not directly comparable with v1.
Historical files remain untouched, and new reports do not dual-write legacy keys. See the
[metric contract](docs/overview.md#structural-spearman-v1) and the
[text](docs/text_training.md) and [HGCN](docs/hgcn_training.md) artifact documentation.
```

In `docs/overview.md`, replace the Spearman table row with:

```markdown
| Structural Spearman v1 (`structural_spearman_v1`) | Average-rank correlation of canonical unordered distance pairs | Defined values approach 1.0 |
```

Insert this complete subsection immediately after the hierarchy-preservation metric table:

```markdown
### Structural Spearman v1

The definition identifier is `structural-spearman-v1`; external fields use
`structural_spearman_v1`. The Python entry point remains
`HierarchyMetrics.spearman_correlation(predicted_distances, target_distances, min_distance=0.1)`.
Inputs are equal-shaped square tensors with the documented real floating or integer dtypes.
They are detached and transferred to CPU before validation and calculation.

Both orientations of every off-diagonal pair must be finite and symmetric. Each matrix uses its
own source-dtype tolerance: float64 `(rtol=1e-7, atol=1e-9)`, float32 `(1e-5, 1e-7)`,
float16/bfloat16 `(1e-3, 1e-3)`, and exact equality for integers. Mirrored values are promoted to
float64 and arithmetically averaged; the strict upper triangle contributes exactly
`N(N-1)/2` candidates. The diagonal, including non-finite diagonal sentinels, is ignored.

Canonical target distances below the finite `min_distance` threshold are removed only after
validation. `n_total` counts candidates before filtering; `n_pairs` counts observations after it.
SciPy uses average ranks for exact ties in the two float64 vectors; tolerance does not group
near-equal observations. The p-value is discarded. The public result's `correlation` is a
detached float32 scalar on the metric's configured device; CUDA and MPS do not select a
different ranking algorithm. Source quantization cannot be reversed by promotion.

Malformed inputs raise `StructuralMetricInputError`, a `ValueError` subclass. Statistically
undefined results have a `NaN` tensor and these ordered reasons: fewer than two filtered
observations (`fewer_than_two_observations`), both vectors constant
(`constant_prediction_and_target`), prediction constant (`constant_prediction`), or target
constant (`constant_target`). Defined results have `status='defined'` and `reason=None`.
An unexpected non-finite SciPy result for otherwise defined inputs raises `RuntimeError`.

The general evaluation runner returns the complete result under `structural_spearman_v1`:
`correlation`, `n_pairs`, `n_total`, `definition`, `status`, and `reason`. Training artifacts
serialize undefined correlation as JSON `null`; Lightning omits that numeric scalar but logs
the pair counts. Stage-4 verification reports pre/post/delta values and metadata but does not
gate acceptance on Spearman.

All unversioned historical fields (`spearman`, `spearman_correlation`,
`val/spearman_correlation`, `val_spearman_correlation`) identify
`legacy-ordinal-rank-v0`. Their order-sensitive ordinal ranks are not valid tied-rank Spearman
and are not directly comparable with v1. Do not rewrite, dual-write, or numerically convert old
artifacts.

This rank repair does not validate the formula that produced the distance matrices. HGCN full
evaluation and the Stage-4 verifier remain fixed at curvature `1.0`; text comparison runs retain
`loss.curvature: 1.0`. Non-unit-curvature metric corrections are a separate change.
```

- [ ] **Step 5: Verify and commit the runner/file boundary**

```bash
./scripts/format_code.sh src/naics_embedder/metrics/runner.py \
  src/naics_embedder/utils/distance_matrix.py tests/unit/test_evaluation.py \
  tests/unit/test_distance_matrix.py
uv run pytest tests/unit/test_structural_spearman.py tests/unit/test_evaluation.py \
  tests/unit/test_distance_matrix.py -q
git diff --check
git add src/naics_embedder/metrics/runner.py src/naics_embedder/utils/distance_matrix.py \
  tests/unit/test_evaluation.py tests/unit/test_distance_matrix.py README.md docs/overview.md
git commit -m "fix(metrics): version runner output and preserve invalid distances" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

Expected: all selected tests pass, only unavailable-device tests skip, and diff checking is clean.

### Task 3: Make Text Validation Fail Closed and Publish Versioned JSON

**Files:**
- Modify: `src/naics_embedder/text_model/mixins/validation.py:13-20,192-199,343-399,474-478`
- Create: `tests/unit/test_text_validation_metrics.py`
- Modify: `docs/text_training.md`

**Interfaces:**
- Consumes:
  `HierarchyMetrics.spearman_correlation(embedding_distances: torch.Tensor, tree_distances: torch.Tensor, min_distance: float = 0.1) -> StructuralSpearmanResult`,
  `STRUCTURAL_SPEARMAN_KEY`, `StructuralMetricInputError`, and both shared fixtures.
- Produces: defined-only Lightning scalar `val/structural_spearman_v1`; always logs
  `val/structural_spearman_v1_n_pairs` and `val/structural_spearman_v1_n_total`.
- Produces: six flat `evaluation_metrics.json` fields: `structural_spearman_v1`,
  `structural_spearman_v1_n_pairs`, `structural_spearman_v1_n_total`,
  `structural_spearman_v1_status`, `structural_spearman_v1_reason`,
  `structural_spearman_v1_definition`. Undefined numeric values are `None` before JSON encoding.
- Preserves:
  `_compute_validation_metrics(embeddings: torch.Tensor, codes: List[str], gt_dists: torch.Tensor, num_samples: int) -> Dict[str, Any]`,
  the existing JSON history
  envelope, clustering behavior, and `finally` cleanup.

- [ ] **Step 1: Write real-hook logging and serialization tests**

Create `tests/unit/test_text_validation_metrics.py`. This harness uses the production validation
and logging mixins, geometry diagnostics, statistics, hierarchy metrics, and JSON writer. It
avoids transformer initialization and replaces only the upstream distance producer.

```python
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from naics_embedder.metrics import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    StructuralMetricInputError,
)
from naics_embedder.text_model.mixins.logging import LoggingMixin
from naics_embedder.text_model.mixins.validation import ValidationMixin

pytestmark = pytest.mark.unit

class ValidationHarness(ValidationMixin, LoggingMixin):

    def __init__(self, directory: Path, target: torch.Tensor, embeddings: torch.Tensor) -> None:
        self.device = torch.device('cpu')
        self.current_epoch = 0
        self.hparams = SimpleNamespace(curvature=1.0, eval_sample_size=4, eval_every_n_epochs=1)
        self.trainer = SimpleNamespace(callback_metrics={})
        self.logger = SimpleNamespace(log_dir=str(directory))
        self.log = Mock()
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()
        for component in (self.embedding_eval, self.embedding_stats, self.hierarchy_metrics):
            component.device = 'cpu'
        self.naics_hierarchy = None
        self.supervision_policy = SimpleNamespace(enable_pseudo_related=False)
        self.ground_truth_distances = target
        self.code_to_idx = {f'n{i}': i for i in range(4)}
        self.validation_embeddings = {f'n{i}': embeddings[i] for i in range(4)}
        self.validation_codes = list(self.validation_embeddings)
        self.evaluation_metrics_history = []

@pytest.mark.parametrize('undefined', [False, True])
def test_text_validation_versioned_logs_and_strict_json(
    tmp_path, caplog, monkeypatch, structural_distance_matrices,
    structural_lorentz_embeddings, undefined
):
    prediction, target = structural_distance_matrices
    if undefined:
        target.fill_(1.0)
        target.fill_diagonal_(0.0)
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    harness.on_validation_epoch_end()

    history = json.loads((tmp_path / 'evaluation_metrics.json').read_text())
    json.dumps(history, allow_nan=False)
    record = history[0]
    key = 'structural_spearman_v1'
    assert record[f'{key}_definition'] == 'structural-spearman-v1'
    assert record[f'{key}_n_pairs'] == record[f'{key}_n_total'] == 6
    assert isinstance(record[f'{key}_n_pairs'], int)
    assert {'spearman', 'spearman_correlation', 'spearman_n_pairs'}.isdisjoint(record)
    logged = {call.args[0]: call.args[1] for call in harness.log.call_args_list}
    assert logged[f'val/{key}_n_pairs'] == logged[f'val/{key}_n_total'] == 6
    assert 'val/spearman_correlation' not in logged
    assert 'val/spearman_n_pairs' not in logged
    assert f'val/{key}_status' not in logged
    warnings = [
        item.getMessage() for item in caplog.records
        if 'structural-spearman-v1 undefined:' in item.getMessage()
    ]
    if undefined:
        assert record[key] is None
        assert record[f'{key}_status'] == 'undefined'
        assert record[f'{key}_reason'] == 'constant_target'
        assert f'val/{key}' not in logged
        assert len(warnings) == 1
        assert 'constant_target' in warnings[0]
    else:
        assert record[key] == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'{key}_status'] == 'defined'
        assert record[f'{key}_reason'] is None
        assert logged[f'val/{key}'] == pytest.approx(record[key])
        assert warnings == []
    assert harness.validation_embeddings == {}
    assert harness.validation_codes == []

@pytest.mark.parametrize('invalid', ['nan', 'asymmetric', 'unaligned_codes'])
def test_text_validation_propagates_input_errors_before_hierarchy_logging(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings, invalid
):
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    if invalid == 'nan':
        target[1, 0] = float('nan')
    elif invalid == 'asymmetric':
        target[1, 0] = 10.0
    else:
        del harness.code_to_idx['n3']
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    with pytest.raises(StructuralMetricInputError):
        harness.on_validation_epoch_end()
    logged_names = [call.args[0] for call in harness.log.call_args_list]
    assert not any(
        'cophenetic' in name or 'ndcg@' in name or 'distortion' in name
        for name in logged_names
    )
    assert harness.evaluation_metrics_history == []
    assert not (tmp_path / 'evaluation_metrics.json').exists()
    assert harness.validation_embeddings == {}
    assert harness.validation_codes == []
```

- [ ] **Step 2: Run the text boundary red**

```bash
uv run pytest tests/unit/test_text_validation_metrics.py -q
```

Expected: missing versioned fields and failure to propagate `StructuralMetricInputError`.
The unaligned-code case additionally exercises a shape mismatch that currently disappears
inside the broad epoch-end exception handler.

- [ ] **Step 3: Move the metric to the first hierarchy boundary and serialize explicit state**

Add this import to `text_model/mixins/validation.py`:

```python
from naics_embedder.metrics.structural_spearman import (
    STRUCTURAL_SPEARMAN_KEY,
    StructuralMetricInputError,
)
```

In `on_validation_epoch_end`, replace the existing `except Exception` block with the following;
leave the existing `finally` block in place:

```python
        except StructuralMetricInputError:
            raise
        except Exception as e:
            logger.error(f'Error during evaluation: {e}', exc_info=True)
```

In `_compute_validation_metrics`, immediately after the existing
`emb_dists = self.embedding_eval.compute_pairwise_distances(...)` call and before
`_log_radius_structure_metrics`, insert:

```python
        spearman_result = self.hierarchy_metrics.spearman_correlation(emb_dists, gt_dists)
        spearman_value = (
            self._to_python_scalar(spearman_result['correlation'])
            if spearman_result['status'] == 'defined' else None
        )
        spearman_fields = {
            STRUCTURAL_SPEARMAN_KEY: spearman_value,
            f'{STRUCTURAL_SPEARMAN_KEY}_n_pairs': spearman_result['n_pairs'],
            f'{STRUCTURAL_SPEARMAN_KEY}_n_total': spearman_result['n_total'],
            f'{STRUCTURAL_SPEARMAN_KEY}_status': spearman_result['status'],
            f'{STRUCTURAL_SPEARMAN_KEY}_reason': spearman_result['reason'],
            f'{STRUCTURAL_SPEARMAN_KEY}_definition': spearman_result['definition'],
        }
        if spearman_value is not None:
            self.log(
                f'val/{STRUCTURAL_SPEARMAN_KEY}',
                spearman_value,
                batch_size=num_samples,
                sync_dist=True,
            )
        else:
            logger.warning(
                '%s undefined: %s (n_pairs=%d, n_total=%d)',
                spearman_result['definition'],
                spearman_result['reason'],
                spearman_result['n_pairs'],
                spearman_result['n_total'],
            )
        for count, value in (
            ('n_pairs', spearman_result['n_pairs']),
            ('n_total', spearman_result['n_total']),
        ):
            self.log(
                f'val/{STRUCTURAL_SPEARMAN_KEY}_{count}',
                value,
                batch_size=num_samples,
                sync_dist=True,
            )
```

Delete the old block beginning `# Compute Spearman for backward compatibility` through the
second `self.log(...)` call, at baseline lines 386-399. That obsolete comment disappears with
the behavior it described; do not leave a second calculation or legacy scalar.

Inside `epoch_metrics`, replace the two old `spearman_correlation` / `spearman_n_pairs` entries
with this dictionary expansion:

```python
            **spearman_fields,
```

Do not change the generic JSON writer to reject every unrelated metric's historic behavior.
Only this metric's undefined scalar is mapped to `None` at its owning boundary.

- [ ] **Step 4: Document text artifacts and comparison configuration**

Add this top-level TOC entry in `docs/text_training.md` before **Resuming and Overrides**:

```markdown
  - [Structural Spearman Validation](#structural-spearman-validation)
```

Insert this section before **Resuming and Overrides**:

````markdown
## Structural Spearman Validation

Text validation reports `structural-spearman-v1` through versioned fields. Each mirrored distance
pair is validated and averaged in CPU float64; only the strict upper triangle (`i < j`) is used,
with the diagonal excluded. Canonical target distances are filtered at `min_distance=0.1`.
SciPy assigns average ranks to exact ties. See the
[complete input and undefined-result contract](overview.md#structural-spearman-v1).

Lightning logs `val/structural_spearman_v1` only when defined and always logs
`val/structural_spearman_v1_n_pairs` and `val/structural_spearman_v1_n_total`.
Undefined results emit one warning with the exact reason and omit the numeric scalar.
Malformed inputs raise `StructuralMetricInputError` and fail validation rather than being
swallowed by the epoch-end evaluation handler.

The existing `evaluation_metrics.json` history includes these fields. For example, a valid
four-node evaluation with a constant target produces:

```json
{
  "structural_spearman_v1": null,
  "structural_spearman_v1_n_pairs": 6,
  "structural_spearman_v1_n_total": 6,
  "structural_spearman_v1_status": "undefined",
  "structural_spearman_v1_reason": "constant_target",
  "structural_spearman_v1_definition": "structural-spearman-v1"
}
```

When defined, the value is numeric, status is `defined`, and reason is `null`. The other
undefined reasons are `fewer_than_two_observations`, `constant_prediction_and_target`, and
`constant_prediction`, in that precedence before `constant_target`.

Unversioned historical `spearman`, `spearman_correlation`, `val/spearman_correlation`, and
`val_spearman_correlation` fields are `legacy-ordinal-rank-v0`: defective, order-sensitive
ordinal-rank results, not directly comparable with v1. Existing files are not rewritten, and
new evaluations do not emit legacy aliases.

For comparisons covered by this repair, retain `loss.curvature: 1.0`. After configuring the
supervision manifest as described above, make the comparison setting explicit:

```bash
uv run naics-embedder train --config conf/config.yaml loss.curvature=1.0
```

The rank fix does not correct non-unit-curvature distances. HGCN full evaluation and Stage-4
verification also remain fixed at `1.0`; Stage-4 reports structural Spearman without using it
as an acceptance threshold.

---
````

- [ ] **Step 5: Verify and commit the text boundary**

```bash
./scripts/format_code.sh src/naics_embedder/text_model/mixins/validation.py \
  tests/unit/test_text_validation_metrics.py
uv run pytest tests/unit/test_structural_spearman.py \
  tests/unit/test_text_validation_metrics.py tests/unit/test_naics_model.py -q
git diff --check
git add src/naics_embedder/text_model/mixins/validation.py \
  tests/unit/test_text_validation_metrics.py docs/text_training.md
git commit -m "fix(validation): publish versioned structural Spearman metadata" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

Expected: the new hook/JSON regressions and existing model tests pass; invalid inputs leave no
partial hierarchy artifact, while epoch buffers are still cleared.

### Task 4: Carry Structured Spearman State Through HGCN Evaluation and History

**Files:**
- Modify: `src/naics_embedder/graph_model/hgcn.py:26-33,350,856-944,998-1019`
- Modify: `tests/unit/test_hgcn_metrics.py`
- Modify: `docs/hgcn_training.md:50-71`

**Interfaces:**
- Consumes:
  `HierarchyMetrics.spearman_correlation(embedding_distances: torch.Tensor, tree_distances: torch.Tensor, min_distance: float = 0.1) -> StructuralSpearmanResult`,
  and Task 2's unsanitized `load_distance_submatrix` return tensor.
- Produces:
  `_compute_full_validation_metrics(embeddings: torch.Tensor) -> Optional[Dict[str, Union[torch.Tensor, int, str, None]]]`.
- Produces: numeric-only Lightning logs with the same `val/structural_spearman_v1` and count
  names as text validation.
- Produces: six history fields prefixed `val_`, including nullable value, integer counts,
  status, reason, and definition. `export_history()` and `save_outputs()` keep their signatures.
- Preserves: fixed `curvature=1.0`, existing graph metrics, optional missing-data behavior,
  sampling/loss/curriculum configuration, and history's epoch envelope.

- [ ] **Step 1: Add full-evaluation and real history-export regressions**

Add imports to `tests/unit/test_hgcn_metrics.py`, preserving its existing imports and test:

```python
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from naics_embedder.graph_model.hgcn import save_outputs
from naics_embedder.metrics import StructuralMetricInputError
```

Append:

```python
@pytest.fixture
def spearman_hgcn(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    prediction, target = structural_distance_matrices
    codes = ['n0', 'n1', 'n2', 'n3']
    distance_path = tmp_path / 'distances.parquet'
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(distance_path)
    cfg = GraphConfig(
        distance_matrix_parquet=str(distance_path),
        relations_parquet=str(tmp_path / 'absent-relations.parquet'),
        curriculum_cache_dir=str(tmp_path),
        curriculum_enabled=False,
        output_parquet=str(tmp_path / 'encodings.parquet'),
        ndcg_k_values=[2],
        full_eval_frequency=1,
        tangent_dim=3,
        n_hgcn_layers=1,
        dropout=0.0,
        learnable_curvature=False,
        learnable_loss_weights=False,
        k_total=2,
        n_positive_samples=1,
        batch_size=1,
    )
    levels = torch.tensor([2, 3, 3, 4])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    metadata = pl.DataFrame({'index': range(4), 'code': codes, 'level': levels.tolist()})
    module = HGCNLightningModule(
        cfg, structural_lorentz_embeddings, levels, edges,
        torch.zeros(edges.shape[1], dtype=torch.long), torch.ones(edges.shape[1]),
        {'edge_type_count': 1, 'sibling_type_id': None}, metadata,
    )
    monkeypatch.setattr(module, 'forward', Mock(return_value=structural_lorentz_embeddings))
    monkeypatch.setattr(module, 'log', Mock())
    monkeypatch.setattr(
        module.embedding_evaluator, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    return module, metadata

@pytest.mark.unit
@pytest.mark.parametrize('undefined', [False, True])
def test_hgcn_spearman_full_metrics_logs_and_history(tmp_path, spearman_hgcn, undefined):
    module, metadata = spearman_hgcn
    if undefined:
        module.tree_distances.fill_(1.0)
        module.tree_distances.fill_diagonal_(0.0)
    full = module._compute_full_validation_metrics(module.forward())
    key = 'structural_spearman_v1'
    assert full is not None
    assert full[f'{key}_n_pairs'] == full[f'{key}_n_total'] == 6
    assert full[f'{key}_definition'] == 'structural-spearman-v1'
    assert {'spearman_correlation', 'spearman_n_pairs'}.isdisjoint(full)

    module.history.append({'epoch': 1, 'loss': 1.0})
    module.on_validation_epoch_start()
    module.validation_step(
        {'anchor_idx': torch.tensor([0]), 'positive_idx': torch.tensor([1]),
         'negative_indices': torch.tensor([[2, 3]])},
        batch_idx=0,
    )
    module.on_validation_epoch_end()
    save_outputs(
        str(tmp_path), module.embeddings.detach(), metadata, module.cfg,
        module.model, module.export_history(),
    )
    history = json.loads((tmp_path / 'training_log.json').read_text())
    json.dumps(history, allow_nan=False)
    record = history[0]
    assert record[f'val_{key}_definition'] == 'structural-spearman-v1'
    assert record[f'val_{key}_n_pairs'] == record[f'val_{key}_n_total'] == 6
    assert isinstance(record[f'val_{key}_n_pairs'], int)
    assert {'val_spearman_correlation', 'val_spearman_n_pairs'}.isdisjoint(record)
    logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert f'val/{key}_n_pairs' in logged
    assert f'val/{key}_n_total' in logged
    assert f'val/{key}_status' not in logged
    assert f'val/{key}_reason' not in logged
    assert f'val/{key}_definition' not in logged
    assert 'val/spearman_correlation' not in logged
    assert 'val/spearman_n_pairs' not in logged
    if undefined:
        assert full[key] is None
        assert record[f'val_{key}'] is None
        assert record[f'val_{key}_status'] == 'undefined'
        assert record[f'val_{key}_reason'] == 'constant_target'
        assert f'val/{key}' not in logged
    else:
        assert full[key].item() == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'val_{key}'] == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'val_{key}_status'] == 'defined'
        assert record[f'val_{key}_reason'] is None
        assert logged[f'val/{key}'].item() == pytest.approx(record[f'val_{key}'])
    module.on_validation_epoch_start()
    assert module._full_val_metrics == {}

@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_file', 'asymmetric', 'shape'])
def test_hgcn_malformed_distances_fail_validation(spearman_hgcn, invalid):
    module, _ = spearman_hgcn
    if invalid == 'nan_file':
        path = Path(module.cfg.distance_matrix_parquet)
        frame = pl.read_parquet(path)
        columns = frame.to_dict(as_series=False)
        columns['idx_0-code_n0'][1] = float('nan')
        pl.DataFrame(columns).write_parquet(path)
        module.tree_distances = module._load_tree_distance_tensor(module.node_codes)
        assert torch.isnan(module.tree_distances[1, 0])
    elif invalid == 'asymmetric':
        module.tree_distances[1, 0] = 10.0
    else:
        module.tree_distances = module.tree_distances[:3, :3]
    with pytest.raises(StructuralMetricInputError):
        module.validation_step(
            {'anchor_idx': torch.tensor([0]), 'positive_idx': torch.tensor([1]),
             'negative_indices': torch.tensor([[2, 3]])},
            batch_idx=0,
        )
    logged_names = [call.args[0] for call in module.log.call_args_list]
    assert 'val/cophenetic_correlation' not in logged_names
    assert 'val/ndcg@2' not in logged_names
    assert module._full_val_metrics == {}

@pytest.mark.unit
def test_hgcn_does_not_skip_unexpected_spearman_failure(monkeypatch, spearman_hgcn):
    module, _ = spearman_hgcn
    monkeypatch.setattr(
        'naics_embedder.metrics.structural_spearman.spearmanr',
        lambda *_: SimpleNamespace(statistic=float('nan')),
    )
    with pytest.raises(RuntimeError, match='structural-spearman-v1.*n_pairs=6'):
        module._compute_full_validation_metrics(module.forward())
```

- [ ] **Step 2: Run the HGCN boundary red**

```bash
uv run pytest tests/unit/test_hgcn_metrics.py -q
```

Expected: versioned output assertions fail and shape mismatch is incorrectly skipped. The new
core already makes some malformed-input cases fail; this task removes the HGCN-specific bypass
and carries metadata through the actual Lightning/history path.

- [ ] **Step 3: Separate metric validation from optional-runtime skip handling**

Add `STRUCTURAL_SPEARMAN_KEY` to the existing import from `naics_embedder.metrics`.
Change only this initialization annotation:

```python
        self._full_val_metrics: Dict[str, Union[float, int, str, None]] = {}
```

Replace `_compute_full_validation_metrics` with the following complete method. In particular,
remove the old shape-mismatch warning/`return None`. The new metric call is outside the existing
optional-runtime catches, so neither malformed input nor an unexpected SciPy computation
failure is relabeled as a skipped hierarchy metric.

```python
    def _compute_full_validation_metrics(
        self, embeddings: torch.Tensor
    ) -> Optional[Dict[str, Union[torch.Tensor, int, str, None]]]:
        if (
            self.tree_distances is None or self.embedding_evaluator is None
            or self.hierarchy_metrics is None
        ):
            return None

        evaluator = self.embedding_evaluator
        hierarchy = self.hierarchy_metrics
        evaluator.device = str(self.device)
        hierarchy.device = str(self.device)

        try:
            with torch.no_grad():
                emb_dists = evaluator.compute_pairwise_distances(
                    embeddings.detach(), metric='lorentz', curvature=1.0
                )
        except RuntimeError as err:
            logger.warning('Skipping hierarchy metrics due to runtime error: %s', err)
            return None

        tree_dists = self.tree_distances
        spearman = hierarchy.spearman_correlation(emb_dists, tree_dists)
        try:
            with torch.no_grad():
                cophenetic = hierarchy.cophenetic_correlation(emb_dists, tree_dists)
                ndcg = hierarchy.ndcg_ranking(emb_dists, tree_dists, k_values=self._ndcg_k_values)
                distortion = hierarchy.distortion(emb_dists, tree_dists)
        except RuntimeError as err:
            logger.warning('Skipping hierarchy metrics due to runtime error: %s', err)
            return None

        metrics: Dict[str, Union[torch.Tensor, int, str, None]] = {
            STRUCTURAL_SPEARMAN_KEY: (
                spearman['correlation'] if spearman['status'] == 'defined' else None
            ),
            f'{STRUCTURAL_SPEARMAN_KEY}_n_pairs': spearman['n_pairs'],
            f'{STRUCTURAL_SPEARMAN_KEY}_n_total': spearman['n_total'],
            f'{STRUCTURAL_SPEARMAN_KEY}_status': spearman['status'],
            f'{STRUCTURAL_SPEARMAN_KEY}_reason': spearman['reason'],
            f'{STRUCTURAL_SPEARMAN_KEY}_definition': spearman['definition'],
        }
        cophenetic_corr = cast(torch.Tensor, cophenetic['correlation'])
        metrics['cophenetic_correlation'] = cophenetic_corr.detach()
        metrics['cophenetic_n_pairs'] = torch.tensor(
            float(cophenetic['n_pairs']), device=cophenetic_corr.device
        )
        for key, value in distortion.items():
            metrics[key] = cast(torch.Tensor, value).detach()
        for k in self._ndcg_k_values:
            ndcg_value = cast(torch.Tensor, ndcg[f'ndcg@{k}'])
            metrics[f'ndcg@{k}'] = ndcg_value.detach()
            n_queries = ndcg.get(f'ndcg@{k}_n_queries', 0)
            metrics[f'ndcg@{k}_n_queries'] = torch.tensor(
                float(n_queries), device=ndcg_value.device
            )

        if (
            self.naics_hierarchy is not None and self.node_codes is not None
            and len(self.node_codes) == embeddings.size(0)
        ):
            radius_metrics = compute_radius_structure_metrics(
                embeddings.detach(), self.node_codes, self.naics_hierarchy
            )
            for name, value in radius_metrics.items():
                metrics[name] = torch.tensor(value, device=embeddings.device)
            retrieval_metrics = compute_hierarchy_retrieval_metrics(
                emb_dists,
                self.node_codes,
                self.naics_hierarchy,
                parent_top_k=self.parent_eval_top_k,
                child_top_k=self.child_eval_top_k,
            )
            for name, value in retrieval_metrics.items():
                metrics[name] = torch.tensor(value, device=embeddings.device)
        elif self.naics_hierarchy is not None:
            logger.warning(
                'Skipping hierarchy diagnostics: code count mismatch (%s codes, %s embeddings)',
                len(self.node_codes) if self.node_codes is not None else 0,
                embeddings.size(0),
            )
        return metrics
```

This is not a general exception-policy rewrite: the existing warning/skip behavior for unrelated
distance-production or other-metric `RuntimeError`s remains intact.

- [ ] **Step 4: Log only numeric results while retaining complete history metadata**

Inside `validation_step`, replace the complete
`if self._should_run_full_eval(batch_idx):` block with:

```python
        if self._should_run_full_eval(batch_idx):
            full_metrics = self._compute_full_validation_metrics(emb_upd)
            if full_metrics:
                self._full_eval_done = True
                for name, value in full_metrics.items():
                    history_key = f'val_{name}'
                    if value is None or isinstance(value, str):
                        self._full_val_metrics[history_key] = value
                        continue
                    tensor_value = (
                        value if isinstance(value, torch.Tensor) else torch.tensor(
                            value, device=self.device
                        )
                    )
                    self.log(
                        f'val/{name}',
                        tensor_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=(
                            name == 'cophenetic_correlation'
                            or name == f'ndcg@{self._primary_ndcg}'
                        ),
                    )
                    self._full_val_metrics[history_key] = (
                        float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
                    )
```

Do not average the new metadata through `_val_epoch_metrics`, convert strings/`None` into
tensors, or carry previous epoch values forward. Existing `on_validation_epoch_start` clearing
and `on_validation_epoch_end` history merging already provide the correct lifecycle.

- [ ] **Step 5: Document HGCN scalar and history fields**

In `docs/hgcn_training.md` replace the Spearman bullet with:

```markdown
- **Structural Spearman v1** (`structural_spearman_v1`) - average-rank structural agreement,
  with filtered and total unique-pair counts.
```

After the metric list insert:

````markdown
`structural-spearman-v1` uses the strict upper triangle of validated square distance matrices,
averages mirrored values in CPU float64, excludes the diagonal, filters canonical target
distances at `min_distance=0.1`, and assigns average ranks to exact ties. See the
[complete contract](overview.md#structural-spearman-v1).

Lightning logs `val/structural_spearman_v1` only when defined, plus
`val/structural_spearman_v1_n_pairs` and `val/structural_spearman_v1_n_total`.
`training_log.json` uses the existing `val_` prefix for the complete state:

```json
{
  "val_structural_spearman_v1": null,
  "val_structural_spearman_v1_n_pairs": 6,
  "val_structural_spearman_v1_n_total": 6,
  "val_structural_spearman_v1_status": "undefined",
  "val_structural_spearman_v1_reason": "constant_target",
  "val_structural_spearman_v1_definition": "structural-spearman-v1"
}
```

Defined values are numeric with status `defined` and a `null` reason. Undefined results remain
non-fatal, with one of the documented reasons, and do not produce a numeric Lightning scalar.
Malformed matrices, including shape mismatches and non-finite off-diagonal entries loaded from
Parquet, raise `StructuralMetricInputError`; they are not treated as unavailable optional
metrics. A missing distance file remains a separate optional-evaluation condition.

Historical `spearman`, `spearman_correlation`, `val/spearman_correlation`, and
`val_spearman_correlation` fields are `legacy-ordinal-rank-v0`. Their order-sensitive ordinal
ranks are not valid tied-rank Spearman coefficients and are not directly comparable with v1.
No historical artifact is rewritten and no new legacy alias is emitted.

Full HGCN evaluation remains explicitly fixed at curvature `1.0`. This rank repair does not
repair non-unit-curvature distances or change the graph architecture, objectives, or curriculum.
````

- [ ] **Step 6: Verify and commit the HGCN boundary**

```bash
./scripts/format_code.sh src/naics_embedder/graph_model/hgcn.py tests/unit/test_hgcn_metrics.py
uv run pytest tests/unit/test_hgcn_metrics.py tests/unit/test_hgcn.py \
  tests/unit/test_distance_matrix.py -q
git diff --check
git add src/naics_embedder/graph_model/hgcn.py tests/unit/test_hgcn_metrics.py docs/hgcn_training.md
git commit -m "fix(hgcn): retain structural Spearman status in validation history" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

Expected: all selected tests pass. Both malformed in-memory shapes and malformed file-backed
observations fail validation; undefined state survives `training_log.json` as `null`.

### Task 5: Report Stage-4 Spearman Without Changing the Acceptance Gate

**Files:**
- Modify: `src/naics_embedder/tools/embeddings_verification.py:6-13,66-89,142-194`
- Modify: `src/naics_embedder/cli/commands/tools.py:273-278,300-311`
- Modify: `tests/unit/test_embeddings_verification.py`
- Modify: `tests/unit/test_cli_commands.py`
- Modify: `docs/hgcn_training.md` under **9. Pre/Post Verification Workflow**
- Modify: `README.md` under **5.5 Pre/Post Verification**

**Interfaces:**
- Consumes:
  `HierarchyMetrics.spearman_correlation(embedding_distances: torch.Tensor, tree_distances: torch.Tensor, min_distance: float = 0.1) -> StructuralSpearmanResult`,
  and Task 2's file-loader behavior.
- Changes the verifier's private helper to:
  `_compute_global_metrics(embeddings: torch.Tensor, tree_distances: torch.Tensor, ndcg_k: int, parent_pairs: Sequence[Tuple[int, int]], evaluator: EmbeddingEvaluator, hierarchy: HierarchyMetrics, top_k: int) -> Tuple[Dict[str, float], StructuralSpearmanResult]`.
- Preserves:
  `verify_stage4(stage3_parquet: Path, stage4_parquet: Path, distance_matrix: Path, relations_parquet: Path, config: Stage4VerificationConfig) -> Dict`.
- Produces: nullable numeric `structural_spearman_v1` in `pre`, `post`, and `delta`; metadata at
  `metric_metadata['structural_spearman_v1']` with `definition`, `pre`, and `post`. Each phase
  metadata record has `status`, `reason`, `n_pairs`, and `n_total`.
- Preserves: `Stage4VerificationConfig` fields/defaults, CLI threshold options, and the exact
  three checks `cophenetic`, `ndcg`, `local_improvement`.

- [ ] **Step 1: Add verifier regressions using real files and the real metric**

Add imports to `tests/unit/test_embeddings_verification.py`, retaining existing imports/helpers:

```python
import json
from dataclasses import asdict

import numpy as np
import pytest
from scipy.stats import spearmanr

from naics_embedder.graph_model.hgcn import load_embeddings
from naics_embedder.metrics import EmbeddingEvaluator, StructuralMetricInputError
```

Append:

```python
@pytest.fixture
def structural_verification_files(tmp_path, monkeypatch, structural_distance_matrices):
    _, target = structural_distance_matrices
    codes = ['n0', 'n1', 'n2', 'n3']
    paths = (
        tmp_path / 'pre.parquet',
        tmp_path / 'post.parquet',
        tmp_path / 'distances.parquet',
        tmp_path / 'relations.parquet',
    )
    vectors = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.3), (0.2, 0.4)]
    _write_embeddings(paths[0], codes, vectors)
    _write_embeddings(paths[1], codes, vectors)
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(paths[2])
    pl.DataFrame({
        'code_i': ['n0', 'n0', 'n0'],
        'code_j': ['n1', 'n2', 'n3'],
        'relation': ['child', 'child', 'child'],
    }).write_parquet(paths[3])
    monkeypatch.setattr(
        'naics_embedder.metrics.core.get_device', lambda: ('cpu', '32-true', 0)
    )
    return paths

def _report_config() -> Stage4VerificationConfig:
    return Stage4VerificationConfig(
        max_cophenetic_degradation=2.0,
        max_ndcg_degradation=1.0,
        min_local_improvement=-1.0,
        ndcg_k=1,
    )

@pytest.mark.unit
def test_verifier_real_distances_match_scipy(
    structural_verification_files, structural_distance_matrices
):
    paths = structural_verification_files
    result = verify_stage4(*paths, _report_config())
    embeddings, _, _ = load_embeddings(str(paths[0]), torch.device('cpu'))
    distances = EmbeddingEvaluator().compute_pairwise_distances(
        embeddings, metric='lorentz', curvature=1.0
    ).double().numpy()
    _, target = structural_distance_matrices
    rows, columns = np.triu_indices(4, k=1)
    observations = (distances[rows, columns] + distances[columns, rows]) / 2.0
    expected = float(spearmanr(observations, target.numpy()[rows, columns]).statistic)
    assert result['pre']['structural_spearman_v1'] == pytest.approx(expected, abs=1e-7)
    assert result['post']['structural_spearman_v1'] == pytest.approx(expected, abs=1e-7)
    assert result['delta']['structural_spearman_v1'] == 0.0

@pytest.mark.unit
def test_negative_spearman_delta_does_not_gate_verification(
    monkeypatch, structural_verification_files, structural_distance_matrices
):
    prediction, _ = structural_distance_matrices
    reversed_prediction = 7.0 - prediction
    reversed_prediction.fill_diagonal_(0.0)
    matrices = iter([prediction, reversed_prediction])
    monkeypatch.setattr(
        EmbeddingEvaluator, 'compute_pairwise_distances', lambda *_, **__: next(matrices)
    )
    result = verify_stage4(*structural_verification_files, _report_config())
    key = 'structural_spearman_v1'
    assert result['pre'][key] == pytest.approx(0.87831006565368, abs=1e-7)
    assert result['post'][key] == pytest.approx(-0.87831006565368, abs=1e-7)
    assert result['delta'][key] == pytest.approx(-1.75662013130736, abs=2e-7)
    assert set(result['checks']) == {'cophenetic', 'ndcg', 'local_improvement'}
    assert result['passed'] is True
    metadata = result['metric_metadata'][key]
    assert metadata['definition'] == 'structural-spearman-v1'
    for phase in ('pre', 'post'):
        assert metadata[phase] == {
            'status': 'defined', 'reason': None, 'n_pairs': 6, 'n_total': 6,
        }
    for phase in ('pre', 'post', 'delta'):
        assert {'spearman', 'spearman_correlation'}.isdisjoint(result[phase])
    assert set(asdict(Stage4VerificationConfig())) == {
        'max_cophenetic_degradation', 'max_ndcg_degradation', 'min_local_improvement',
        'ndcg_k', 'parent_top_k',
    }
    json.dumps(result, allow_nan=False)

@pytest.mark.unit
@pytest.mark.parametrize('undefined_phases', [('pre',), ('post',), ('pre', 'post')])
def test_verifier_undefined_values_and_delta_are_null(
    monkeypatch, structural_verification_files, structural_distance_matrices, undefined_phases
):
    prediction, _ = structural_distance_matrices
    constant = torch.ones_like(prediction)
    constant.fill_diagonal_(0.0)
    matrices = iter([
        constant if phase in undefined_phases else prediction for phase in ('pre', 'post')
    ])
    monkeypatch.setattr(
        EmbeddingEvaluator, 'compute_pairwise_distances', lambda *_, **__: next(matrices)
    )
    result = verify_stage4(*structural_verification_files, _report_config())
    key = 'structural_spearman_v1'
    for phase in ('pre', 'post'):
        metadata = result['metric_metadata'][key][phase]
        assert metadata['n_pairs'] == metadata['n_total'] == 6
        if phase in undefined_phases:
            assert result[phase][key] is None
            assert metadata['status'] == 'undefined'
            assert metadata['reason'] == 'constant_prediction'
        else:
            assert result[phase][key] == pytest.approx(0.87831006565368, abs=1e-7)
            assert metadata['status'] == 'defined'
            assert metadata['reason'] is None
        assert {'spearman', 'spearman_correlation'}.isdisjoint(result[phase])
    assert result['delta'][key] is None
    assert set(result['checks']) == {'cophenetic', 'ndcg', 'local_improvement'}
    assert result['passed'] is True
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)['delta'][key] is None

@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_lower', 'asymmetric', 'prediction_shape'])
def test_verifier_rejects_malformed_distances(
    monkeypatch, structural_verification_files, invalid
):
    paths = structural_verification_files
    if invalid == 'prediction_shape':
        monkeypatch.setattr(
            EmbeddingEvaluator, 'compute_pairwise_distances',
            lambda *_, **__: torch.ones(3, 3),
        )
    else:
        frame = pl.read_parquet(paths[2])
        columns = frame.to_dict(as_series=False)
        columns['idx_0-code_n0'][1] = float('nan') if invalid == 'nan_lower' else 10.0
        pl.DataFrame(columns).write_parquet(paths[2])
    with pytest.raises(StructuralMetricInputError):
        verify_stage4(*paths, _report_config())
```

- [ ] **Step 2: Add CLI formatting and fatal-input tests**

Add this import to `tests/unit/test_cli_commands.py`:

```python
from naics_embedder.metrics import StructuralMetricInputError
```

Append:

```python
@pytest.mark.unit
@pytest.mark.parametrize('undefined', [False, True])
def test_verify_stage4_formats_versioned_spearman(monkeypatch, runner, undefined):
    key = 'structural_spearman_v1'
    value = None if undefined else 0.87831006565368
    delta = None if undefined else 0.125
    payload = {
        'pre': {key: value},
        'post': {key: value},
        'delta': {key: delta},
        'checks': {'cophenetic': True, 'ndcg': True, 'local_improvement': True},
        'passed': True,
    }
    monkeypatch.setattr(tools_cli, 'verify_stage4', lambda *_, **__: payload)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 0, result.output
    if undefined:
        assert result.output.count(f'{key}: N/A') == 3
    else:
        assert result.output.count(f'{key}: 0.8783') == 2
        assert f'{key}: +0.1250' in result.output
    assert 'spearman_correlation' not in result.output

@pytest.mark.unit
def test_verify_stage4_input_error_is_fatal(monkeypatch, runner):

    def invalid(*_args, **_kwargs):
        raise StructuralMetricInputError('structural-spearman-v1: invalid tree_distances')

    monkeypatch.setattr(tools_cli, 'verify_stage4', invalid)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 1
    assert 'Verification failed' in result.output
    assert 'invalid tree_distances' in result.output

@pytest.mark.unit
def test_verify_stage4_has_no_spearman_threshold_option(runner):
    result = runner.invoke(tools_cli.app, ['verify-stage4', '--help'])
    assert result.exit_code == 0
    assert '--max-spearman-drop' not in result.output
    assert '--min-spearman' not in result.output
```

Run both files:

```bash
uv run pytest tests/unit/test_embeddings_verification.py tests/unit/test_cli_commands.py -q
```

Expected: verifier versioned-key/metadata assertions fail, and formatting a `None` scalar fails.
The old threshold-failure test and explicit CLI error propagation should remain green.

- [ ] **Step 3: Calculate structural Spearman before the verifier's existing hierarchy metrics**

Add imports to `tools/embeddings_verification.py`:

```python
from naics_embedder.metrics.structural_spearman import (
    STRUCTURAL_SPEARMAN_DEFINITION,
    STRUCTURAL_SPEARMAN_KEY,
    StructuralSpearmanResult,
)
```

Replace `_compute_global_metrics` with:

```python
def _compute_global_metrics(
    embeddings: torch.Tensor,
    tree_distances: torch.Tensor,
    ndcg_k: int,
    parent_pairs: Sequence[Tuple[int, int]],
    evaluator: EmbeddingEvaluator,
    hierarchy: HierarchyMetrics,
    top_k: int,
) -> Tuple[Dict[str, float], StructuralSpearmanResult]:
    with torch.no_grad():
        emb_dists = evaluator.compute_pairwise_distances(
            embeddings, metric='lorentz', curvature=1.0
        )

    spearman = hierarchy.spearman_correlation(emb_dists, tree_distances)
    cophenetic = hierarchy.cophenetic_correlation(emb_dists, tree_distances)
    ndcg = hierarchy.ndcg_ranking(emb_dists, tree_distances, k_values=[ndcg_k])
    parent_retrieval = _parent_retrieval_accuracy(emb_dists, parent_pairs, top_k=top_k)
    metrics = {
        'cophenetic_correlation': _to_float(cophenetic['correlation']),
        f'ndcg@{ndcg_k}': _to_float(ndcg[f'ndcg@{ndcg_k}']),
        f'parent_retrieval@{top_k}': parent_retrieval,
    }
    return metrics, spearman
```

Keep the original arguments to both helper calls in `verify_stage4`, changing only their
assignment targets to:

```python
    pre_metrics, pre_spearman = _compute_global_metrics(
        emb_stage3,
        tree_distances,
        config.ndcg_k,
        parent_pairs,
        evaluator,
        hierarchy,
        config.parent_top_k,
    )
    post_metrics, post_spearman = _compute_global_metrics(
        emb_stage4,
        tree_distances,
        config.ndcg_k,
        parent_pairs,
        evaluator,
        hierarchy,
        config.parent_top_k,
    )
```

Leave the existing `ndcg_key`, `parent_key`, `delta`, and `checks` assignments unchanged.
Immediately after `checks`, replace the original return block with:

```python
    pre_value = (
        _to_float(pre_spearman['correlation']) if pre_spearman['status'] == 'defined' else None
    )
    post_value = (
        _to_float(post_spearman['correlation']) if post_spearman['status'] == 'defined' else None
    )
    spearman_delta = (
        post_value - pre_value if pre_value is not None and post_value is not None else None
    )
    spearman_metadata = {
        'definition': STRUCTURAL_SPEARMAN_DEFINITION,
        'pre': {
            'status': pre_spearman['status'],
            'reason': pre_spearman['reason'],
            'n_pairs': pre_spearman['n_pairs'],
            'n_total': pre_spearman['n_total'],
        },
        'post': {
            'status': post_spearman['status'],
            'reason': post_spearman['reason'],
            'n_pairs': post_spearman['n_pairs'],
            'n_total': post_spearman['n_total'],
        },
    }
    return {
        'pre': {**pre_metrics, STRUCTURAL_SPEARMAN_KEY: pre_value},
        'post': {**post_metrics, STRUCTURAL_SPEARMAN_KEY: post_value},
        'delta': {**delta, STRUCTURAL_SPEARMAN_KEY: spearman_delta},
        'metric_metadata': {STRUCTURAL_SPEARMAN_KEY: spearman_metadata},
        'checks': checks,
        'thresholds': asdict(config),
        'passed': all(checks.values()),
        'codes': codes,
    }
```

The separate numeric dictionaries preserve the original gate's non-nullable inputs and prevent
Spearman from accidentally becoming a fourth threshold.

- [ ] **Step 4: Render undefined values and deltas as `N/A`**

In `cli/commands/tools.py`, replace the verifier command's docstring with:

```python
    '''
    Compare Stage 3 and Stage 4 embeddings at curvature 1.0.

    Enforce cophenetic, NDCG, and parent-retrieval thresholds. Report structural
    Spearman v1 separately; undefined values and deltas display as N/A.
    '''
```

Replace its three metric-printing loops, keeping the surrounding section headings:

```python
    for key, value in result['pre'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')
```

```python
    for key, value in result['post'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')
```

```python
    for key, value in result['delta'].items():
        formatted = 'N/A' if value is None else f'{value:+.4f}'
        console.print(f'  • {key}: {formatted}')
```

Keep every existing option/default, exception-to-exit-code mapping, and threshold result loop.

- [ ] **Step 5: Document report-only pre/post/delta metadata**

Replace the final verifier paragraph in `README.md` **5.5 Pre/Post Verification** with:

```markdown
The verifier reports cophenetic correlation, NDCG\@K, parent-retrieval accuracy, and
`structural_spearman_v1` pre/post/delta values at fixed curvature `1.0`. Only cophenetic, NDCG,
and local parent-retrieval checks determine pass/fail; structural Spearman is report-only, with
no threshold option. Undefined values or deltas display as `N/A` and serialize as JSON `null`.
Definition, status, reason, and pair counts live under
`metric_metadata['structural_spearman_v1']`; see the
[verification output contract](docs/hgcn_training.md#9-prepost-verification-workflow).
```

Append the following subsection under **9. Pre/Post Verification Workflow** in
`docs/hgcn_training.md`:

````markdown
### Structural Spearman reporting

The verifier adds `structural_spearman_v1` to `pre`, `post`, and `delta`. A delta is computed
only if both phase values are defined. Otherwise it is `null` in the Python/JSON report and
`N/A` in the CLI. Definition, statuses, reasons, and pair counts are recorded separately:

```json
{
  "pre": {"structural_spearman_v1": 0.8783101},
  "post": {"structural_spearman_v1": null},
  "delta": {"structural_spearman_v1": null},
  "metric_metadata": {
    "structural_spearman_v1": {
      "definition": "structural-spearman-v1",
      "pre": {
        "status": "defined", "reason": null, "n_pairs": 6, "n_total": 6
      },
      "post": {
        "status": "undefined", "reason": "constant_prediction", "n_pairs": 6, "n_total": 6
      }
    }
  }
}
```

This excerpt omits the existing non-Spearman metrics, checks, thresholds, pass/fail flag, and
code list; those remain in the full report. Structural Spearman does not add a fourth check.
Only cophenetic degradation, NDCG degradation, and local parent-retrieval improvement govern
acceptance, using the same options and defaults as before.

Malformed inputs raise `StructuralMetricInputError`; the CLI prints the failure and exits
nonzero instead of printing a partial success report. Undefined correlation alone is
non-fatal. Both phase evaluations stay explicitly fixed at curvature `1.0`, regardless of
training configuration. The rank correction does not fix non-unit-curvature geometry.
````

- [ ] **Step 6: Run the complete focused caller suite and touched-file checks**

```bash
./scripts/format_code.sh src/naics_embedder/tools/embeddings_verification.py \
  src/naics_embedder/cli/commands/tools.py tests/unit/test_embeddings_verification.py \
  tests/unit/test_cli_commands.py
uv run pytest tests/unit/test_structural_spearman.py tests/unit/test_evaluation.py \
  tests/unit/test_distance_matrix.py tests/unit/test_text_validation_metrics.py \
  tests/unit/test_hgcn_metrics.py tests/unit/test_embeddings_verification.py \
  tests/unit/test_cli_commands.py -q
./scripts/format_code.sh --check src/naics_embedder/metrics/structural_spearman.py \
  src/naics_embedder/metrics/core.py src/naics_embedder/metrics/__init__.py \
  src/naics_embedder/metrics/runner.py src/naics_embedder/utils/distance_matrix.py \
  src/naics_embedder/text_model/mixins/validation.py src/naics_embedder/graph_model/hgcn.py \
  src/naics_embedder/tools/embeddings_verification.py src/naics_embedder/cli/commands/tools.py \
  tests/conftest.py tests/unit/test_structural_spearman.py tests/unit/test_evaluation.py \
  tests/unit/test_distance_matrix.py tests/unit/test_text_validation_metrics.py \
  tests/unit/test_hgcn_metrics.py tests/unit/test_embeddings_verification.py \
  tests/unit/test_cli_commands.py
```

Expected: every selected test passes apart from unavailable-device skips, and every touched
Python file passes Ruff and YAPF. Do not run `ruff format`.

- [ ] **Step 7: Run the required repository-level gates**

Run these commands individually so one failure does not hide another gate's result:

```bash
uv run pytest tests/ -v --cov=src/naics_embedder --cov-report=xml:coverage.xml --cov-report=term
uv run ruff check src/ tests/
uv run mkdocs build --strict
git diff --check
```

Expected: full tests, strict documentation build, and diff check pass. Ruff must have no new
failures relative to the captured execution baseline; a deferred historical count is not proof.
Use the existing Python 3.10/3.12 CI matrix for both supported interpreter checks. Do not modify
dependency manifests or workflows just for this repair.

Verify no implementation writer still emits the old keys:

```bash
rg -n "'(val/)?spearman(_correlation|_n_pairs)'|'val_spearman_correlation'" \
  src/naics_embedder/metrics/runner.py src/naics_embedder/text_model/mixins/validation.py \
  src/naics_embedder/graph_model/hgcn.py \
  src/naics_embedder/tools/embeddings_verification.py
```

Expected: no matches (ripgrep exit status `1` means no matches). The compatibility Python method
name and historical documentation/tests are intentionally not searched as forbidden output keys.

- [ ] **Step 8: Re-run the full-population benchmark manually**

Use the current validated Stage-3 bundle's distance matrix, not synthetic targets or files in
another checkout. No real bundle is present in the planning worktree. If the execution worktree
has exactly one bundle matrix, the command below selects it; otherwise supply its exact path
through `NAICS_SPEARMAN_DISTANCE_MATRIX`. If the data is unavailable, report the benchmark
blocked and obtain the artifact rather than claiming the small tests establish performance.

This command measures SciPy alone and the public validation/canonicalization wrapper separately.
It does not train a model, change artifacts, or assert a wall-clock threshold:

```bash
uv run python - <<'PY'
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import polars as pl
import scipy
import torch
from scipy.stats import spearmanr

from naics_embedder.metrics import HierarchyMetrics

configured = os.environ.get('NAICS_SPEARMAN_DISTANCE_MATRIX')
if configured:
    path = Path(configured)
else:
    paths = sorted(
        Path('data/supervision/stage3-supervision-v1').glob('*/naics_distance_matrix.parquet')
    )
    if len(paths) != 1:
        raise SystemExit(
            'Set NAICS_SPEARMAN_DISTANCE_MATRIX to the current validated bundle matrix.'
        )
    path = paths[0]

target = torch.from_numpy(pl.read_parquet(path).to_numpy()).float()
if target.shape != (2125, 2125):
    raise SystemExit(f'Expected the current 2125-node population, got {tuple(target.shape)}')
rows, columns = np.triu_indices(2125, k=1)
prediction = torch.zeros_like(target)
seed = sum(map(ord, 'structural-spearman-v1-benchmark'))
generator = torch.Generator().manual_seed(seed)
values = torch.randperm(len(rows), generator=generator).float() + 1.0
prediction[rows, columns] = values
prediction[columns, rows] = values
metric = HierarchyMetrics()
metric.device = 'cpu'
checked = metric.spearman_correlation(prediction, target)
assert checked['status'] == 'defined'
assert checked['n_total'] == 2256750

target64 = target.double().numpy()
canonical_target = (target64[rows, columns] + target64[columns, rows]) / 2.0
selected = canonical_target >= 0.1
observed = values.double().numpy()[selected]
expected = canonical_target[selected]
oracle = float(spearmanr(observed, expected).statistic)
assert np.isclose(checked['correlation'].item(), oracle, rtol=1e-6, atol=1e-7)
assert checked['n_pairs'] == int(selected.sum())

scipy_seconds = []
public_seconds = []
for _ in range(5):
    start = perf_counter()
    spearmanr(observed, expected)
    scipy_seconds.append(perf_counter() - start)
    start = perf_counter()
    metric.spearman_correlation(prediction, target)
    public_seconds.append(perf_counter() - start)
distinct, counts = np.unique(expected, return_counts=True)
print(json.dumps({
    'definition': checked['definition'],
    'matrix_path': str(path),
    'n_total': checked['n_total'],
    'n_pairs': checked['n_pairs'],
    'target_distinct': int(distinct.size),
    'largest_tie_count': int(counts.max()),
    'raw_float32_vector_bytes': int(values.numel() * values.element_size() * 2),
    'prediction_fixture': 'Seeded unique float32 distances, not trained embeddings',
    'prediction_seed': seed,
    'scipy_seconds_median': median(scipy_seconds),
    'public_seconds_median': median(public_seconds),
    'platform': platform.platform(),
    'torch': torch.__version__,
    'scipy': scipy.__version__,
}, indent=2))
PY
```

Expected: `n_total=2256750`, the current bundle's 12 distinct target distances, and its dominant
1,984,647-pair tie are visible. The raw float32 vectors total 18,054,000 bytes. Retain the output
in the execution session's artifacts and compare like-for-like SciPy timing with the design's
approximately 0.22-second observation; it is not a portable pass threshold. Investigate and
record any material regression before completing the repair.

- [ ] **Step 9: Commit verifier reporting and final documentation**

```bash
git add src/naics_embedder/tools/embeddings_verification.py \
  src/naics_embedder/cli/commands/tools.py tests/unit/test_embeddings_verification.py \
  tests/unit/test_cli_commands.py docs/hgcn_training.md README.md
git commit -m "feat(metrics): report structural Spearman in Stage-4 verification" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
git status --short --branch
```

Expected: only deliberately preserved user changes remain. Generated test/build artifacts must
not enter the commit.

---

## Acceptance Coverage and Execution Completion

| Spec acceptance criterion | Implementing task and evidence |
| --- | --- |
| 1. Correct motivating result and all permutations | Task 1 fixed oracle and 24 permutations |
| 2. Exactly one observation per unordered non-self pair | Task 1 counts, diagonal sentinels, filtering |
| 3. One CPU-float64 dtype/device contract | Task 1 source tolerance tests, SciPy spy, conditional accelerators |
| 4. Fatal malformed inputs at every public boundary | Tasks 1-5, including text's exception handler, HGCN's shape bypass, and CLI exit |
| 5. Explicit undefined state and JSON `null` | Task 1 reason precedence; Tasks 3-5 artifact/CLI regressions |
| 6. Versioned fields only | Tasks 2-5 positive and negative key assertions, final writer scan |
| 7. Verifier reports without changing the gate | Task 5 negative/undefined reports with passing original checks |
| 8. Historical results labeled and non-comparable | Tasks 2-5 documentation; no historical rewrites |
| 9. Focused/full tests, quality checks, strict docs, diff check | Task 5 Steps 6-8 and each task's focused cycle |
| 10. No unrelated metric/model/supervision changes | File scope, unchanged curvature calls/config, existing suites |

After every task and its review are complete, run the `writing-plans` **Plan Completion
Protocol** before `finishing-a-development-branch`. This is an execution requirement, not a
claim that this planning session implemented anything:

1. Resolve skipped work and unresolved review findings with the user before deferring or marking
   the plan complete. An unavailable full-scale benchmark is an unresolved input, not a pass.
2. Mark completed checkboxes once the completed/deferred partition is final; record deviations
   and add the dated completion header.
3. Run the existing-backlog closure pass in `specs/deferred_items.md`, even if nothing new was
   deferred. Append only actual, self-contained deferred items under the skill's schema.
4. Run the skill's deferred-backlog statistics and required read-only triage; report backlog
   health. Do not silently absorb unrelated backlog items into this metric change.
5. Retire this plan with `git mv` to
   `specs/plans/completed/2-structural-spearman-metric-integrity.md`. If no other live plan
   implements the source spec, retire it to
   `specs/completed/structural-spearman-metric-integrity.md` in the same
   `chore(specs): retire plan 2` commit, with the required co-author trailer.

When both files retire, update this plan's source-spec link from
`../structural-spearman-metric-integrity.md` to
`../../completed/structural-spearman-metric-integrity.md`. If the spec remains live, its link
instead becomes `../../structural-spearman-metric-integrity.md`.
