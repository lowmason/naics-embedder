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
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
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
            np.allclose(upper, lower, rtol=rtol, atol=atol) and np.allclose(
                lower, upper, rtol=rtol, atol=atol
            )
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
        return StructuralSpearmanComputation(float('nan'), n_pairs, n_total, 'undefined', reason)

    correlation = float(spearmanr(prediction, target).statistic)
    if not math.isfinite(correlation):
        raise RuntimeError(
            f'{STRUCTURAL_SPEARMAN_DEFINITION}: non-finite SciPy correlation for n_pairs={n_pairs}'
        )
    return StructuralSpearmanComputation(correlation, n_pairs, n_total, 'defined', None)
