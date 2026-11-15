'''
Classical scaling of a distance matrix (spec economic-candidates, 4.2: ``tree-dN``).

With J the centring matrix, B = −½ J (D ∘ D) J holds the Gram matrix of a Euclidean embedding of
D. Its d leading eigenvectors, each scaled by the square root of its eigenvalue, are the
coordinates. The tree metric D* is not Euclidean, so B has negative eigenvalues too, and the cut
at d is checked, not assumed.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Tuple

import numpy as np

# The relative gap below which two eigenvalues are tied, and the fraction of the first
# eigenvalue at or below which one is not positive
CUT_GAP = 1e-6

# -------------------------------------------------------------------------------------------------
# Scaling
# -------------------------------------------------------------------------------------------------

def classical_scaling(distances: np.ndarray, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    The classical scaling of ``distances`` to ``dimension`` coordinates.

    B = −½ J (D ∘ D) J, and ``numpy.linalg.eigh`` gives its eigenpairs. The coordinates are the
    ``dimension`` leading eigenvectors, each times the square root of its eigenvalue. Each
    column's sign is fixed so that its entry of largest magnitude, the first under
    ``numpy.argmax``, is positive.

    Args:
        distances: A symmetric (n, n) matrix of distances.
        dimension: The number of coordinates, d.

    Returns:
        The (n, d) float64 coordinates, and the d + 1 leading eigenvalues of B in descending
        order.

    Raises:
        ValueError: If d is not in 1 … n − 1; if the d-th eigenvalue is not positive, that is at
            most ``CUT_GAP`` times the first; or if the d-th and (d + 1)-th eigenvalues tie, that
            is differ by less than ``CUT_GAP`` of the d-th, which would make the cut arbitrary.
    '''

    squared = np.asarray(distances, dtype=np.float64)**2
    n = squared.shape[0]
    if not 1 <= dimension < n:
        raise ValueError(f'the dimension {dimension} is not in 1 … {n - 1} for {n} points')
    # J D2 J without forming J; the pair sum is commutative, so B is exactly symmetric
    means = squared.mean(axis=1)
    gram = -0.5 * (squared - (means[:, None] + means[None, :]) + means.mean())
    values, vectors = np.linalg.eigh(gram)
    values, vectors = values[::-1], vectors[:, ::-1]
    last_kept, first_dropped = values[dimension - 1], values[dimension]
    if last_kept <= CUT_GAP * values[0]:
        raise ValueError(
            f'the eigenvalue {last_kept:.6g} at dimension {dimension} is not positive '
            f'(at most {CUT_GAP:g} of the first, {values[0]:.6g})'
        )
    if (last_kept - first_dropped) / last_kept < CUT_GAP:
        raise ValueError(
            f'the cut at dimension {dimension} ties the eigenvalues {last_kept:.6g} and '
            f'{first_dropped:.6g}: their relative gap is below {CUT_GAP:g}, so the truncation '
            f'is arbitrary'
        )
    coordinates = vectors[:, :dimension] * np.sqrt(values[:dimension])
    # argmax takes the first of equal magnitudes, so a tie is broken by row order
    largest = np.abs(coordinates).argmax(axis=0)
    coordinates *= np.where(coordinates[largest, np.arange(dimension)] < 0, -1.0, 1.0)
    return coordinates, values[:dimension + 1].copy()
