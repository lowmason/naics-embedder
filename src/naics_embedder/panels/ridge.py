'''
Ridge regression on standardized features along a penalty grid (Req 2 "Fitting"; D1).

``standardized_ridge_path`` matches scikit-learn's ``make_pipeline(StandardScaler(),
Ridge(alpha))`` at every penalty on the grid, from one singular value decomposition of the
standardized fit set. The mean and scale come from the fit rows only, and a constant column
keeps scale 1, as ``StandardScaler`` does.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Sequence

import numpy as np

# -------------------------------------------------------------------------------------------------
# Path and selection
# -------------------------------------------------------------------------------------------------

def standardized_ridge_path(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_score: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    '''
    Predictions for the score rows at every penalty.

    Args:
        x_fit: Fit features (n, p).
        y_fit: Fit outcomes (n,).
        x_score: Features to predict (m, p).
        alphas: Penalties (A,), each positive.

    Returns:
        Predictions (m, A), column a at ``alphas[a]``.
    '''

    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    x_score = np.asarray(x_score, dtype=np.float64)
    grid = np.asarray(alphas, dtype=np.float64)
    if x_fit.ndim != 2 or x_score.ndim != 2 or x_fit.shape[1] != x_score.shape[1]:
        raise ValueError(f'feature shapes {x_fit.shape} and {x_score.shape} do not match')
    if x_fit.shape[0] != y_fit.shape[0] or x_fit.shape[0] < 2:
        raise ValueError(f'{x_fit.shape[0]} fit rows for {y_fit.shape[0]} outcomes')
    if grid.ndim != 1 or not grid.size or (grid <= 0).any():
        raise ValueError('alphas must be a non-empty list of positive penalties')

    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 10 * np.finfo(np.float64).eps] = 1.0
    z_fit = (x_fit - mean) / scale
    z_score = (x_score - mean) / scale
    y_mean = y_fit.mean()

    u, singular, vt = np.linalg.svd(z_fit, full_matrices=False)
    projected = u.T @ (y_fit - y_mean)
    shrink = singular[:, None] / (singular[:, None]**2 + grid[None, :])
    coefficients = vt.T @ (shrink * projected[:, None])
    return z_score @ coefficients + y_mean

def squared_errors(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Summed squared error per penalty: ``y_true`` (m,) against ``predictions`` (m, A).'''

    residuals = np.asarray(predictions) - np.asarray(y_true, dtype=np.float64)[:, None]
    return (residuals**2).sum(axis=0)

def best_alpha_index(errors: np.ndarray) -> int:
    '''The penalty with the least error; a tie goes to the larger penalty (more shrinkage).'''

    errors = np.asarray(errors, dtype=np.float64)
    return int(len(errors) - 1 - np.argmin(errors[::-1]))
