'''
Ridge on standardized features along a penalty grid (Req 2 "Fitting"; D1).
'''

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from naics_embedder.panels.ridge import (
    best_alpha_index,
    squared_errors,
    standardized_ridge_path,
)

pytestmark = pytest.mark.unit

ALPHAS = [0.001, 0.1, 1.0, 10.0, 1000.0]

def _data(n=40, m=12, p=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n + m, p)) * rng.uniform(0.1, 10.0, size=p) + rng.normal(size=p)
    y = x @ rng.normal(size=p) + rng.normal(scale=0.5, size=n + m)
    return x[:n], y[:n], x[n:]

def _sklearn(x_fit, y_fit, x_score, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(x_fit, y_fit).predict(x_score)

def test_the_path_matches_scikit_learns_standardized_ridge_at_every_penalty():
    x_fit, y_fit, x_score = _data()

    path = standardized_ridge_path(x_fit, y_fit, x_score, ALPHAS)

    assert path.shape == (len(x_score), len(ALPHAS))
    for column, alpha in enumerate(ALPHAS):
        np.testing.assert_allclose(
            path[:, column], _sklearn(x_fit, y_fit, x_score, alpha), rtol=1e-9, atol=1e-9
        )

def test_more_columns_than_rows_still_matches():
    x_fit, y_fit, x_score = _data(n=8, p=20)

    path = standardized_ridge_path(x_fit, y_fit, x_score, ALPHAS)

    for column, alpha in enumerate(ALPHAS):
        np.testing.assert_allclose(
            path[:, column], _sklearn(x_fit, y_fit, x_score, alpha), rtol=1e-9, atol=1e-9
        )

def test_a_constant_column_keeps_scale_one():
    x_fit, y_fit, x_score = _data()
    x_fit[:, 2] = 3.0

    path = standardized_ridge_path(x_fit, y_fit, x_score, [1.0])

    assert np.isfinite(path).all()
    np.testing.assert_allclose(path[:, 0], _sklearn(x_fit, y_fit, x_score, 1.0), atol=1e-9)

def test_a_huge_penalty_predicts_the_fit_mean():
    x_fit, y_fit, x_score = _data()

    path = standardized_ridge_path(x_fit, y_fit, x_score, [1e12])

    np.testing.assert_allclose(path[:, 0], y_fit.mean(), atol=1e-6)

def test_squared_errors_are_summed_per_penalty():
    predictions = np.array([[1.0, 2.0], [3.0, 3.0]])

    np.testing.assert_allclose(squared_errors(np.array([1.0, 1.0]), predictions), [4.0, 5.0])

def test_a_tie_goes_to_the_larger_penalty():
    assert best_alpha_index(np.array([3.0, 1.0, 1.0, 2.0])) == 2
    assert best_alpha_index(np.array([0.5, 1.0, 2.0])) == 0

@pytest.mark.parametrize(
    ('x_fit', 'y_fit', 'x_score', 'alphas', 'message'),
    [
        (np.ones((4, 2)), np.ones(4), np.ones((2, 3)), [1.0], 'shapes'),
        (np.ones((1, 2)), np.ones(1), np.ones((2, 2)), [1.0], 'fit rows'),
        (np.ones((4, 2)), np.ones(3), np.ones((2, 2)), [1.0], 'fit rows'),
        (np.ones((4, 2)), np.ones(4), np.ones((2, 2)), [], 'positive penalties'),
        (np.ones((4, 2)), np.ones(4), np.ones((2, 2)), [0.0], 'positive penalties'),
    ],
)
def test_bad_inputs_are_refused(x_fit, y_fit, x_score, alphas, message):
    with pytest.raises(ValueError, match=message):
        standardized_ridge_path(x_fit, y_fit, x_score, alphas)
