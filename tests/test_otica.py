import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from otica import OTICA
from otica._utils import gauss_quantiles


def _signals() -> np.ndarray:
    """Creates deterministic mixed non-Gaussian signals.

    Returns:
        np.ndarray: Observations with samples as rows.
    """
    sources = np.array(
        [
            [-1.5, -1.0],
            [-1.0, 0.5],
            [-0.2, 1.4],
            [0.3, -1.2],
            [0.9, 0.1],
            [1.4, 0.9],
        ]
    )
    mixing = np.array([[1.0, 0.4], [-0.3, 1.0]])

    return sources @ mixing.T


@pytest.mark.parametrize(
    ("whiten", "w_init", "max_iter"),
    [(True, "random", 2), (False, np.eye(2), 1)],
)
def test_fit_transform(whiten, w_init, max_iter):
    """Exercises fitting, transformation, inverse transformation, and cloning."""
    X = _signals()
    estimator = OTICA(
        whiten=whiten,
        w_init=w_init,
        max_iter=max_iter,
        random_state=0,
    )

    with pytest.raises(NotFittedError):
        estimator.transform(X)

    assert clone(estimator).get_params()["max_iter"] == max_iter
    with pytest.warns(ConvergenceWarning, match="OTICA did not converge"):
        assert estimator.fit(X) is estimator

    transformed = estimator.transform(X)
    reconstructed = estimator.inverse_transform(transformed)

    assert transformed.shape == X.shape
    assert reconstructed.shape == X.shape
    assert estimator.components_.shape == (2, 2)
    assert estimator.mixing_.shape == (2, 2)
    assert estimator.unmixing_.shape == (2, 2)
    assert estimator.n_iter_ >= 1

    if whiten:
        assert estimator.mean_.shape == (2,)
        assert estimator.whitening_.shape == (2, 2)
    else:
        assert "mean_" not in estimator.__dict__
        assert "whitening_" not in estimator.__dict__


def test_w_init_shape():
    """Checks that explicit initial unmixing matrices match the fitted dimension."""
    X = _signals()

    with pytest.raises(ValueError, match="w_init must have shape"):
        OTICA(whiten=False, w_init=np.eye(3), max_iter=1).fit(X)


def test_fit_convergence_warning():
    """Checks that fitting warns only when the solver reports non-convergence."""
    X = _signals()

    with pytest.warns(ConvergenceWarning, match="OTICA did not converge"):
        OTICA(whiten=False, w_init=np.eye(2), max_iter=1, tol=0.0).fit(X)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        OTICA(whiten=False, w_init=np.eye(2), max_iter=2, tol=1e9).fit(X)

    assert not [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]


def test_fastica_init():
    """Checks the FastICA initialization branch."""
    X = _signals()

    unmixing = OTICA(w_init="fastica")._init_unmixing(
        X,
        np.random.RandomState(0),
    )

    assert np.allclose(unmixing @ unmixing.T, np.eye(2))


def test_whiten_dimension():
    """Checks principal-component whitening with dimension reduction."""
    X = np.column_stack([_signals(), np.linspace(-1.0, 1.0, 6)])
    estimator = OTICA(n_components=2, max_iter=1, w_init=np.eye(2))

    Z, mean, whitening = estimator._whiten(X, n_components=2)

    assert Z.shape == (6, 2)
    assert mean.shape == (3,)
    assert whitening.shape == (2, 3)


def test_component_validation():
    """Checks feature and component counts in transform methods."""
    X = np.column_stack([_signals(), np.linspace(-1.0, 1.0, 6)])
    with pytest.warns(ConvergenceWarning, match="OTICA did not converge"):
        estimator = OTICA(n_components=2, max_iter=1, w_init=np.eye(2)).fit(X)

    transformed = estimator.transform(X)
    reconstructed = estimator.inverse_transform(transformed)

    assert transformed.shape == (X.shape[0], 2)
    assert reconstructed.shape == X.shape

    with pytest.raises(ValueError, match="X has 2 features"):
        estimator.transform(X[:, :2])

    with pytest.raises(ValueError, match="must have 2 components"):
        estimator.inverse_transform(X)


def test_gradient_direction():
    """Checks objective, gradient, and L-BFGS direction calculations."""
    X = _signals()
    estimator = OTICA(whiten=False, max_iter=1, w_init=np.eye(2))
    quantiles = gauss_quantiles(X.shape[0])
    objective, grad = estimator._objective_and_grad(np.eye(2), X, quantiles)
    history = [(0.1 * grad, 0.2 * grad, 1.0)]

    direction = estimator._direction(grad, history)

    assert np.isfinite(objective)
    assert np.allclose(grad + grad.T, 0.0)
    assert np.allclose(direction + direction.T, 0.0)


def test_line_search_solver(monkeypatch):
    """Exercises line-search failure, fallback direction, and solver termination."""
    X = _signals()
    quantiles = gauss_quantiles(X.shape[0])
    estimator = OTICA(
        whiten=False,
        max_iter=2,
        max_line_search_steps=1,
        w_init=np.eye(2),
    )
    objective, grad = estimator._objective_and_grad(np.eye(2), X, quantiles)
    step, new_objective, new_unmixing = estimator._line_search(
        np.eye(2),
        X,
        quantiles,
        objective,
        grad,
        grad,
    )

    assert step in {0.0, 1.0}
    assert np.isfinite(new_objective)
    assert new_unmixing.shape == (2, 2)

    monkeypatch.setattr(estimator, "_direction", lambda grad, _history: -grad)
    estimator._solve(X, quantiles, np.eye(2))

    assert estimator.unmixing_.shape == (2, 2)
    assert estimator.n_iter_ >= 1
    assert not estimator.converged_

    def decreasing_objective(_unmixing, _X, _quantiles):
        return objective - 1.0, grad

    monkeypatch.setattr(estimator, "_objective_and_grad", decreasing_objective)
    failed_step, failed_objective, failed_unmixing = estimator._line_search(
        np.eye(2),
        X,
        quantiles,
        objective,
        grad,
        grad,
    )

    assert failed_step == 0.0
    assert failed_objective == objective
    assert np.allclose(failed_unmixing, np.eye(2))


def test_solver_stopping(monkeypatch):
    """Checks the remaining solver stopping branches."""
    X = _signals()
    quantiles = gauss_quantiles(X.shape[0])
    init_unmixing = np.eye(2)

    estimator = OTICA(whiten=False, max_iter=2, tol=1e9)
    estimator._solve(
        X,
        quantiles,
        init_unmixing,
    )
    assert np.allclose(estimator.unmixing_, init_unmixing)
    assert estimator.n_iter_ == 1
    assert estimator.converged_

    estimator = OTICA(whiten=False, max_iter=2, tol=0.0)
    monkeypatch.setattr(
        estimator,
        "_line_search",
        lambda *_: (0.0, 0.0, init_unmixing),
    )
    estimator._solve(
        X,
        quantiles,
        init_unmixing,
    )
    assert np.allclose(estimator.unmixing_, init_unmixing)
    assert estimator.n_iter_ == 1
    assert not estimator.converged_

    estimator = OTICA(whiten=False, max_iter=2, tol=0.0)
    monkeypatch.setattr(
        estimator,
        "_line_search",
        lambda *_: (1.0, 1.0, init_unmixing),
    )
    estimator._solve(
        X,
        quantiles,
        init_unmixing,
    )
    assert np.allclose(estimator.unmixing_, init_unmixing)
    assert estimator.n_iter_ == 1
    assert estimator.converged_


def test_gauss_quantiles():
    """Checks basic Gaussian rank-statistic properties."""
    quantiles = gauss_quantiles(6)

    assert quantiles.shape == (6,)
    assert np.all(np.diff(quantiles) > 0.0)
    assert np.isclose(quantiles.mean(), 0.0)
