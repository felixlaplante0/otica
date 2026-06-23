"""Optimal transport independent component analysis estimator."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Self, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._lbfgs import LBFGSMixin
from ._utils import gauss_quantiles


class OTICA(LBFGSMixin, TransformerMixin, BaseEstimator):
    """Optimal transport ICA using Riemannian L-BFGS.

    The estimator whitens the observations and maximizes the sum of empirical squared
    Wasserstein distances between the recovered components and a standard Gaussian.
    Optimization uses a Picard-style L-BFGS method on the orthogonal group with an
    Armijo backtracking line search.

    Attributes:
        mean_ (np.ndarray): Feature means removed during fitting.
        whitening_ (np.ndarray): Symmetric whitening matrix.
        rotation_ (np.ndarray): Orthogonal rotation in the whitened space.
        components_ (np.ndarray): Estimated unmixing matrix.
        mixing_ (np.ndarray): Pseudo-inverse of the unmixing matrix.
        sources_ (np.ndarray): Recovered sources for the training observations.
        n_iter_ (int): Number of L-BFGS iterations.
    """

    mean_: np.ndarray
    whitening_: np.ndarray
    rotation_: np.ndarray
    components_: np.ndarray
    mixing_: np.ndarray
    sources_: np.ndarray
    n_iter_: int

    _parameter_constraints = {  # noqa: RUF012
        "init": [StrOptions({"fastica", "identity", "random"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "memory": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "initial_step": [Interval(Real, 0.0, None, closed="neither")],
        "contraction": [Interval(Real, 0.0, 1.0, closed="neither")],
        "sufficient_increase": [Interval(Real, 0.0, 1.0, closed="neither")],
        "max_line_search_steps": [Interval(Integral, 1, None, closed="left")],
        "curvature_tol": [Interval(Real, 0.0, None, closed="neither")],
        "gradient_tol": [Interval(Real, 0.0, None, closed="left")],
        "min_eigenvalue": [Interval(Real, 0.0, None, closed="neither")],
        "fastica_fun": [StrOptions({"logcosh", "exp", "cube"})],
        "fastica_max_iter": [Interval(Integral, 1, None, closed="left")],
        "fastica_tol": [Interval(Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        init: str = "fastica",
        max_iter: int = 200,
        memory: int = 7,
        tol: float = 1e-7,
        initial_step: float = 1.0,
        contraction: float = 0.5,
        sufficient_increase: float = 1e-4,
        max_line_search_steps: int = 30,
        curvature_tol: float = 1e-12,
        gradient_tol: float = 1e-8,
        min_eigenvalue: float = 1e-12,
        fastica_fun: str = "logcosh",
        fastica_max_iter: int = 2000,
        fastica_tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        """Initializes OTICA.

        Args:
            init (str, optional): Initialization method. Defaults to `"fastica"`.
            max_iter (int, optional): Maximum L-BFGS iterations. Defaults to 200.
            memory (int, optional): L-BFGS memory size. Defaults to 7.
            tol (float, optional): Relative objective-increase tolerance. Defaults to
                1e-7.
            initial_step (float, optional): Initial Armijo step size. Defaults to 1.0.
            contraction (float, optional): Armijo contraction factor. Defaults to 0.5.
            sufficient_increase (float, optional): Armijo sufficient-increase factor.
                Defaults to 1e-4.
            max_line_search_steps (int, optional): Maximum Armijo backtracking steps.
                Defaults to 30.
            curvature_tol (float, optional): Minimum accepted L-BFGS curvature and
                ascent slope. Defaults to 1e-12.
            gradient_tol (float, optional): Gradient-norm tolerance. Defaults to 1e-8.
            min_eigenvalue (float, optional): Relative covariance eigenvalue threshold.
                Defaults to 1e-12.
            fastica_fun (str, optional): FastICA contrast. Defaults to `"logcosh"`.
            fastica_max_iter (int, optional): Maximum FastICA iterations. Defaults to
                2000.
            fastica_tol (float, optional): FastICA tolerance. Defaults to 1e-6.
            random_state (int | None, optional): Random seed. Defaults to None.
        """
        self.init = init
        self.max_iter = max_iter
        self.memory = memory
        self.tol = tol
        self.initial_step = initial_step
        self.contraction = contraction
        self.sufficient_increase = sufficient_increase
        self.max_line_search_steps = max_line_search_steps
        self.curvature_tol = curvature_tol
        self.gradient_tol = gradient_tol
        self.min_eigenvalue = min_eigenvalue
        self.fastica_fun = fastica_fun
        self.fastica_max_iter = fastica_max_iter
        self.fastica_tol = fastica_tol
        self.random_state = random_state

    def _whiten(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centers and symmetrically whitens observations."""
        mean = X.mean(axis=0)
        centered = X - mean
        covariance = np.atleast_2d(np.cov(centered, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[0] <= self.min_eigenvalue * max(
            1.0,
            float(eigenvalues[-1]),
        ):
            raise ValueError("X must have a full-rank sample covariance matrix.")
        whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        return centered @ whitening, mean, whitening

    def _initial_rotation(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Computes the initial orthogonal rotation."""
        if self.init == "identity":
            return np.eye(X.shape[1])
        if self.init == "random":
            return np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))[0]

        estimator = FastICA(
            whiten=False,
            fun=self.fastica_fun,
            max_iter=self.fastica_max_iter,
            tol=self.fastica_tol,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).fit(X)
        left, _, right = np.linalg.svd(estimator.components_, full_matrices=False)
        return left @ right

    def fit(
        self,
        X: np.typing.ArrayLike,
        y: object = None,  # noqa: ARG002
    ) -> Self:
        """Fits the optimal transport ICA model.

        Args:
            X (np.typing.ArrayLike): Training observations.
            y (object, optional): Ignored. Defaults to None.

        Returns:
            Self: The fitted estimator.
        """
        self._validate_params()
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64, ensure_min_samples=2),
        )
        whitened, self.mean_, self.whitening_ = self._whiten(X)
        rng = np.random.default_rng(self.random_state)
        initial_rotation = self._initial_rotation(whitened, rng)
        self.rotation_, self.n_iter_ = self._solve(
            whitened,
            gauss_quantiles(X.shape[0]),
            initial_rotation,
        )
        self.components_ = self.rotation_ @ self.whitening_
        self.mixing_ = np.linalg.pinv(self.components_)
        self.sources_ = whitened @ self.rotation_.T

        return self

    def transform(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Recovers independent components from observations.

        Args:
            X (np.typing.ArrayLike): Observations to transform.

        Returns:
            np.ndarray: Recovered independent components.
        """
        check_is_fitted(self, attributes=["components_"])
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64, reset=False),
        )
        return (X - self.mean_) @ self.components_.T
