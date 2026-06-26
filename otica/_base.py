from numbers import Integral, Real
from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    validate_data,
)

from ._lbfgs import LBFGSMixin
from ._utils import gauss_quantiles


class OTICA(LBFGSMixin, TransformerMixin, BaseEstimator):
    """Optimal transport ICA using orthogonal L-BFGS.

    The estimator whitens the observations and maximizes the sum of empirical squared
    Wasserstein distances between the recovered components and a standard Gaussian.
    Optimization uses an L-BFGS approximation on the orthogonal group.

    The estimator supports an optional rank reduction step through `n_components`.
    When `n_components` is smaller than the ambient dimension, the whitening matrix
    becomes rectangular and the optimization runs in the reduced component space.

    Optimization behavior is controlled by `max_iter`, `history_size`, `tol`,
    `max_line_search_steps`, and `armijo_min_increase`.

    Attributes:
        n_components (int | None): Number of retained components before fitting.
        init (str): Initialization method.
        max_iter (int): Maximum L-BFGS iterations.
        history_size (int): Maximum number of L-BFGS correction pairs.
        tol (float): Convergence tolerance.
        max_line_search_steps (int): Maximum Armijo backtracking steps.
        armijo_min_increase (float): Armijo sufficient increase constant.
        random_state (int | None): Random seed.
        mean_ (np.ndarray): Feature means removed during fitting.
        whitening_ (np.ndarray): Whitening matrix used to project onto the reduced
            component space.
        components_ (np.ndarray): Estimated unmixing matrix.
        mixing_ (np.ndarray): Pseudo-inverse of the unmixing matrix.
        objective_ (float): Final optimized objective value.
        n_iter_ (int): Number of L-BFGS iterations.
    """

    n_components: int | None
    init: str
    max_iter: int
    history_size: int
    tol: float
    max_line_search_steps: int
    armijo_min_increase: float
    random_state: int | None
    mean_: np.ndarray
    whitening_: np.ndarray
    components_: np.ndarray
    mixing_: np.ndarray
    objective_: float
    n_iter_: int

    @validate_params(
        {
            "n_components": [Interval(Integral, 1, None, closed="left"), None],
            "init": [StrOptions({"fastica", "identity", "random"})],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "history_size": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0.0, None, closed="left")],
            "max_line_search_steps": [Interval(Integral, 1, None, closed="left")],
            "armijo_min_increase": [Interval(Real, 0.0, 1.0, closed="neither")],
            "random_state": ["random_state"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_components: int | None = None,
        init: str = "fastica",
        max_iter: int = 200,
        history_size: int = 10,
        tol: float = 1e-5,
        max_line_search_steps: int = 20,
        armijo_min_increase: float = 1e-4,
        random_state: int | None = None,
    ):
        """Initializes the OTICA model.

        Args:
            n_components (int | None, optional): Number of retained components.
                Defaults to None.
            init (str, optional): Initialization method. Defaults to `"fastica"`.
            max_iter (int, optional): Maximum L-BFGS iterations. Defaults to 200.
            history_size (int, optional): Maximum number of L-BFGS correction pairs.
                Defaults to 10.
            tol (float, optional): Convergence tolerance. Defaults to 1e-5.
            max_line_search_steps (int, optional): Maximum Armijo backtracking steps.
                Defaults to 20.
            armijo_min_increase (float, optional): Armijo sufficient increase constant.
                Defaults to 1e-4.
            random_state (int | None, optional): Random seed. Defaults to None.
        """
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.history_size = history_size
        self.tol = tol
        self.max_line_search_steps = max_line_search_steps
        self.armijo_min_increase = armijo_min_increase
        self.random_state = random_state

    def _whiten(
        self,
        X: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centers observations and projects them onto a whitened subspace.

        Args:
            X (np.ndarray): Input observations.
            n_components (int): Number of retained components.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Whitened data, feature means,
                and the whitening matrix.
        """
        mean = X.mean(axis=0)
        centered = X - mean
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        whitening = (
            eigenvectors[:, :n_components] / np.sqrt(eigenvalues[:n_components])
        ).T

        return centered @ whitening.T, mean, whitening

    def _init_unmixing(
        self,
        X: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Computes the initial orthogonal unmixing matrix.

        Args:
            X (np.ndarray): Whitened observations.
            rng (np.random.RandomState): Random number generator used for stochastic
                initialization.

        Returns:
            np.ndarray: Initial orthogonal unmixing matrix.
        """
        if self.init == "identity":
            return np.eye(X.shape[1])
        if self.init == "random":
            return np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))[0]

        estimator = FastICA(
            whiten=False,
            random_state=rng,
        ).fit(X)
        left, _, right = np.linalg.svd(estimator.components_, full_matrices=False)

        return left @ right

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
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
        X = validate_data(self, X, ensure_min_samples=2)

        n = X.shape[0]

        n_components = X.shape[1] if self.n_components is None else self.n_components
        n_components = min(n_components, X.shape[1], n - 1)

        whitened, self.mean_, self.whitening_ = self._whiten(X, n_components)
        rng = check_random_state(self.random_state)
        init_unmixing = self._init_unmixing(whitened, rng)

        unmixing, self.n_iter_, self.objective_ = self._solve(
            whitened,
            gauss_quantiles(n),
            init_unmixing,
        )
        self.components_ = unmixing @ self.whitening_
        self.mixing_ = np.linalg.pinv(self.components_)

        return self

    @validate_params(
        {
            "X": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def transform(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Recovers independent components from observations.

        Args:
            X (np.typing.ArrayLike): Observations to transform.

        Returns:
            np.ndarray: Recovered independent components.
        """
        check_is_fitted(self, ["components_"])
        X = validate_data(self, X, reset=False)

        return (X - self.mean_) @ self.components_.T

    @validate_params(
        {
            "X": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def inverse_transform(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Reconstructs observations from independent components.

        Args:
            X (np.typing.ArrayLike): Independent components to reconstruct.

        Returns:
            np.ndarray: Reconstructed observations in the original feature space.
        """
        check_is_fitted(self, ["mixing_", "mean_"])
        X = check_array(X)  # type: ignore

        return X @ self.mixing_.T + self.mean_
