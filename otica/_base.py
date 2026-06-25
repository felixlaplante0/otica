from numbers import Integral, Real
from typing import Self, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from ._lbfgs import LBFGSMixin
from ._utils import gauss_quantiles


class OTICA(LBFGSMixin, TransformerMixin, BaseEstimator):
    """Optimal transport ICA using Riemannian L-BFGS.

    The estimator whitens the observations and maximizes the sum of empirical squared
    Wasserstein distances between the recovered components and a standard Gaussian.
    Optimization uses torch L-BFGS with an orthogonal parametrization of the unmixing
    matrix.

    The estimator supports an optional rank reduction step through `n_components`.
    When `n_components` is smaller than the ambient dimension, the whitening matrix
    becomes rectangular and the optimization runs in the reduced component space.

    Torch L-BFGS behavior is controlled by `orthogonal_map`, `lr`, `max_iter`,
    `history_size`, `tolerance_grad`, and `tolerance_change`.

    Attributes:
        n_components (int | None): Number of retained components before fitting.
        n_components_ (int): Number of retained components after fitting.
        mean_ (np.ndarray): Feature means removed during fitting.
        whitening_ (np.ndarray): Whitening matrix used to project onto the reduced
            component space.
        unmixing_ (np.ndarray): Orthogonal unmixing matrix in the whitened space.
        components_ (np.ndarray): Estimated unmixing matrix.
        mixing_ (np.ndarray): Pseudo-inverse of the unmixing matrix.
        objective_ (float): Final optimized objective value.
        n_iter_ (int): Number of L-BFGS iterations.
    """

    n_components: int | None
    mean_: np.ndarray
    whitening_: np.ndarray
    unmixing_: np.ndarray
    components_: np.ndarray
    mixing_: np.ndarray
    objective_: float
    n_iter_: int
    n_components_: int

    @validate_params(
        {
            "n_components": [Interval(Integral, 1, None, closed="left"), None],
            "init": [StrOptions({"fastica", "identity", "random"})],
            "orthogonal_map": [StrOptions({"matrix_exp", "cayley"})],
            "lr": [Interval(Real, 0.0, None, closed="neither")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "history_size": [Interval(Integral, 1, None, closed="left")],
            "tolerance_grad": [Interval(Real, 0.0, None, closed="left")],
            "tolerance_change": [Interval(Real, 0.0, None, closed="left")],
            "random_state": ["random_state"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_components: int | None = None,
        init: str = "fastica",
        orthogonal_map: str = "matrix_exp",
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 100,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        random_state: int | None = None,
    ):
        """Initializes the OTICA model.

        Args:
            n_components (int | None, optional): Number of retained components.
                Defaults to None.
            init (str, optional): Initialization method. Defaults to `"fastica"`.
            orthogonal_map (str, optional): Orthogonal parametrization used by torch
                L-BFGS. Defaults to `"matrix_exp"`.
            lr (float, optional): Torch L-BFGS learning rate. Defaults to 1.0.
            max_iter (int, optional): Maximum L-BFGS iterations. Defaults to 200.
            history_size (int, optional): Torch L-BFGS history size. Defaults to 100.
            tolerance_grad (float, optional): Torch L-BFGS gradient tolerance.
                Defaults to 1e-7.
            tolerance_change (float, optional): Torch L-BFGS change tolerance.
                Defaults to 1e-9.
            random_state (int | None, optional): Random seed. Defaults to None.
        """
        self.n_components = n_components
        self.init = init
        self.orthogonal_map = orthogonal_map
        self.lr = lr
        self.max_iter = max_iter
        self.history_size = history_size
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
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
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64, ensure_min_samples=2),
        )

        n_components = X.shape[1] if self.n_components is None else self.n_components
        n_components = min(n_components, X.shape[1], X.shape[0] - 1)

        self.n_components_ = n_components
        whitened, self.mean_, self.whitening_ = self._whiten(X, n_components)
        rng = check_random_state(self.random_state)
        init_unmixing = self._init_unmixing(whitened, rng)

        self.unmixing_, self.n_iter_, self.objective_ = self._solve(
            whitened,
            gauss_quantiles(X.shape[0]),
            init_unmixing,
        )
        self.components_ = self.unmixing_ @ self.whitening_
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
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64, reset=False),
        )

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
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64, reset=False),
        )

        return X @ self.mixing_.T + self.mean_
