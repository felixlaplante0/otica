from numbers import Integral, Real
from typing import ClassVar, Self, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.decomposition import FastICA  # type: ignore
from sklearn.utils._param_validation import (  # type: ignore
    Interval,  # type: ignore
    StrOptions,  # type: ignore
    validate_params,  # type: ignore
)
from sklearn.utils.validation import (  # type: ignore
    check_array,  # type: ignore
    check_is_fitted,  # type: ignore
    check_random_state,  # type: ignore
    validate_data,  # type: ignore
)

from ._lbfgs import LBFGSMixin
from ._utils import gauss_quantiles


class OTICA(LBFGSMixin, TransformerMixin, BaseEstimator):
    """Optimal transport ICA using orthogonal L-BFGS.

    The estimator whitens the observations and maximizes the sum of empirical squared
    Wasserstein distances between the recovered components and a standard Gaussian.
    Optimization uses an L-BFGS approximation on the orthogonal group.

    Data preprocessing settings:
        - `n_components`: Number of components retained during whitening. When this is
          smaller than the ambient dimension, optimization runs in the reduced space.
        - `whiten`: Whether to center and whiten the observations before fitting. When
          disabled, the observations are assumed to already be whitened.

    Initialization settings:
        - `w_init`: Initialization method or square initial unmixing matrix.
        - `random_state`: Seed used for random initialization and FastICA.

    Optimization settings:
        - `max_iter`: Maximum number of L-BFGS iterations.
        - `history_size`: Maximum number of L-BFGS correction pairs.
        - `tol`: Convergence tolerance.
        - `max_line_search_steps`: Maximum number of Armijo backtracking steps.
        - `armijo_min_increase`: Armijo sufficient increase constant.

    Attributes:
        n_components (int | None): Number of retained components before fitting.
        whiten (bool): Whether to whiten the data before fitting.
        w_init (str | np.typing.ArrayLike): Initialization method or initial unmixing
            matrix.
        max_iter (int): Maximum L-BFGS iterations.
        history_size (int): Maximum number of L-BFGS correction pairs.
        tol (float): Convergence tolerance.
        max_line_search_steps (int): Maximum Armijo backtracking steps.
        armijo_min_increase (float): Armijo sufficient increase constant.
        random_state (int | None): Random seed.
        mean_ (np.ndarray): Feature means removed during fitting. Available only when
            `whiten=True`.
        whitening_ (np.ndarray): Whitening matrix used to project onto the reduced
            component space. Available only when `whiten=True`.
        components_ (np.ndarray): Estimated unmixing matrix.
        mixing_ (np.ndarray): Pseudo-inverse of the unmixing matrix.
        n_iter_ (int): Number of L-BFGS iterations.
        converged_ (bool): Indicator of estimator convergence.
    """

    n_components: int | None
    whiten: bool
    w_init: str | np.typing.ArrayLike
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
    n_iter_: int

    _parameter_constraints: ClassVar[dict] = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "whiten": ["boolean"],
        "w_init": [StrOptions({"fastica", "random"}), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "history_size": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "max_line_search_steps": [Interval(Integral, 1, None, closed="left")],
        "armijo_min_increase": [Interval(Real, 0.0, 1.0, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components: int | None = None,
        *,
        whiten: bool = True,
        w_init: str | np.typing.ArrayLike = "fastica",
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
            whiten (bool, optional): Whether to whiten the data. Defaults to True.
            w_init (str | np.typing.ArrayLike, optional): Initialization method or
                square initial unmixing matrix. Defaults to `"fastica"`.
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
        self.whiten = whiten
        self.w_init = w_init
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
        d = X.shape[1]

        if not isinstance(self.w_init, str):
            w_init = check_array(self.w_init, ensure_2d=True)
            if w_init.shape != (d, d):
                raise ValueError(
                    f"w_init must have shape {(d, d)}, but got {w_init.shape}."
                )

            return w_init

        if self.w_init == "random":
            return np.linalg.qr(rng.standard_normal((d, d)))[0]

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
        self._validate_params()
        X = cast(np.ndarray, validate_data(self, X))  # type: ignore

        n, d = X.shape

        n_components = d if self.n_components is None else self.n_components

        if self.whiten:
            X, self.mean_, self.whitening_ = self._whiten(X, n_components)
        else:
            self.__dict__.pop("mean_", None)
            self.__dict__.pop("whitening_", None)

        rng = check_random_state(self.random_state)
        init_unmixing = self._init_unmixing(X, rng)

        self._solve(
            X,
            gauss_quantiles(n),
            init_unmixing,
        )

        self.components_ = (
            self.unmixing_ @ self.whitening_ if self.whiten else self.unmixing_
        )
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
        X = cast(np.ndarray, validate_data(self, X, reset=False))  # type: ignore

        if self.whiten:
            X = X - self.mean_

        return X @ self.components_.T

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
        check_is_fitted(self, ["mixing_"])
        X = cast(np.ndarray, check_array(X))  # type: ignore
        n_components = self.mixing_.shape[1]
        if X.shape[1] != n_components:
            raise ValueError(
                f"X must have {n_components} components, but got {X.shape[1]}."
            )

        X = X @ self.mixing_.T
        if self.whiten:
            X = X + self.mean_

        return X
