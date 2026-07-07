import numpy as np
from scipy.linalg import expm


class LBFGSMixin:
    """Provides a minimal orthogonal L-BFGS solver."""

    max_iter: int
    history_size: int
    tol: float
    max_line_search_steps: int
    armijo_min_increase: float
    unmixing_: np.ndarray
    n_iter_: int
    converged_: bool

    def _objective_and_grad(
        self,
        unmixing: np.ndarray,
        X: np.ndarray,
        quantiles: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Computes the objective and its tangent gradient.

        Args:
            unmixing (np.ndarray): Current orthogonal unmixing matrix.
            X (np.ndarray): Whitened observations.
            quantiles (np.ndarray): Standard-normal reference quantiles.

        Returns:
            tuple[float, np.ndarray]: Wasserstein objective and skew-symmetric tangent
                gradient in exponential coordinates.
        """
        n = X.shape[0]

        sources = X @ unmixing.T
        order = np.argsort(sources, axis=0)
        residual = np.take_along_axis(sources, order, axis=0) - quantiles[:, None]

        score_grad = np.empty_like(residual)
        np.put_along_axis(score_grad, order, 2.0 * residual / n, axis=0)

        euclidean_grad = score_grad.T @ X
        raw_grad = euclidean_grad @ unmixing.T

        return np.sum(residual * residual) / n, 0.5 * (raw_grad - raw_grad.T)

    def _direction(
        self,
        grad: np.ndarray,
        history: list[tuple[np.ndarray, np.ndarray, float]],
    ) -> np.ndarray:
        """Computes the L-BFGS ascent direction.

        Args:
            grad (np.ndarray): Current tangent gradient.
            history (list[tuple[np.ndarray, np.ndarray, float]]): Stored correction
                pairs and inverse inner products.

        Returns:
            np.ndarray: Skew-symmetric ascent direction.
        """
        direction = grad.copy()
        step_weights = []

        for step_delta, grad_delta, inverse_curvature in reversed(history):
            step_weight = inverse_curvature * np.sum(step_delta * direction)
            step_weights.append(step_weight)
            direction -= step_weight * grad_delta

        if history:
            step_delta, grad_delta, _ = history[-1]
            direction *= np.sum(step_delta * grad_delta) / np.sum(grad_delta**2)

        for (step_delta, grad_delta, inverse_curvature), step_weight in zip(
            history,
            reversed(step_weights),
            strict=True,
        ):
            grad_weight = inverse_curvature * np.sum(grad_delta * direction)
            direction += step_delta * (step_weight - grad_weight)

        return direction

    def _line_search(
        self,
        unmixing: np.ndarray,
        X: np.ndarray,
        quantiles: np.ndarray,
        objective: float,
        grad: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """Finds an Armijo step for objective maximization.

        Args:
            unmixing (np.ndarray): Current orthogonal unmixing matrix.
            X (np.ndarray): Whitened observations.
            quantiles (np.ndarray): Standard-normal reference quantiles.
            objective (float): Current objective value.
            grad (np.ndarray): Current tangent gradient.
            direction (np.ndarray): Proposed skew-symmetric ascent direction.

        Returns:
            tuple[float, float, np.ndarray]: Step size, new objective value, and new
                orthogonal unmixing matrix.
        """
        derivative = np.sum(grad * direction)
        step = 1.0

        for _ in range(self.max_line_search_steps):
            new_unmixing = expm(step * direction) @ unmixing
            new_objective, _ = self._objective_and_grad(
                new_unmixing,
                X,
                quantiles,
            )

            if (
                new_objective
                >= objective + self.armijo_min_increase * step * derivative
            ):
                return step, new_objective, new_unmixing

            step *= 0.5

        return 0.0, objective, unmixing

    def _solve(
        self,
        X: np.ndarray,
        quantiles: np.ndarray,
        init_unmixing: np.ndarray,
    ):
        """Maximizes Wasserstein non-Gaussianity over orthogonal matrices.

        Args:
            X (np.ndarray): Whitened observations.
            quantiles (np.ndarray): Standard-normal reference quantiles.
            init_unmixing (np.ndarray): Initial orthogonal unmixing matrix.
        """
        unmixing = init_unmixing.copy()
        history: list[tuple[np.ndarray, np.ndarray, float]] = []
        objective, grad = self._objective_and_grad(unmixing, X, quantiles)
        converged = False

        for n_iter in range(1, self.max_iter + 1):  # noqa: B007
            if np.max(np.abs(grad)) <= self.tol:
                converged = True
                break

            direction = self._direction(grad, history)
            if np.sum(grad * direction) <= 0.0:
                direction = grad
                history.clear()

            step, new_objective, new_unmixing = self._line_search(
                unmixing,
                X,
                quantiles,
                objective,
                grad,
                direction,
            )
            if step == 0.0:
                break

            _, new_grad = self._objective_and_grad(new_unmixing, X, quantiles)
            step_delta = step * direction
            grad_delta = grad - new_grad
            curvature = np.sum(step_delta * grad_delta)

            if curvature > np.finfo(np.float64).eps:
                history.append((step_delta, grad_delta, 1.0 / curvature))
                history = history[-self.history_size :]

            norm = np.linalg.norm(new_unmixing - unmixing)

            unmixing = new_unmixing
            grad = new_grad
            objective = new_objective

            if norm <= self.tol:
                converged = True
                break

        self.unmixing_ = unmixing
        self.n_iter_ = n_iter
        self.converged_ = converged
