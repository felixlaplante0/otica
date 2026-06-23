from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class LBFGSMixin:
    """Provides a parameter-driven Riemannian L-BFGS solver."""

    max_iter: int
    memory: int
    tol: float
    initial_step: float
    contraction: float
    sufficient_increase: float
    max_line_search_steps: int
    curvature_tol: float
    gradient_tol: float

    def _objective(self, sources: np.ndarray, quantiles: np.ndarray) -> float:
        """Computes the total squared Wasserstein non-Gaussianity."""
        differences = np.sort(sources, axis=1) - quantiles
        return float(np.mean(differences**2, axis=1).sum())

    def _relative_gradient(
        self,
        sources: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        """Computes the relative gradient on the orthogonal group."""
        n_samples = sources.shape[1]
        order = np.argsort(sources, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(n_samples)[None, :], axis=1)
        scores = (2.0 / n_samples) * (sources - quantiles[ranks])
        gradient = scores @ sources.T
        return 0.5 * (gradient - gradient.T)

    def _direction(
        self,
        gradient: np.ndarray,
        history: tuple[list[np.ndarray], list[np.ndarray], list[float]],
    ) -> np.ndarray:
        """Computes an L-BFGS direction with identity vector transport."""
        steps, differences, inverse_curvatures = history
        direction = gradient.copy()
        coefficients: list[float] = []

        for step, difference, inverse_curvature in zip(
            reversed(steps),
            reversed(differences),
            reversed(inverse_curvatures),
            strict=True,
        ):
            coefficient = inverse_curvature * float(np.sum(step * direction))
            coefficients.append(coefficient)
            direction -= coefficient * difference

        if differences:
            difference = differences[-1]
            scale = float(
                np.sum(steps[-1] * difference) / np.sum(difference * difference)
            )
            direction *= scale

        for step, difference, inverse_curvature, coefficient in zip(
            steps,
            differences,
            inverse_curvatures,
            reversed(coefficients),
            strict=True,
        ):
            correction = inverse_curvature * float(np.sum(difference * direction))
            direction += (coefficient - correction) * step

        return direction

    def _armijo(
        self,
        rotation: np.ndarray,
        direction: np.ndarray,
        objective: float,
        slope: float,
        X: np.ndarray,
        quantiles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Returns the first update satisfying the Armijo condition."""
        step = self.initial_step
        for _ in range(self.max_line_search_steps):
            candidate_rotation = expm(step * direction) @ rotation
            candidate_sources = candidate_rotation @ X
            candidate_objective = self._objective(candidate_sources, quantiles)
            if (
                candidate_objective
                >= objective + self.sufficient_increase * step * slope
            ):
                return (
                    candidate_rotation,
                    candidate_sources,
                    candidate_objective,
                    step,
                )
            step *= self.contraction
        return None

    def _solve(
        self,
        X: np.ndarray,
        quantiles: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Maximizes Wasserstein non-Gaussianity by Riemannian L-BFGS."""
        X = X.T
        sources = rotation @ X
        gradient = self._relative_gradient(sources, quantiles)
        objective = self._objective(sources, quantiles)
        history: tuple[list[np.ndarray], list[np.ndarray], list[float]] = ([], [], [])

        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            direction = self._direction(gradient, history)
            slope = float(np.sum(gradient * direction))
            if slope <= 0.0:
                direction = gradient
                slope = float(np.sum(gradient * gradient))
            if slope < self.curvature_tol:
                break

            update = self._armijo(
                rotation,
                direction,
                objective,
                slope,
                X,
                quantiles,
            )
            if update is None:
                break

            new_rotation, new_sources, new_objective, step_size = update
            new_gradient = self._relative_gradient(new_sources, quantiles)
            step = step_size * direction
            difference = gradient - new_gradient
            curvature = float(np.sum(step * difference))

            if curvature > self.curvature_tol:
                steps, differences, inverse_curvatures = history
                steps.append(step)
                differences.append(difference)
                inverse_curvatures.append(1.0 / curvature)
                if len(steps) > self.memory:
                    steps.pop(0)
                    differences.pop(0)
                    inverse_curvatures.pop(0)

            relative_gain = (new_objective - objective) / (
                abs(objective) + self.curvature_tol
            )
            rotation = new_rotation
            sources = new_sources
            gradient = new_gradient
            objective = new_objective
            if relative_gain < self.tol or np.linalg.norm(gradient) < self.gradient_tol:
                break

        return rotation, n_iter
