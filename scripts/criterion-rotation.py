from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from otica.utils._wasserstein import gauss_quantiles
from scipy.special import ndtri

from _utils import _gen_quadrature, _gen_sources

# Set plot parameters
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

# Set defaults
np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 3000
QUADRATURE_ORDER = 160
N_RUNS = 20
N_ANGLES = 181
DISTRIBUTIONS = ("Laplace", "Uniform", "Exponential", "Uniform-Exponential Mixture")


def log_cosh(x):
    return np.logaddexp(x, -x) - np.log(2.0)


def weighted_wasserstein(values, weights):
    order = np.argsort(values, axis=0)
    values = np.take_along_axis(values, order, axis=0)
    weights = np.take_along_axis(
        np.broadcast_to(weights[:, None], values.shape), order, axis=0
    )
    boundaries = np.clip(
        np.vstack((np.zeros(values.shape[1]), np.cumsum(weights, axis=0))), 0.0, 1.0
    )
    boundaries[-1] = 1.0
    density = np.exp(-0.5 * ndtri(boundaries) ** 2) / np.sqrt(2.0 * np.pi)

    return (
        np.sum(weights * values**2, axis=0)
        + 1.0
        - 2.0 * np.sum(values * (density[:-1] - density[1:]), axis=0)
    )


def gaussian_log_cosh():
    nodes, weights = _gen_quadrature(QUADRATURE_ORDER, "Gaussian")
    return np.sum(weights * log_cosh(nodes))


def quadrature_scores(distribution, angles):
    nodes, weights = _gen_quadrature(QUADRATURE_ORDER, distribution)
    pairs = np.array(np.meshgrid(nodes, nodes, indexing="ij")).reshape(2, -1).T
    product_weights = np.outer(weights, weights).ravel()
    gaussian_expectation = gaussian_log_cosh()
    scores = np.empty((2, len(angles)))

    for index, angle in enumerate(angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        values = pairs @ np.array(((cosine, -sine), (sine, cosine))).T
        expected_log_cosh = product_weights @ log_cosh(values)
        scores[:, index] = (
            np.sum((expected_log_cosh - gaussian_expectation) ** 2),
            np.sum(weighted_wasserstein(values, product_weights)),
        )

    return scores[0], scores[1]


def empirical_scores(
    samples: np.ndarray, angles: np.ndarray, gaussian_log_cosh: float
) -> tuple[np.ndarray, np.ndarray]:
    samples = samples - samples.mean(axis=0)
    quantiles = gauss_quantiles(len(samples))
    correction = 1.0 - np.mean(quantiles**2)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotations = np.array(((cosine, -sine), (sine, cosine))).transpose(2, 0, 1)
    rotated = np.einsum("nd,akd->ank", samples, rotations)
    log_cosh_scores = np.sum(
        (np.mean(log_cosh(rotated), axis=1) - gaussian_log_cosh) ** 2, axis=1
    )
    residual = np.sort(rotated, axis=1) - quantiles[None, :, None]
    wasserstein_scores = np.mean(residual**2, axis=1).sum(axis=1) + 2.0 * correction

    return log_cosh_scores, wasserstein_scores


def empirical_results(angles: np.ndarray, gaussian_log_cosh: float) -> pd.DataFrame:
    records = []
    degrees = np.rad2deg(angles)

    for run in range(N_RUNS):
        np.random.seed(42 + run)
        for distribution in DISTRIBUTIONS:
            samples = _gen_sources(N_SAMPLES, 2, distribution)
            scores = empirical_scores(samples, angles, gaussian_log_cosh)
            for criterion, values in zip(
                ("log-cosh", "Wasserstein"), scores, strict=True
            ):
                records.extend(
                    {
                        "Run": run,
                        "Distribution": distribution,
                        "Criterion": criterion,
                        "Angle": angle,
                        "Score": score,
                    }
                    for angle, score in zip(degrees, values, strict=True)
                )

    return pd.DataFrame(records)


def main():
    angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, N_ANGLES)
    gaussian_expectation = gaussian_log_cosh()
    oracles = {
        distribution: quadrature_scores(distribution, angles)
        for distribution in DISTRIBUTIONS
    }
    results = empirical_results(angles, gaussian_expectation)

    figure, axes = plt.subplots(1, 4, figsize=(28, 5), layout="constrained")
    degrees = np.rad2deg(angles)
    for axis, (distribution, oracle_scores) in zip(axes, oracles.items(), strict=True):
        title = distribution.replace(" Mixture", " mixture")
        wasserstein_axis = axis.twinx()
        for criterion, score_axis, color, oracle_score in zip(
            ("log-cosh", "Wasserstein"),
            (axis, wasserstein_axis),
            ("tab:blue", "tab:orange"),
            oracle_scores,
            strict=True,
        ):
            subset = results[
                (results["Distribution"] == distribution)
                & (results["Criterion"] == criterion)
            ]
            sns.lineplot(
                data=subset,
                x="Angle",
                y="Score",
                errorbar=("ci", 95),
                seed=42,
                color=color,
                label=f"{criterion} empirical",
                linewidth=2.0,
                ax=score_axis,
            )
            score_axis.plot(
                degrees,
                oracle_score,
                ":",
                color=color,
                label=f"{criterion} Oracle",
                linewidth=2.0,
            )
            score_axis.tick_params(axis="y", colors=color)

        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
        axis.set(
            xlabel="Rotation angle (degrees)",
            ylabel="Objective ↑",
            title=title,
        )
        wasserstein_axis.set_ylabel("")
        axis.set_xticks(np.arange(-45, 46, 15))
        axis.grid(alpha=0.3)
        lines = [
            line
            for line in axis.get_lines() + wasserstein_axis.get_lines()
            if not line.get_label().startswith("_")
        ]
        if axis is axes[0]:
            axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
        else:
            axis.get_legend().remove()
        wasserstein_axis.get_legend().remove()

    figure.suptitle("Objective values by rotation angle")
    figure.savefig(ROOT / "figures" / "criterion-rotation.pdf")
    plt.show()


if __name__ == "__main__":
    main()
