import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from otica import OTICA
from otica.utils import amari_index
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from _utils import gen_data, gen_gaussianity

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

# Filter warnings from FastICA not converging
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set defaults
np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"OT-ICA": OTICA, "FastICA": FastICA}
DISTRIBUTIONS = ("Laplace", "Uniform", "Exponential", "Uniform-Exponential Mixture")
N_RUNS = 20
N_RANGE = (100, 250, 500, 1000, 1500)
D_RANGE = (5, 10, 15, 20, 30)
FIXED_N = 1000
FIXED_D = 8
GAUSSIANITY_N = 3000
GAUSSIANITY_D = 8
EPSILON_RANGE = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)


def nd_results(distribution):
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", N_RANGE, None, FIXED_D),
        ("d", D_RANGE, FIXED_N, None),
    ):
        for value in grid:
            for run in range(N_RUNS):
                data, mixing = gen_data(
                    fixed_n or value,
                    fixed_d or value,
                    distribution,
                )
                for name, factory in MODELS.items():
                    model = factory(random_state=run).fit(data)
                    results.append(
                        {
                            "Sweep": sweep,
                            "Value": value,
                            "Method": name,
                            "Amari index": amari_index(mixing, model.components_),
                        }
                    )

    return pd.DataFrame(results)


def gaussianity_results(distribution):
    results = []
    for epsilon in EPSILON_RANGE:
        for run in range(N_RUNS):
            data, mixing = gen_gaussianity(
                GAUSSIANITY_N,
                GAUSSIANITY_D,
                distribution,
                epsilon,
            )
            for name, factory in MODELS.items():
                model = factory(random_state=run).fit(data)
                results.append(
                    {
                        "Value": epsilon,
                        "Method": name,
                        "Amari index": amari_index(mixing, model.components_),
                    }
                )

    return pd.DataFrame(results)


def plot(axis, results, xlabel, title, legend):
    sns.lineplot(
        data=results,
        x="Value",
        y="Amari index",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axis,
        legend=legend,
    )
    axis.set(xlabel=xlabel, ylabel="Amari index ↓", title=title)
    if legend:
        axis.legend(loc="upper left")
    axis.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nd", action="store_true")
    mode.add_argument("--gaussianity", action="store_true")
    args = parser.parse_args()

    if args.nd:
        figure, axes = plt.subplots(2, 4, figsize=(28, 10), layout="constrained")
        figure.suptitle("Amari index with varying sample size and dimension")
        for column, distribution in enumerate(DISTRIBUTIONS):
            results = nd_results(distribution)
            for row, (sweep, xlabel) in enumerate(
                (
                    ("n", f"n (sample size), d = {FIXED_D}"),
                    ("d", f"d (dimension), n = {FIXED_N}"),
                )
            ):
                plot(
                    axes[row, column],
                    results[results["Sweep"] == sweep],
                    xlabel,
                    distribution,
                    row == 0 and column == 0,
                )
        output = ROOT / "figures" / "varying-nd-amari-index.pdf"
    else:
        figure, axes = plt.subplots(1, 4, figsize=(28, 5), layout="constrained")
        figure.suptitle("Amari index as sources approach Gaussianity")
        for column, distribution in enumerate(DISTRIBUTIONS):
            results = gaussianity_results(distribution)
            plot(
                axes[column],
                results,
                rf"$\varepsilon$, n = {GAUSSIANITY_N}, d = {GAUSSIANITY_D}",
                distribution,
                column == 0,
            )
        output = ROOT / "figures" / "gaussianity-amari-index.pdf"

    figure.savefig(output)
    plt.show()


if __name__ == "__main__":
    main()
