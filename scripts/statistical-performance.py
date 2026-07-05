import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from utils import amari_index, gen_data, gen_t

from otica import OTICA

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
MODELS = {"OT-ICA": OTICA, "FastICA": FastICA}
DISTRIBUTIONS = ("Laplace", "Uniform", "Exponential", "Discrete")
N_RUNS = 20


def nd_results(distribution):
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", (100, 250, 500, 1000, 1500), None, 7),
        ("d", (5, 10, 15, 20, 30), 1000, None),
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
                            "Amari index": amari_index(model.components_, mixing),
                        }
                    )

    return pd.DataFrame(results)


def heterogeneity_results(condition_number):
    results = []
    for maximum_df in (2.5, 5, 10, 20, 40):
        for run in range(N_RUNS):
            data, mixing = gen_t(
                3000,
                8,
                np.linspace(2.5, maximum_df, 8),
                condition_number,
            )
            for name, factory in MODELS.items():
                model = factory(random_state=run).fit(data)
                results.append(
                    {
                        "Value": maximum_df,
                        "Method": name,
                        "Amari index": amari_index(model.components_, mixing),
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
    axis.set(xlabel=xlabel, ylabel="Amari index", title=title)
    if legend:
        axis.legend(loc="upper left")
    axis.grid(alpha=0.3)


parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--nd", action="store_true")
mode.add_argument("--heterogeneity", action="store_true")
args = parser.parse_args()

if args.nd:
    figure, axes = plt.subplots(2, 4, figsize=(28, 10), layout="constrained")
    figure.suptitle("Amari index with varying sample size and dimension")
    for column, distribution in enumerate(DISTRIBUTIONS):
        results = nd_results(distribution)
        for row, (sweep, xlabel) in enumerate(
            (("n", "n (sample size), d = 7"), ("d", "d (dimension), n = 1000"))
        ):
            plot(
                axes[row, column],
                results[results["Sweep"] == sweep],
                xlabel,
                distribution,
                row == 0 and column == 0,
            )
    output = "../figures/varying-nd-amari-index.pdf"
else:
    figure, axes = plt.subplots(1, 4, figsize=(28, 5), layout="constrained")
    figure.suptitle("Amari index under source and mixing-matrix heterogeneity")
    for column, condition_number in enumerate((1, 2, 3, 4)):
        results = heterogeneity_results(condition_number)
        plot(
            axes[column],
            results,
            "Maximum degrees of freedom",
            f"Condition number = {condition_number}",
            column == 0,
        )
    output = "../figures/mixing-heterogeneity-amari-index.pdf"

figure.savefig(output)
plt.show()
