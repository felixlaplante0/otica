import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from otica import OTICA
from sklearn.decomposition import FastICA

from _utils import gen_data

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
MODELS = {"OT-ICA": OTICA, "FastICA": FastICA}
MARKERS = ("o", "s")
N_RUNS = 20
N_RANGE = (250, 500, 1000, 2000, 4000)
D_RANGE = (5, 10, 15, 20, 30)
FIXED_N = 1000
FIXED_D = 8
DISTRIBUTION = "Uniform"


def main():
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", N_RANGE, None, FIXED_D),
        ("d", D_RANGE, FIXED_N, None),
    ):
        for value in grid:
            for run in range(N_RUNS):
                data, _ = gen_data(
                    fixed_n or value,
                    fixed_d or value,
                    DISTRIBUTION,
                )
                for name, factory in MODELS.items():
                    model = factory(random_state=run)
                    start = time.perf_counter()
                    model.fit(data)
                    results.append(
                        {
                            "Sweep": sweep,
                            "Value": value,
                            "Method": name,
                            "Runtime (seconds)": time.perf_counter() - start,
                        }
                    )

    results = pd.DataFrame(results)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, (sweep, xlabel, title) in zip(
        axes,
        (
            (
                "n",
                f"n (sample size), d = {FIXED_D}",
                "Runtime scaling with sample size",
            ),
            ("d", f"d (dimension), n = {FIXED_N}", "Runtime scaling with dimension"),
        ),
    ):
        subset = results[results["Sweep"] == sweep]
        values = subset["Value"].drop_duplicates()
        sns.pointplot(
            data=subset,
            x="Value",
            y="Runtime (seconds)",
            hue="Method",
            dodge=0.25,
            linestyles="-",
            errorbar="sd",
            capsize=0.1,
            ax=axis,
            legend=axis is axes[0],
        )
        lines = [line for line in axis.lines if len(line.get_xdata()) == len(values)]
        for line, marker in zip(lines, MARKERS, strict=True):
            line.set_marker(marker)
        if axis is axes[0]:
            for method, marker in zip(MODELS, MARKERS, strict=True):
                next(
                    line for line in axis.lines if line.get_label() == method
                ).set_marker(marker)
        axis.set(xlabel=xlabel, ylabel="Runtime (seconds) ↓", title=title)
        if axis is axes[0]:
            axis.legend(loc="upper left")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    (ROOT / "figures").mkdir(exist_ok=True)
    figure.savefig(ROOT / "figures" / "runtime-scaling.pdf")
    plt.show()


if __name__ == "__main__":
    main()
