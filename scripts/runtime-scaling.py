import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA
from utils import gen_data

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

# Set defaults
np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"OT-ICA": OTICA, "FastICA": FastICA}
N_RUNS = 20

results = []
for sweep, grid, fixed_n, fixed_d in (
    ("n", (250, 500, 1000, 2000, 4000), None, 7),
    ("d", (5, 10, 15, 20, 30), 1000, None),
):
    for value in grid:
        for run in range(N_RUNS):
            data, _ = gen_data(
                fixed_n or value,
                fixed_d or value,
                "Uniform",
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
figure, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
for axis, (sweep, xlabel, title) in zip(
    axes,
    (
        ("n", "n (sample size), d = 7", "Runtime scaling with sample size"),
        ("d", "d (dimension), n = 1000", "Runtime scaling with dimension"),
    ),
):
    sns.lineplot(
        data=results[results["Sweep"] == sweep],
        x="Value",
        y="Runtime (seconds)",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axis,
    )
    axis.set(xlabel=xlabel, ylabel="Runtime (seconds)", title=title)
    axis.legend(loc="upper left")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
figure.savefig(ROOT / "figures" / "runtime-scaling.pdf")
plt.show()
