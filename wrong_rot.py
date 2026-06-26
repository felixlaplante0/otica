import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from otica import OTICA


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def gen_data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([0.0, 3.0, -3.0], size=(n, 2), p=[8 / 9, 1 / 18, 1 / 18])


def logcosh(x):
    return np.logaddexp(x, -x) - np.log(2.0)


x = gen_data()

thetas = np.linspace(0, np.pi / 4, 2001)
scores = np.array([logcosh(x @ rot(t).T).sum(axis=1).mean() for t in thetas])
w_argmax = rot(thetas[np.argmax(scores)])

otica = OTICA(n_components=2, random_state=0, init="random", whiten=False).fit(x)
w_otica = otica.components_

fig, ax = plt.subplots(figsize=(5, 5))

for row in w_argmax:
    ax.arrow(
        0,
        0,
        row[0],
        row[1],
        color="tab:blue",
        head_width=0.04,
        length_includes_head=True,
    )

for row in w_otica:
    ax.arrow(
        0,
        0,
        row[0],
        row[1],
        color="tab:orange",
        head_width=0.04,
        length_includes_head=True,
    )

ax.axhline(0, lw=0.8)
ax.axvline(0, lw=0.8)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

ax.legend(
    handles=[
        Line2D([0], [0], color="tab:blue", lw=2, label="Log-cosh argmax"),
        Line2D([0], [0], color="tab:orange", lw=2, label="OTICA"),
    ]
)

plt.tight_layout()
plt.show()
