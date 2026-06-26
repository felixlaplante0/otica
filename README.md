# OTICA

`otica` provides an optimal transport independent component analysis estimator.

## Installation

The package depends on `numpy`, `scipy`, and `scikit-learn`.

```bash
pip install .
```

## Example

```python
import numpy as np

from otica import OTICA

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5))
X = X @ rng.standard_normal((5, 5)).T

model = OTICA(n_components=3, random_state=42)
model.fit(X)
S = model.transform(X)
```

## Demo

Run the top-level `demo.py` script to see a synthetic source-separation example. It
generates three non-Gaussian latent sources, mixes them into five observed features,
fits `OTICA`, and prints a small recovery summary.
