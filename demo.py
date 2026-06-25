import numpy as np

from otica import OTICA


rng = np.random.default_rng(0)
n_samples = 2000
sources = np.column_stack(
    [
        rng.laplace(size=n_samples),
        rng.uniform(-1.0, 1.0, size=n_samples),
        rng.standard_normal(size=n_samples),
    ]
)
mixing = rng.standard_normal((3, 3))
X = sources @ mixing.T

model = OTICA()
model.fit(X)
estimated = model.transform(X)

correlations = np.abs(np.corrcoef(sources.T, estimated.T)[:3, 3:])

print("objective_:", model.objective_)
print("n_components_:", model.n_components_)
print("components_ shape:", model.components_.shape)
print("mixing_ shape:", model.mixing_.shape)
print("transformed shape:", estimated.shape)
print("max abs correlation per true source:", correlations.max(axis=1))
