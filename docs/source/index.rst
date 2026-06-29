Contrast-Free ICA
=================

**otica** is a Python package for linear independent component analysis (ICA) based on optimal transport. It recovers latent sources by maximizing their empirical squared 2-Wasserstein distances to the standard Gaussian, using a fixed non-Gaussianity criterion that requires no user-chosen contrast function or nonlinearity.

Features
--------

- **Contrast-free source separation**: Uses the squared 2-Wasserstein distance to the standard Gaussian as a fixed non-Gaussianity criterion.
- **Exact empirical objective**: Computes the one-dimensional Wasserstein criterion directly from ordered samples and Gaussian quantiles, without density estimation.
- **Riemannian optimization**: Optimizes the whitened ICA objective on the orthogonal group with a Picard-style limited-memory BFGS method and Armijo backtracking.
- **Dimension reduction**: Supports extraction of a specified number of components through principal-component whitening.
- **scikit-learn integration**: Implements the standard transformer API, including ``fit``, ``transform``, ``fit_transform``, and ``inverse_transform``.

Method
------

For observations :math:`X \in \mathbb{R}^{n \times d}`, the linear ICA model assumes

.. math::

   X = S A^\mathsf{T},

where the columns of :math:`S` are mutually independent latent sources and :math:`A` is an invertible mixing matrix. The sources are identifiable only up to permutation, sign, and scale.

OTICA centers and symmetrically whitens the observations. For each standardized candidate source, it computes the empirical squared 2-Wasserstein distance to a standard Gaussian by matching ordered observations with optimal equally weighted Gaussian quantiles. For whitened observations :math:`Z`, the fitted orthogonal rotation maximizes

.. math::

   \sum_{k = 1}^{d} \frac{1}{n} \sum_{i = 1}^{n} \left( Y_{k(i)} - q_i \right)^2, \quad Y = W Z^\mathsf{T}, \quad W W^\mathsf{T} = I_d.

Here, :math:`Y_{k(i)}` is the :math:`i`-th order statistic of component :math:`k`, and :math:`q_i` is the mean standard-normal quantile over the :math:`i`-th equal-probability interval. Under the usual ICA assumptions, including mutually independent sources with at most one Gaussian component, the population objective identifies the sources up to the unavoidable ambiguities.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install otica

Usage
-----

The following example generates three independent non-Gaussian signals, mixes them linearly, and recovers them with ``OTICA``. Because ICA is identifiable only up to permutation and sign, the Hungarian algorithm aligns the estimated components with the true sources before plotting and reporting their absolute correlations.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.optimize import linear_sum_assignment

   from otica import OTICA

   rng = np.random.default_rng(42)
   n_samples = 5_000
   time = np.linspace(0.0, 8.0, n_samples)

   # Generate independent, non-Gaussian latent sources.
   sources = np.column_stack(
       [
           rng.laplace(size=n_samples),
           rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples),
           rng.standard_t(df=5, size=n_samples) * np.sqrt(3.0 / 5.0),
       ]
   )

   # Mix the sources into the observed signals.
   mixing = np.array(
       [
           [1.0, 0.5, -0.2],
           [0.2, 1.0, 0.4],
           [-0.4, 0.1, 1.0],
       ]
   )
   X = sources @ mixing.T

   # Fit OTICA and recover the latent components.
   model = OTICA(random_state=42)
   estimated_sources = model.fit_transform(X)

   # Align components for evaluation, resolving ICA's permutation and sign ambiguity.
   correlations = np.corrcoef(sources.T, estimated_sources.T)[:3, 3:]
   source_indices, estimated_indices = linear_sum_assignment(-np.abs(correlations))
   aligned_sources = estimated_sources[:, estimated_indices]
   signs = np.sign(correlations[source_indices, estimated_indices])
   aligned_sources *= signs
   scores = np.abs(correlations[source_indices, estimated_indices])

   print("Absolute correlations:", np.round(scores, 3))

   # Compare a short segment of the standardized true and recovered sources.
   true_standardized = sources / sources.std(axis=0)
   estimated_standardized = aligned_sources / aligned_sources.std(axis=0)
   fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
   for component, ax in enumerate(axes):
       ax.plot(time[:500], true_standardized[:500, component], label="True", alpha=0.8)
       ax.plot(
           time[:500],
           estimated_standardized[:500, component],
           label="Recovered",
           alpha=0.8,
       )
       ax.set_ylabel(f"Source {component + 1}")
   axes[0].legend()
   axes[-1].set_xlabel("Time")
   fig.tight_layout()
   plt.show()

Configuration
-------------

The ``init`` parameter selects ``"fastica"`` or ``"random"`` initialization. The L-BFGS iteration count, memory, stopping tolerance, Armijo line search, and random seed are estimator parameters. Scikit-learn utilities such as ``get_params``, ``set_params``, ``clone``, and pipelines therefore handle the complete configuration.

The empirical objective is nonconvex and piecewise smooth because component ranks change at ties. Different initializations may reach different local optima, and the gradient need not vanish exactly at an order-cell boundary.

API Reference
-------------

.. autoclass:: otica.OTICA
   :members:
   :show-inheritance:
