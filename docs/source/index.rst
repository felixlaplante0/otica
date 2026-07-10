Contrast-Free ICA
=================

**otica** is a Python package for linear independent component analysis (ICA) based on optimal transport. It recovers latent sources by maximizing their empirical squared 2-Wasserstein distances to the standard Gaussian, using a fixed non-Gaussianity criterion that requires no user-chosen contrast function or nonlinearity.

Features
--------

- **Contrast-free source separation**: Uses the squared 2-Wasserstein distance to the standard Gaussian as a fixed non-Gaussianity criterion.
- **Exact empirical objective**: Computes the one-dimensional Wasserstein criterion directly from ordered samples and Gaussian quantiles, without density estimation.
- **Riemannian optimization**: Optimizes the whitened ICA objective on the orthogonal group with a Picard-style limited-memory BFGS method and Armijo backtracking.
- **Dimension reduction**: Supports extraction of a specified number of components through principal-component whitening.
- **Flexible initialization**: Accepts FastICA, random, or user-provided initial unmixing matrices through ``w_init``.
- **scikit-learn integration**: Implements the standard transformer API, including ``fit``, ``transform``, ``fit_transform``, and ``inverse_transform``.

Method
------

For observations :math:``X \in \mathbb{R}^{n \times d}``, the linear ICA model assumes

.. math::

   X = S A^\top,

where the columns of :math:``S`` are mutually independent latent sources and :math:``A`` is an invertible mixing matrix. The sources are identifiable only up to permutation, sign, and scale.

Let :math:``Z \in \mathbb{R}^{d}`` be the centered and whitened random vector. For a candidate orthogonal unmixing matrix :math:``W \in \mathbb{R}^{d \times d}``, the population Wasserstein objective is

.. math::

   F(W) = \sum_{k = 1}^{d} \mathcal{W}_2\left( \left( Z W^\top \right)_k, \mathcal{N}(0, 1) \right)^2, \quad W W^\top = I_d.

Given whitened observations :math:``Z^{(1)}, \ldots, Z^{(n)}`` collected as the rows of :math:``Z \in \mathbb{R}^{n \times d}``, let :math:``Y = Z W^\top``. OTICA maximizes the empirical objective

.. math::

   \widehat{W}_n \in \operatorname*{\arg\max}_{W W^\top = I_d} \widehat{F}_n(W) = \sum_{k = 1}^{d} \mathcal{W}_2\left( \frac{1}{n} \sum_{i = 1}^{n} \delta_{Y_{ik}}, \mathcal{N}(0, 1) \right)^2.

For each component :math:``k``, OTICA sorts the entries of the :math:``k``-th column independently as :math:``Y_{(1)k} \leq \cdots \leq Y_{(n)k}``. It evaluates each empirical Wasserstein distance exactly by matching :math:``Y_{(i)k}`` with :math:``q_i``, the mean of the standard-normal quantile function over the :math:``i``-th equal-probability interval. Under the usual ICA assumptions, including mutually independent sources with at most one Gaussian component, the population objective identifies the sources up to the unavoidable ambiguities.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   python -m pip install otica

Usage
-----

The following example generates three independent non-Gaussian signals, mixes them linearly, and recovers them with ``OTICA``. Because ICA is identifiable only up to permutation and sign, recovery is evaluated using the best absolute correlation for each true source.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from otica import OTICA

   rng = np.random.default_rng(42)
   n_samples = 5000
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

   correlations = np.corrcoef(sources.T, estimated_sources.T)[:3, 3:]
   best_indices = np.abs(correlations).argmax(axis=1)
   best_correlations = correlations[np.arange(3), best_indices]
   recovered_sources = estimated_sources[:, best_indices] * np.sign(best_correlations)
   print("Best absolute correlation per source:", np.abs(best_correlations))

   fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 6))
   for component in range(3):
       axes[component, 0].plot(time[:500], sources[:500, component])
       axes[component, 1].plot(
           time[:500], recovered_sources[:500, component], color="tab:orange"
       )
       axes[component, 0].set_ylabel(f"Source {component + 1}")

   axes[0, 0].set_title("True sources")
   axes[0, 1].set_title("Recovered sources")
   axes[-1, 0].set_xlabel("Time")
   axes[-1, 1].set_xlabel("Time")
   fig.tight_layout()
   plt.show()

Configuration
-------------

The ``w_init`` parameter selects ``"fastica"`` or ``"random"`` initialization, or accepts a square array-like initial unmixing matrix whose dimensions match the number of fitted components. The L-BFGS iteration count, memory, stopping tolerance, Armijo line search, and random seed are estimator parameters. Scikit-learn utilities such as ``get_params``, ``set_params``, ``clone``, and pipelines therefore handle the complete configuration.

The empirical objective is nonconvex and piecewise smooth because component ranks change at ties. Different initializations may reach different local optima, and the gradient need not vanish exactly at an order-cell boundary.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   modules
