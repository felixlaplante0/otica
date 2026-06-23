OTICA
=====

OTICA performs linear independent component analysis by maximizing empirical
Wasserstein non-Gaussianity. The estimator follows the scikit-learn transformer API
and uses a Picard-style Riemannian L-BFGS solver with Armijo backtracking.

Model and objective
-------------------

For observations :math:`X \in \mathbb{R}^{n \times d}`, linear ICA assumes

.. math::

   X = S A^\mathsf{T},

where the columns of :math:`S` are mutually independent latent sources and :math:`A`
is an invertible mixing matrix. The sources are identifiable only up to permutation,
sign, and scale.

OTICA centers and symmetrically whitens the observations. For each standardized
candidate source, it computes the empirical squared 2-Wasserstein distance to a
standard Gaussian by matching the ordered observations with the optimal equally
weighted Gaussian quantiles

.. math::

   q_i = n \int_{(i - 1)/n}^{i/n} \Phi^{-1}(u) \, du.

For whitened observations :math:`Z`, the fitted orthogonal rotation maximizes

.. math::

   \sum_{k = 1}^{d} \frac{1}{n} \sum_{i = 1}^{n} \left( Y_{k(i)} - q_i \right)^2,
   \qquad Y = W Z^\mathsf{T},
   \qquad W W^\mathsf{T} = I_d.

Here, :math:`Y_{k(i)}` is the :math:`i`-th order statistic of component :math:`k`.
The solver computes relative gradients on the orthogonal group, applies L-BFGS with
identity vector transport, and retracts updates through the matrix exponential.

Quickstart
----------

.. code-block:: python

   import numpy as np

   from otica import OTICA

   rng = np.random.default_rng(42)
   sources = np.column_stack(
       [
           rng.laplace(size=5_000),
           rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=5_000),
           rng.standard_t(df=5, size=5_000),
       ]
   )
   mixing = np.array(
       [
           [1.0, 0.4, -0.2],
           [0.2, 1.0, 0.3],
           [-0.3, 0.1, 1.0],
       ]
   )
   X = sources @ mixing.T

   model = OTICA(random_state=42)
   estimated_sources = model.fit_transform(X)
   transformed_sources = model.transform(X[:100])

Configuration
-------------

The ``init`` parameter selects ``"fastica"``, ``"identity"``, or ``"random"``
initialization. The L-BFGS iteration count, memory, stopping tolerances, Armijo line
search, covariance rank threshold, and FastICA warm-start settings are all estimator
parameters. Scikit-learn utilities such as ``get_params``, ``set_params``, ``clone``,
and pipelines therefore handle the complete configuration.

The empirical objective is nonconvex and piecewise smooth because component ranks
change at ties. Different initializations may reach different local optima, and the
gradient need not vanish exactly at an order-cell boundary.

API reference
-------------

.. autoclass:: otica.OTICA
   :members:
   :show-inheritance:
