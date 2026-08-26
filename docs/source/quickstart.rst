Quick start
===========

Install OTICA with pip:

.. code-block:: bash

   python -m pip install otica

Generate three independent non-Gaussian sources, mix them, and recover them:

.. code-block:: python

   import numpy as np
   from otica import OTICA

   rng = np.random.default_rng(42)
   sources = np.column_stack(
       [
           rng.laplace(size=2_000) / np.sqrt(2.0),
           rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=2_000),
           rng.standard_t(df=5, size=2_000) * np.sqrt(3.0 / 5.0),
       ]
   )
   mixing = np.array(
       [
           [1.0, 0.5, -0.2],
           [0.2, 1.0, 0.4],
           [-0.4, 0.1, 1.0],
       ]
   )
   X = sources @ mixing.T

   model = OTICA(random_state=42).fit(X)
   estimated_sources = model.transform(X)
   print(estimated_sources.shape)

The fitted estimator exposes ``components_`` and ``mixing_`` together with the
standard scikit-learn transformer methods ``fit``, ``transform``,
``fit_transform``, and ``inverse_transform``. ICA identifies components only up
to permutation and sign, so compare recovered sources after alignment.

The mathematics in one paragraph
---------------------------------

OTICA centers and whitens observations before optimizing an orthogonal unmixing
matrix. For whitened data :math:`Z` and candidate :math:`W`, it maximizes the
sum of squared one-dimensional Wasserstein distances between each component
and a standard Gaussian:

.. math::

   F(W) = \sum_{k=1}^{d} \mathcal{W}_2\left((ZW^\top)_k,\mathcal{N}(0,1)\right)^2,
   \qquad WW^\top = I_d.

The empirical distances are evaluated exactly from sorted samples and Gaussian
rank quantiles. Under the usual ICA assumptions, including at most one Gaussian
source, the population objective identifies the sources up to permutation, sign,
and scale.

Next steps
----------

Run the complete, pre-executed :doc:`tutorial` or inspect the
:doc:`modules` API reference. The method and its assumptions are described in
`Contrast-Free ICA and Causal Inference via Wasserstein Distances to the Gaussian
<https://arxiv.org/abs/2607.12832>`_.
