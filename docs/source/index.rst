OTICA
=====

**otica** is a Python package for linear independent component analysis (ICA) based
on optimal transport. It recovers latent sources by maximizing their empirical
squared 2-Wasserstein distances to the standard Gaussian, a fixed non-Gaussianity
criterion that needs no user-chosen contrast function.

.. code-block:: bash

   pip install otica

See :doc:`quickstart` for a first example, or the :doc:`tutorial`. The package is
available on `PyPI <https://pypi.org/project/otica/>`_, and the method is described in
the `paper <https://arxiv.org/abs/2607.12832>`_.

Why OTICA?
----------

Linear ICA seeks a representation :math:`X = SA^\top` whose latent components are mutually independent. OTICA centers and whitens the observations, then searches for an orthogonal unmixing matrix whose projected components are maximally non-Gaussian under a fixed Wasserstein criterion.

The method avoids choosing a problem-specific contrast function. Its empirical objective is computed directly from sorted samples and standard-Gaussian rank statistics, making the score explicit and reproducible. The whitened objective is optimized on the orthogonal group with limited-memory quasi-Newton updates, behind the scikit-learn ``fit``, ``transform``, ``fit_transform`` and ``inverse_transform`` API.

Quick example
-------------

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

The :doc:`quickstart` explains the objective, and the :doc:`tutorial` walks through a
full source-separation example.

Learn
-----

.. grid:: 1 1 1 3
   :gutter: 3

   .. grid-item-card:: Quick start
      :link: quickstart.html

      Install OTICA, fit a model, and understand the objective.

   .. grid-item-card:: Tutorial notebook
      :link: tutorial.html

      Follow a complete synthetic source-separation example with executed outputs and plots.

Citation
--------

If you use OTICA, please cite:

.. code-block:: bibtex

   @article{laplante2026otica,
     title   = {Contrast-Free ICA and Causal Inference via Wasserstein Distances
                to the Gaussian},
     author  = {Laplante, F{\'e}lix and Ambroise, Christophe and Humbert, Pierre},
     journal = {arXiv preprint arXiv:2607.12832},
     year    = {2026},
     doi     = {10.48550/arXiv.2607.12832}
   }

.. toctree::
   :hidden:

   quickstart
   tutorial
   modules
