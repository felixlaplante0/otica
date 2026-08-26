OTICA
=====

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/otica-logo.svg" alt="OTICA logo">
     <p class="eyebrow">INDEPENDENT COMPONENT ANALYSIS · OPTIMAL TRANSPORT</p>
     <h1>Recover hidden signals from mixed observations.</h1>
     <p class="hero-copy">OTICA finds independent non-Gaussian sources with an exact empirical Wasserstein objective and a scikit-learn-compatible API.</p>
     <div class="hero-actions">
       <a class="primary" href="quickstart.html">Get started</a>
       <a href="https://github.com/felixlaplante0/otica">View on GitHub</a>
     </div>
   </section>

.. raw:: html

   <div class="pypi-card">
     <div><span class="pypi-kicker">OPEN SOURCE PYTHON PACKAGE</span><strong>Install OTICA in seconds</strong><p>Works with NumPy, scikit-learn, and standard ICA workflows.</p></div>
     <a href="quickstart.html">Read the quick start</a>
   </div>

Highlights
----------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Contrast-free objective
      :class-card: feature-card

      Measure non-Gaussianity with the squared one-dimensional Wasserstein distance to a standard Gaussian.

   .. grid-item-card:: Exact empirical scoring
      :class-card: feature-card

      Sort samples and match them with Gaussian rank quantiles without density estimation or a chosen contrast function.

   .. grid-item-card:: Riemannian optimization
      :class-card: feature-card

      Optimize the whitened ICA objective on the orthogonal group with limited-memory quasi-Newton updates.

   .. grid-item-card:: Familiar API
      :class-card: feature-card

      Use ``fit``, ``transform``, ``fit_transform``, and ``inverse_transform`` with scikit-learn utilities.

Why OTICA?
----------

Linear ICA seeks a representation :math:`X = SA^\top` whose latent components are mutually independent. OTICA centers and whitens the observations, then searches for an orthogonal unmixing matrix whose projected components are maximally non-Gaussian under a fixed Wasserstein criterion.

The method avoids choosing a problem-specific contrast function. Its empirical objective is computed directly from sorted samples and standard-Gaussian rank statistics, making the score explicit and reproducible.

Learn
-----

.. grid:: 1 1 1 3
   :gutter: 3

   .. grid-item-card:: Quick start
      :link: quickstart
      :class-card: feature-card

      Install OTICA, fit a model, and understand the objective.

   .. grid-item-card:: Tutorial notebook
      :link: tutorial
      :class-card: feature-card

      Follow a complete synthetic source-separation example with executed outputs and plots.

.. raw:: html

   <p><a class="tutorial-link" href="https://github.com/felixlaplante0/otica/blob/main/examples/tutorial.ipynb">Open the pre-executed tutorial notebook source on GitHub</a></p>

API reference
-------------

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   tutorial
   modules
