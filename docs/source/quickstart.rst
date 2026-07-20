Quick start
===========

Installation
------------

CLASSIX is distributed as ``classixclustering``.

.. code-block:: bash

   pip install classixclustering

Conda users can install from conda-forge:

.. code-block:: bash

   conda install -c conda-forge classixclustering

The core runtime depends on NumPy, SciPy, pandas, matplotlib, scikit-learn, and
requests. Source builds also use Cython and a C++ compiler for the optional
accelerated extensions.

You can check whether the accelerated Cython modules are available with:

.. code-block:: python

   import classix

   classix.cython_is_available(verbose=True)

Basic clustering
----------------

CLASSIX follows the estimator conventions used by scikit-learn. Create an
estimator, call ``fit`` or ``fit_transform``, and read the fitted labels from
``labels_``.

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from classix import CLASSIX

   X, y_true = make_blobs(
       n_samples=1000,
       centers=3,
       n_features=2,
       random_state=1,
   )

   clx = CLASSIX(radius=0.5, minPts=10, verbose=0)
   labels = clx.fit_transform(X)

   plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap="tab10")
   plt.axis("equal")
   plt.show()

Useful fitted attributes include:

``labels_``
    Final cluster label for each input sample.

``groups_``
    Aggregation group label for each sorted sample.

``splist_``
    Internal starting-point information for the aggregation groups.

``groupCenters_``
    Original input indices of the aggregation group starting points.

``clusterSizes_``
    Number of samples in each final cluster.

Choosing the metric
-------------------

The default ``metric='euclidean'`` works well for standard numerical feature
data. CLASSIX also supports Manhattan and Tanimoto distances.

.. code-block:: python

   clx_l1 = CLASSIX(metric="manhattan", radius=0.4, minPts=5, verbose=0)
   labels_l1 = clx_l1.fit_transform(X)

For non-negative binary or count data, use Tanimoto distance:

.. code-block:: python

   clx_tanimoto = CLASSIX(metric="tanimoto", radius=0.25, minPts=5, verbose=0)
   labels_tanimoto = clx_tanimoto.fit_transform(binary_or_count_matrix)

Explaining results
------------------

CLASSIX can explain the global clustering, a single sample, or the relationship
between two samples. Explanations are printed by default and can also be plotted.

.. code-block:: python

   clx = CLASSIX(radius=0.5, minPts=10, verbose=0).fit(X)

   # Global summary
   clx.explain(X)

   # Explain one sample
   clx.explain(X, index1=42)

   # Explain whether two samples are connected through aggregation groups
   clx.explain(X, index1=42, index2=100, plot=True)

Cluster centers and built-in data
---------------------------------

The helper functions exported from ``classix`` cover common exploratory tasks:

.. code-block:: python

   from classix import loadData

   X_vdu = loadData("vdu_signals")

   clx = CLASSIX(radius=1.0, verbose=0).fit(X_vdu)
   cluster_centers = clx.load_cluster_centers(X_vdu)
   group_centers = clx.load_group_centers(X_vdu)

The available named datasets include ``'vdu_signals'``, ``'Iris'``,
``'Dermatology'``, ``'Ecoli'``, ``'Glass'``, ``'Banknote'``, ``'Seeds'``,
``'Phoneme'``, ``'Wine'``, ``'CovidENV'``, and ``'Covid3MC'``.
