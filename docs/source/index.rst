CLASSIX documentation
=====================

CLASSIX is a fast, scalable, and explainable clustering package with a
scikit-learn-like estimator interface. The algorithm is built around a simple
two-stage idea:

1. **Aggregation** sorts the data and greedily groups nearby samples around
   representative starting points.
2. **Merging** connects those aggregation groups into final clusters by either
   a distance criterion or, for Euclidean data, a density-overlap criterion.

This design gives CLASSIX its main strengths: it is easy to tune, uses only a
small number of interpretable parameters, can process large datasets with low
memory overhead, and can explain cluster assignments through groups, starting
points, and connecting paths.

Highlights
----------

* A familiar estimator API with ``fit``, ``fit_transform``, ``predict`` and
  fitted attributes such as ``labels_``.
* Euclidean, Manhattan, and Tanimoto distance options.
* Sorting strategies for Euclidean aggregation, including PCA-based sorting and
  norm-based sorting.
* Distance-based merging for speed and density-based merging for Euclidean
  cluster-shape sensitivity.
* Explanations for individual samples, sample pairs, and global clustering
  structure.
* Built-in datasets and examples for quick experimentation.
* Optional Cython extensions for accelerated aggregation and merging.

When to use CLASSIX
-------------------

CLASSIX is especially useful when you need a clustering method that is fast
enough for large exploratory workflows but still transparent enough to explain.
It is a good fit for numerical feature data, low-dimensional segmentation tasks,
large scatter-like datasets, and non-negative binary or count vectors where
Tanimoto distance is meaningful.

The two most important parameters are:

``radius``
    Controls how far a sample may be from a starting point during aggregation.
    Smaller values create more groups and finer clusters; larger values create
    fewer, coarser groups.

``minPts``
    Sets the minimum valid cluster size. Clusters smaller than this threshold
    are dissolved and their groups are reassigned when possible.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   features
   clustering_tutorial
   matlab
   comparison
   outlier_detection

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api_reference

Project
-------

.. toctree::
   :maxdepth: 2

   acknowledgement
   license
   contact

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. image:: images/nla_group.png
   :width: 360
