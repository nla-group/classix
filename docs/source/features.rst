Features and concepts
=====================

Algorithm overview
------------------

CLASSIX compresses the clustering problem before it performs the final merge.
Instead of comparing every sample with every other sample, it sorts the data and
creates aggregation groups around starting points. The merge step then works on
those group representatives rather than on all samples directly.

This produces three useful levels of structure:

``data points``
    The original samples supplied by the user.

``groups``
    Local neighborhoods formed during aggregation. Each group has a starting
    point that represents it.

``clusters``
    Final connected components formed by merging groups.

Because the intermediate groups are retained, CLASSIX can explain a result in
terms of the starting point that captured a sample and the graph of groups that
connects samples inside a cluster.

Distance metrics
----------------

``metric='euclidean'``
    The default metric for continuous numerical data. Euclidean clustering
    supports PCA sorting, norm sorting, distance merging, and density merging.

``metric='manhattan'``
    Uses L1 distance. The implementation shifts data to the non-negative
    orthant and uses sum-based sorting to prune candidate comparisons.

``metric='tanimoto'``
    Uses Tanimoto distance, ``1 - similarity``. This is useful for non-negative
    binary, count, or fingerprint-like vectors.

Sorting options
---------------

Sorting is most relevant for Euclidean clustering:

``sorting='pca'``
    Sort samples by their first principal component. This is the default and is
    a strong general-purpose choice.

``sorting='norm-mean'``
    Mean-center the data and sort by Euclidean norm.

``sorting='norm-orthant'``
    Shift the data to the non-negative orthant and sort by Euclidean norm.

``sorting=None``
    Disable sorting and aggregate the raw order.

For ``metric='manhattan'``, CLASSIX uses sum-based sorting automatically. For
``metric='tanimoto'``, row sums are used internally for efficient candidate
pruning.

Merging options
---------------

``group_merging='distance'``
    Merge groups when their starting points are within
    ``mergeScale * radius``. This is the default and the fastest general-purpose
    strategy.

``group_merging='density'``
    Merge Euclidean groups using an intersection-density criterion. This can be
    useful when local density structure is important.

``group_merging=None``
    Skip the merge step and return aggregation groups as clusters.

Outlier and small-cluster handling
----------------------------------

The ``minPts`` parameter defines the minimum valid cluster size. Clusters with
fewer samples are treated as small clusters and are reassigned to nearby valid
clusters when possible. If ``post_alloc=False`` is used with a supported merge
path, filtered samples can be labeled ``-1`` instead.

Hyperparameter tuning
---------------------

A practical tuning strategy is to start with ``radius=1`` and then reduce
``radius`` until the number of clusters is only slightly larger than expected.
After that, increase ``minPts`` until small clusters are dissolved and the
desired number of clusters is obtained.

The :meth:`classix.CLASSIX.minPtsChange` method makes the second step fast. It
updates ``minPts`` after fitting and recomputes only the small-cluster filtering
phase when possible:

.. code-block:: python

   from classix import CLASSIX

   clx = CLASSIX(radius=0.7, minPts=1, verbose=0).fit(X)

   for minPts in [3, 5, 10, 20]:
       clx.minPtsChange(minPts)
       print(minPts, len(set(clx.labels_)), clx.clusterSizes_)

When ``mergeTinyGroups=True`` (the default), distance-based merging does not
depend on ``minPts``, so ``minPtsChange`` reuses the existing aggregation and
merge structure. When ``mergeTinyGroups=False``, the distance merge graph
depends on ``minPts`` because tiny groups are excluded from merge edges; in that
case ``minPtsChange`` recomputes the group-merging step but still reuses the
existing preprocessing and aggregation results.

Explanation tools
-----------------

The ``CLASSIX.explain`` method can be used in three modes:

* ``clx.explain(X)`` prints a global summary of the fitted clustering.
* ``clx.explain(X, index1=i)`` explains the group and cluster containing one
  sample.
* ``clx.explain(X, index1=i, index2=j)`` explains whether two samples are in the
  same cluster and, when available, prints a path through connected groups.

For two-dimensional data, or for higher-dimensional data projected to two
principal components, ``plot=True`` visualizes the explanation.

Cython acceleration
-------------------

CLASSIX can use compiled Cython extensions for the core aggregation and merging
steps. The public Python API is the same whether Cython is enabled or not. Use
``classix.cython_is_available(verbose=True)`` to check which implementation is
active, and set ``classix.__enable_cython__ = False`` before constructing an
estimator if you need to force the pure-Python path for debugging or comparison.
