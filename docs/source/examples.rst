Examples
========

This page collects short examples that show how CLASSIX is used in practice and
why its sorting-based design is useful. The examples are intentionally close to
the scikit-learn workflow: construct an estimator, fit it, inspect ``labels_``,
and then use CLASSIX-specific explanation tools when needed.

Example 1: Fast clustering with a scikit-learn-like API
-------------------------------------------------------

For standard numerical data, the default Euclidean configuration is often a
good first run. CLASSIX exposes familiar methods while also keeping information
about the aggregation groups that produced the clusters.

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_blobs
   from classix import CLASSIX

   X, y_true = make_blobs(
       n_samples=5000,
       centers=8,
       n_features=6,
       cluster_std=0.8,
       random_state=42,
   )

   clx = CLASSIX(radius=0.45, minPts=10, verbose=0)
   labels = clx.fit_transform(X)

   print(np.unique(labels).size)
   print(clx.clusterSizes_)
   print(clx.nrDistComp_)

Why CLASSIX helps here:

* Like k-means, CLASSIX is easy to run and returns labels directly.
* Unlike k-means, it does not require specifying the number of clusters in
  advance.
* Unlike many pairwise-distance workflows, aggregation reduces the number of
  comparisons that need to be carried into the merge step.

Example 2: Non-convex structure with density merging
----------------------------------------------------

Distance-based merging is the fastest default. For two-dimensional Euclidean
data with narrow bridges, curved structure, or non-convex shapes, density-based
merging can be a useful alternative.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn.datasets import make_blobs, make_moons
   from classix import CLASSIX

   moons, _ = make_moons(n_samples=1000, noise=0.05, random_state=1)
   blobs, _ = make_blobs(
       n_samples=1200,
       centers=[(-0.85, 2.75), (1.75, 2.25)],
       cluster_std=0.45,
       random_state=1,
   )
   X = np.vstack([moons, blobs])

   clx = CLASSIX(
       sorting="pca",
       group_merging="density",
       radius=0.10,
       minPts=10,
       verbose=0,
   ).fit(X)

   plt.scatter(X[:, 0], X[:, 1], c=clx.labels_, s=12, cmap="tab10")
   plt.axis("equal")
   plt.show()

Why CLASSIX helps here:

* DBSCAN can also find non-convex clusters, but it may require many neighbor
  queries on large data.
* CLASSIX first compresses the data into aggregation groups, then merges those
  representatives.
* The resulting groups remain available for explanation and visualization.

Example 3: Explain why two points are in the same cluster
---------------------------------------------------------

Many clustering algorithms return only a label vector. CLASSIX keeps the group
graph, so it can print a local explanation for one sample or a path connecting
two samples.

.. code-block:: python

   from sklearn.datasets import make_blobs
   from classix import CLASSIX

   X, _ = make_blobs(
       n_samples=1000,
       centers=5,
       n_features=2,
       random_state=7,
   )

   clx = CLASSIX(radius=0.35, minPts=8, verbose=0).fit(X)

   # Explain the full clustering.
   clx.explain(X)

   # Explain a single sample.
   clx.explain(X, index1=25)

   # Explain the relationship between two samples and draw the connected groups.
   clx.explain(X, index1=25, index2=260, plot=True)

This is useful during exploratory analysis because it answers questions such as
"which starting point captured this sample?" and "which groups connect these two
samples?" without requiring users to inspect implementation internals.

Example 4: Manhattan distance for L1 geometry
---------------------------------------------

When L1 distance is a better description of similarity, use
``metric='manhattan'``. CLASSIX automatically uses sum-based sorting for this
metric.

.. code-block:: python

   from sklearn.preprocessing import MinMaxScaler
   from classix import CLASSIX, loadData

   X, y = loadData("Wine")
   X = MinMaxScaler().fit_transform(X)

   clx = CLASSIX(
       metric="manhattan",
       radius=0.25,
       minPts=5,
       verbose=0,
   ).fit(X)

   print(clx.labels_)
   print(clx.clusterSizes_)

Why CLASSIX helps here:

* k-means is tied to centroid-style Euclidean geometry.
* DBSCAN supports alternative metrics, but parameter tuning can become
  expensive when many neighbor searches are required.
* CLASSIX keeps the same two-stage aggregation-and-merge workflow while
  changing the distance geometry.

Example 5: Tanimoto distance for binary or count data
-----------------------------------------------------

Tanimoto distance is common for non-negative sparse, binary, or count vectors.
CLASSIX supports this metric directly.

.. code-block:: python

   import numpy as np
   from classix import CLASSIX

   rng = np.random.default_rng(0)
   X = rng.binomial(1, p=0.08, size=(600, 128)).astype(float)

   clx = CLASSIX(
       metric="tanimoto",
       radius=0.35,
       minPts=4,
       verbose=0,
   ).fit(X)

   print(np.unique(clx.labels_).size)

Why CLASSIX helps here:

* Tanimoto distance is more meaningful than Euclidean distance for many
  fingerprint-like vectors.
* CLASSIX uses row sums to prune candidate comparisons during aggregation.
* The same explanation API remains available after fitting.

Practical comparison with common alternatives
---------------------------------------------

No clustering algorithm is best for every dataset. CLASSIX is designed for the
case where speed, simple parameters, and interpretability matter together.

``k-means``
    Very fast and widely available, but users must choose the number of
    clusters and the method favors centroid-shaped clusters. CLASSIX does not
    require ``n_clusters`` and can merge local groups into more flexible shapes.

``DBSCAN``
    Good for density-based structure and explicit noise labels, but runtime and
    memory can become challenging on very large datasets. CLASSIX uses sorted
    aggregation to reduce the work done in the merge stage and can still filter
    small clusters with ``minPts``.

``HDBSCAN``
    Powerful for variable-density data, but has more complex behavior and an
    additional dependency. CLASSIX emphasizes a smaller set of interpretable
    parameters and exposes the intermediate groups used to form clusters.

``Agglomerative clustering``
    Gives hierarchical structure, but usually requires substantial pairwise
    distance work. CLASSIX targets a flatter, faster workflow while retaining
    enough group-level structure to explain assignments.

For reliable model selection, compare algorithms on the data distribution,
distance metric, and downstream interpretation needs that matter for your
application.
