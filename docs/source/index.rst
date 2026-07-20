CLASSIX documentation
=====================

.. raw:: html

   <section class="clx-hero">
     <div class="clx-hero-copy">
       <p class="clx-eyebrow">Fast, scalable, explainable clustering</p>
       <h1>CLASSIX turns sorted data into interpretable clusters.</h1>
       <p class="clx-lede">
         CLASSIX combines a familiar scikit-learn-like API with a two-stage
         sorting, aggregation, and merging workflow. It is designed for large
         exploratory clustering tasks where speed and explanation both matter.
       </p>
       <div class="clx-actions">
         <a class="clx-button clx-button-primary" href="quickstart.html">Quick start</a>
         <a class="clx-button" href="examples.html">Examples</a>
         <a class="clx-button" href="api_reference.html">API reference</a>
       </div>
     </div>
     <div class="clx-hero-panel">
       <div class="clx-pipeline">
         <div><span>1</span><strong>Sort</strong><small>PCA, norm, sum, or metric-aware order</small></div>
         <div><span>2</span><strong>Aggregate</strong><small>Group nearby samples around starting points</small></div>
         <div><span>3</span><strong>Merge</strong><small>Connect groups by distance or density</small></div>
         <div><span>4</span><strong>Explain</strong><small>Inspect groups, paths, and assignments</small></div>
       </div>
     </div>
   </section>

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

.. raw:: html

   <div class="clx-card-grid">
     <article class="clx-card">
       <h3>Small parameter surface</h3>
       <p>Start with radius and minPts instead of selecting the number of clusters in advance.</p>
     </article>
     <article class="clx-card">
       <h3>Fast sorted aggregation</h3>
       <p>Reduce the clustering problem to representative groups before the final merge step.</p>
     </article>
     <article class="clx-card">
       <h3>Multiple metrics</h3>
       <p>Use Euclidean, Manhattan, or Tanimoto distance without leaving the CLASSIX estimator workflow.</p>
     </article>
     <article class="clx-card">
       <h3>Readable explanations</h3>
       <p>Trace samples through groups, starting points, connected paths, and final cluster labels.</p>
     </article>
   </div>

Compared with common clustering algorithms, CLASSIX is designed for the
intersection of speed, simple tuning, and interpretability. It does not require
``n_clusters`` like k-means, it keeps an explicit explanation structure that
many density methods do not expose, and it can use sorted aggregation to reduce
the amount of work carried into the merge stage.

Introductory video
------------------

The short video below gives a visual introduction to the idea behind CLASSIX:
sorting data, aggregating points around starting points, and merging those
groups into explainable clusters. It is a useful companion to the examples in
this documentation because it shows why CLASSIX keeps intermediate groups
available for interpretation instead of returning only final labels.

.. raw:: html

   <div class="clx-video-shell">
     <iframe width="720" height="420" src="https://www.youtube.com/embed/K94zgRjFEYo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

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

.. raw:: html

   <div class="clx-link-strip">
     <a href="features.html"><strong>Concepts</strong><span>Metrics, sorting, merging, and explanations</span></a>
     <a href="examples.html"><strong>Examples</strong><span>Practical workflows and algorithm comparisons</span></a>
     <a href="comparison.html"><strong>Benchmarks</strong><span>Runtime and quality comparisons on large datasets</span></a>
   </div>

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   features
   examples
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
