Explainable Clustering
======================================

CLASSIX provides an appealing explanation for clustering results, either in global view or by specific indexing. 

If we would like to make plot accompany just remember to set ``plot`` to ``True``.

Global insight
------------------------------

.. code:: python

    from sklearn import datasets
    import numpy as np
    from classix import CLASSIX

    X, y = datasets.make_blobs(n_samples=5000, centers=2, n_features=2, cluster_std=1, random_state=1)

    clx = CLASSIX(sorting='pca', group_merging='density', radius=0.5, verbose=1, minPts=4)
    clx.fit(X)

    clx.explain(plot=True, savefig=True, figsize=(10,10))



The output is:

.. parsed-literal::

    A clustering of 5000 data points with 2 features has been performed. 
    The radius parameter was set to 0.50 and MinPts was set to 4. 
    As the provided data has been scaled by a factor of 1/6.01,
    data points within a radius of R=0.50*6.01=3.01 were aggregated into groups. 
    In total 7903 comparisons were required (1.58 comparisons per data point). 
    This resulted in 14 groups, each uniquely associated with a starting point. 
    These 14 groups were subsequently merged into 2 clusters. 
    A list of all starting points is shown below.
    ----------------------------------------
    Group  NrPts  Cluster  Coordinates 
    0     398      0     -1.19 -1.09 
    1    1073      0     -0.65 -1.15 
    2     553      0     -1.17 -0.56 
    3     466      0     -0.67 -0.65 
    4       6      0     -0.19 -0.88 
    5       3      0     -0.72 -0.03 
    6       1      0     -0.22 -0.28 
    7     470      1       0.31 0.21 
    8     675      1       0.18 0.71 
    9     579      1       0.86 0.19 
    10     763      1       0.69 0.67 
    11       6      1       0.42 1.35 
    12       5      1       1.24 0.59 
    13       2      1        1.0 1.08 
    ----------------------------------------
    In order to explain the clustering of individual data points, 
    use .explain(ind1) or .explain(ind1, ind2) with indices of the data points.
.. image:: images/explain_viz.png


Track single data
------------------------------

Following the previous steps, we can analyze the specific data by refering to the index, for example here, we want to track the data with index 0:

.. code:: python

    clx.explain(0,  plot=True, savefig=True, fmt='PNG')


Output:

.. parsed-literal::
    The data point is in group 2, which has been merged into cluster #0.
.. image:: images/None0.png

Comparison insight
------------------------------
We give two examples to compare the data pair cluster assignment as follows.

.. code:: python
    
    clx.explain(0, 2000,  plot=True, savefig=True, fmt='png')

.. parsed-literal::
    The data point 0 is in group 2, which has been merged into cluster 0.
    The data point 2000 is in group 10, which has been merged into cluster 1.
    There is no path of overlapping groups between these clusters.
.. image:: images/None0_2000.png


.. code:: python
    
    clx.explain(0, 2008,  plot=True, savefig=True, fmt='png')

.. parsed-literal::
    The data point 0 is in group 2 and the data point 2008 is in group 4, 
    both of which were merged into cluster #0. 
    These two groups are connected via groups 2 <-> 1 <-> 4.
.. image:: images/None0_2008.png