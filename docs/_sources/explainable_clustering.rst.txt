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




Case study of industry data
------------------------------
Here, we turn our attention on practical data. 
Similar to above, we load the necessary data to produce the analytical result.

.. code:: python

    import time
    import numpy as np
    import classix


To load the industry data provided by Kamil, we can simply use the API ``load_data`` and require the paramter as ``vdu_signals``
we leave the default parameters except setting radius to 1.

.. code:: python

    data = classix.loadData('vdu_signals')
    clx = classix.CLASSIX(radius=1, group_merging='distance')

.. admonition:: Note

    The method ``loadData`` also supports other typical UCI datasets for clustering, which include ``'vdu_signals'``, ``'Iris'``, ``'Dermatology'``, ``'Ecoli'``, ``'Glass'``, ``'Banknote'``, ``'Seeds'``, ``'Phoneme'``, and ``'Wine'``.


Then, we employ classix model to train the data and record the timing:

.. code:: python

    st = time.time()
    clx.fit_transform(data)
    et = time.time()
    print("consume time:", et - st)

.. parsed-literal::

    CLASSIX(sorting='pca', radius=1, minPts=0, group_merging='distance')
    The 2028780 data points were aggregated into 36 groups.
    In total 3920623 comparisons were required (1.93 comparisons per data point). 
    The 36 groups were merged into 4 clusters with the following sizes: 
        * cluster 0 : 2008943
        * cluster 1 : 16920
        * cluster 2 : 1800
        * cluster 3 : 1117
    Try the .explain() method to explain the clustering.
    consume time: 1.1904590129852295

If you set radius to 0.5, you can get the output:
.. parsed-literal::

    CLASSIX(sorting='pca', radius=0.5, minPts=0, group_merging='distance')
    The 2028780 data points were aggregated into 93 groups.
    In total 6252385 comparisons were required (3.08 comparisons per data point). 
    The 93 groups were merged into 7 clusters with the following sizes: 
        * cluster 0 : 2008943
        * cluster 1 : 16909
        * cluster 2 : 1800
        * cluster 3 : 900
        * cluster 4 : 180
        * cluster 5 : 37
        * cluster 6 : 11
    Try the .explain() method to explain the clustering.
    consume time: 1.3505780696868896

From this, we can see there is big gap between the number of cluster 4 and cluster 5, by which we can assume the data within a cluster with size smaller than 38 are outliers. Therefore, we set 
``minPts`` to 38. After that, we can get the same result as that with radius of 1. You can also set the parameter of ``post_alloc`` to ``False``, then all outliers will be marked as label of -1 instead of 
executing the allocation strategy. Though in most cases outliers are hard to define and capture, this case tells us how to select an appropriate value for `minPts` to separate outliers or deal with outliers based on distance. 

As above, we view the whole picture for data simply by 

.. code:: python

    clx.explain(plot=True)

You can also specify other parameters to personalize the visualization to make it easier to analyze. For example, you can enlarge the fontsize of starting points labels by 
setting ``sp_fontsize`` larger or change the shape by tunning appropriate value for ``figsize``. For more details about parameter settings, we refer to our API Reference. So, we try:

.. code:: python

    clx.explain(plot=True, figsize=(24,10), sp_fontsize=12)

.. image:: images/kamil_explain_viz.png

.. parsed-literal::

    A clustering of 2028780 data points with 2 features has been performed. 
    The radius parameter was set to 1.00 and MinPts was set to 0. 
    As the provided data has been scaled by a factor of 1/2.46,
    data points within a radius of R=1.00*2.46=2.46 were aggregated into groups. 
    In total 3920623 comparisons were required (1.93 comparisons per data point). 
    This resulted in 36 groups, each uniquely associated with a starting point. 
    These 36 groups were subsequently merged into 4 clusters. 
    A list of all starting points is shown below.
    ----------------------------------------
    Group   NrPts  Cluster  Coordinates 
    0     10560     1      16.35 3.26 
    1      1800     2      15.81 1.85 
    2      2580     1      15.38 3.47 
    3       656     1      14.83 4.33 
    4       177     1      13.87 4.59 
    5      1058     1       12.9 4.23 
    6       392     1        12.0 4.8 
    7       664     1      11.98 2.94 
    8       806     1       11.6 3.88 
    9        18     1      10.89 3.15 
    10         9     1      10.66 2.05 
    11       128     3        9.0 1.93 
    12        45     3       8.04 1.51 
    13        23     3       7.82 2.55 
    14       183     3       6.97 0.56 
    15       146     3       6.93 2.06 
    16       138     3       6.23 1.33 
    17        47     3       6.16 2.79 
    18        40     3      5.81 -0.33 
    19       317     3        5.4 0.69 
    20        50     3       5.31 2.03 
    21       576     0      3.06 -0.02 
    22     12001     0      2.25 -0.61 
    23         2     0        2.0 0.94 
    24     76469     0      1.87 -1.56 
    25     47743     0      1.38 -0.07 
    26    500225     0      1.04 -1.01 
    27    145955     0        0.7 0.69 
    28     16456     0       0.6 -1.91 
    29    506281     0      0.38 -0.25 
    30    455788     0      -0.04 1.37 
    31     13196     0     -0.05 -1.16 
    32    110364     0      -0.36 0.42 
    33    123548     0      -0.89 1.92 
    34       274     0       -1.2 0.96 
    35        65     0       -1.87 1.7 
    ----------------------------------------
    In order to explain the clustering of individual data points, 
    use .explain(ind1) or .explain(ind1, ind2) with indices of the data points.

We can see most of data objects are allocated to groups 26~33, which correspond to cluster 0. 


Then to track or compare any data by indexing, you can enter like

.. code:: python

    clx.explain(14940, 16943,  plot=True, savefig=True, sp_fontsize=10)

.. image:: images/kamil_14940_16943.png

.. parsed-literal::

    The data point 14940 is in group 7, which has been merged into cluster 1.
    The data point 16943 is in group 11, which has been merged into cluster 3.
    There is no path of overlapping groups between these clusters.

The output documentation describes how two data objects are separated into two clusters, and also how far or close they are.
