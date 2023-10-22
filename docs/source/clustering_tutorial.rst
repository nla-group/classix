Clustering Tutorial
======================================

The CLASSIX with density method implements density clustering in an explicit way while the one with distance method implements density clustering in an implicit way. We will illustrate how to use them separately.  

The examples here are demonstrated with a sample with a more complicated shape:

.. code:: python

    import matplotlib.pyplot as plt
    from sklearn import datasets
    import numpy as np
    random_state = 1
    moons, _ = datasets.make_moons(n_samples=1000, noise=0.05, random_state=random_state)
    blobs, _ = datasets.make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], cluster_std=0.5, random_state=random_state)
    X = np.vstack([blobs, moons])


.. admonition:: Note

    Setting radius lower, the more groups will form, and the groups tend to be separated instead of merging as clusters, and therefore runtime will increase. 
    

Density clustering
------------------------------
Instead of leaving the default option as the previous example, in this example, we can explicitly set a parameter ``group_merging`` to specify which merging strategy we would like to adopt. 
Also, we employ ``sorting='pca'`` or other choices such as 'norm-orthant' or 'norm-mean'. In most cases, we recommend PCA sorting. Other available parameters include ``radius`` and ``minPts``. The parameter of ``radius`` is a tolerance value dominating the aggregation phase, which immediately affects the merging phase. 
Usually, the thinner boundary between the clusters are, the lower ``radius`` is required. In addition, we would explain why we set `minPts` to 10 later. Also, we can output the log by setting verbose to 1, then it clearly shows how many clusters and the associated size we get:

.. code:: python

    from classix import CLASSIX
    clx = CLASSIX(sorting='pca', group_merging='density', radius=0.1, minPts=10)
    clx.fit(X)

The output of the code is:

.. parsed-literal::

    CLASSIX(sorting='pca', radius=0.1, minPts=10, group_merging='density')
    The 2500 data points were aggregated into 316 groups.
    In total 16395 comparisons were required (6.56 comparisons per data point). 
    The 316 groups were merged into 47 clusters with the following sizes: 
        * cluster 0 : 717
        * cluster 1 : 711
        * cluster 2 : 500
        * cluster 3 : 500
        * cluster 4 : 10
        * cluster 5 : 6
        * cluster 6 : 3
        ......
        * cluster 45 : 1
        * cluster 46 : 1
    As MinPts is 10, the number of clusters has been further reduced to 5.
    Try the .explain() method to explain the clustering.




The reason why we set ``minPts`` to 10 is that we want the clusters with a size smaller than 10 to agglomerate to other big clusters which are partitioned significantly.

``minPts`` is a parameter which we denote it as outliers threshold, and we will illustrate it in the section of ``Outlier Detection``.

The visualization of clustering results is reasonable:

.. code:: python

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clx.labels_)
    plt.show()

.. image:: images/demo2.png
    :width: 360



Distance clustering
------------------------------
The distance-based CLASSIX has the same steps as density-based CLASSIX except that the density comparison steps, in such a way distance-based CLASSIX does not require calculating the density, hence intuitively would be faster. By contrast, it just compares the pair of the clusters one at a time to determine if they should merge. 
Also, we propose a distance-based clustering exempted from calculating the density but with one more parameter for appropriate smoothing ``scale``. By tuning the ``scale``, we only calculate the distance between pairs of starting points and define the distance as the weights in the graph, and the distance that is smaller than $\texttt{scale}*\radius$ is assigned to 1 otherwise 0. The next step, similarly, is to find the connected components in the graph as clusters.

Similar to the previous example, we refer ``group_merge`` to 'distance', then adopt distance-based CLASSIX, the code is as below:


.. code:: python

    clx= CLASSIX(sorting='pca', group_merging='distance', radius=0.1, minPts=4)
    clx.fit(X)



.. parsed-literal::

    CLASSIX(sorting='pca', radius=0.1, method='distance')
    The 2500 data points were aggregated into 316 groups.
    In total 16395 comparisons were required (6.56 comparisons per data point). 
    The 316 groups were merged into 28 clusters with the following sizes: 
        * cluster 0 : 733
        * cluster 1 : 730
        * cluster 2 : 501
        * cluster 3 : 500
        * cluster 4 : 4
        * cluster 5 : 4
        * cluster 6 : 3
        * cluster 7 : 2
        * cluster 8 : 2
        * cluster 9 : 2
        * cluster 10 : 2
        * cluster 11 : 1
        * cluster 12 : 1
        * cluster 13 : 1
        * cluster 14 : 1
        * cluster 15 : 1
        * cluster 16 : 1
        * cluster 17 : 1
        * cluster 18 : 1
        * cluster 19 : 1
        * cluster 20 : 1
        * cluster 21 : 1
        * cluster 22 : 1
        * cluster 23 : 1
        * cluster 24 : 1
        * cluster 25 : 1
        * cluster 26 : 1
        * cluster 27 : 1
    As MinPts is 4, the number of clusters has been further reduced to 4.
    Try the .explain() method to explain the clustering.


Visualize the result:

.. code:: python

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clx.labels_)
    plt.show()

.. image:: images/demo3.png
    :width: 360

.. admonition:: Note

    The density-based merging criterion usually results in slightly better clusters than the distance-based criterion, but the latter has a significant speed advantage.


Visualize connecting edge
------------------------------
Now we use the same example to demonstrate how cluster are formed by computing starting points and edge connections. We can output the information by

.. code:: python

    clx.visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, fmt='png')

.. image:: images/linkage_scale_1.5_tol_0.1.png


.. admonition:: Note

    The starting points can be interpreted as a reduced-density estimator of the data. 

There is one more parameter that affects distance-based CLASSIX, that is ``scale``.  By simply adding the parameter ``plot_boundary`` and setting it to ``True``, then we can obtain the starting points with their group boundary. The visualization of the connecting edge between starting points with varying ``scale`` is plotted as below:

.. code:: python

    for scale in np.arange(1.1, 2, 0.1):
        clx = CLASSIX(sorting='pca', radius=0.1, group_merging='distance', verbose=0)
        clx.fit_transform(X)
        clx.visualize_linkage(scale=round(scale,1), figsize=(8,8), labelsize=24, plot_boundary=True, fmt='png')

.. image:: images/single_linkage.png


Considering a graph constructed by the starting points, as ``scale`` increases, the number of edges increases, therefore, the connected components area enlarges while the number of connected components decreases.
Though in most cases, the scale setting is not necessary, when the small ``radius`` needed, adopting distance-based CLASSIX with an appropriate ``scale`` can greatly speed up the clustering application, such as image segmentation.









Explainable Clustering
------------------------------

CLASSIX provides an appealing explanation for clustering results, either in global view or by specific indexing. 

If we would like to make plot accompany just remember to set ``plot`` to ``True``.

We now have a global view of it:

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

