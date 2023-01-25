Performance Comparison
======================================

In this tutorial we compare CLASSIX with some other widely used density clustering algorithms like DBSCAN,HDBSCAN and Quickshift++. We perform this experiment on the VDU (Vacuum Distillation Unit) dataset that comes with the CLASSIX installation and for synthetic Gaussian blobs.

Example 1: VDU Data
##################

We first import the required modules and load the data:

.. code:: python

    import time
    import math
    import hdbscan
    import warnings
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn import metrics
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    from classix import CLASSIX, loadData
    from quickshift.QuickshiftPP import * # download from https://github.com/google/quickshift
    
    data = loadData('vdu_signals') # load the data
    print(data.shape) # (2028780, 2)

The data set has more than 2 million data points. Despite being only two-dimensional, this is challenging for many clustering algorithms as we will see. The timings reported below were obtained in this computational environment:

    * Dell PowerEdge R740 Server
    * 2x Intel Xeon Silver 4114 2.2G (total 20 cores, 40 threads)
    * 2x NVIDIA Tesla P100 16GB GPU
    * 1.5 TB RAM (=1536 GB RAM)
    * 1.8 TB disk space (expandable)

The DBSCAN, HDBSCAN, Quickshift++ fail in this experiment (runtime > 24 hr) while CLASSIX use around 1.2 seconds for clustering of whole data.
Therefore, to compare the the four algorithms (ensure they can finish clustering within a day), we need to preprocess the data for downsampling:

.. code:: python
    
    # This block of code is provided by Kamil Oster
    final_len = 0.05 * data.shape[0] % use 5% selected data
    outliers_position = np.where(data[:,0] > 7.5)[0]
    no_outliers_position = np.delete(np.arange(0, len(data[:,0])), outliers_position, axis=0)

    outlier_len = len(outliers_position)
    data_no_outliers_length = int(final_len - outlier_len)

    data_outliers = data[outliers_position, :]
    data_no_outliers = np.delete(data, outliers_position, axis=0)

    random_integers = np.arange(0, len(no_outliers_position))
    np.random.shuffle(random_integers)

    data_no_outliers_out = data_no_outliers[random_integers[data_no_outliers_length:],:]
    data_no_outliers =  data_no_outliers[random_integers[:data_no_outliers_length],:]

    X = np.concatenate((data_no_outliers, data_outliers))
    print(X.shape)

Cause other clustering algorithms almost cannot complete this clustering on the full data. So we employ CLASSIX clustering on the whole data while employing other clustering algorithms on down-sampling data, and get their average runtime for comparison:

.. code:: python
    
    sample_size = 10 # repeats each algorithm's performing for 10 times.

    sum_time = 0
    timing = []

    for i in range(sample_size):
        st = time.time()
        dbscan = DBSCAN(eps=0.7, min_samples=6)
        dbscan.fit(X)
        et = time.time()
        sum_time = sum_time + et - st

    timing.append(sum_time/sample_size)
    print("Average consume time: ", sum_time/sample_size)
    plt.figure(figsize=(24,10))
    plt.scatter(X[:,0], X[:,1], c=dbscan.labels_, cmap='jet')
    plt.tick_params(axis='both',  labelsize=15)
    plt.title('DBSCAN',  fontsize=20)
    plt.show()

    sum_time = 0
    for i in range(sample_size):
        st = time.time()
        _hdbscan = hdbscan.HDBSCAN(min_cluster_size=1100, core_dist_n_jobs=1)
        hdbscan_labels = _hdbscan.fit_predict(X)
        et = time.time()
        sum_time = sum_time + et - st

    timing.append(sum_time/sample_size)
    print("Average consume time: ", sum_time/sample_size)
    plt.figure(figsize=(24,10))
    plt.scatter(X[:,0], X[:,1], c=hdbscan_labels, cmap='jet')
    plt.tick_params(axis='both',  labelsize=15)
    plt.title('HDBSCAN',  fontsize=20)
    plt.show()

    sum_time = 0
    for i in range(sample_size):
        st = time.time()
        quicks = QuickshiftPP(k=450, beta=0.75)
        quicks.fit(X.copy(order='C'))
        quicks_labels = quicks.memberships
        et = time.time()
        sum_time = sum_time + et - st

    timing.append(sum_time/sample_size)
    print("Average consume time: ", sum_time/sample_size)
    plt.figure(figsize=(24,10))
    plt.scatter(X[:,0], X[:,1], c=quicks_labels, cmap='jet')
    plt.tick_params(axis='both',  labelsize=15)
    plt.title('Quickshift++',  fontsize=20)
    plt.show()

    sum_time = 0
    for i in range(sample_size):
        st = time.time()
        clx = CLASSIX(sorting='pca', radius=0.45, verbose=0, group_merging='distance')
        clx.fit_transform(data)
        et = time.time()
        sum_time = sum_time + et - st

    timing.append(sum_time/sample_size)
    print("Average consume time: ", sum_time/sample_size)
    plt.figure(figsize=(24,10))
    plt.scatter(data[:,0], data[:,1], c=clx.labels_, cmap='jet')
    plt.tick_params(axis='both',  labelsize=15)
    plt.title('CLASSIX',  fontsize=20)
    plt.show()
    
.. image:: images/DBSCAN_kamil.png
.. image:: images/HDBSCAN_kamil.png
.. image:: images/Quickshiftpp_kamil.png
.. image:: images/CLASSIX_kamil.png

We can simply visualize the runtime:

.. code:: python

    bardf = pd.DataFrame()
    names = ['DBSCAN \n(5%)', 'HDBSCAN \n(5%)', 'Quickshift++ \n(5%)', 'CLASSIX \n(100%)']
    bardf['clustering'] = names
    bardf['runtime'] = timing

    def colors_from_values(values, palettes):
        norm = (values - min(values)) / (max(values) - min(values))
        indices = np.round(norm * (len(values) - 1)).astype(np.int32)
        palettes = sns.color_palette(palettes, len(values))
        return np.array(palettes).take(indices, axis=0)


    pvals = np.array([0.1,0.2,0.4,0.6]) # np.array(timing)/np.sum(timing)
    plt.figure(figsize=(14, 9))
    sns.set(font_scale=1.5, style="whitegrid")
    ax = sns.barplot(x="clustering", y="runtime", data=bardf, width=0.6, 
                     palette=colors_from_values(pvals, 'Set1'))

    ax.bar_label(ax.containers[0], fmt='%.2f s')
    ax.set(xlabel=None)
    ax.set_ylabel("runtime", fontsize=28)
    plt.tick_params(axis='both', labelsize=19)
    plt.show()

The runtime bar plot is as below, we can see that CLASSIX achieves the fastest speed even if it runs with the whole data.

.. image:: images/runtime.png


Gaussian blobs
##################

To provide another insight for clustering comparison with respect to runtime, we compare these algorithms by fixing optimal parameter setting on synthetic Gaussian blobs data with increasing size and dimension. So as to obtain a fair comparison of their runtime, we hope the clustering accuracy for all algorithms remains the same as much as possible as the data change. On the other hand, this experiment manifests the sensitivity of parameter settings to environmental settings. This Gaussian blobs test can be referred to in CLASSIX's paper. The test is referenced from https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html. 

.. image:: images/performance.png


As we see from the figure, CLASSIX compares favorably against other algorithms while achieving the fastest speed and stable runtime among them.  
