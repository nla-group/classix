Performance Comparison
======================================

In this tutorial we compare CLASSIX with some widely used density-based clustering algorithms like DBSCAN, HDBSCAN and Quickshift++. We perform experiments on the VDU (Vacuum Distillation Unit) dataset that comes with the CLASSIX installation and with synthetically generated Gaussian blobs.

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

When run on the full data set, DBSCAN, HDBSCAN, Quickshift++ fail in this experiment (runtime > 24 hr) while CLASSIX requires around 1.5 seconds for the clustering of whole data. Therefore, to compare the four algorithms (and ensure they finish their computation within a day), we select 5% of the data for all competing algorithms except CLASSIX:

.. code:: python
    
    np.random.seed(0)
    sample = np.random.choice(data.shape[0], size=int(np.round(0.05*data.shape[0])))
    X = data[sample]
    print(X.shape)


We repeatedly run each algorithm 10 times and get the average runtime for comparison. All algorithms run on a single thread and the parameter settings for each algorithm are tuned for the best visual clustering result. 

.. code:: python
    
    sample_size = 10 # run each algorithm 10 times.
    timing = []
    
    sum_time = 0
    for i in range(sample_size):
        st = time.time()
        dbscan = DBSCAN(eps=0.6, min_samples=12)
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
        _hdbscan = hdbscan.HDBSCAN(min_cluster_size=420, core_dist_n_jobs=1)
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
        quicks = QuickshiftPP(k=450, beta=0.85)
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
        clx = CLASSIX(sorting='pca', radius=1, verbose=0, group_merging='distance')
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
    
.. image:: images/DBSCAN.png
.. image:: images/HDBSCAN.png
.. image:: images/Quickshiftpp.png
.. image:: images/CLASSIX.png

The runtime of all algorithms is visualized in the below bar chart. Recall that CLASSIX has been run on the full data set, while the other algorithms were run only on 5 percent of the data.

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

.. image:: images/runtime.png


Example 2: Gaussian blobs
##################

We now compare the algorithms on synthetic Gaussian blobs with increasing number of data points and dimension. Further details on this experiment can be found in the CLASSIX paper (https://arxiv.org/abs/2202.01456).  

.. image:: images/performance.png
