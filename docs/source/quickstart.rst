
Get Started with CLASSIX
======================================
Clustering is a widely-used unsupervised learning technique to find patterns and structures in data. The applications of clustering are wide-ranging,  including areas like finance, traffic, civil engineering, and bioinformatics.  Clustering algorithms aim to group the data points into distinct clusters such that points within a cluster share similar characteristics on the basis of spatial properties, while points in two distinct clusters are less similar.  It might be easy for a human to perceive the clusters with a small sample in a small (1 or 2) dimensional space, however, in practice, the real world data with increasing dimensions and size of the data usually make a human out of reach. Considering data provided with labels are considerably rare and expensive, reliable and easy-to-tune explainable clustering methods are highly desirable. 

A novel clustering method called CLASSIX was proposed which shares features with both distance and density-based methods. It can handle arbitrarily shaped clusters without specifying the number of clusters in advance.  The method comprises two phases: aggregation and merging. 
In the aggregation phase, data points are sorted along their first principal axis and then grouped using a greedy aggregation technique.  The aggregation phase is followed by the merging of overlapping groups into clusters using either a distance or density-based criterion. The density-based merging criterion usually results in slightly better clusters than the distance-based criterion, but the latter has a significant speed advantage. CLASSIX has two parameters to be defined and its tuning is relatively straightforward. In brief, there is a distance parameter $R$ that serves as a tolerance for the grouping in the aggregation phase, while an $\minPts$ parameter specifies the smallest acceptable cluster size.

Owing to the initial sorting of the data points, CLASSIX does not perform spatial range queries for each data point. CLASSIX's inherent simplicity allows us to derive a procedure to explain the clustering result. We believe that this feature, explainability, in addition to the fast clustering time of CLASSIX, might make this method very attractive to users.

Here we show you how to get a quick start!

Installation guide
------------------------------
To install the current release via PIP use:

.. parsed-literal::
    
    pip install ClassixClustering


Download CLASSIX code:

.. parsed-literal::
    
    git clone https://github.com/nla-group/CLASSIX.git


Quick start
------------------------------


CLASSIX follows a similar API design as scikit-learn library. So if you are familiar with scikit-learn, you can quickly master the CLASSIX library to do a wonderful clustering. 
We demonstrate a toy application of CLASSIX's clustering on simple data. 

After importing the required python libraries, we generate isotropic Gaussian blobs with 2 clusters using sklearn.datasets tool. 
The sample is exhibited with 2 clusters of 1000 2-dimensional data. Then, we employ CLASSIX with the simple setting:


.. code:: python

    from sklearn.datasets import make_blobs
    from classix import CLASSIX

    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)    
    clx = CLASSIX(sorting='pca', radius=0.5, verbose=0, minPts=13)
    clx.fit(X)

After that, to get the clustering result, we just need to load ``clx.labels_``. Also you can return the cluster labels directly by ``labels = clx.fit_transform(X)``.
Now we plot the clustering result:

.. code:: python

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0], X[:,1], c=clx.labels_)
    plt.show()

The result is as belows:

.. image:: images/demo1.png

That is a basic setting tutorial of CLASSIX, which applied to most cases. If you want to learn more, please go through other sections of the documentation.