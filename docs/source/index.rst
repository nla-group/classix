.. CLASSIX documentation

Welcome to CLASSIX's documentation!
===================================

Clustering is a widely-used unsupervised learning technique to find patterns and structures in data. Clustering algorithms group the data points into distinct clusters such that points within a cluster share similar characteristics on the basis of their distance, density, or other spatial properties, while points in two distinct clusters are less similar. Since obtaining data labels requires a huge investment of human labor and cost, clustering analysis, as a general task to be solved, has a broad range of applications in many scientific and engineering fields, e.g., time series analysis, social network analysis, market segmentation, anomaly detection, and image segmentation.  We propose a novel clustering method called CLASSIX which shares features with both distance and density-based methods.  CLASSIX is an explainable sorting-based clustering algorithm towards a fast and scalable setting. 

In this documentation, we will illustrate the use of CLASSIX and introduce its basic applications and provide fundamental guidance for its parameter settings. Simply put, CLASSIX comprises two phases to conduct clustering, namely aggregation and merging. The aggregation performs a quick partition of data associated with starting points which are a reduced estimator of general density. After that. the aggregation phase is followed by the merging of overlapping groups into clusters using either a distance or density-based criterion. For parameter settings, CLASSIX is dominated by two parameters, namely ``radius`` and ``minPts``, and their tuning is straightforward and simple, which we will illustrate later. In brief, ``radius`` is a distance parameter that serves as a tolerance for the grouping in the aggregation phase, while ``minPts`` parameter specifies the minimum acceptable cluster size for the final picture.

The documentation mainly contains five chapters about user guidance and one chapter for API reference, which is organized as follows: The first chapter demonstrates a quick introduction to the installment of CLASSIX and its deployment; The second chapter compares CLASSIX with other sought-after clustering algorithms with respect to speed and accuracy on built-in data; The third chapter illustrates a vivid tutorial for density and distance merging applications; The fourth chapter illustrates the interpretable clustering result obtained from CLASSIX; The final chapter demonstrates how to use CLASSIX to find outliers in data; The API details can be found at the independent section titled as “API reference”. The documentation is still under construction, any suggestions from users are appreciated, please be free to email us for any questions on CLASSIX. 

.. raw:: html

   <iframe width="720" height="420" src="https://www.youtube.com/embed/K94zgRjFEYo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Guide
-------------

.. toctree::
   :maxdepth: 2
   
   quickstart
   clustering_tutorial
   comparison
   outlier_detection
   

API Reference
-------------
.. toctree::
   :maxdepth: 2

   api_reference


Others
-------------
.. toctree::
   :maxdepth: 2
   
   acknowledgement
   license
   contact


Indices and Tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. image:: images/nla_group.png
    :width: 360
