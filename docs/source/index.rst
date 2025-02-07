.. CLASSIX documentation

Welcome to CLASSIX's documentation!
===================================


Clustering is a fundamental unsupervised learning technique used to identify patterns and structures in data. It groups data points into distinct clusters so that points within the same cluster share similar characteristics based on **distance, density, or other spatial properties**, while points in separate clusters are less similar.

Since acquiring labeled data is often costly and time-consuming, clustering serves as a powerful tool for exploratory data analysis and pattern discovery across various scientific and engineering fields, including:

- **Time series analysis** – Identifying trends, anomalies, and recurring patterns in sequential data.
- **Social network analysis** – Detecting communities and influential entities within networks.
- **Market segmentation** – Grouping customers based on behavior, demographics, or purchasing patterns.
- **Anomaly detection** – Recognizing unusual patterns in cybersecurity, fraud detection, and medical diagnosis.
- **Image segmentation** – Dividing images into meaningful regions for object recognition, medical imaging, and computer vision tasks.


We introduce **CLASSIX**, a novel, explainable clustering algorithm that integrates features of both **distance-based** and **density-based** methods. Unlike many traditional clustering techniques, CLASSIX is designed for **speed, scalability, and interpretability**, making it particularly well-suited for large datasets.

Key Features of CLASSIX
-----------------------

- **Sorting-based approach** – CLASSIX leverages data sorting as a core mechanism, enabling efficient clustering with minimal computational overhead.
- **Fast and scalable** – The algorithm is optimized for large-scale data processing without sacrificing accuracy.
- **Explainable** – The clustering process remains transparent, allowing users to understand and interpret how clusters are formed.

This documentation provides a comprehensive guide on using CLASSIX, its practical applications, and best practices for parameter tuning.

How CLASSIX Works
-----------------------

CLASSIX follows a two-phase clustering process:

1. **Aggregation** – A fast, initial partitioning of data based on **starting points**, which serve as reduced estimators of density.
2. **Merging** – Overlapping groups from the aggregation phase are merged into clusters based on either a **distance-based** or **density-based** criterion.

This two-step approach ensures that CLASSIX remains both computationally efficient and robust across various data distributions.

Understanding CLASSIX Parameters
-----------------------

The behavior of CLASSIX is primarily controlled by two key parameters: **radius** and **minPts**, both of which are easy to interpret and tune.

- **radius**: A distance-based threshold that governs the tolerance for grouping points during the aggregation phase.
- **minPts**: Specifies the minimum number of points required for a valid cluster in the final output, ensuring small, insignificant clusters are filtered out.

These parameters provide users with **flexible control** over clustering granularity and computational efficiency. In the following sections, we will explore their effects in greater detail and provide guidance on optimal parameter selection.



.. raw:: html

   <iframe width="720" height="420" src="https://www.youtube.com/embed/K94zgRjFEYo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Guide
-------------


The documentation mainly contains five chapters about user guidance and one chapter for API reference, which is organized as follows: The first chapter demonstrates a quick introduction to the installment of CLASSIX and its deployment; The second chapter compares CLASSIX with other sought-after clustering algorithms with respect to speed and accuracy on built-in data; The third chapter illustrates a vivid tutorial for density and distance merging applications; The fourth chapter illustrates the interpretable clustering result obtained from CLASSIX; The final chapter demonstrates how to use CLASSIX to find outliers in data; The API details can be found at the independent section titled as “API reference”. The documentation is still under construction, any suggestions from users are appreciated, please be free to email us for any questions on CLASSIX. 


.. toctree::
   :maxdepth: 2
   
   quickstart
   clustering_tutorial
   matlab
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
