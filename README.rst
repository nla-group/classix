.. image:: https://codecov.io/gh/nla-group/classix/branch/master/graph/badge.svg?token=D4MQZS67H1
    :target: https://codecov.io/gh/nla-group/classix
    :alt: codecov
.. image:: https://img.shields.io/pypi/v/ClassixClustering?color=orange
    :target: https://pypi.org/project/ClassixClustering/
    :alt: pypi
.. image:: https://static.pepy.tech/badge/ClassixClustering
    :target: https://pypi.org/project/ClassixClustering/
    :alt: Download Status
.. image:: https://readthedocs.org/projects/classix/badge/?version=latest
    :target: https://classix.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/nla-group/classix/blob/master/LICENSE
    :alt: License: MIT



CLASSIX is a fast and explainable clustering algorithm based on sorting. Here are a few highlights:

* Ability to cluster low and high-dimensional data of arbitrary shape efficiently.
* Ability to detect and deal with outliers in the data.
* Ability to provide textual explanations for the generated clusters.
* Full reproducibility of all tests in the accompanying paper.
* Support of Cython compilation.

``CLASSIX`` is a contrived acronym of CLustering by Aggregation with Sorting-based Indexing and the letter X for explainability. CLASSIX clustering consists of two phases, namely a greedy aggregation phase of the sorted data into groups of nearby data points, followed by a merging phase of groups into clusters. The algorithm is controlled by two parameters, namely the distance parameter radius for the group aggregation and a minPts parameter controlling the minimal cluster size.


-----------------------
Installing and example
-----------------------

CLASSIX has the following dependencies for its clustering functionality:

* cython
* numpy
* scipy
* requests

and requires the following packages for data visualization:

* matplotlib
* pandas


To install the current CLASSIX release via PIP use:

.. code:: bash
    
    pip install classixclustering

To check the CLASSIX installation you can use:

.. code:: bash
    
    python -m pip show classixclustering


Download the repository via:

.. code:: bash
    
    git clone https://github.com/nla-group/classix.git
    
    

Example usage:

.. code:: python

    from sklearn import datasets
    from classix import CLASSIX

    # Generate synthetic data
    X, y = datasets.make_blobs(n_samples=2000000, centers=4, n_features=10, random_state=1)

    # Employ CLASSIX clustering
    clx = CLASSIX(sorting='pca', verbose=1)
    clx.fit(X)


----------
Citation
----------

.. code:: bibtex

    @techreport{CG22b,
      title   = {Fast and explainable clustering based on sorting},
      author  = {Chen, Xinye and G\"{u}ttel, Stefan},
      year    = {2022},
      number  = {arXiv:2202.01456},
      pages   = {25},
      institution = {The University of Manchester},
      address = {UK},
      type    = {arXiv EPrint},
      url     = {https://arxiv.org/abs/2202.01456}
    }