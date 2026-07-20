API reference
=============

This page documents the public Python API exported by ``classix`` and the main
pure-Python implementation modules. Most users should start with
:class:`classix.CLASSIX`.

Estimator
---------

.. autoclass:: classix.CLASSIX
   :members:
   :undoc-members:
   :show-inheritance:

Top-level helpers
-----------------

.. autofunction:: classix.cython_is_available

.. autofunction:: classix.loadData

.. autofunction:: classix.preprocessing

.. autofunction:: classix.calculate_cluster_centers

Implementation modules
----------------------

The functions below are primarily internal building blocks used by
:class:`classix.CLASSIX`. They are documented for advanced users who want to
inspect or extend the algorithm.

Euclidean aggregation
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: classix.aggregate_ed
   :members:
   :undoc-members:

Euclidean merging
~~~~~~~~~~~~~~~~~

.. automodule:: classix.merge_ed
   :members:
   :undoc-members:

Manhattan aggregation and merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: classix.aggregate_md
   :members:
   :undoc-members:

.. automodule:: classix.merge_md
   :members:
   :undoc-members:

Tanimoto aggregation and merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: classix.aggregate_td
   :members:
   :undoc-members:

.. automodule:: classix.merge_td
   :members:
   :undoc-members:
