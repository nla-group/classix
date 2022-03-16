<h1 align="center">
  CLASSIX :honeybee:
  
  
   

</h1>

<h3 align="center">
  <strong> Fast and explainable clustering based on sorting </strong>  
</h3>


[![Publish](https://github.com/nla-group/classix/actions/workflows/package_release.yml/badge.svg?branch=master)](https://github.com/nla-group/classix/actions/workflows/package_release.yml)
[![codecov](https://codecov.io/gh/nla-group/classix/branch/master/graph/badge.svg?token=D4MQZS67H1)](https://codecov.io/gh/nla-group/classix)
[![!pypi](https://img.shields.io/pypi/v/ClassixClustering?color=orange)](https://pypi.org/project/ClassixClustering/)
[![Download Status](https://static.pepy.tech/badge/ClassixClustering)](https://pypi.org/project/ClassixClustering/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nla-group/classix/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/classix/badge/?version=latest)](https://classix.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/nla.stefan.xinye/classix/badges/version.svg)](https://anaconda.org/nla.stefan.xinye/classix)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nla-group/classix/HEAD)
[![Anaconda-Server Badge](https://anaconda.org/nla.stefan.xinye/classix/badges/latest_release_date.svg)](https://anaconda.org/nla.stefan.xinye/classix)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ClassixClustering.svg)](https://pypi.python.org/pypi/ClassixClustering/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6329792.svg)](https://doi.org/10.5281/zenodo.6329792)

## :sparkles: Features

CLASSIX is a fast and explainable clustering algorithm based on sorting. Here are a few highlights:

- Ability to cluster low and high-dimensional data of arbitrary shape efficiently.
- Ability to detect and deal with outliers in the data.
- Ability to provide textual explanations for the generated clusters.
- Full reproducibility of all tests in the accompanying paper.
- Support of Cython compilation.



``CLASSIX`` is a contrived acronym of *CLustering by Aggregation with Sorting-based Indexing* and the letter *X* for *explainability*. CLASSIX clustering consists of two phases, namely a greedy aggregation phase of the sorted data into groups of nearby data points, followed by a merging phase of groups into clusters. The algorithm is controlled by two parameters, namely the distance parameter ``radius`` for the group aggregation and a ``minPts`` parameter controlling the minimal cluster size. 

**Here is a video abstract of CLASSIX:** 

[<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/classix_video_screenshot.png width=520 />](https://www.youtube.com/watch?v=K94zgRjFEYo)
 

A detailed documentation, including tutorials, is available at [![Dev](https://img.shields.io/badge/docs-latest-blue.svg)](https://classix.readthedocs.io/en/latest/). 

## :rocket: Install

CLASSIX has the following dependencies for its clustering functionality:

- cython (recommend >=0.27)
- numpy>=1.3.0 (recommend >=1.20.0)
- scipy>=1.2.1
- requests

and requires the following packages for data visualization:

- matplotlib
- pandas

To install the current CLASSIX release via PIP use:
```
pip install ClassixClustering
```


To check the CLASSIX installation you can use:
```
python -m pip show ClassixClustering
```

Download this repository via:
```
git clone https://github.com/nla-group/classix.git
```


##  :checkered_flag: Quick start

We start with a simple synthetic dataset: 

```Python
from sklearn import datasets
from classix import CLASSIX

# Generate synthetic data
X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)

# Employ CLASSIX clustering
clx = CLASSIX(sorting='pca', radius=0.5, verbose=0)
clx.fit(X)
```

Get the clustering result by ``clx.labels_`` and visualize the clustering:  
```Python
plt.figure(figsize=(10,10))
plt.rcParams['axes.facecolor'] = 'white'
plt.scatter(X[:,0], X[:,1], c=clx.labels_)
plt.show()
```

You can also load the aggregation groups via ``clx.groups_`` (version >=0.5.5).

## :mortar_board: The explain method

CLASSIX provides an API for the easy visualization of clusters, and to explain the assignment of data points to their clusters. To get an overview of the data points, the location of starting points, and their associated groups, simply type:

```Python
clx.explain(plot=True)
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/explain_viz.png width=500 />

The starting points are marked as the small red boxes. The method also returns a textual summary as follows:

```
A clustering of 5000 data points with 2 features has been performed. 
The radius parameter was set to 0.50 and MinPts was set to 0. 
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
  ---      lines omitted        ---
  11       6      1       0.42 1.35 
  12       5      1       1.24 0.59 
  13       2      1        1.0 1.08 
----------------------------------------
In order to explain the clustering of individual data points, 
use .explain(ind1) or .explain(ind1, ind2) with indices of the data points. 
```

In the above table, *Group* denotes the group label, *NrPts* denotes the number of data points in the group, *Cluster* is the cluster label assigned to the group, and the final column shows the normalized *Coordinates* of the starting point. In order to explain the cluster assignment of a particular data point, we provide its index to the explain method:

```Python
clx.explain(0, plot=True)
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/None0.png width=720 />
Output:

```
The data point 0 is in group 2, which has been merged into cluster #0.
```

We can also query why two data points ended up in the same cluster, or not: 

```Python
clx.explain(0, 2000, plot=True)
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/None0_2000.png width=720 />
Output:

```
The data point 0 is in group 2, which has been merged into cluster 0.
The data point 2000 is in group 10, which has been merged into cluster 1.
There is no path of overlapping groups between these clusters.
```


Besides, CLASSIX allows the dataframes input and explain the corresponding index, we give an example as below.
Similarly, first step is to generate data and apply CLASSIX.

```Python
X, _ = make_blobs(n_samples=5, centers=2, n_features=2, cluster_std=1.5, random_state=1)
X = pd.DataFrame(X, index=['Anna', 'Bert', 'Carl', 'Tom', 'Bob'])
```

We obtain the dataframe look like:

|  | 0 | 1 |
| ----------- | ----------- | ----------- |
| 'Anna'   | -7.804551 | -7.043560|
| 'Bert'   | -9.519154 | -4.327404|
| 'Carl'   | -0.361448 | 0.954182|
| 'Tom'    | 0.957658 | 3.264680|
| 'Bob'    | -2.451818 | 2.797037|


Then, run CLASSIX with ``radius`` of 0.5:
```Python
clx = CLASSIX(radius=0.5, group_merging='distance')
clx.fit_transform(X)
```

Output:
```
CLASSIX(sorting='pca', radius=0.5, minPts=0, group_merging='distance')
The 5 data points were aggregated into 4 groups.
In total 3 comparisons were required (0.60 comparisons per data point). 
The 4 groups were merged into 2 clusters with the following sizes: 
      * cluster 0 : 3
      * cluster 1 : 2
Try the .explain() method to explain the clustering.
```

Let's take a look at the data distribution and clustering result:
```Python
clx.explain(plot=True, figsize=(2,2), sp_fontsize=12)
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/explain_viz_df.png width=720 />

```
A clustering of 5 data points with 2 features has been performed. 
The radius parameter was set to 0.50 and MinPts was set to 0. 
As the provided data has been scaled by a factor of 1/6.33,
data points within a radius of R=0.50*6.33=3.17 were aggregated into groups. 
In total 3 comparisons were required (0.60 comparisons per data point). 
This resulted in 4 groups, each uniquely associated with a starting point. 
These 4 groups were subsequently merged into 2 clusters. 
A list of all starting points is shown below.
----------------------------------------
 Group  NrPts  Cluster  Coordinates 
   0      1       1     -0.63 -0.97 
   1      1       1      -0.9 -0.55 
   2      2       0       0.22 0.58 
   3      1       0       0.76 0.65 
----------------------------------------
In order to explain the clustering of individual data points, 
use .explain(ind1) or .explain(ind1, ind2) with indices of the data points.
```

If we want to check individual data object, just use:

```Python
clx.explain('Bert', plot=True, sp_fontsize=12)
```

Output:

<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/NoneBert.png width=500 />

The table below indicates group information of Bert, the ``Coordinates`` corresponds to the coordinates of starting points for the group, the values are normalized associated with the sorting way we choose (here is ``sorting='pca'``).

```
----------------------------------------
 Group  NrPts  Cluster Coordinates  Label
   1      1       1     -0.9 -0.55   Bert
----------------------------------------
The data point Bert is in group 1, which has been merged into cluster #1.
```

Similarly, we can track the pair of objects and ccompare them:

```Python
clx.explain(index1='Tom', index2='Bert', plot=True, sp_fontsize=12)
```

Output:

<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/NoneTom_Bert.png width=500 />

```
----------------------------------------
 Group  NrPts  Cluster Coordinates  Label
   3      1       0      0.76 0.65    Tom
   1      1       1     -0.9 -0.55   Bert
----------------------------------------
The data point Tom is in group 3, which has been merged into cluster 0.
The data point Bert is in group 1, which has been merged into cluster 1.
There is no path of overlapping groups between these clusters.
```

## :raising_hand: Frequently Asked Questions

### Does density based merging work better than distance based merging?

It depends. For high dimensional data, distance based merging is recommended. Usually, density based merging works slower than distance based merging. But if data can be dealT with high `radius` for density based merging, density based merging might be faster than the distance based merging with a low `radius` while achiving better performance. For example, the data in the next question may not be suitable for distance based merging.


### How to tune the `minPts`?

Mostly, `minPts` is not required, though it is very important part in CLASSIX. To obtain a good clustering, users usually use `minPts` accompanying with `verbose=1`. Then, we can specify the minPts to an appropriate level for those isolated clusters. For example, the dataset like 

```Python
from sklearn import datasets
# generate synthetic data
X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
# run CLASSIX
clx = CLASSIX(sorting='pca', radius=0.15, group_merging='density', verbose=1, minPts=14, post_alloc=False)
clx.fit(X)
# draw figure
plt.figure(figsize=(8,8))
plt.rcParams['axes.facecolor'] = 'white'
plt.scatter(X[:,0], X[:,1], c=clx.labels_)
plt.xticks([])
plt.yticks([])
plt.show()
```

We can easily perceive that the appropriate `minPts` is 14, i.e., the clusters that cardinality is smaller than 14 will be treated as outliers. 

```
CLASSIX(sorting='pca', radius=0.15, minPts=14, group_merging='density')
The 1000 data points were aggregated into 212 groups.
In total 5675 comparisons were required (5.67 comparisons per data point). 
The 212 groups were merged into 41 clusters with the following sizes: 
      * cluster 0 : 454
      * cluster 1 : 440
      * cluster 2 : 13
      * cluster 3 : 10
      * cluster 4 : 7
      * cluster 5 : 7
      * cluster 6 : 6
      * cluster 7 : 5
      * cluster 8 : 5
      * cluster 9 : 4
      * cluster 10 : 4
      * cluster 11 : 4
      * cluster 12 : 3
      * cluster 13 : 3
      * cluster 14 : 3
      * cluster 15 : 2
      * cluster 16 : 2
      * cluster 17 : 2
      * cluster 18 : 2
      * cluster 19 : 2
      * cluster 20 : 2
      * cluster 21 : 1
      * cluster 22 : 1
      * cluster 23 : 1
      * cluster 24 : 1
      * cluster 25 : 1
      * cluster 26 : 1
      * cluster 27 : 1
      * cluster 28 : 1
      * cluster 29 : 1
      * cluster 30 : 1
      * cluster 31 : 1
      * cluster 32 : 1
      * cluster 33 : 1
      * cluster 34 : 1
      * cluster 35 : 1
      * cluster 36 : 1
      * cluster 37 : 1
      * cluster 38 : 1
      * cluster 39 : 1
      * cluster 40 : 1
As MinPts is 14, the number of clusters has been further reduced to 3.
```

The figure below shows the clustering result with `post_alloc=False`:

<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/demo5.png  width=360 />

So the next step is how we process these outliers, we can either marked as independent color as the figure above shows (the labels for outliers denote -1) or allocate them to the nearby clusters (each outlier will be assigned a label). If we determine to allow the outliers exist, we can set `post_alloc=False`. Otherwise outliers will be reassigned by setting `post_alloc=True`. The performance of `post_alloc=True` is as below. The post-processing depends on the problem context. 

<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/demo5_post.png />


### How to determine an appropriate value for radius?

In most cases, `radius` ranging from 0.1 to 1 can handle most cases. The higher the dimensionality of the data is, the higher the `radius` we should use (better try the `radius` > 1).  For density based merging, the `radius` can be set a bit higher than distance based merging. 




## :art: Reproducible experiment
All experiment in the paper referenced below are reproducible by running the code in the folder of ["exp"](https://github.com/nla-group/classix/tree/master/exp).
Before running, ensure the dependencies `scikit-learn` and `hdbscan` are installed, and compile the ``Quickshift++`` code ([Quickshift++: Provably Good Initializations for Sample-Based Mean Shift](https://github.com/google/quickshift)). After configuring all of these, run the commands below. 

```
cd exp
python3 run exp_main.py
```

All results will be stored on ["exp/results"](https://github.com/nla-group/classix/tree/master/exp/results). Please let us know if you have any questions.




## :paperclip: Citation 

```bibtex
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
```


## :memo: License
This project is licensed under the terms of the [MIT license](https://github.com/nla-group/classix/blob/master/LICENSE).


<p align="left">
  <a>
    <img alt="CLASSIX" src="https://raw.githubusercontent.com/nla-group/classix/master/docs/_images/nla_group.png" width="240" />
  </a>
</p>
