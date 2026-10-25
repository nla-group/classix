<h1 align="center">
  CLASSIX：Python 中快速且可解释的聚类算法
</h1>

[English](README.md) | **简体中文**

[![Publish](https://github.com/nla-group/classix/actions/workflows/package_release.yml/badge.svg?branch=master)](https://github.com/nla-group/classix/actions/workflows/package_release.yml)
[![!pypi](https://img.shields.io/pypi/v/classixclustering?color=tomato)](https://pypi.org/project/classixclustering/)
![Static Badge](https://img.shields.io/badge/Compiler-8A2BE2?label=Cython-Accelerated)
[![Workflow for Codecov (with Cython coverage)](https://github.com/nla-group/classix/actions/workflows/codecov.yml/badge.svg)](https://github.com/nla-group/classix/actions/workflows/codecov.yml)
[![License: MIT](https://anaconda.org/conda-forge/classixclustering/badges/license.svg)](https://github.com/nla-group/classix/blob/master/LICENSE)


[![Anaconda-Server Badge](https://anaconda.org/conda-forge/classixclustering/badges/version.svg)](https://anaconda.org/conda-forge/classixclustering)
[![azure](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/classixclustering-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=15797&branchName=main)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/classixclustering.svg)](https://anaconda.org/conda-forge/classixclustering)
[![Download Status](https://static.pepy.tech/badge/classixclustering)](https://pypi.org/project/classixclustering/)
[![Download Status](https://img.shields.io/pypi/dm/classixclustering.svg?label=PyPI%20downloads)](https://pypi.org/project/classixclustering/)
[![Documentation Status](https://readthedocs.org/projects/classix/badge/?version=stable)](https://classix.readthedocs.io/en/latest/?badge=stable)


<!---[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ClassixClustering.svg)](https://pypi.python.org/pypi/ClassixClustering/)-->

__CLASSIX__ 是一种快速、内存高效且可解释的聚类算法，主要特点如下：

- 高效地对任意形状的低维和高维数据进行聚类
- 能够检测并处理数据中的异常值
- 能够为聚类结果提供文本和可视化解释
- 配套论文中的所有实验均可完全复现
- 支持欧几里得、Tanimoto 和曼哈顿距离
- 支持 Cython 编译

__CLASSIX__ 是由 *CLustering by Aggregation with Sorting-based Indexing*（基于排序索引的聚合式聚类）的缩写与代表 *explainability*（可解释性）的字母 *X* 组合而成的名称。

## 安装

可以通过 PIP（推荐）或 Conda 安装 __CLASSIX__：

| PyPI | conda-forge |
| :---: |:---: |
|[![PyPI version](https://badge.fury.io/py/classixclustering.svg)](https://pypi.org/project/classixclustering/) | [![conda-forge version](https://anaconda.org/conda-forge/classixclustering/badges/version.svg)](https://anaconda.org/conda-forge/classixclustering) |
| NumPy<=1.26.4: `pip install classixclustering` <br/> NumPy>2: `pip install classixclustering --no-cache-dir `| `conda install -c conda-forge classixclustering` |


我们推荐使用 `pip install classixclustering --no-cache-dir ` 安装。

<br/>

__编程语言__ | __依赖项__
:---:|:---:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) |  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


## 快速开始

以下简单的代码示例可以帮助您快速上手：

```Python
import classix
from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=0)
clx = classix.CLASSIX(radius=0.1, minPts=10, verbose=0)
clx.fit(data)
print(clx.labels_) # clustering labels
clx.explain(data=data, plot=True)
```

下面是一个更贴近实际应用的示例，使用 CLASSIX 提供的演示 DataFrame：

```Python
import classix
data, labels = classix.loadData('Covid3MC')
clx = classix.CLASSIX(radius=0.2, minPts=500, verbose=0)
clx.fit(data)
clx.explain(data=data, plot=True)
```

模型拟合后，还可以使用 ``predict()`` 对样本外数据进行聚类，例如 `clx.predict(data.iloc[:1000])`。

## explain 方法

CLASSIX 是一种*可解释*的聚类方法。要查看计算所得聚类结果的概况，请输入：

```Python
clx.explain(data)
```
输出：
```
CLASSIX clustered 5726839 data points with 3 features.
The radius parameter was set to 0.20 and minPts was set to 500.
As the provided data was auto-scaled by a factor of 1/12.27,
points within a radius R=0.20*12.27=2.45 were grouped together.
In total, 71926291 distances were computed (12.6 per data point).
This resulted in 274 groups, each with a unique group center.
These 274 groups were subsequently merged into 25 clusters.
```

我们可以让 CLASSIX 解释为什么两个数据点最终被分到同一个簇，或为什么没有被分到同一个簇：

```Python
clx.explain(data, 'hCoV-19/Norway/6348/2021', 'hCoV-19/USA/WY-WYPHL-20146677/2020', plot=True)
```
输出：
```
Data point hCoV-19/Norway/6348/2021 is in group 256.
Data point hCoV-19/USA/WY-WYPHL-20146677/2020 is in group 200.
Both groups were merged into cluster #15.

The two groups are connected via groups
 256 <-> 257 <-> 248 <-> 242 <-> 225 <-> 217 <-> 200.

Here is a list of connected data points with
their global data indices and group numbers:

   Index Distance  Group                                     Label
    600       --    256                  hCoV-19/Norway/6348/2021
 186838     0.17    256   hCoV-19/Ireland/LD-NVRL-89IRL62956/2021
5261957     0.28    257 hCoV-19/Northern_Ireland/NIRE-003afb/2021
5086569     0.20    248        hCoV-19/Ukraine/Kyiv-PHC-0002/2021
4600143     0.23    242            hCoV-19/Belgium/rega-2620/2021
1599046     0.20    225      hCoV-19/Germany/NW-RKI-I-088626/2021
 273188     0.21    217             hCoV-19/USA/CA-ALSR-8586/2020
 167935     0.24    200            hCoV-19/Spain/CT-HUB00588/2021
    100     0.13    200        hCoV-19/USA/WY-WYPHL-20146677/2020

The distance between consecutive data points is at most R=3.68.
Here, R=0.2*12.27*1.5, where 0.2 is the chosen radius parameter,
dataScale_=12.27 is a data scaling factor determined by CLASSIX,
and mergeScale_=1.5 (the default value).
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/sample.png width=800 />

CLASSIX 聚类包含两个阶段：首先通过贪心聚合，将相邻的数据点组成组；然后将这些组合并成簇。``radius`` 参数控制组的大小，``minPts`` 控制簇的最小大小。CLASSIX 的解释表明，从数据点 `hCoV-19/Costa_Rica/HNN-0400/2021` 到数据点 `hCoV-19/Denmark/DCGC-269073/2021` 存在一条经过所计算的组中心（256、248 等）的路径。


#### 数据框（DataFrame）
CLASSIX 支持 DataFrame，并支持使用 DataFrame 的索引来指定数据点：

```Python
from sklearn import datasets
from classix import CLASSIX
import pandas as pd

X, _ = datasets.make_blobs(n_samples=5, centers=2, n_features=2, cluster_std=1.5, random_state=1)
X = pd.DataFrame(X, index=['Anna', 'Bert', 'Carl', 'Tom', 'Bob'])

#            0     |     1
# Anna | -7.804551 | -7.043560
# Bert | -9.519154 | -4.327404
# Carl | -0.361448 |  0.954182
# Tom  |  0.957658 |  3.264680
# Bob  | -2.451818 |  2.797037

clx = CLASSIX(radius=0.6)
clx.fit_transform(X)
clx.explain(X, index1='Carl', index2='Bert', plot=True, showallgroups=True, sp_fontsize=12)
```
输出：
```
The data point Carl is in group 1, which has been merged into cluster 1.
The data point Bert is in group 0, which has been merged into cluster 0.
There is no path of overlapping groups between these clusters.
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/NoneTom_Bert.png width=800 />

## 常见问题

### 如何调整 `radius` 和 `minPts` 参数？

我们建议先使用较大的 `radius` 参数（例如 `radius=1`）和 `minPts=1` 运行 CLASSIX。设置 `verbose=1` 也有助于获取更详细的运行反馈。通常，`radius` 越大，算法运行得越快，计算得到的簇越少。如果簇的数量太少，可以逐步减小 `radius`，直到获得数量合适且有意义的簇；具体标准取决于应用场景。

当选定的半径使簇的数量略多于预期时，可以使用 `CLASSIX.minPtsChange(minPts)` 调整 `minPts`，无需从头拟合。该方法会复用拟合时的预处理和聚合结果，从而显著加快第二步调参：

```Python
clx = CLASSIX(radius=1.0, minPts=1, verbose=0)
clx.fit(X)

for minPts in [2, 5, 10, 20, 40]:
    clx.minPtsChange(minPts)
    print(minPts, len(set(clx.labels_)), clx.clusterSizes_)
```

在默认设置 `mergeTinyGroups=True` 下，`minPtsChange` 只会重新运行小簇处理阶段。如果设置 `mergeTinyGroups=False`，CLASSIX 还会重新计算组的合并，因为合并图本身依赖于 `minPts`。

可以通过 `clx.clusterSizes_` 查看各个簇的大小。如果存在仅包含少量数据点且不需要的“噪声”簇，可以增大 `minPts` 参数来消除它们。例如，当 `minPts=14` 时，所有少于 14 个数据点的簇都会被重新分配到更大的簇中。以下示例展示了 `minPts` 的作用：

```Python
# generate synthetic data
X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
# run CLASSIX
clx = CLASSIX(sorting='pca', radius=0.15, group_merging='density', verbose=1, minPts=14, post_alloc=False)
clx.fit(X)
# (omit plot code)
```

输出：
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
    --- lines omitted ---
      * cluster 38 : 1
      * cluster 39 : 1
      * cluster 40 : 1
As MinPts is 14, the number of clusters has been further reduced to 3.
```
<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/demo5.png  width=400 />

请注意，这里有许多簇包含的数据点少于 14 个。由于设置了 `minPts=14` 和 `post_alloc=False`，这些小簇都被标记为噪声，标签为 `-1`。也可以通过设置 `post_alloc=True`（默认值），将噪声簇重新分配到各自附近的簇中。此时会得到以下聚类结果：

<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/demo5_post.png width=400 />


### 如何解读和调整可视化结果？

当数据点很多时，`.explain` 方法生成的可视化结果可能难以解读。可以通过一些选项改善绘图效果。例如，当起始点的标注框过大、遮挡了数据点时，可以减小 ``sp_alpha``，提高标注框的透明度；也可以减小 ``sp_pad``，缩小标注框；或者通过 ``sp_fcolor`` 将标注框设为较浅的颜色。更多细节请参阅文档。此外，还可以设置 `cmap`（例如 `'bmh'`）、`cmin` 和 `cmax`，自定义簇的颜色映射。

### 为什么会收到有关 Cython 的警告？

CLASSIX 使用 Cython 加速部分计算。如果收到 Cython 警告，说明当前的 Python 安装不支持 Cython，或者[系统中没有 C 编译器](https://cython.readthedocs.io/en/latest/src/quickstart/install.html)。安装 C 编译器的方式因系统而异，例如 Linux 上可使用 GNU C Compiler（gcc），Windows 上可使用 Microsoft Visual C/C++（MSVC）。CLASSIX 在不使用 Cython 时也能正常运行，但速度可能比通常慢，有时耗时可达 40 倍。可以通过以下方式确认是否正在使用 Cython：
```Python
import classix
classix.cython_is_available(verbose=1)
```

如有需要，可以通过以下方式禁用 Cython：
```Python
classix.__enable_cython__ = False
```

### CLASSIX 的工作原理是什么？
请参阅下方引用的 CLASSIX 论文。简而言之，CLASSIX 沿数据点的第一主轴对其排序，然后将其聚合为半径为 `radius` 的组。组的合并利用高层 BLAS 的最佳实践，并使用提前停止准则结束组搜索，从而加快计算。随后，将计算得到的组合并成簇。最后，对于数据点少于 `minPts` 的簇，其包含的组会被重新分配到更大的簇中。

以下是介绍 CLASSIX 的简短视频摘要：

[<img src=https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/classix_video_screenshot.png width=600 />](https://www.youtube.com/watch?v=K94zgRjFEYo)

包含教程的详细文档见 [![Dev](https://img.shields.io/badge/docs-latest-blue.svg)](https://classix.readthedocs.io/en/latest/)。也欢迎查看 `demos` 子文件夹中的众多演示。

## 实验复现
运行 ["exps"](https://github.com/nla-group/classix/tree/master/exps) 文件夹中的代码，即可复现下方引用的 CLASSIX 论文中的所有实验。
运行前，请确保已安装依赖项 `scikit-learn` 和 `hdbscan`，并已编译 ``Quickshift++`` 代码（[Quickshift++: Provably Good Initializations for Sample-Based Mean Shift](https://github.com/google/quickshift)）。完成这些配置后，运行以下命令：
```
cd exp
python3 run exp_main.py
```
所有结果都会写入 ["exps/results"](https://github.com/nla-group/classix/tree/master/exp/results) 文件夹。如有任何问题，请联系我们。

## 贡献
欢迎任何形式的贡献，尤其欢迎以 Jupyter Notebook 形式提交新的 `demos`。如果希望帮助改进文档或代码，欢迎提交 issue 和 pull request。参与贡献时，请先 fork 本项目，再为您的修改提交 pull request。我们会尽力处理各项问题和请求，并将您的代码合并到主分支。贡献者将在发行说明中得到致谢。

## 引用

如果使用 CLASSIX，请引用：
```bibtex
@article{CG24,
    title = {Fast and explainable clustering based on sorting},
    author = {Xinye Chen and Stefan Güttel},
    journal = {Pattern Recognition},
    volume = {150},
    pages = {110298},
    year = {2024},
    doi = {https://doi.org/10.1016/j.patcog.2024.110298}
}

@article{CGK26,
  title   = {{CLASSIX}: A High-Performance Library for Explainable Clustering},
  author = {Xinye Chen and Stefan Güttel and Kaustubh Roy},
  journal = {Journal of Machine Learning Research},
  year    = {2026},
  note    = {Accepted for publication}
}
```


如果使用 Tanimoto 和曼哈顿距离，请引用：
```bibtex
@misc{GK26,
    title={Fast and explainable clustering in the Manhattan and Tanimoto distance},
    author={Stefan Güttel and Kaustubh Roy},
    year={2026},
    eprint={2601.08781},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2601.08781},
}
```

## 许可证
本项目采用 [MIT 许可证](https://github.com/nla-group/classix/blob/master/LICENSE)。


<p align="left">
  <a>
    <img alt="CLASSIX" src="https://raw.githubusercontent.com/nla-group/classix/master/docs/source/images/nla_logo.jpg" width="240" />
  </a>
</p>
