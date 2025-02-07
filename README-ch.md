 [![es](https://img.shields.io/badge/lang-es-black.svg)](https://github.com/nla-group/classix/blob/master/README.md)
 [![ch](https://img.shields.io/badge/lang-ch-white.svg)](https://github.com/nla-group/classix/blob/master/README-ch.md)
  

<h1 align="center">
  CLASSIX: 快速且可解释的聚类算法
</h1>

[![发布](https://github.com/nla-group/classix/actions/workflows/package_release.yml/badge.svg?branch=master)](https://github.com/nla-group/classix/actions/workflows/package_release.yml)
[![!pypi](https://img.shields.io/pypi/v/classixclustering?color=red)](https://pypi.org/project/classixclustering/)
![Static Badge](https://img.shields.io/badge/Compiler-8A2BE2?label=Cython-Accelerated)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/classixclustering/badges/version.svg)](https://anaconda.org/conda-forge/classixclustering)
[![codecov](https://codecov.io/gh/nla-group/classix/branch/master/graph/badge.svg?token=D4MQZS67H1)](https://codecov.io/gh/nla-group/classix)
[![License: MIT](https://anaconda.org/conda-forge/classixclustering/badges/license.svg)](https://github.com/nla-group/classix/blob/master/LICENSE)
[![azure](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/classixclustering-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=15797&branchName=main)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/classixclustering.svg)](https://anaconda.org/conda-forge/classixclustering)
[![下载状态](https://static.pepy.tech/badge/classixclustering)](https://pypi.org/project/classixclustering/)
[![下载量](https://img.shields.io/pypi/dm/classixclustering.svg?label=PyPI%20downloads)](https://pypi.org/project/classixclustering/)
[![文档状态](https://readthedocs.org/projects/classix/badge/?version=stable)](https://classix.readthedocs.io/en/latest/?badge=stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10257432.svg)](https://doi.org/10.5281/zenodo.10257432)

__CLASSIX__ 是一种快速、内存高效、且可解释的聚类算法。主要特点如下：

- 高效处理任意形状的低维和高维数据
- 能够检测并处理数据中的异常值
- 提供聚类的文本和可视化解释
- 附带论文中的所有实验均可完全复现
- 支持 Cython 加速编译

__CLASSIX__ 是 *CLustering by Aggregation with Sorting-based Indexing* 和解释性 (*X* for explainability) 的组合词。

---

## 安装方法

您可以通过 PIP（推荐）或 Conda 安装 __CLASSIX__：

| PyPI | conda-forge |
| :---: |:---: |
|[![PyPI 版本](https://badge.fury.io/py/classixclustering.svg)](https://pypi.org/project/classixclustering/) | [![conda-forge 版本](https://anaconda.org/conda-forge/classixclustering/badges/version.svg)](https://anaconda.org/conda-forge/classixclustering) |
| NumPy<=1.26.4: `pip install classixclustering` <br/> NumPy>2: `pip install classixclustering --no-cache-dir `| `conda install -c conda-forge classixclustering` |

我们推荐使用命令 `pip install classixclustering --no-cache-dir` 安装。

<br/>

__语言支持__ | __依赖库支持__  
:---:|:---:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) |  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 

---

## 快速开始

以下示例展示如何使用 CLASSIX 对内置的演示数据集进行聚类：

```python
import classix

data, labels = classix.loadData('Covid3MC')
# 或者您可以生成高斯数据以快速体验：
# from sklearn.datasets import make_blobs
# data, labels = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# 使用 CLASSIX
clx = classix.CLASSIX(radius=0.2, minPts=500, verbose=0)
clx.fit(data)
print(clx.labels_) # 输出聚类标签
```
此外，您可以在模型训练后通过 predict() 方法对样本外数据进行聚类，例如 clx.predict(data.iloc[:1000])。



---

## 特性亮点

CLASSIX 提供以下主要功能：

1. **灵活的参数调整**：
   - `radius` 和 `minPts` 可自定义，用于调节聚类结果的颗粒度。
   - 适用于高维和低维数据。

2. **异常值检测**：
   - 自动识别并处理数据集中的异常值。

3. **高性能**：
   - 使用 Cython 优化，实现快速的运行时间。

4. **可解释性**：
   - 聚类结果提供文本描述和可视化。

5. **代码可复现性**：
   - 附带的实验和案例完全可复现。

---

## 使用案例

### 1. 聚类低维数据

以下示例展示了 CLASSIX 对 2D 数据的聚类效果：

```python
import classix
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=1000, noise=0.05)

clx = classix.CLASSIX(radius=0.15, minPts=10, verbose=0)
clx.fit(data)

# 可视化结果
plt.scatter(data[:, 0], data[:, 1], c=clx.labels_, cmap='viridis', s=5)
plt.show()
```

### 2. 聚类高维数据

以下示例展示了 CLASSIX 对高维数据的聚类能力：

```python
from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=5000, centers=5, n_features=20, random_state=0)

clx = classix.CLASSIX(radius=10, minPts=100, verbose=1)
clx.fit(data)
print(clx.labels_)
```

### 3. 检测异常值

CLASSIX 提供了异常值检测功能：

```python
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=0)

# 添加一些异常值
np.random.seed(0)
outliers = np.random.uniform(low=-10, high=10, size=(50, 2))
data_with_outliers = np.vstack([data, outliers])

clx = classix.CLASSIX(radius=0.5, minPts=10, verbose=1)
clx.fit(data_with_outliers)

# 标记异常值 (-1 表示异常值)
print(clx.labels_)
```

### 贡献方式

欢迎社区用户贡献代码和提交问题。以下是参与方式：

    克隆仓库：
```
git clone https://github.com/nla-group/classix.git
cd classix
```

创建虚拟环境并安装依赖项：
```
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install .
```
运行测试：
```
pytest unittests.py
```
我们期待您的参与！请阅读 贡献指南 以了解更多信息。
文档和支持

如果您有任何疑问或建议，请通过以下方式联系我们：

    在 GitHub Issues 提交问题。
    通过电子邮件与我们联系：stefan.guettel@manchester.ac.uk。


### 引用

如果您在研究中使用了 CLASSIX，请引用以下论文：
```
@article{CG24,
title = {Fast and explainable clustering based on sorting},
author = {Xinye Chen and Stefan Güttel},
journal = {Pattern Recognition},
volume = {150},
pages = {110298},
year = {2024},
doi = {https://doi.org/10.1016/j.patcog.2024.110298}
}
```
