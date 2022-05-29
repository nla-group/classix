# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2022 Stefan Güttel, Xinye Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings

import os
import re
import copy
import requests
import collections
import numpy as np
import pandas as pd
from numpy.linalg import norm
#import pyximport; pyximport.install()
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.sparse.linalg import svds
# from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, _sparsetools
# from .merging import minimum_spanning_tree_agglomerate 
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import shortest_path
np.random.seed(0)




            
def loadData(name='vdu_signals'):
    """Obtain the built-in data.
    
    Parameters
    ----------
    name: str, {'vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 'Banknote', 'Seeds', 'Phoneme', 'Wine'}, default='vdu_signals'
        The support build-in datasets for CLASSIX.

    Returns
    -------
    X, y: numpy.ndarray
        Data and ground-truth labels.    
    """
    
    current_dir, current_filename = os.path.split(__file__)
    
    if not os.path.isdir(os.path.join(current_dir, 'data')):
        os.mkdir(os.path.join(current_dir, 'data/'))
        
    if name == 'vdu_signals':
        DATA_PATH = os.path.join(current_dir, 'data/vdu_signals.npy')
        if not os.path.isfile(DATA_PATH):
            get_data(current_dir)
        return np.load(DATA_PATH)
    
    if name == 'Iris':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Irirs.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Irirs.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Iris')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Dermatology':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Dermatology.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Dermatology.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Dermatology')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Ecoli':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Ecoli.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Ecoli.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Ecoli')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Glass':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Glass.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Glass.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Glass')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Banknote':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Banknote.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Banknote.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Banknote')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Seeds':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Seeds.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Seeds.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Seeds')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Phoneme':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Phoneme.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Phoneme.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Phoneme')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Wine':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Wine.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Wine.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            get_data(current_dir, 'Wine')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name not in ['vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 'Banknote', 'Seeds', 'Phoneme', 'Wine']:
        warnings.warn("Currently not support this data.")


        

def get_data(current_dir='', name='vdu_signals'):
    """Download the built-in data."""
    if name == 'vdu_signals':
        url_parent = "https://github.com/nla-group/classix/raw/master/classix/source/vdu_signals.npy"
        vdu_signals = requests.get(url_parent).content
        with open(os.path.join(current_dir, 'data/vdu_signals.npy'), 'wb') as handler:
            handler.write(vdu_signals)
         
    elif name == 'Iris':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Irirs.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Irirs.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Irirs.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Irirs.npy'), 'wb') as handler:
            handler.write(y)
            
    elif name == 'Dermatology':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Dermatology.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Dermatology.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Dermatology.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Dermatology.npy'), 'wb') as handler:
            handler.write(y)
    
    elif name == 'Ecoli':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Ecoli.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Ecoli.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Ecoli.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Ecoli.npy'), 'wb') as handler:
            handler.write(y)
    
    elif name == 'Glass':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Glass.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Glass.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Glass.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Glass.npy'), 'wb') as handler:
            handler.write(y)
    
    elif name == 'Banknote':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Banknote.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Banknote.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Banknote.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Banknote.npy'), 'wb') as handler:
            handler.write(y)
    
    elif name == 'Seeds':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Seeds.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Seeds.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Seeds.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Seeds.npy'), 'wb') as handler:
            handler.write(y)
            
    elif name == 'Phoneme':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Phoneme.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Phoneme.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Phoneme.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Phoneme.npy'), 'wb') as handler:
            handler.write(y)
    
    elif name == 'Wine':
        url_parent_x = "https://github.com/nla-group/classix/raw/master/classix/source/X_Wine.npy"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Wine.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Wine.npy'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Wine.npy'), 'wb') as handler:
            handler.write(y)
                    
# ******************************************** the main wrapper ********************************************
class CLASSIX:
    """CLASSIX: Fast and explainable clustering based on sorting.
    
    The user only need to concern the hyperparameters of ``sorting``, ``radius``, and ``minPts`` in the most cases.
    If want a flexible clustering, might consider other hyperparameters such as ``group_merging``, ``scale``, and ``post_alloc``.
    
    Parameters
    ----------
    sorting : str, {'pca', 'norm-mean', 'norm-orthant', None}，default='pca'
        Sorting method used for the aggregation phase.
        
        - 'pca': sort data points by their first principal component
        
        - 'norm-mean': shift data to have zero mean and then sort by 2-norm values
        
        - 'norm-orthant': shift data to positive orthant and then sort by 2-norm values
        
        - None: aggregate the raw data without any sorting
        
    radius : float, default=0.5
        Tolerance to control the aggregation. If the distance between a starting point 
        and an object is less than or equal to the tolerance, the object will be allocated 
        to the group which the starting point belongs to. For details, we refer users to [1].
    
    group_merging : str, {'density', 'distance', None}, default='distance'
        The method for merging the groups. 
        
        - 'density': two groups are merged if the density of data points in their intersection 
           is at least as high the smaller density of both groups. This option uses the disjoint 
           set structure to speedup agglomerate.
        
        - 'distance': two groups are merged if the distance of their starting points is at 
           most scale*radius (the parameter above). This option uses the disjoint 
           set structure to speedup agglomerate.
        
        For more details, we refer to [1].
        If the users set group_merging to None, the clustering will only return the labels formed by aggregation as cluster labels.
    
    minPts : int, default=0
        Clusters with less than minPts points are classified as abnormal clusters.  
        The data points in an abnormal cluster will be redistributed to the nearest normal cluster. 
        When set to 0, no redistribution is performed. 
    
    norm : boolean, default=True
        If normalize the data associated with the sorting, default as True. 
        
    scale : float
        Design for distance-clustering, when distance between the two starting points 
        associated with two distinct groups smaller than scale*radius, then the two groups merge.

    post_alloc : boolean, default=True
        If allocate the outliers to the closest groups, hence the corresponding clusters. 
        If False, all outliers will be labeled as -1.
    

    verbose : boolean or int, default=1
        Whether print the logs or not.
             
             
    Attributes
    ----------
    groups_ : numpy.ndarray
        Groups labels of aggregation.
    
    splist_ : numpy.ndarray
        List of starting points formed in the aggregation.
        
    labels_ : numpy.ndarray
        Clustering class labels for data objects 

    group_outliers_ : numpy.ndarray
        Indices of outliers (aggregation groups level), 
        i.e., indices of abnormal groups within the clusters with fewer 
        data points than minPts points.
        
    clean_index_ : numpy.ndarray
        The data without outliers. Given data X,  the data without outliers 
        can be exported by X_clean = X[classix.clean_index_,:] while the outliers can be exported by 
        Outliers = X[~classix.clean_index_,:] 
        
    connected_pairs_ : list
        List for connected group labels.


    Methods:
    ----------
    fit(data):
        Cluster data while the parameters of the model will be saved. The labels can be extracted by calling ``self.labels_``.
        
    fit_transform(data):
        Cluster data and return labels. The labels can also be extracted by calling ``self.labels_``.
        
    predict(data):
        After clustering the in-sample data, predict the out-sample data.
        Data will be allocated to the clusters with the nearest starting point in the stage of aggregation. Default values.
        
    explain(index1, index2): 
        Explain the computed clustering. 
        The indices index1 and index2 are optional parameters (int) corresponding to the 
        indices of the data points. 
        
    References
    ----------
    [1] X. Chen and S. Güttel. Fast and explainable sorted based clustering, 2022
    """
        
    # deprecated descriptions
    # group_merging : str, {'density', 'distance', 'scc-distance', 'mst-distance'}, default='distance'
    #     The method for merging the groups. 
    #     - 'density': two groups are merged if the density of data points in their intersection 
    #        is at least as high the smaller density of both groups. This option uses the disjoint 
    #        set structure to speedup agglomerate.
    #     - 'distance': two groups are merged if the distance of their starting points is at 
    #        most scale*radius (the parameter above). This option uses the disjoint 
    #        set structure to speedup agglomerate.
    #     - 'scc-distance': calculate the distance-based merging by Tarjan's algorithm for finding
    #        strongly connected components.
    #     - 'mst-distance': it is also a distance based group_merging, but use minimum spanning tree instead
    #        in the second stage with cutoff_scale scale*radius.
    # n_jobs : int, default=-1
    #     This parameter only works for method called 'scc-distance', which refers to the
    #     number of jobs to use for the computation by breaking down the pairwise matrix into n_jobs 
    #     even slices and computing them in parallel.
        
    def __init__(self, sorting="pca", radius=0.5, minPts=0, group_merging="distance", norm=True, scale=1.5, post_alloc=True, verbose=1): 
        
        # deprecated parameter (15/07/2021): noise_percent=0, distance_scale=1, eta=1, cython=True
        # eta (deprecated): float, default=1.0
        #     the value for the density-based groups merging, the groups will 
        #     merge together if their starting points p1 and p2 satisy both:
        #     (1) "density in of intersection between p1 or p2 * eta < density in p1" and "density in of intersection between p1 and p2 * eta <     density in p2"
        #     (2) "distance between p1 and p1" < 2*radius (tolerance)
        # 
        # if distance_scale > 1 and group_merging == "density": # since there is no intersection.
        #     raise ValueError(
        #         "Please feed an correct value for distance_scale for density group merging.")
        # 
        # distance_scale (deprecated): float, default=1.0
        #     The value for the distance-based groups merging, the groups will 
        #     merge together if their starting points  distance is less than or equal to 2*distance_scale*radius (tolerance)
        #     The higher it is, the higher possibility the groups will merge together.
        #     *Note that for density group merging, distance_scale can not be higher than 1, otherwise there is no intersection between two groups.
        #   
        # noise_percent (deprecated): int, default=50
        #     The threshold, in the range of [0,100) to determine the noise degree.
        #     Decide 'noise_percent' percentile of the number of data in clusters as outliers threshold. 
        #     The intuition is that if the number of the objects in a cluster is very small, 
        #     the cluster is very likely to be a abnormal cluster.

        self.verbose = verbose
        self.minPts = minPts

        # if self.minPts == 0:
        #     if self.verbose:
        #         print("No anomaly detection setting.")
        # else:
        #     if self.verbose:
        #        print("Noises check with minPts:{:.6f}".format(minPts))

        self.sorting = sorting
        self.radius = radius
        self.group_merging = group_merging
        self.sp_to_c_info = False # combine with visualization and data analysis, ensure call PCA and form groups information table only once
        # self.cython = cython
        # self.eta = eta
        # self.distance_scale = distance_scale
        self.centers = None
        self.norm = norm # usually, we do not use this parameter
        self.scale = scale # For distance measure, usually, we do not use this parameter
        self.post_alloc = post_alloc
        # self.n_jobs = n_jobs

        self.groups_ = None
        self.clean_index_ = None
        self.labels_ = None
        self.connected_pairs_ = None
        self.cluster_color = None
        
        if self.verbose:
            print(self)
        
        self.splist_indices = [None]
        self.index_data = None
        
        from . import __enable_cython__
        self.__enable_cython__ = __enable_cython__
        
        if self.__enable_cython__:
            try:
                # %load_ext Cython
                # !python3 setup.py build_ext --inplace
                import scipy, numpy
                if scipy.__version__ == '1.8.0' or numpy.__version__ < '1.22.0':
                    from .aggregation_c import aggregate 
                    # cython without memory view, solve the error from scipy ``TypeError: type not understood``
                else:
                    from .aggregation_cm import aggregate
                    # cython with memory view
                from .merging_cm import fast_agglomerate

            except (ModuleNotFoundError, ValueError):
                from .aggregation import aggregate 
                from .merging import fast_agglomerate
                warnings.warn("This CLASSIX installation is not using Cython.")
        else:
            from .aggregation import aggregate 
            from .merging import fast_agglomerate
            warnings.warn("This run of CLASSIX is not using Cython.")

        self.aggregate = aggregate
        self.fast_agglomerate = fast_agglomerate
        
        
            
    def fit(self, data):
        """ 
        Cluster the data and return the associated cluster labels. 
        
        Parameters
        ----------
        data : numpy.ndarray
            The ndarray-like input of shape (n_samples,)
        
            
        """
        if isinstance(data, pd.core.frame.DataFrame):
            self.index_data = data.index
            
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            if len(data.shape) == 1:
                data = data.reshape(-1,1)
                
        if data.dtype !=  'float64':
            data = data.astype('float64')
            
        if self.sorting == "norm-mean":
            # self._mu, self._std = data.mean(axis=0), data.std()
            self._mu = data.mean(axis=0)
            self.data = data - self._mu
            self._scl = self.data.std()
            if self._scl == 0: # prevent zero-division
                self._scl = 1
            self.data = self.data / self._scl
        
        elif self.sorting == "pca":
            self._mu = data.mean(axis=0)
            self.data = data - self._mu # mean center
            rds = norm(self.data, axis=1) # distance of each data point from 0
            self._scl = np.median(rds) # 50% of data points are within that radius
            if self._scl == 0: # prevent zero-division
                self._scl = 1
            self.data = self.data / self._scl # now 50% of data are in unit ball 
            
        elif self.sorting == "norm-orthant":
            # self._mu, self._std = data.min(axis=0), data.std()
            self._mu = data.min(axis=0)
            self.data = data - self._mu
            self._scl = self.data.std()
            if self._scl == 0: # prevent zero-division
                self._scl = 1
            self.data = self.data / self._scl
            
        else:
            # self._mu, self._std = data.mean(axis=0), data.std(axis=0) # z-score
            self._mu, self._scl = 0, 1 # no normalization
            self.data = (data - self._mu) / self._scl
        
        # aggregation
        self.groups_, self.splist_, self.dist_nr = self.aggregate(data=self.data, sorting=self.sorting, tol=self.radius) 
        self.splist_ = np.array(self.splist_)
        
        self.clean_index_ = np.full(self.data.shape[0], True) # claim clean data indices
        # clustering
        
        if self.group_merging is None:
            self.labels_ = copy.deepcopy(self.groups_) 
        elif self.group_merging.lower()=='none':
            self.labels_ = copy.deepcopy(self.groups_) 
        else:
            self.labels_ = self.clustering(
                data=self.data,
                agg_labels=self.groups_, 
                splist=self.splist_,             
                sorting=self.sorting, 
                radius=self.radius, 
                method=self.group_merging, # eta=self.eta, distance_scale=self.distance_scale, 
                minPts=self.minPts # percent=self.noise_percent, noise_scale=self.noise_scale,
            ) 
        return self


        
    def fit_transform(self, data):
        """ 
        Cluster the data and return the associated cluster labels. 
        
        Parameters
        ----------
        data : numpy.ndarray
            The ndarray-like input of shape (n_samples,)
        
        Returns
        -------
        labels : numpy.ndarray
            Index of the cluster each sample belongs to.
            
        """
        
        return self.fit(data).labels_
        
        
        
    def predict(self, data):
        """
        Allocate the data to their nearest clusters.
        
        data : numpy.ndarray
            The ndarray-like input of shape (n_samples,)

        Returns
        -------
        labels : numpy.ndarray
            The predicted clustering labels.
        """
        # deprecated documentation (24/07/2021):    
        # 
        # method : str, {'centers', 'agg_centers', 'agg_sps'}, default='agg_centers'
        #     the way chosen for data been allocated to the corresponding way. 
        # 
        #     1) 'centers': data will be allocated to the clusters with the nearest center. 
        #     May lead to misallocation
        # 
        #     2) 'agg_centers': data will be allocated to the clusters with the nearest center
        #     in the stage of aggregation.  May lead to misallocation
        #     
        #     3) 'agg_sps': data will be allocated to the clusters with the nearest starting point 
        #     in the stage of aggregation. Default values (Recommended).
            
        labels = list()
        
        # if method == "agg_centers":
        #     agg_centers = self.calculate_group_centers(self.data, self.labels_)
        #     data = (data - self._mu) / self._std
        #     for i in range(len(data)):
        #         agglabel = np.argmin(np.linalg.norm(agg_centers[:,3:] - data[i], axis=1, ord=2))
        #         labels.append(self.label_change[agglabel])
                
        # elif method == "centers": # may lead to misallocation! avoid using it
        #     for i in range(len(data)):
        #         labels.append(np.argmin(np.linalg.norm(self.centers - data[i], axis=1, ord=2)))

        # elif method == "agg_sps": 
        
        data = (data - self._mu) / self._scl
        for i in range(len(data)):
            splabel = np.argmin(np.linalg.norm(data[self.splist_[:,0]] - data[i], axis=1, ord=2)) 
            #  np.argmin(np.linalg.norm(self.splist_[:,3:] - data[i], axis=1, ord=2))
            labels.append(self.label_change[splabel])

        # else:
        #     raise ValueError("Please refer to an correct value for 'group_merging'")
            
        return labels
    
    
    
    def clustering(self, data, agg_labels, splist, sorting="pca", radius=0.5, method="distance", minPts=0):
        """
        Merge groups after aggregation. 

        Parameters
        ----------
        data : numpy.ndarray
            The input that is array-like of shape (n_samples,).
        
        agg_labels: numpy.ndarray
            Groups labels of aggregation.
        
        splist: numpy.ndarray
            List formed in the aggregation storing starting points.
        
        sorting : str
            The sorting way refered for aggregation, 
            default='pca', other options: 'norm-mean', 'norm-orthant', 'z-pca', or None.
        
        radius : float, default=0.5
            Tolerance to control the aggregation hence the whole clustering process. For aggregation, 
            if the distance between a starting point and an object is less than or equal to the tolerance, 
            the object will be allocated to the group which the starting point belongs to. 
        
        method : str
            The method for groups merging, 
            default='distance', other options: 'density', 'mst-distance', and 'scc-distance'.

        minPts : int, default=0
            The threshold, in the range of [0, infity] to determine the noise degree.
            When assgin it 0, algorithm won't check noises.


        Returns
        -------
        centers : numpy.ndarray
            The return centers of clusters
        
        clabels : numpy.ndarray 
            The clusters labels of the data
        """
        
        # deprecated (24/07/2021):    
        # (optional)
        # eta : float, default=1.0
        #     the value for the density-based groups merging, the groups will 
        #     merge together if their starting points p1 and p2 satisy both:
        #    (1) "density in of intersection between p1 or p2 * eta < density in p1" and "density in of intersection between p1 and p2 * eta < density in p2"
        #    (2) "distance between p1 and p1" < 2*radius (tolerance)
        #
        # distance_scale : float, default=1.0
        #    the value for the distance-based groups merging, the groups will 
        #    merge together if their starting points  distance is less than or equal to 2*distance_scale*radius (tolerance)
        #    The higher it is, the higher possibility the groups will merge together.
        #    *Note that for density method, distance_scale can not be higher than 1, otherwise there is no intersection between two groups.
            
        # percent : int, default=50
        #     The threshold, in the range of [0,100) to determine the noise degree.
        #     Decide 'percent' percentile of the number of data in clusters as outliers threshold. 
        #     The intuition is that if the number of the objects in a cluster is very small, 
        #     the cluster is very likely to be a abnormal cluster.
        
        # try:
            # from .cagg import aggregate
            # # %load_ext Cython
            # !python3 setup.py build_ext --inplace
        #     from .cagg_memview import aggregate # memory view
        # except ModuleNotFoundError:
        #     from .agg import aggregate

        # print("aggregation initialize...")
        # if isinstance(data, pd.core.frame.DataFrame):
        #     data = data.values
        #     if len(data.values.shape) == 1:
        #         data = data.reshape(-1,1)
                
        # if sorting == "pca": # mean-shift normalization + pca sorting
            # data, parameters = self.normalize(data, shift='mean')
        #     labels, splist, self.dist_c = self.aggregate(data=data, sorting=sorting, tol=radius)
        # elif sorting == "norm-mean": # mean-shift normalization + norm sorting
            # data, parameters = self.normalize(data, shift='mean')
        #     labels, splist, self.dist_c = self.aggregate(data=data, sorting=sorting, tol=radius)
        # elif sorting == "norm-orthant": # min-shift normalization + norm sorting
            # data, parameters = self.normalize(data, shift='min')
        #     labels, splist, self.dist_c = self.aggregate(data=data, sorting=sorting, tol=radius)
        # elif sorting == "z-pca": # z-score normalization + pca sorting
            # data, parameters = self.normalize(data, shift='z-score')
        #    labels, splist, self.dist_c = self.aggregate(data=data, sorting="pca", tol=radius)
        # else: # z-score normalization without sorting
            # data, parameters = self.normalize(data, shift='z-score')
        #     labels, splist, self.dist_c = self.aggregate(data=data, sorting=sorting, tol=radius)     
        

        #print('splist:\n', splist)
        # labels = np.array(labels)
        labels = copy.deepcopy(agg_labels) 
        
        # calculate the overall volume of the combination of the clusters,
        # hence only compare their neighbor clusters
        # if method == "mst-distance":
        #     self.merge_groups = minimum_spanning_tree_agglomerate(splist, radius=radius, scale=self.scale)
        # 
        # elif method == 'scc-distance':
        #     self.merge_groups = scc_agglomerate(splist, radius=radius, scale=self.scale, n_jobs=self.n_jobs)
        
        # --Deprecated 
        # elif method == 'trivial-distance': # deprecated method: brute force
        #     self.connected_pairs_ = agglomerate_trivial(data, splist, radius, "distance", scale=self.scale)
            # self.check_labels = labels
            # reassign lalels, start from 0, since the cluster number not start with 0.

            # we employ an intutive and simple way to implement merging groups, resulting in a fast clustering
            # self.merge_groups = merge_pairs_dr(self.connected_pairs_)
        
        # elif method == 'trivial-density': # deprecated method: brute force
            # self.connected_pairs_ = agglomerate_trivial(data, splist, radius, "density", scale=self.scale)
            # self.check_labels = labels
            # reassign lalels, start from 0, since the cluster number not start with 0.

            # we employ an intutive and simple way to implement merging groups, resulting in a fast clustering
            # self.merge_groups = merge_pairs_dr(self.connected_pairs_)
        
        # else:
            # print("clusters merging initialize...")
            # self.merge_groups, self.connected_pairs_ = self.fast_agglomerate(data, splist, radius, method, scale=self.scale)
            # self.check_labels = labels
            # reassign lalels, start from 0, since the cluster number not start with 0.

            # we employ an intutive and simple way to implement merging groups, resulting in a fast clustering
            # self.merge_groups = merge_pairs(self.connected_pairs_)
        
        self.merge_groups, self.connected_pairs_ = self.fast_agglomerate(data, splist, radius, method, scale=self.scale)
        maxid = max(labels) + 1
        
        # after this step, the connected pairs (groups) will be transformed into merged clusters, 
        for sublabels in self.merge_groups: # some of aggregated groups might be independent which are not included in self.merge_groups
            # not labels[sublabels] = maxid !!!
            for j in sublabels:
                labels[labels == j] = maxid
            maxid = maxid + 1
        
        # but the existent clusters may have some very independent clusters which are possibly be "noise" clusters.
        # so the next step is extracting the clusters with very rare number of objects as potential "noises".
        # we calculate the percentiles of the number of clusters objects. For example, given the dataset size of 100,
        # there are 4 clusters, the associated number of objects inside clusters are repectively of 5, 20, 25, 50. 
        # The 10th percentlie (we set percent=10, noise_scale=0.1) of (5, 20, 25, 50) is 14, 
        # and we calculate threshold = 100 * noise_scale =  10. Obviously, the first cluster with number of objects 5
        # satisfies both condition 5 < 14 and 5 < 10, so we classify the objects inside first cluster as outlier.
        # And then we allocate the objects inside the outlier cluster into other closest cluster.
        # This method is quite effective at solving the noise arise from small tolerance (radius).
        

        self.old_cluster_count = collections.Counter(labels)
        
        if minPts >= 1:
            potential_noise_labels = self.outlier_filter(labels, min_samples=minPts) # calculate the min_samples directly
            SIZE_NOISE_LABELS = len(potential_noise_labels) 
            if SIZE_NOISE_LABELS == len(np.unique(labels)):
                warnings.warn(
                    "Setting of noise related parameters is not correct, degenerate to the method without noises dectection.", 
                DeprecationWarning)
            else:
                for i in np.unique(potential_noise_labels):
                    labels[labels == i] = maxid # marked as noises, 
                                                # the label number is not included in any of existing labels (maxid).
        else:
            potential_noise_labels = list()
            SIZE_NOISE_LABELS = 0
        # remove noise cluster, avoid connecting two separate to a single cluster
        #---------------------------------------------------------------------------------------------------------
        # if minPts >= 1:
        #     possible_noise_labels = self.noises_filter(labels, min_samples=minPts) # calculate the min_samples directly
        # else: 
        #     possible_noise_labels = self.noises_filter(labels, min_samples_rate=minPts) # calculate the min_samples as minPts*data.shape[0]
        #---------------------------------------------------------------------------------------------------------
        

        
        # label_return_noises = copy.deepcopy(labels) # all noises are marked as maxid, similar to DBSCAN return noises.
        # the label with the maxid is label marked noises

        # the following is centers calculation step, which is not our current concern
        # centers = np.zeros((0, data.shape[1]))
        
        # self.outliers = np.array([]) # abnormal groups from aggregation
        
        if SIZE_NOISE_LABELS > 0:
            # noise_id = max(labels)
            # pure_labels = [i for i in np.unique(labels) if i != maxid]
            # self.outliers = np.where(labels == maxid)[0]
            # self.ne_outliers = np.where(labels != maxid)[0]
            
            # assign outliers in terms of group level
            self.clean_index_ = labels != maxid
            agln = agg_labels[self.clean_index_]
            # agg_clean_index_ = np.unique(agln) # extract the unique number of aggregation groups.
            # agg_noise_index = np.unique(self.groups_[self.outliers]) # extract the unique number of aggregation groups.
            # print("aggregation clean index:", agg_clean_index_)
            self.label_change = dict(zip(agln, labels[self.clean_index_])) # how object change group to cluster.
            # print("label change:", self.label_change)
            # allocate the outliers to the corresponding closest cluster.
            
            self.group_outliers_ = np.unique(agg_labels[~self.clean_index_]) # abnormal groups
            unique_agln = np.unique(agln)
            splist_clean = splist[unique_agln]
            # splist_outliers = splist[self.group_outliers_] 
            if self.post_alloc:
                for nsp in self.group_outliers_:
                    alloc_class = np.argmin(np.linalg.norm(data[splist_clean[:, 0].astype(int)] - data[int(splist[nsp, 0])], axis=1, ord=2))
                    # alloc_class = np.argmin(np.linalg.norm(splist_clean[:, 3:] - splist[nsp, 3:], axis=1, ord=2))
                    labels[agg_labels == nsp] = self.label_change[unique_agln[alloc_class]]
            else:
                labels[np.isin(agg_labels, self.group_outliers_)] = -1
            # previous code works on the point level
            # ----------------------------------- old code 1-----------------------------------
            # self.outliers = np.where(labels == maxid)[0]
            # agln = agg_labels[labels != maxid]           
            # splist_clean = splist[np.unique(agln)]
            # 
            # self.label_change = dict(zip(agln, labels[labels != maxid])) # how object change group to cluster.
            # if self.outliers.size > 0:
            #     # marked outliers location, and allocate the outliers to their closest clusters.
            #     for i in self.outliers:
            #         alloc_class = np.argmin(np.linalg.norm(splist_clean[:,3:] - data[i], axis=1, ord=2))
            #         labels[i] = self.label_change[splist_clean[alloc_class, 1]]
            # -------------------------------------------------------------------------------- 
            
            # ----------------------------------- old code 2-----------------------------------
            # splist_clean = splist[agg_clean_index_]
            # print("splist_clean:", splist_clean)
            # if self.outliers.size > 0:
            #     # marked outliers location, and allocate the outliers to their closest clusters.
            #     for i in self.outliers:
            #         alloc_class = np.argmin(np.linalg.norm(splist_clean[:,3:] - data[i], axis=1, ord=2))
            #         # print("alloc_class:", alloc_class)
            #         # print("labels[i]:", labels[i], "change to:", self.label_change[splist_clean[alloc_class, 1]])
            #         labels[i] = self.label_change[splist_clean[alloc_class, 1]]
            # -------------------------------------------------------------------------------- 
            
        # else:
            # self.outliers = np.array([])
            # pure_labels = np.unique(labels)
           
            # noise_id = -1   
        
        # we ensure the centers calculation does not include noises.
        # for c in sorted(pure_labels):
        #    indc = np.argwhere(labels==c)
        #    center = np.mean(data[indc,:], axis=0)
        #    centers = np.r_[ centers, center ]
        
                
        labels = self.reassign_labels(labels) 
        self.label_change = dict(zip(agg_labels, labels)) # how object change from group to cluster.
        
        if self.verbose == 1:
            print("""The {datalen} data points were aggregated into {num_group} groups.""".format(datalen=len(data), num_group=splist.shape[0]))
            print("""In total {dist:.0f} comparisons were required ({avg:.2f} comparisons per data point). """.format(dist=self.dist_nr, avg=self.dist_nr/len(data)))
            print("""The {num_group} groups were merged into {c_size} clusters with the following sizes: """.format(
                num_group=splist.shape[0], c_size=len(self.old_cluster_count)))
            #print("Initialize clusters with size:")
            
            self.pprint_format(self.old_cluster_count)
            if self.minPts != 0 and SIZE_NOISE_LABELS > 0:
                print("As MinPts is {minPts}, the number of clusters has been further reduced to {r}.".format(
                    minPts=self.minPts, r=len(np.unique(labels))
                ))
                
            print("Try the .explain() method to explain the clustering.")

        # if self.verbose == 1:
        #     print("As minPts is 20, the clusters are compressed from {l} to {r}.".format(
        #         l=len(np.unique(agg_labels)), r=len(np.unique(labels))
        #     ))
            
        return labels #, centers
    
    
    
    def explain(self, index1=None, index2=None, showsplist=True, max_colwidth=None, replace_name=None, 
                plot=False, figsize=(11, 6), figstyle="ggplot", savefig=False, ind_color="k", ind_marker_size=150,
                sp_fcolor='tomato',  sp_alpha=0.05, sp_pad=0.5, sp_fontsize=None, sp_bbox=None, 
                dp_fcolor='bisque',  dp_alpha=0.6, dp_pad=2, dp_fontsize=None, dp_bbox=None,
                cmap='turbo', cmin=0.07, cmax=0.97, color='red', connect_color='green', alpha=0.5, 
                cline_width=0.5, axis='off', figname=None, fmt='pdf', *argv, **kwargs):
        """
        'self.explain(object/index) # prints an explanation for why a point object1 is in its cluster (or an outlier)
        'self.explain(object1/index1, object2/index2) # prints an explanation why object1 and object2 are either in the same or distinct clusters
        
        
        Here we unify the terminology:
            [-] data points
            [-] groups (made up of data points, formed by aggregation)
            [-] clusters (made up of groups)
        
        
        Parameters
        ----------
        index1 : int or numpy.ndarray, optional
            Input object1 [with index 'index1'] for explanation.
        
        index2 : int or numpy.ndarray, optional
            Input object2 [with index 'index2'] for explanation, and compare objects [with indices 'index1' and 'index2'].
        
        showsplist : boolean
            Determine if show the starting points information, which include the number of data points (NumPts), 
            corresponding clusters, and associated coordinates. This only applies to both index1 and index2 are "NULL".
            Default as True. 
        
        max_colwidth : int, optional
            Max width to truncate each column in characters. By default, no limit.
            
        replace_name : str or list, optional
            Replace the index with name. 
            * For example: as for indices 1 and 1300 we have 
            
            ``classix.explain(1, 1300, plot=False, figstyle="seaborn") # or classix.explain(obj1, obj4)``
            
            The data point 1 is in group 9 and the data point 1300 is in group 8, both of which were merged into cluster #0. 
            These two groups are connected via groups 9 -> 2 -> 8.
            * if we specify the replace name, then the output will be
            
            ``classix.explain(1, 1300, replace_name=["Peter Meyer", "Anna Fields"], figstyle="seaborn")``
            
            The data point Peter Meyer is in group 9 and the data point Anna Fields is in group 8, both of which were merged into cluster #0. 
            These two groups are connected via groups 9 -> 2 -> 8.

        plot : boolean, default=False
            Determine if visulize the explaination. 
        
        figsize : tuple, default=(10, 6)
            Determine the size of visualization figure. 

        figstyle : str, default="ggplot"
            Determine the style of visualization.
            see reference: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        
        savefig : boolean, default=False
            Determine if save figure, the figure will be saved in the folder named "img".
        
        indices_color : str, default as 'k'
            Color for visualization of data with indices index1 and index2.
        
        ind_marker_size : float, optional:
            Size for visualization of data with indices index1 and index2.
    
        sp_fcolor : str, default='tomato'
            The color marked for starting points text box. 
            
        sp_alpha : float, default=0.3
            The value setting for transprency of text box for starting points. 
            
        sp_pad : int, default=2
            The size of text box for starting points. 
        
        sp_bbox : dict, optional
            Dict with properties for patches.FancyBboxPatch for starting points.
    
        sp_fontsize : int, optional
            The fontsize for text marked for starting points. 
    
        dp_fcolor : str, default='bisque'
            The color marked for specified data objects text box. 
            
        dp_alpha : float, default=0.3
            The value setting for transprency of text box for specified data objects. 
            
        dp_pad : int, default=2
            The size of text box for specified data objects. 
            
        dp_fontsize : int, optional
            The fontsize for text marked for specified data objects.    
                      
        dp_bbox : dict, optional
            Dict with properties for patches.FancyBboxPatch for specified data objects.
        
        cmap : str, default='turbo'
            The colormap to be employed.
        
        cmin : int or float, default=0.07
            The minimum color range.
         
        cmax : int or float, default=0.97
            The maximum color range.
            
        color : str, default='red'
            Color for text of starting points labels in visualization. 
        
        alpha : float, default=0.5
            Scalar or None. 
    
        cline_width : float, default=0.5
            Set the patch linewidth of circle for starting points.
        
        figname : str, optional
            Set the figure name for the image to be saved.
            
        fmt : str
            Specify the format of the image to be saved, default as 'pdf', other choice: png.
        
        """
        
        # deprecated (24/07/2021)
        # cols = ["NrPts"] 
        # for i in range(self.splist_.shape[1] - 2):
        #     cols = cols + ["feat." + str(i + 1)]
        
        # deprecated (24/07/2021)
        # -----------------------------alternative method--------------------------------
        # if self.splist_.shape[1] - 2 !=2:
        #     pca = PCA(n_components=2)
        #     x_pca = pca.fit_transform(self.splist_[:, 3:])
        #     spdisplay = pd.DataFrame(np.hstack((self.splist_[:, 0:2], x_pca)), columns=cols)
        # else:
        #     spdisplay = pd.DataFrame(self.splist_[:, 1:], columns=cols)
        
        # -----------------------------second method--------------------------------
        if sp_bbox is None:
            sp_bbox = dict()
            sp_bbox['facecolor'] = sp_fcolor
            sp_bbox['alpha'] = sp_alpha
            sp_bbox['pad'] = sp_pad
        # else: sp_bbox = {'facecolor': 'tomato', 'alpha': 0.2, 'pad': 2}
        
        if dp_bbox is None:
            dp_bbox = dict()
            dp_bbox['facecolor'] = dp_fcolor
            dp_bbox['alpha'] = dp_alpha
            dp_bbox['pad'] = dp_pad
        # else: dp_bbox = {'facecolor': 'bisque', 'alpha': 0.2, 'pad': 2}
            
        if dp_fontsize is None and sp_fontsize is not None:
            dp_fontsize = sp_fontsize
        
        # if self.cluster_color is None:
        _cmap = plt.cm.get_cmap(cmap)
        _interval = np.linspace(cmin, cmax, len(set(self.labels_))) 
        self.cluster_color = list()
        for c in _interval:
            rgba = _cmap(c) 
            color_hex = colors.rgb2hex(rgba) 
            self.cluster_color.append(str(color_hex)) 
                
        # if self.cluster_color is None:
        #     self.cluster_color = dict()
        #     for i in np.unique(self.labels_):
        #         self.cluster_color[i] = '#%06X' % np.random.randint(0, 0xFFFFFF)
                
        if not self.sp_to_c_info: #  ensure call PCA and form groups information table only once
            
            if self.data.shape[1] > 2:
                warnings.warn("The group radius in the visualization might not be accurate.")
                # self.pca = PCA(n_components=2)
                # self.x_pca = self.pca.fit_transform(self.data)
                # self.s_pca = self.pca.transform(self.data[self.splist_[:, 0].astype(int)])
                scaled_data = self.data - self.data.mean(axis=0)
                _U, _s, self._V = svds(scaled_data, k=2, return_singular_vectors="u")
                self.x_pca = np.matmul(scaled_data, self._V[np.argsort(_s)].T)
                self.s_pca = self.x_pca[self.splist_[:, 0].astype(int)]
                
            elif self.data.shape[1] == 2:
                self.x_pca = self.data.copy()
                self.s_pca = self.data[self.splist_[:, 0].astype(int)] # self.splist_[:, 3:].copy()

            else: # when data is one-dimensional, no PCA transform
                self.x_pca = np.ones((len(self.data.copy()), 2))
                self.x_pca[:, 0] = self.data[:, 0]
                self.s_pca = np.ones((len(self.splist_), 2))
                self.s_pca[:, 0] = self.data[self.splist_[:, 0].astype(int)].reshape(-1) # self.splist_[:, 2]
                
                # remove (24/07/2021):
                # print("This function is restricted to multidimensional (dimension greater than or equal to 2) data.")
            
            self.form_starting_point_clusters_table()
            
        if index1 is None and index2 is not None:
            raise ValueError("Please enter a valid value for index1.")
        
        
        # pd.options.display.max_colwidth = colwidth
        dash_line = "--------"*5 # "--------"*(self.splist_.shape[1])
            
        indexlist = [i for i in kwargs.keys() if 'index' in re.split('(\d+)',i)]

        
        if index1 is None: # analyze in the general way with a global view
            if plot == True:
                self.explain_viz(figsize=figsize, figstyle=figstyle, savefig=savefig, fontsize=sp_fontsize, bbox=sp_bbox, axis=axis, fmt=fmt)
                
            data_size = self.data.shape[0]
            feat_dim = self.data.shape[1]
            
            print("""A clustering of {length:.0f} data points with {dim:.0f} features has been performed. """.format(length=data_size, dim=feat_dim))
            print("""The radius parameter was set to {tol:.2f} and MinPts was set to {minPts:.0f}. """.format(tol=self.radius, minPts=self.minPts))
            print("""As the provided data has been scaled by a factor of 1/{scl:.2f},\ndata points within a radius of R={tol:.2f}*{scl:.2f}={tolscl:.2f} were aggregated into groups. """.format(
                scl=self._scl, tol=self.radius, tolscl=self._scl*self.radius
            ))
            print("""In total {dist:.0f} comparisons were required ({avg:.2f} comparisons per data point). """.format(dist=self.dist_nr, avg=self.dist_nr/data_size))
            print("""This resulted in {groups:.0f} groups, each uniquely associated with a starting point. """.format(groups=self.splist_.shape[0]))
            print("""These {groups:.0f} groups were subsequently merged into {num_clusters:.0f} clusters. """.format(groups=self.splist_.shape[0], num_clusters=len(np.unique(self.labels_))))
            
            if showsplist:
                print("""A list of all starting points is shown below.""")
                print(dash_line)
                print(self.sp_info.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                print(dash_line)       
            else:
                print("""In order to see a visual representation of the clustered data, use .explain(plot=True). """)
                
            print("""In order to explain the clustering of individual data points, \n"""
                  """use .explain(ind1) or .explain(ind1, ind2) with indices of the data points.""")
            
        else: # index is not None, explain(index1)
            if isinstance(index1, int):
                object1 = self.x_pca[index1] # self.data has been normalized
                agg_label1 = self.groups_[index1] # get the group index for object1
            
            elif isinstance(index1, str):
                if index1 in self.index_data:
                    if len(set(self.index_data)) != len(self.index_data):
                        warnings.warn("The index of data is duplicate.")
                        object1 = self.x_pca[np.where(self.index_data == index1)[0]][0]
                        agg_label1 = self.groups_[np.where(self.index_data == index1)[0][0]]
                    else:
                        object1 = self.x_pca[self.index_data == index1][0]
                        agg_label1 = self.groups_[self.index_data == index1][0]
                        
                else:
                    raise ValueError("Please enter a legal value for index1.")
                    
            elif isinstance(index1, list) or isinstance(index1, np.ndarray):
                index1 = np.array(index1)
                object1 = (index1 - self._mu) / self._scl # allow for out-sample data
                agg_label1 = np.argmin(np.linalg.norm(self.s_pca - object1, axis=1, ord=2)) # get the group index for object1
                if self.data.shape[1] >= 2:
                    object1 = self.pca.transform(object1)
                    
            else:
                raise ValueError("Please enter a legal value for index1.")
                
            plt.style.use('default')
            plt.style.use(style=figstyle)
            
            
            # explain one object
            # cluster_centers = self.calculate_group_centers(self.data, self.labels_)
            # print("Starting point list of {} data:".format(len(self.groups_)))

            if index2 is None:
                if replace_name is not None:
                    if isinstance(replace_name, list):
                        index1 = replace_name[0]
                    else:
                        index1 = replace_name
                else:
                    index1 = index1

                cluster_label1 = self.label_change[agg_label1]
                sp_str = self.s_pca[agg_label1] # self.splist_[agg_label1, 3:]
                
                if plot == True:
                    plt.style.use(style=figstyle)
                    fig, ax = plt.subplots(figsize=figsize)
                    plt.rcParams['axes.facecolor'] = 'white'
                    # select indices
                    x_pca = self.x_pca[self.labels_ == cluster_label1]
                    s_pca = self.s_pca[self.sp_info.Cluster == cluster_label1]
                    
                    ax.scatter(x_pca[:, 0], x_pca[:, 1], marker=".", c=self.cluster_color[cluster_label1])
                    # ax.scatter(x_pca[:, 0], x_pca[:, 1], marker="*", c='pink')
                    ax.scatter(s_pca[:, 0], s_pca[:, 1], marker="p")
                    
                    
                    if dp_fontsize is None:
                        ax.text(object1[0], object1[1], s=str(index1), bbox=dp_bbox, color=ind_color)
                    else:
                        ax.text(object1[0], object1[1], s=str(index1), fontsize=dp_fontsize, bbox=dp_bbox, color=ind_color)
                    
                    ax.scatter(object1[0], object1[1], marker="*", s=ind_marker_size)
                    
                    for i in range(s_pca.shape[0]):
                        ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                        ax.set_aspect('equal', adjustable='datalim')
                        if sp_fontsize is None:
                            ax.text(s_pca[i, 0], s_pca[i, 1],
                                    s=str(self.sp_info.Group[self.sp_info.Cluster == cluster_label1].astype(int).values[i]),
                                    # self.splist_[self.sp_info.Cluster == cluster_label1, 1][i].astype(int).astype(str), 
                                    bbox=sp_bbox
                            )
                        else:
                            ax.text(s_pca[i, 0], s_pca[i, 1],
                                    s=str(self.sp_info.Group[self.sp_info.Cluster == cluster_label1].astype(int).values[i]),
                                    # self.splist_[self.sp_info.Cluster == cluster_label1, 1][i].astype(int).astype(str), 
                                    fontsize=sp_fontsize, bbox=sp_bbox
                            )
                    ax.plot()
                    ax.axis('off') # the axis here may not be consistent, so hide.
                    
                    if savefig:
                        if not os.path.exists("img"):
                            os.mkdir("img")
                        if fmt == 'pdf':
                            fm = 'img/' + str(figname) + str(index1) +'.pdf'
                            plt.savefig(fm, bbox_inches='tight')
                        elif fmt == 'png':
                            fm = 'img/' + str(figname) + str(index1) +'.png'
                            plt.savefig(fm, bbox_inches='tight')
                        else:
                            fm = 'img/' + str(figname) + str(index1) +'.'+fmt
                            plt.savefig(fm, bbox_inches='tight')
                        
                        print("image successfully save as", fm)
                    
                    plt.show()
                    
                    # deprecated (24/07/2021):
                    # if self.data.shape[1] != 2:
                    #     pca = PCA(n_components=2)
                    #     x_pca = pca.fit_transform(self.data)
                    #     s_pca = pca.transform(self.splist_[:, 3:])
                    #     
                    #     # select indices
                    #     x_pca = x_pca[self.labels_ == cluster_label1]
                    #     s_pca = s_pca[self.sp_info.Cluster == cluster_label1]
                    #     
                    #     ax.scatter(x_pca[:, 0], x_pca[:, 1],
                    #                 marker="*", c=self.labels_[self.labels_ == cluster_label1])
                    #     ax.scatter(s_pca[:,0], s_pca[:,1], marker="p")
                    #     
                    #     ax.scatter(object1[0],  object1[1], marker="*", s=ind_marker_size)
                    #     ax.text(object1[0], object1[1], s=str(index1), color=ind_color)
                    #     
                    #     for i in range(s_pca.shape[0]):
                    #         ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                    #         ax.set_aspect('equal', adjustable='datalim')
                    #         ax.text(s_pca[i, 0], s_pca[i, 1],
                    #                  s=self.splist_[self.sp_info.Cluster == cluster_label1,1][i].astype(int).astype(str), 
                    #                  bbox=sp_bbox)
                    #     ax.plot()
                    #     if savefig:
                    #         if not os.path.exists("img"):
                    #             os.mkdir("img")
                    #             plt.savefig('img/ind1.pdf')
                    #             print("successfully save")
                    #     plt.show()
                        
                    # elif self.data.shape[1] == 2:
                    #     x_pca = self.data.copy()
                    #     s_pca = self.splist_[:, 3:].copy()
                    #     
                    #     # select indices
                    #     x_pca = x_pca[self.labels_ == cluster_label1]
                    #     s_pca = s_pca[self.sp_info.Cluster == cluster_label1]
                    #     
                    #     ax.scatter(x_pca[:,0], x_pca[:,1], 
                    #                 marker="*", c=self.labels_[self.labels_ == cluster_label1])
                    #     ax.scatter(s_pca[:, 0], s_pca[:, 1], marker="p")
                    #     
                    #     ax.scatter(object1[0],  object1[1], marker="*", s=ind_marker_size)
                    #     ax.text(object1[0], object1[1], s=str(index1), color=ind_color)
                    #     
                    #     for i in range(s_pca.shape[0]):
                    #         ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                    #         ax.set_aspect('equal', adjustable='datalim')
                    #         plt.text(s_pca[i, 0], s_pca[i, 1],
                    #                  s=self.splist_[self.sp_info.Cluster == cluster_label1,1][i].astype(int).astype(str), 
                    #                  bbox=sp_bbox)
                    #     ax.plot() 
                    #     if savefig:
                    #         if not os.path.exists("img"):
                    #             os.mkdir("img")
                    #             plt.savefig('img/ind1.pdf')
                    #             print("successfully save")
                    #     plt.show()
                
                if showsplist:
                    # print(f"A list of information for {index1} is shown below.")
                    select_sp_info = self.sp_info.iloc[[agg_label1]].copy(deep=True)
                    select_sp_info.loc[:, 'Label'] = index1
                    print(dash_line)
                    print(select_sp_info.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                    print(dash_line)       

                print(
                    """The data point %(index1)s is in group %(agg_id)i, which has been merged into cluster #%(m_c)i."""% {
                        "index1":index1, "agg_id":agg_label1, "m_c":cluster_label1
                    }
                )

                #  print("then assigned to cluster %(m_c)i." % {
                #      "m_c":cluster_label1
                # })

                # if showsplist:
                #     print(dash_line)
                #     spdisplay = self.sp_info.iloc[[agg_label1],:]
                #     spdisplay["Data points"] = [str(index1)]
                #     spdisplay = spdisplay[["Data points", "Group", "NrPts", "Cluster", "Coordinates"]] 
                #     print(spdisplay.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                #     print(dash_line)
                

            # explain two objects relationship
            
            else: # explain(index1, index2, ...)

                if len(indexlist) > 0: # A more general case, index1=.., index2=.., index3=..
                    kwargs['index1'] = index1
                    kwargs['index2'] = index2
                    indexlist = ['index1', 'index2'] + indexlist
                    group_labels_m = np.array([int(self.groups_[kwargs[i]]) for i in indexlist])
                    cluster_labels_m = np.array([self.label_change[i] for i in group_labels_m])
 
                    # group_ind = np.zeros(self.data.shape[0])
                    # group_ind[list(set(group_labels_m))] = 1

                    plt.style.use(style=figstyle)
                    fig, ax = plt.subplots(figsize=figsize)
                    plt.rcParams['axes.facecolor'] = 'white'
                    
                    for i in set(cluster_labels_m):
                        x_pca = self.x_pca[self.labels_ == i, :]
                        ax.scatter(x_pca[:, 0], x_pca[:, 1], marker=".", c=self.cluster_color[i])
                    
                    for i in set(group_labels_m):
                        s_pca = self.s_pca[i]
                        ax.scatter(s_pca[0], s_pca[1], marker="p")
                        ax.add_patch(plt.Circle((s_pca[0], s_pca[1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                        
                    if dp_fontsize is None:
                        for index in indexlist:
                            _index = kwargs[index]
                            if isinstance(_index, int):
                                _object = self.x_pca[_index] # self.data has been normalized

                            elif isinstance(_index, str):
                                if _index in self.index_data:
                                    if len(set(self.index_data)) != len(self.index_data):
                                        warnings.warn("The index of data is duplicate.")
                                        _object = self.x_pca[np.where(self.index_data == _index)[0][0]][0]
                                    else:
                                        _object = self.x_pca[self.index_data == _index][0]
                                else:
                                    raise ValueError("Please enter a legal value for index.")

                            elif isinstance(_index, list) or isinstance(_index, np.ndarray):
                                _index = np.array(_index)
                                _object = (_index - self._mu) / self._scl # allow for out-sample data
                                if self.data.shape[1] >= 2:
                                    _object = self.pca.transform(_object)
                            else:
                                raise ValueError("Please enter a legal value for index.")
                        
                            # _object = self.x_pca[index]
                            ax.text(_object[0], _object[1], s=str(_index), bbox=dp_bbox, color=ind_color)
                            ax.scatter(_object[0], _object[1], marker="*", s=ind_marker_size)
                    else:
                        for index in indexlist:
                            _index = kwargs[index]
                            if isinstance(_index, int):
                                _object = self.x_pca[_index] # self.data has been normalized

                            elif isinstance(_index, str):
                                if _index in self.index_data:
                                    if len(set(self.index_data)) != len(self.index_data):
                                        warnings.warn("The index of data is duplicate.")
                                        _object = self.x_pca[np.where(self.index_data == _index)[0][0]][0]
                                    else:
                                        _object = self.x_pca[self.index_data == _index][0]
                                else:
                                    raise ValueError("Please enter a legal value for index.")

                            elif isinstance(_index, list) or isinstance(_index, np.ndarray):
                                _index = np.array(_index)
                                _object = (_index - self._mu) / self._scl # allow for out-sample data
                                if self.data.shape[1] >= 2:
                                    _object = self.pca.transform(_object)
                            else:
                                raise ValueError("Please enter a legal value for index.")
                                
                            ax.text(_object[0], _object[1], s=str(_index), fontsize=dp_fontsize, bbox=dp_bbox, color=ind_color)
                            ax.scatter(_object[0], _object[1], marker="*", s=ind_marker_size)
                    

                    ax.axis('off') # the axis here may not be consistent, so hide.
                    ax.plot()
                    if savefig:
                        if not os.path.exists("img"):
                            os.mkdir("img")
                        if fmt == 'pdf':
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'_m.pdf'
                            plt.savefig(fm, bbox_inches='tight')
                        elif fmt == 'png':
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'_m.png'
                            plt.savefig(fm, bbox_inches='tight')
                        else:
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'_m.'+fmt
                            plt.savefig(fm, bbox_inches='tight')
                        
                        print("image successfully save as ",fm)
                    
                    plt.show()
                    return 
                
                if isinstance(index2, int):
                    object2 = self.x_pca[index2] # self.data has been normalized
                    agg_label2 = self.groups_[index2] # get the group index for object2
                    
                elif isinstance(index2, str):
                    if index2 in self.index_data:
                        if len(set(self.index_data)) != len(self.index_data):
                            warnings.warn("The index of data is duplicate.")
                            object2 = self.x_pca[np.where(self.index_data == index2)[0][0]][0]
                            agg_label2 = self.groups_[np.where(self.index_data == index2)[0][0]]
                        else:
                            object2 = self.x_pca[self.index_data == index2][0]
                            agg_label2 = self.groups_[self.index_data == index2][0]
                    else:
                        raise ValueError("Please enter a legal value for index2.")
                        
                elif isinstance(index2, list) or isinstance(index2, np.ndarray):
                    index2 = np.array(index2)
                    object2 = (index2 - self._mu) / self._scl # allow for out-sample data
                    if self.data.shape[1] >= 2:
                        object2 = self.pca.transform(object2)
                    agg_label2 = np.argmin(np.linalg.norm(self.s_pca - object2, axis=1, ord=2)) # get the group index for object2
                
                else:
                    raise ValueError("Please enter a legal value for index2.")

                if showsplist:
                    # print(f"A list of information for {index1} is shown below.")
                    select_sp_info = self.sp_info.iloc[[agg_label1, agg_label2]].copy(deep=True)
                    select_sp_info.loc[:, 'Label'] = [index1, index2]
                    print(dash_line)
                    print(select_sp_info.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                    print(dash_line)       

                if replace_name is not None:
                    if isinstance(replace_name, list) or isinstance(replace_name, np.ndarray):
                        try:
                            index1 = replace_name[0]
                            index2 = replace_name[1]
                        except:
                            index1 = replace_name[0]
                            
                else:
                    index1 = index1
                    index2 = index2

                cluster_label1, cluster_label2 = self.label_change[agg_label1], self.label_change[agg_label2]

                connected_groups = {}
                for subs in self.merge_groups:
                    if agg_label1 in subs and agg_label2 in subs:
                        connected_groups["object 1"] = connected_groups["object 2"] = np.array(subs, dtype=int)
                    elif agg_label1 in subs:
                        connected_groups["object 1"] = np.array(subs, dtype=int)
                    elif agg_label2 in subs:
                        connected_groups["object 2"] = np.array(subs, dtype=int)
               
                
                if agg_label1 == agg_label2: # when ind1 & ind2 are in the same group
                    sp_str = np.array2string(self.splist_[agg_label1, 3:], separator=',')
                    print("The data points %(index1)s and %(index2)s are in the same group %(agg_id)i, hence were merged into the same cluster #%(m_c)i"%{
                        "index1":index1, "index2":index2, "agg_id":agg_label1, "m_c":cluster_label1}
                    )
                    connected_paths = [agg_label1]
                else:
                    sp1_str = self.s_pca[agg_label1] # self.splist_[agg_label1, 3:]
                    sp2_str = self.s_pca[agg_label2] # self.splist_[agg_label2, 3:]
                    # print("""The two objects are assigned to different groups through aggregation.""")

                    # print(
                    # """The first object is aggregated to group %(agg_id)i associated with starting point %(sp)s with respect to radius %(radius)s, then assigned to cluster label %(m_c)i.""" % {
                    #     "agg_id":agg_label1, "sp":sp1_str, "radius":self.radius, "m_c":cluster_label1
                    # })

                    # print(
                    # """The second object is aggregated to group %(agg_id)i associated with starting point %(sp)s with respect to radius %(radius)s, then assigned to cluster label %(m_c)i.""" % {
                    #     "agg_id":agg_label2, "sp":sp2_str, "radius":self.radius, "m_c":cluster_label2
                    # })
                    
                    if self.connected_pairs_ is None:
                        # distm = pairwise_distances(self.splist_[:,3:], Y=None, metric='euclidean', n_jobs=n_jobs)
                        distm = pairwise_distances(self.s_pca, Y=None, metric='euclidean', n_jobs=n_jobs)
                        distm = (distm <= radius*scale).astype(int)
                        self.connected_pairs_ = return_csr_matrix_indices(csr_matrix(distm)).tolist() # list
                        
                    if cluster_label1 == cluster_label2: # when ind1 & ind2 are in the same cluster but diff group
                        # deprecated (24/07/2021)  from scipy.sparse.csgraph import shortest_path -> Dijkstra’s algorithm 
                        # path_graph = pairs_to_graph(self.connected_pairs_, N=self.splist_.shape[0], sparse=True)
                        ## apply Dijkstra’s algorithm with Fibonacci heaps for shortest path finding
                        # dist_matrix, predecessors = shortest_path(csgraph=path_graph, method="D", directed=False, return_predecessors=True) 
                        # connected_paths = get_shortest_path(predecessors, agg_label1, agg_label2)
                        connected_paths = find_shortest_path(agg_label1,
                                                             self.connected_pairs_,
                                                             self.splist_.shape[0],
                                                             agg_label2
                        )
                        
                        # -- Deprecated
                        # if agg_label1 != 0:
                        #     connected_paths_vis = " -> ".join([str(group) for group in connected_paths]) 
                        # else:
                        #     connected_paths_vis = " -> ".join(["0"] + [str(group) for group in connected_paths]) 
                        
                        connected_paths_vis = " <-> ".join([str(group) for group in connected_paths]) 
                        print(
                        """The data point %(index1)s is in group %(agg_id1)s and the data point %(index2)s is in group %(agg_id2)s, """
                            """\nboth of which were merged into cluster #%(cluster)i. """% {
                            "index1":index1, "index2":index2, "cluster":cluster_label1, "agg_id1":agg_label1, "agg_id2":agg_label2}
                        )
                        
                        print("""These two groups are connected via groups %(connected)s.""" % {
                               "connected":connected_paths_vis}
                        )
                    else: 
                        # alternative expression 1
                        # print("The two objects are allocated to two different cluster.")
                        # print(
                        # """The first object is allocated to the cluster %(c_id1)s via %(connected1)s with cluster center %(cen1)s
                        # while the second object is allocated to the cluster %(c_id2)s via %(connected2)s with cluster center %(cen2)s""" % {
                        # "c_id1":cluster_label1, "connected1": connected_groups["object 1"], "cen1":cen1_str,
                        # "c_id2":cluster_label2, "connected2": connected_groups["object 2"], "cen1":cen2_str
                        #     }
                        # )
                        # print("The two objects are allocated to two different cluster.")
                        
                        # alternative expression 2
                        # print("""The data point %(index1)s is in group %(agg_id1)i, which has been merged into cluster %(c_id1)s via connected groups %(connected1)s.""" % {
                        #     "index1":index1, "agg_id1":agg_label1, "c_id1":cluster_label1, "connected1": connected_groups["object 1"]
                        # })

                        # print("""\nThe data point %(index2)s is in group %(agg_id2)i, which has been merged into cluster %(c_id2)s via connected groups %(connected2)s.""" % {
                        #     "index2":index2, "agg_id2":agg_label2, "c_id2":cluster_label2, "connected2": connected_groups["object 2"]
                        # })
                        
                        connected_paths = []
                        print("""The data point %(index1)s is in group %(agg_id1)i, which has been merged into cluster %(c_id1)s.""" % {
                            "index1":index1, "agg_id1":agg_label1, "c_id1":cluster_label1})

                        print("""The data point %(index2)s is in group %(agg_id2)i, which has been merged into cluster %(c_id2)s.""" % {
                            "index2":index2, "agg_id2":agg_label2, "c_id2":cluster_label2})   
                        
                        print("""There is no path of overlapping groups between these clusters.""")
                        
                if plot == True:
                    plt.style.use(style=figstyle)
                    fig, ax = plt.subplots(figsize=figsize)
                    plt.rcParams['axes.facecolor'] = 'white'
                    # select indices
                    union_ind = np.where((self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2))[0]
                    x_pca1 = self.x_pca[self.labels_ == cluster_label1]
                    x_pca2 = self.x_pca[self.labels_ == cluster_label2]
                    s_pca = self.s_pca[union_ind]
                    # self.s_pca[]
                    
                    ax.scatter(x_pca1[:, 0], x_pca1[:, 1], marker=".", c=self.cluster_color[cluster_label1])
                    ax.scatter(x_pca2[:, 0], x_pca2[:, 1], marker=".", c=self.cluster_color[cluster_label2])
                    
                    # ax.scatter(x_pca1[:, 0], x_pca1[:, 1],
                    #             marker="*", c='lightgreen')
                    # ax.scatter(x_pca2[:, 0], x_pca2[:, 1],
                    #             marker="*", c='lightblue')
                    
                    ax.scatter(s_pca[:,0], s_pca[:,1], marker="p")
                    
                    if dp_fontsize is None:
                        ax.text(object1[0], object1[1], s=str(index1), bbox=dp_bbox, color=ind_color)
                        ax.text(object2[0], object2[1], s=str(index2), bbox=dp_bbox, color=ind_color)
                    else:
                        ax.text(object1[0], object1[1], s=str(index1), fontsize=dp_fontsize, bbox=dp_bbox, color=ind_color)
                        ax.text(object2[0], object2[1], s=str(index2), fontsize=dp_fontsize, bbox=dp_bbox, color=ind_color)
                    
                    ax.scatter(object1[0], object1[1], marker="*", s=ind_marker_size)
                    ax.scatter(object2[0], object2[1], marker="*", s=ind_marker_size)
                    
                    for i in range(s_pca.shape[0]):
                        if union_ind[i] in connected_paths:
                            ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=connect_color, alpha=alpha, lw=cline_width, clip_on=False))
                        else:
                            ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                        ax.set_aspect('equal', adjustable='datalim')
                        if sp_fontsize is None:
                            ax.text(s_pca[i, 0], s_pca[i, 1], 
                                    s=self.sp_info.Group[(self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)].values[i].astype(int).astype(str),
                                    bbox=sp_bbox
                            )
                        else:
                            ax.text(s_pca[i, 0], s_pca[i, 1], 
                                    s=self.sp_info.Group[union_ind].values[i].astype(int).astype(str),
                                    fontsize=sp_fontsize, bbox=sp_bbox
                            )
                    ax.axis('off') # the axis here may not be consistent, so hide.
                    ax.plot()
                    if savefig:
                        if not os.path.exists("img"):
                            os.mkdir("img")
                        if fmt == 'pdf':
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'.pdf'
                            plt.savefig(fm, bbox_inches='tight')
                        elif fmt == 'png':
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'.png'
                            plt.savefig(fm, bbox_inches='tight')
                        else:
                            fm = 'img/' + str(figname) + str(index1) + '_' + str(index2) +'.'+fmt
                            plt.savefig(fm, bbox_inches='tight')
                            
                        print("image successfully save as", fm)
                        
                    plt.show()
                    
                    # deprecated (24/07/2021):
                    # if self.data.shape[1] > 2:
                    #     pca = PCA(n_components=2)
                    #     x_pca = pca.fit_transform(self.data)
                    #     s_pca = pca.transform(self.splist_[:, 3:])
                    #     
                    #     # select indices
                    #     x_pca = x_pca[(self.labels_ == cluster_label1) | (self.labels_ == cluster_label2)]
                    #     s_pca = s_pca[(self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)]
                    #     
                    #     ax.scatter(x_pca[:, 0], x_pca[:, 1],
                    #                 marker="*", c=self.labels_[(self.labels_ == cluster_label1) | (self.labels_ == cluster_label2)])
                    #     ax.scatter(s_pca[:,0], s_pca[:,1], marker="p")
                    #     
                    #     ax.scatter(object1[0],  object1[1], marker="*", s=ind_marker_size)
                    #     ax.text(object1[0], object1[1], s=str(index1), color=ind_color)
                    #     
                    #     ax.scatter(object2[0],  object2[1], marker="*", s=ind_marker_size)
                    #     ax.text(object2[0], object2[1], s=str(index2), color=ind_color)
                    #         
                    #     for i in range(s_pca.shape[0]):
                    #         ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                    #         ax.set_aspect('equal', adjustable='datalim')
                    #         ax.text(s_pca[i, 0], s_pca[i, 1], 
                    #             s=self.sp_info.Group[(self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)].values[i].astype(int).astype(str),
                    #             bbox=sp_bbox)
                    #     
                    #     ax.plot()
                    #     if savefig:
                    #         if not os.path.exists("img"):
                    #             os.mkdir("img")
                    #             plt.savefig('img/ind1_ind2.pdf')
                    #             print("successfully save")
                    #     plt.show()
                    #     
                    # elif self.data.shape[1] == 2:
                    #     x_pca = self.data.copy()
                    #     s_pca = self.splist_[:, 3:].copy()
                    #     
                    #     # select indices
                    #     x_pca = x_pca[(self.labels_ == cluster_label1) | (self.labels_ == cluster_label2)]
                    #     s_pca = s_pca[(self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)]
                    #     
                    #     ax.scatter(x_pca[:,0], x_pca[:,1], 
                    #                 marker="*", c=self.labels_[(self.labels_ == cluster_label1) | (self.labels_ == cluster_label2)])
                    #     ax.scatter(s_pca[:, 0], s_pca[:, 1], marker="p")
                    #     
                    #     ax.scatter(object1[0],  object1[1], marker="*", s=ind_marker_size)
                    #     ax.text(object1[0], object1[1], str(index1), color=ind_color)
                    #     
                    #     ax.scatter(object2[0],  object2[1], marker="*", s=ind_marker_size)
                    #     ax.text(object2[0], object2[1], str(index2), color=ind_color)
                    #     
                    #     for i in range(s_pca.shape[0]):
                    #         ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color, alpha=alpha, lw=cline_width, clip_on=False))
                    #         ax.set_aspect('equal', adjustable='datalim')
                    #         ax.text(s_pca[i, 0], s_pca[i, 1],
                    #             s=self.sp_info.Group[(self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)].values[i].astype(int).astype(str), 
                    #             bbox=sp_bbox)
                    #     
                    #     ax.plot()
                    #     if savefig:
                    #         if not os.path.exists("img"):
                    #             os.mkdir("img")
                    #             plt.savefig('img/ind1_ind2.pdf')
                    #             print("successfully save")
                    #     plt.show()
                    #     
                    # else:
                    #     print("This function is restricted to multidimensional (dimension greater than or equal to 2) data.")
                        

                        
                # if showsplist:
                #     print(dash_line)
                #     spdisplay = self.sp_info.iloc[[agg_label1, agg_label2],:]
                #     spdisplay["Data points"] = [str(index1), str(index2)]
                #     spdisplay = spdisplay[["Data points", "Group", "NrPts", "Cluster", "Coordinates"]] 
                #     print(spdisplay.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                #     print(dash_line)
                
        return 
    
    
    
    def explain_viz(self, figsize=(12, 8), figstyle="ggplot", savefig=False, fontsize=None, bbox={'facecolor': 'tomato', 'alpha': 0.3, 'pad': 2}, axis='off', fmt='pdf'):
        """Visualize the starting point and data points"""
        
        if self.cluster_color is None:
            self.cluster_color = dict()
            for i in np.unique(self.labels_):
                self.cluster_color[i] = '#%06X' % np.random.randint(0, 0xFFFFFF)

        # if self.data.shape[1] >= 2:
        plt.style.use('default') # clear the privous figure style
        plt.style.use(style=figstyle)
        plt.figure(figsize=figsize)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.scatter(self.s_pca[:,0], self.s_pca[:,1], marker="p")
        for i in np.unique(self.labels_):
            x_pca_part = self.x_pca[self.labels_ == i,:]
            plt.scatter(x_pca_part[:,0], x_pca_part[:,1], marker="*", c=self.cluster_color[i])
            # plt.scatter(x_pca_part[:,0], x_pca_part[:,1], marker="*", c='#%06X' % np.random.randint(0, 0xFFFFFF))
            for j in range(self.s_pca.shape[0]):
                if fontsize is None:
                    plt.text(self.s_pca[j, 0], self.s_pca[j, 1], str(j), bbox=bbox)
                else:
                    plt.text(self.s_pca[j, 0], self.s_pca[j, 1], str(j), fontsize=fontsize, bbox=bbox)

        plt.axis(axis) # the axis here may not be consistent, so hide.
        plt.xlim([np.min(self.x_pca[:,0])-0.1, np.max(self.x_pca[:,0])+0.1])
        plt.ylim([np.min(self.x_pca[:,1])-0.1, np.max(self.x_pca[:,1])+0.1])

        if savefig:
            if not os.path.exists("img"):
                os.mkdir("img")
            if fmt == 'pdf':
                fm = 'img/explain_viz.pdf'
                plt.savefig(fm, bbox_inches='tight')
            elif fmt == 'png':
                fm = 'img/explain_viz.png'
                plt.savefig(fm, bbox_inches='tight')
            else:
                fm = 'img/explain_viz.'+fmt
                plt.savefig(fm, bbox_inches='tight')
                
            print("successfully save")

        plt.show()
            
        # else:
        #     print("Visualization is restricted to multidimensional (dimension greater than or equal to 2) data.")
            
        return
        
        

    def form_starting_point_clusters_table(self, aggregate=False):
        """form the columns details for starting points and clusters information"""
        
        # won't change the original order of self.splist_
        cols = ["Group", "NrPts"]
        coord = list()
        
        if aggregate:
            # for i in np.around(self.splist_[:, 3:], 2).tolist():
            for i in np.around(self.s_pca, 2).tolist():
                fill = ""
                if len(i) <= 5:
                    for j in i:
                        fill = fill + str(j) + " "
                else:
                    for j in i[:5]:
                        fill = fill + str(j) + " "
                    fill = fill + "..."
                fill += ""
                coord.append(fill)

            # self.sp_info = pd.DataFrame(self.splist_[:, 1:2], columns=cols)

        else:
            for i in self.splist_[:, 0]:
                fill = ""
                sp_item = np.around(self.data[int(i), :], 2).tolist()
                if len(sp_item) <= 5:
                    for j in sp_item:
                        fill = fill + str(j) + " "
                else:
                    for j in sp_item[:5]:
                        fill = fill + str(j) + " "
                    fill = fill + "..."
                fill += ""
                coord.append(fill)
                
        self.sp_info = pd.DataFrame(columns=cols)
        self.sp_info["Group"] = np.arange(0, self.splist_.shape[0])
        self.sp_info["NrPts"] = self.splist_[:, 2].astype(int)
        self.sp_info["Cluster"] = [self.label_change[i] for i in range(self.splist_.shape[0])]
        self.sp_info["Coordinates"] = coord # np.around(self.splist_[:,3:], 2).tolist()
        self.sp_to_c_info = True
        return
        
        
    
    def visualize_linkage(self, scale=1.5, 
                          figsize=(8,8),
                          labelsize=24, 
                          markersize=320,
                          plot_boundary=False,
                          bound_color='red', path='.', fmt='pdf'):
        """Visualize the linkage in the distance clustering.
        
        
        Parameters
        ----------
        scale : float
            Design for distance-clustering, when distance between the two starting points 
            associated with two distinct groups smaller than scale*radius, then the two groups merge.
        
        labelsize : int 
            The fontsize of ticks. 
            
        markersize : int 
            The size of the markers for starting points.
            
        plot_boundary : boolean
            If it is true, will plot the boundary of groups for the starting points.
            
        bound_color : str
            The color for the boundary for groups with the specified radius.
            
        path : str
            Relative file location for figure storage.
            
        fmt : str
            Specify the format of the image to be saved, default as 'pdf', other choice: png.
        
        """
        
        distm, n_components, labels = visualize_connections(self.data, self.splist_, radius=self.radius, scale=round(scale,2))
        # plt.figure(figsize=figsize)
        plt.rcParams['axes.facecolor'] = 'white'
        # if norm:
        #     P = self.splist_[:, 3:]
        # else:
        #     P = self.splist_[:, 3:]*self._scl + self._mu
        P = self.data[self.splist[:, 0]]
        link_list = return_csr_matrix_indices(csr_matrix(distm))
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(self.splist_.shape[0]):
            ax.scatter(P[i,0], P[i,1], s=markersize, c='k', marker='.')
            if plot_boundary:
                ax.add_patch(plt.Circle((P[i, 0], P[i, 1]), self.radius, 
                                        color=bound_color, fill=False, clip_on=False)
                            )
            ax.set_aspect('equal', adjustable='datalim')

        for edge in link_list:
            i, j = edge
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], linewidth=3, c='k') 
        
        ax.tick_params(axis='both', labelsize=labelsize, colors='k')
        if not os.path.isdir(path):
            os.makedirs(path)
        if fmt == 'pdf':
            fig.savefig(path + '/linkage_scale_'+str(round(scale,2))+'_tol_'+str(round(self.radius,2))+'.pdf', bbox_inches='tight')
        else:
            fig.savefig(path + '/linkage_scale_'+str(round(scale,2))+'_tol_'+str(round(self.radius,2))+'.png', bbox_inches='tight')
        # plt.show()
    
    
    
    def load_splist_indices(self):
        if self.splist_indices is not None:
            self.splist_indices = np.full(self.data.shape[0], 0, dtype=int)
            self.splist_indices[self.splist_[:,0].astype(int)] = 1
        return self.splist_indices
        
        
        
    def load_cluster_centers(self):
        if self.centers is None:
            self.centers = calculate_cluster_centers(self.data*self._scl + self._mu, self.labels_)
            return self.centers
        else:
            return self.centers
        
        
    def calculate_group_centers(self, data, labels):
        centers = list() 
        for c in set(labels):
            indc = [i for i in range(data.shape[0]) if labels[i] == c]
            indc = (labels==c)
            center = [-1, c] + np.mean(data[indc,:], axis=1).tolist()
            centers.append( center )
        return centers

    
    
    def outlier_filter(self, labels, min_samples=None, min_samples_rate=0.1): # percent
        """Filter outliers in terms of min_samples"""
        # print("Min_samples (minPts):{:.0f}".format(min_samples))
        # self.cluster_count_dict = collections.Counter(labels) # move
        if min_samples == None:
            min_samples = min_samples_rate*sum(self.old_cluster_count.values())
            
        # threshold = np.percentile(
        #    a=np.array(list(count_dict.values())), q=percent,
        #    interpolation='linear', axis=-1)
        
        # return [i[0] for i in count_dict.items() 
        #         if i[1] <= threshold and i[1] <= min_samples]
        
        # filter minimum number of points each cluster smaller than scale
        
        return [i[0] for i in self.old_cluster_count.items() if i[1] < min_samples]
    
    
    # deprecated (24/07/2021)
    # def merge_pairs(self, pairs):
    #     """Transform connected pairs to connected groups (list)"""
    #     
    #     ulabels = np.full(len_p, -1, dtype=int)
    #     labels = list()
    #     maxid = 0
    #     for i in range(len(pairs)):
    #         if ulabels[i] == -1:
    #             sub = pairs[i]
    #             ulabels[i] = maxid
    # 
    #             for j in range(i+1, len(pairs)):
    #                 com = pairs[j]
    #                 if check_if_intersect(sub, com):
    #                     sub = sub + com
    #                     if ulabels[j] == -1:
    #                         ulabels[j] = maxid
    #                     else:
    #                         ulabels[ulabels == maxid] = ulabels[j]
    # 
    #             maxid = maxid + 1
    # 
    #     for i in np.unique(ulabels):
    #         sub = list()
    #         for j in np.where(ulabels == i)[0]:
    #             sub = sub + pairs[int(j)]
    #         labels.append(list(set(sub)))
    #     return labels
    
    

    # """
    # def normalize(self, data, shift = 'z-score'):
    #     if shift not in ['mean', 'min', 'z-score']:
    #         raise ValueError("please enter a correct shift.")
    #         
    #     if shift == 'min':
    #         _mu = data.min(axis=0)
    #         _std = data.std()
    #         
    #     elif shift == 'mean':
    #         _mu = data.mean(axis=0)
    #         _std = data.std()
    #     
    #     else: #  shift == 'z-score':
    #         _mu = data.mean(axis=0)
    #         _std = data.std(axis=0)
    #         
    #     data = (data - _mu)/_std
    #     return data, (_mu, _std)    
    # """


        
    # def check_if_intersect(self, l1, l2):
    #     return set(l1).intersection(l2) != set()
  


    def reassign_labels(self, labels):
        # unique_labels = sorted(np.unique(labels))
        sorted_dict = sorted(self.old_cluster_count.items(), key=lambda x: x[1], reverse=True)

        clabels = copy.deepcopy(labels)
        for i in range(len(sorted_dict)):
            clabels[labels == sorted_dict[i][0]]  = i
        return clabels

    

    def pprint_format(self, items):
        # lite-function to print dict or turple/list
        cluster = 0
        if isinstance(items, dict):
            for key, value in sorted(items.items(), key=lambda x: x[1], reverse=True): 
                # print("      [-] cluster {} : {}".format(key, value))
                print("      * cluster {} : {}".format(cluster, value))
                cluster = cluster + 1
                
        elif isinstance(items, list) or isinstance(items, tuple):
            for item in items:
                print("      * ", item)
            
        return 
            

            
    def __repr__(self):
        _name = "CLASSIX(sorting={0.sorting!r}, radius={0.radius!r}, minPts={0.minPts!r}, group_merging={0.group_merging!r})".format(self)
        return _name 

    
    
    def __str__(self):
        _name = 'CLASSIX(sorting={0.sorting!r}, radius={0.radius!r}, minPts={0.minPts!r}, group_merging={0.group_merging!r})'.format(self)
        return _name
    
    
    
    @property
    def radius(self):
        return self._radius
    
    
    
    @radius.setter
    def radius(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type')
        if value <= 0:
            raise ValueError(
                "Please feed an correct value (>0) for tolerance.")
        # if value > 1:
        #     warnings.warn("Might lead to bad aggregation", DeprecationWarning)
        self._radius = value
    
    
        
    @property
    def sorting(self):
        return self._sorting
    
    
    
    @sorting.setter
    def sorting(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string type')
        if value not in ['pca', 'norm-mean', 'norm-orthant'] and value != None:
            raise ValueError(
                "Please refer to an correct sorting way, namely 'pca', 'norm-mean' and 'norm-orthant'.")
        self._sorting = value

        
    
    @property
    def group_merging(self):
        return self._group_merging
    
    
    
    @group_merging.setter
    def group_merging(self, value):
        if not isinstance(value, str) and not isinstance(None, type(None)):
            raise TypeError('Expected a string type or None.')
        if value not in ['density', 
                         'distance'
                        ] and value is not None: # 'mst-distance', 'scc-distance', 'trivial-distance', 'trivial-density'
            if value.lower()!='none':
                raise ValueError(
                "Please refer to an correct sorting way, namely 'density' and 'distance' or None."
                ) # 'scc-distance' and 'mst-distance'
        self._group_merging = value
        


    # @property
    # def distance_scale(self):
    #     return self._distance_scale
    
    
    
    # @distance_scale.setter
    # def distance_scale(self, value):
    #     if not isinstance(value, float) and not isinstance(value, int):
    #         raise TypeError('Expected a float or int type')
    #     if value <= 0:
    #         raise ValueError(
    #             "Please feed an correct value for distance_scale")

    #     self._distance_scale = value
    
        
    # @property
    # def eta(self):
    #     return self._eta
    
    
    
    # @eta.setter
    # def eta(self, value):
    #     if not isinstance(value, float) and not isinstance(value, int):
    #         raise TypeError('Expected a float or int type')
    #     if value <= 0:
    #         raise ValueError(
    #             "Please feed an correct value for eta")
    #     self._eta = value


        
    # @property
    # def noise_percent(self):
    #     return self._noise_percent
    
    
    
    # @noise_percent.setter
    # def noise_percent(self, value):
    #     if not isinstance(value, float) and not isinstance(value,int):
    #         raise TypeError('Expected a float or int type')
    #     if value < 0 or value >= 100:
    #         raise ValueError('Percentiles must be in the range [0, 100)')
    #     self._noise_percent = value
    
    
    
    @property
    def minPts(self):
        return self._minPts
    
    
    
    @minPts.setter
    def minPts(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type')
        if value < 0 or (0 < value & value < 1):
            raise ValueError('Noise_scale must be 0 or greater than 1.')
        self._minPts = value
    
    

# deprecated (24/07/2021)
# python implementation for aggregation
# def aggregate(data, sorting="pca", tol=0.5, verbose=1):
#     """aggregate the data
# 
#     Parameters
#     ----------
#     data : numpy.ndarray
#         the input that is array-like of shape (n_samples,).
# 
#     sorting : str
#         the sorting way refered for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.
# 
#     tol : float
#         the tolerance to control the aggregation, if the distance between the starting point 
#         and the object is less than or equal than the tolerance,
#         the object should allocated to the group which starting point belongs to.  
# 
#     Returns
#     -------
#     labels (list) : the group category of the data after aggregation
#     splist (list) : the list of the starting points
#     agg_centers (list) : store the centers of aggregation groups
#     """
# 
#     # nr_dist = 0 # if necessary, count the distance computation
#     splist = list() # store the starting points
#     len_ind = data.shape[0]
# 
#     if sorting == "norm-mean":
#         sort_vals = np.linalg.norm(data, ord=2, axis=1)
#         ind = np.argsort(sort_vals)
# 
#     elif sorting == "norm-orthant":
#         sort_vals = np.linalg.norm(data, ord=2, axis=1)
#         ind = np.argsort(sort_vals)
# 
#     elif sorting == "pca":
#         pca = PCA(n_components=1)
#         sort_vals = pca.fit_transform(data).reshape(-1)
#         ind = np.argsort(sort_vals)
# 
#     else: # no sorting
#         ind = np.arange(len_ind)
# 
#     lab = 0
#     labels = [-1] * len_ind
# 
#     for i in tqdm(range(len_ind), disable=not verbose):
#         sp = ind[i] # starting point
#         if labels[sp] >= 0:
#             continue
#         else:
#             clustc = data[sp,:] 
#             labels[sp] = lab
#             num_group = 0
# 
#         for j in ind[i:]:
#             if labels[j] >= 0:
#                 continue
# 
#             # sort_val_c = sort_vals[sp]
#             # sort_val_j = sort_vals[j]
# 
#             if (sort_vals[j] - sort_vals[sp] > tol):
#                 break       
# 
#             # nr_dist += 1
#             # dist = np.sum((clustc - data[j,:])**2)    # slow
# 
#             dat = clustc - data[j,:]
#             dist = np.inner(dat, dat)
# 
#             if dist <= tol**2:
#                 num_group = num_group + 1
#                 labels[j] = lab
# 
#         splist.append([sp, lab] + [num_group] + list(clustc) ) # respectively store starting point
#                                                                # index, label, number of neighbor objects, center (starting point).
#         lab = lab + 1
# 
#     agg_centers = list() 
#     for c in range(lab):
#         indc = [i for i in range(len_ind) if labels[i] == c]
#         center = np.mean(data[indc,:], axis=0)
#         center = [-1, c] + center.tolist()
#         agg_centers.append( center )
# 
#     # if verbose == 1:
#     #    print("aggregate {} groups".format(len(np.unique(labels))))
# 
#     return labels, splist, agg_centers



def pairwise_distances(X):
    """Calculate the Euclidean distance matrix."""
    distm = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            distm[i, j] = distm[j, i] = norm(X[i,:]-X[j,:], ord=2)
    return distm



def visualize_connections(data, splist, radius=0.5, scale=1.5):
    """Calculate the connected components for graph constructed by starting points given radius and scale."""
    distm = pairwise_distances(data[splist[:,0].astype(int)])
    tol = radius*scale
    distm = (distm <= tol).astype(int)
    n_components, labels = connected_components(csgraph=distm, directed=False, return_labels=True)
    return distm, n_components, labels
    
    
    
def novel_normalization(data, base):
    """Initial data preparation of CLASSIX."""
    if base == "norm-mean":
        # self._mu, self._std = data.mean(axis=0), data.std()
        _mu = data.mean(axis=0)
        ndata = data - _mu
        _scl = ndata.std()
        ndata = ndata / _scl

    elif base == "pca":
        _mu = data.mean(axis=0)
        ndata = data - _mu # mean center
        rds = norm(ndata, axis=1) # distance of each data point from 0
        _scl = np.median(rds) # 50% of data points are within that radius
        ndata = ndata / _scl # now 50% of data are in unit ball 

    elif base == "norm-orthant":
        # self._mu, self._std = data.min(axis=0), data.std()
        _mu = data.min(axis=0)
        ndata = data - _mu
        _scl = ndata.std()
        ndata = ndata / _scl

    else:
        # self._mu, self._std = data.mean(axis=0), data.std(axis=0) # z-score
        _mu, _scl = 0, 1 # no normalization
        ndata = (data - _mu) / _scl
    return ndata, (_mu, _scl)



def calculate_cluster_centers(data, labels):
    """Calculate the mean centers of clusters from given data."""
    ulabels = set(labels)
    centers = np.zeros((len(ulabels), data.shape[1]))
    for c in ulabels:
        indc = (labels==c)
        centers[c] = np.mean(data[indc,:], axis=0)
    return centers




# ##########################################################################################################
# *************************** <!-- the independent functions of checking overlap ***************************
# *******************************  determine if two groups should be merged ********************************






def find_shortest_path(source_node=None, connected_pairs=None, num_nodes=None, target_node=None):
    """Get single-sourse shortest paths as well as distance from source node,
    design especially for unweighted undirected graph. The time complexity is O(|V| + |E|)
    where |V| is the number of vertices and |E| is the number of edges.
    
    Parameters
    ----------
    source_node: int
        A given source vertex.
    
    connected_pairs: list
        The list stores connected nodes pairs.
    
    num_nodes: int
        The number of nodes existed in the graph.
        
    target_node: int, default=None
        Find the shortest paths from source node to target node.
        If not None, function returns the shortest path between source node and target node,
        otherwise returns table storing shortest path information.
        
    Returns
    -------
    dist_info: numpy.ndarray
        The table storing shortest path information.
    
    shortest_path_to_target: list
        The shortest path between source node and target node
    
    """
    visited_nodes = [False]*num_nodes
    queque = list()
    graph = pairs_to_graph(connected_pairs, num_nodes) # return sparse matrix
    dist_info = np.empty((num_nodes, 3), dtype=int) # node, dist, last node
    dist_info[:,0] = np.arange(num_nodes)
    dist_info[:,1] = num_nodes # np.iinfo(np.int64).max
    dist_info[:,2] = -1
    source_node = int(source_node)
    queque.append(source_node+1)
    dist_info[source_node,1] = 0
    while(np.any(queque)):
        node = queque.pop(0)
        if not visited_nodes[node-1]:
            neighbor = list()
            visited_nodes[node-1] = True
            for i in range(int(num_nodes)):
                if graph[node-1, i] == 1 and not visited_nodes[i] and not i+1 in queque:
                    neighbor.append(i+1)
                    dist_info[i, 1], dist_info[i, 2] = dist_info[node-1, 1]+1, node-1
            queque = queque + neighbor
            
    if target_node != None:
        shortest_path_to_target = list()
        if dist_info[target_node,1] == np.iinfo(np.int64).min:
            print("no path between {} and {}".format(source_node, target_node))
            return None
        
        predecessor = target_node
        while(dist_info[predecessor, 2] != -1):
            shortest_path_to_target.append(predecessor)
            predecessor = dist_info[predecessor, 2]
            
        shortest_path_to_target.append(source_node)
        shortest_path_to_target.reverse()
        return shortest_path_to_target
    else:
        return dist_info

    

def pairs_to_graph(pairs, num_nodes, sparse=True):
    """Transform the pairs represented by list into graph."""
    # from scipy.sparse import csr_matrix
    graph = np.full((num_nodes, num_nodes), np.inf, dtype=int)
    for i in range(num_nodes):
        graph[i, i] = 0
    pairs = np.array(pairs, dtype=int)
    for pair in pairs:
        graph[pair[0], pair[1]] = graph[pair[1], pair[0]] = 1
    if sparse:
        graph = csr_matrix(graph)
    return graph



def return_csr_matrix_indices(csr_matrix): 
    """Return sparce matrix indices."""
    shape_dim1, shape_dim2 = csr_matrix.shape
    length_range = csr_matrix.indices

    indices = np.empty(len(length_range), dtype=csr_matrix.indices.dtype)
    _sparsetools.expandptr(shape_dim1, csr_matrix.indptr, indices)
    return np.array(list(zip(indices, length_range)))



# deprecated (24/07/2021) -> for scipy shortest distance use
# def get_shortest_path(predecessors, i, j):
#     """Get the shortest path between two nodes in the graph"""
#     path = [j]
#     k = j
#     while predecessors[i, k] != -9999:
#         path.append(predecessors[i, k])
#         k = predecessors[i, k]
#     return path[::-1]









