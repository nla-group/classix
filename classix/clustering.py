# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2026 Stefan Güttel, Xinye Chen


import warnings

import os
import numbers
import collections
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial import distance
import scipy.sparse as sparse
from time import time

def cython_is_available(verbose=0):
    """Return whether the optimized Cython implementation is available.

    Parameters
    ----------
    verbose : int or bool, default=0
        If non-zero, print which implementation variant is being used.

    Returns
    -------
    available : bool
        ``True`` when the package is configured to use Cython and the compiled
        extension modules can be imported, otherwise ``False``.
    """
    
    __cython_type__ = "memoryview"
    
    from . import __enable_cython__
    if __enable_cython__:
        try:
            # %load_ext Cython
            # !python3 setup.py build_ext --inplace
            import numpy
            
            try: # check if Cython packages are loaded properly
                from .aggregate_ed_cm import general_aggregate, pca_aggregate
                from .merge_ed_cm import density_merge, distance_merge
                # cython with memoryviews
                # Typed memoryviews allow efficient access to memory buffers, such as those underlying NumPy arrays, without incurring any Python overhead. 
            
            except ModuleNotFoundError:
                from .aggregate_ed_c import general_aggregate, pca_aggregate
                
                __cython_type__ =  "trivial"

            if verbose:
                if __cython_type__ == "memoryview":
                    print("This CLASSIX is using Cython typed memoryviews.")
                else:
                    print("This CLASSIX is not using Cython typed memoryviews.")
            
            return True

        except (ModuleNotFoundError, ValueError):
            if verbose:
                print("CLASSIX is currently not using Cython.")
            return False
    else:
        if verbose:
            print("Cython is currently disabled. Please set ``__enable_cython__`` to True to enable Cython.")
        return False
    
    
def loadData(name='vdu_signals'):
    """Load a built-in CLASSIX sample dataset.
    
    Parameters
    ----------
    name : {'vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', \
'Banknote', 'Seeds', 'Phoneme', 'Wine', 'Covid3MC', 'CovidENV'}, \
default='vdu_signals'
        Dataset identifier.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        Returned when ``name='vdu_signals'``.

    X : ndarray or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix for labeled datasets.

    y : ndarray of shape (n_samples,)
        Ground-truth labels for labeled datasets.

    Notes
    -----
    Missing data files are downloaded automatically and cached in the package's
    ``data`` directory.
    """

    import logging
    logging.basicConfig()

    current_dir, current_filename = os.path.split(__file__)
    
    if not os.path.isdir(os.path.join(current_dir, 'data')):
        os.mkdir(os.path.join(current_dir, 'data/'))
        
    if name == 'vdu_signals':
        DATA_PATH = os.path.join(current_dir, 'data/vdu_signals.npy')
        if not os.path.isfile(DATA_PATH):
            print('Loading data files from the web...')
            get_data(current_dir)
            print('Done.')
        return np.load(DATA_PATH)
    
    if name == 'Iris':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Irirs.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Irirs.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Iris')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Dermatology':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Dermatology.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Dermatology.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Dermatology')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Ecoli':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Ecoli.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Ecoli.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Ecoli')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Glass':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Glass.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Glass.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Glass')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Banknote':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Banknote.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Banknote.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Banknote')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Seeds':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Seeds.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Seeds.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Seeds')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Phoneme':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Phoneme.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Phoneme.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Phoneme')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Wine':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Wine.npy')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Wine.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Wine')
            print('Done.')
        return np.load(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'CovidENV':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_CovidENV.pkl')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_CovidENV.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'CovidENV')
            print('Done.')
        return pd.read_pickle(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    if name == 'Covid3MC':
        DATA_PATH_X = os.path.join(current_dir, 'data/X_Covid3MC.pkl')
        DATA_PATH_Y = os.path.join(current_dir, 'data/y_Covid3MC.npy')
        if not os.path.isfile(DATA_PATH_X) or not os.path.isfile(DATA_PATH_Y):
            print('Loading data files from the web...')
            get_data(current_dir, 'Covid3MC')
            print('Done.')
        return pd.read_pickle(DATA_PATH_X), np.load(DATA_PATH_Y)
    
    
    if name not in ['vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 
                    'Banknote', 'Seeds', 'Phoneme', 'Wine', 'CovidENV', 'Covid3MC']:
        
        warnings.warn("Invalid dataset identifier.")


def get_data(current_dir='', name='vdu_signals'):
    """Download one of the built-in sample datasets.

    Parameters
    ----------
    current_dir : str, default=''
        Directory containing the ``data`` subdirectory where downloaded files
        are written.

    name : {'vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', \
'Banknote', 'Seeds', 'Phoneme', 'Wine', 'CovidENV', 'Covid3MC'}, \
default='vdu_signals'
        Name of the dataset to download.

    Notes
    -----
    This helper is called by :func:`loadData` when a bundled dataset is not
    present locally. It writes files to disk and does not return a value.
    """
    import requests
    
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
                    
    elif name == 'CovidENV':
        url_parent_x = "https://www.dropbox.com/scl/fi/tulksb60iu22q7sj9gef3/X_CovidENV.pkl?rlkey=olndlzyxz69i28srfuclsecrc&st=5ojreume&dl=1"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_CovidENV.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_CovidENV.pkl'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_CovidENV.npy'), 'wb') as handler:
            handler.write(y)
                    

    elif name == 'Covid3MC':
        url_parent_x = "https://www.dropbox.com/scl/fi/qcc60zplu7vtv2l3fee8z/X_Covid3MC.pkl?rlkey=y3y2k90itgjd98odytlf6aehy&st=edodmc5b&dl=1"
        url_parent_y = "https://github.com/nla-group/classix/raw/master/classix/source/y_Covid3MC.npy"
        x = requests.get(url_parent_x).content
        y = requests.get(url_parent_y).content
        with open(os.path.join(current_dir, 'data/X_Covid3MC.pkl'), 'wb') as handler:
            handler.write(x)
        with open(os.path.join(current_dir, 'data/y_Covid3MC.npy'), 'wb') as handler:
            handler.write(y)
                    
                
                
class NotFittedError(ValueError, AttributeError):
    """Exception raised when a CLASSIX estimator is used before fitting."""
        
        

class CLASSIX:
    """Fast and explainable clustering based on sorting.

    CLASSIX is a scikit-learn-like clustering estimator. It first sorts and
    aggregates nearby samples into small groups represented by starting points,
    then optionally merges those groups into final clusters. The implementation
    emphasizes speed, low memory use, and post-hoc explanations of why samples
    belong to the same or different clusters.

    Parameters
    ----------
    metric : str, {'euclidean', 'manhattan', 'tanimoto'}, default='euclidean'
        Distance metric used during aggregation and merging.

        ``'euclidean'`` uses the Euclidean distance and supports ``'pca'``,
        ``'norm-mean'``, ``'norm-orthant'``, and ``None`` sorting. ``'manhattan'``
        uses the L1 distance and automatically switches to sum-based sorting.
        ``'tanimoto'`` uses Tanimoto distance, which is intended for non-negative
        binary, count, or fingerprint-like data.

    sorting : str, {'pca', 'norm-mean', 'norm-orthant', 'sum', None}, default='pca'
        Sorting method used before aggregation. ``'pca'`` sorts by the first
        principal component, ``'norm-mean'`` mean-centers the data and sorts by
        Euclidean norm, ``'norm-orthant'`` shifts the data to the non-negative
        orthant and sorts by Euclidean norm, ``'sum'`` sorts by the sum of
        feature values, and ``None`` disables sorting.

    radius : float, default=0.5
        Aggregation radius. A sample is assigned to the current starting point's
        group when its distance to that starting point is at most ``radius`` in
        the preprocessed feature space.

    group_merging : str, {'density', 'distance', None}, default='distance'
        Strategy used to merge aggregation groups. ``'distance'`` merges groups
        whose starting points are within ``mergeScale * radius``. ``'density'``
        merges Euclidean groups using an intersection-density criterion. ``None``
        or ``'none'`` skips merging and returns aggregation groups as clusters.

    minPts : int, default=1
        Minimum valid cluster size. Clusters smaller than ``minPts`` are
        dissolved and their groups are reassigned to the nearest valid cluster
        when possible. Set to 1 to disable this filtering.

    mergeScale : float, default=1.5
        Multiplicative factor for distance-based merging.

    post_alloc : bool, default=True
        If ``True``, groups filtered by ``minPts`` are assigned to the nearest
        valid cluster. If ``False``, those samples receive label ``-1`` where the
        selected merging implementation supports explicit outlier labels.

    mergeTinyGroups : bool, default=True
        If ``False``, groups smaller than ``minPts`` do not initiate
        distance-based merge connections.

    verbose : bool or int, default=1
        Controls progress output. Set to 0 for silent mode.

    short_log_form : bool, default=True
        If ``True``, truncate long cluster-size lists in progress output.


    Attributes
    ----------
    groups_ : numpy.ndarray
        Aggregation group labels in sorted-data order.

    splist_ : numpy.ndarray
        Starting-point information for aggregation groups. For Euclidean
        clustering this stores sorted-data indices and group sizes; for
        Manhattan and Tanimoto clustering this stores sorted-data indices.

    labels_ : numpy.ndarray
        Final cluster labels in the original input order.

    group_outliers_ : numpy.ndarray
        Indices of small aggregation groups dissolved during minPts processing.

    nrDistComp_ : int
        Number of distance computations performed.

    dataScale_ : float
        Scaling factor applied during preprocessing.

    References
    ----------
    Chen, X. and Guettel, S. (2022). Fast and explainable clustering based on
    sorting. https://arxiv.org/abs/2202.01456
    """
        
    def __init__(self, sorting="pca", metric='euclidean', radius=0.5, minPts=1, group_merging="distance", mergeScale=1.5, 
                 post_alloc=True, mergeTinyGroups=True, verbose=1, short_log_form=True): 
        """Initialize the CLASSIX clustering estimator."""
        
        self.metric = metric.lower()
        if self.metric not in ['euclidean', 'manhattan', 'tanimoto']:
            raise ValueError("Only 'euclidean', 'manhattan', and 'tanimoto' are supported now.")
        
        if self.metric == 'manhattan' and sorting not in ['sum', 'popcount', None]:
            sorting = 'sum'  # Manhattan clustering uses sum-based sorting.
        self.__verbose = verbose
        self.minPts = int(minPts)

        self.sorting = sorting
        self.radius = radius
        self.group_merging = group_merging

        self.mergeScale_ = mergeScale # For distance measure, usually, we do not use this parameter
        self.__post_alloc = post_alloc
        self.__mergeTinyGroups = mergeTinyGroups
        self.__truncate = short_log_form
        self.labels_ = None
        
        self._gcIndices = np.frompyfunc(self.gc2ind, 1, 1)
                     
        if self.__verbose:
            print(self)
        

        from . import __enable_cython__
        self.__enable_cython__ = __enable_cython__
        self.__enable_aggregate_cython__ = False
        import platform
        
        if self.__enable_cython__:
            try:
                try:
                    from .aggregate_ed_cm import general_aggregate, pca_aggregate, lm_aggregate
                    
                except ModuleNotFoundError:
                    from .aggregate_ed_c import general_aggregate, pca_aggregate, lm_aggregate
                
                self.__enable_aggregate_cython__ = True

                
                
                if platform.system() == 'Windows':
                    from .merge_ed_cm_win import density_merge, distance_merge, distance_merge_mtg
                else:
                    from .merge_ed_cm import density_merge, distance_merge, distance_merge_mtg

            except (ModuleNotFoundError, ValueError):
                if not self.__enable_aggregate_cython__:
                    from .aggregate_ed import general_aggregate, pca_aggregate, lm_aggregate
                
                from .merge_ed import density_merge, distance_merge, distance_merge_mtg
                warnings.warn("This CLASSIX installation is not using Cython.")

        else:
            from .aggregate_ed import general_aggregate, pca_aggregate, lm_aggregate
            from .merge_ed import density_merge, distance_merge, distance_merge_mtg
            warnings.warn("This run of CLASSIX is not using Cython.")

        
        if platform.system() == 'Windows':
            if sorting == 'pca':
                self._aggregate = pca_aggregate
            else:
                self._aggregate = general_aggregate
        else:
            self._aggregate = lm_aggregate

        self._density_merge = density_merge
        
        if self.__mergeTinyGroups:
            self._distance_merge = distance_merge
        else:
            self._distance_merge = distance_merge_mtg

            
    def fit(self, data):
        """Fit CLASSIX to data.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features) or (n_samples,)
            Training data. One-dimensional input is reshaped to
            ``(n_samples, 1)``.

        Returns
        -------
        self : CLASSIX
            Fitted estimator.

        """
        if isinstance(data, pd.core.frame.DataFrame):
            self._index_data = data.index
            
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            if len(data.shape) == 1:
                data = data.reshape(-1,1)

        self.t1_prepare = time()        
        if data.dtype !=  'float64':
            data = data.astype('float64')
        
        # preprocessing phase
        if self.metric == 'euclidean':
            if self.sorting == "norm-mean":
                self.mu_ = data.mean(axis=0)
                data = data - self.mu_
                self.dataScale_ = data.std()
                if self.dataScale_ == 0: # prevent zero-division
                    self.dataScale_ = 1
                data = data / self.dataScale_
            
            elif self.sorting == "pca":
                self.mu_ = data.mean(axis=0)
                data = data - self.mu_ # mean center
                rds = norm(data, axis=1) # distance of each data point from 0
                self.dataScale_ = np.median(rds) # 50% of data points are within that radius
                if self.dataScale_ == 0: # prevent zero-division
                    self.dataScale_ = 1
                data = data / self.dataScale_ # now 50% of data are in unit ball 
                
            elif self.sorting == "norm-orthant":
                self.mu_ = data.min(axis=0)
                data = data - self.mu_
                self.dataScale_ = data.std()
                if self.dataScale_ == 0: # prevent zero-division
                    self.dataScale_ = 1
                data = data / self.dataScale_
                
            else:
                self.mu_, self.dataScale_ = 0, 1 # no preprocessing
                data = (data - self.mu_) / self.dataScale_

        elif self.metric == 'manhattan':
            # Manhattan-specific preprocessing: shift to the non-negative
            # orthant and normalize by the median feature sum.
            self.mu_ = data.min(0)
            data = data - self.mu_
            sort_vals = np.sum(data, axis=1)
            mext = np.median(sort_vals) or 1.0
            data /= mext
            dataScale_ = mext
            self.dataScale_ = mext
            sort_vals /= mext
            
        elif self.metric == 'tanimoto':
            # Tanimoto data should already be non-negative; keep placeholders
            # so downstream preprocessing-related attributes are available.
            self.mu_ = np.zeros(data.shape[1])
            self.dataScale_ = 1.0

        self.t1_prepare = time() - self.t1_prepare
        self.t2_aggregate = time()
        # aggregation
        
        if self.metric == 'euclidean':
            self.groups_, self.splist_, self.nrDistComp_, self.ind, sort_vals, data, self.__half_nrm2 = self._aggregate(
                                                                                    data=data,
                                                                                    sorting=self.sorting, 
                                                                                    tol=self.radius
                                                                                ) 
            
            if self.__half_nrm2 is None: # the self.aggregate will return None if certain strategy is applied.
                self.__half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5

        elif self.metric == 'tanimoto':
            # Tanimoto aggregation.
            try:
                if self.__enable_cython__:
                    from .aggregate_td_cm import aggregate_tanimoto
                else:
                    from .aggregate_td import aggregate_tanimoto
            except (ModuleNotFoundError, ImportError):
                from .aggregate_td import aggregate_tanimoto
                warnings.warn("aggregation_td module is required for tanimoto metric, roll back to Python version.")
        
            agg_res = aggregate_tanimoto(data, self.radius)
            self.groups_ = agg_res['labels']
            self.splist_ = agg_res['splist']
            self.nrDistComp_ = agg_res['nr_dist']
            self.ind = agg_res['ind']
            sort_vals = agg_res['sort_vals']
            data = agg_res['data_sorted']
            self.group_sizes_ = agg_res['group_sizes']

        elif self.metric == 'manhattan':
            # Manhattan aggregation.
            try:
                if self.__enable_cython__:
                    from .aggregate_md_cm import aggregate_manhattan
                else:
                    from .aggregate_md import aggregate_manhattan
                    
            except (ModuleNotFoundError, ImportError):
                from .aggregate_md import aggregate_manhattan
                warnings.warn("aggregation_md module is required for manhattan metric, roll back to Python version.")

            agg_res = aggregate_manhattan(data, self.radius)
            self.groups_ = agg_res['labels']
            self.splist_ = agg_res['splist']
            self.nrDistComp_ = agg_res['nr_dist']
            self.ind = agg_res['ind']
            sort_vals = agg_res['sort_vals']
            data = agg_res['data_sorted']          # this is sorted 
            self.group_sizes_ = agg_res['group_sizes']
            

        self.splist_ = np.array(self.splist_)
        self._fit_data_ = data
        self._fit_sort_vals_ = sort_vals
        self._fit_agg_labels_ = np.asarray(self.groups_)
        self.t2_aggregate = time() - self.t2_aggregate

        self.t3_merge = time()
        if self.group_merging is None:
            self.inverse_ind = np.argsort(self.ind)
            self.labels_ = np.asarray(self.groups_)[self.inverse_ind]
        
        elif self.group_merging.lower()=='none':
            self.inverse_ind = np.argsort(self.ind)
            self.labels_ = np.asarray(self.groups_)[self.inverse_ind]
        
        else:
            if self.metric == 'euclidean':
                self.labels_ = self.merging(
                    data=data,
                    agg_labels=self.groups_, 
                    splist=self.splist_,  
                    ind=self.ind, sort_vals=sort_vals, 
                    radius=self.radius, 
                    method=self.group_merging, 
                    minPts=self.minPts
                ) 
                
                if self.__verbose:
                    print(f"Euclidean merging completed: {len(np.unique(self.labels_))} clusters")

                self.sp_data_pts = data[self.splist_[:, 0].astype(int),:]
                self._update_minpts_base_state()
                
            elif self.metric == 'manhattan':
                try:
                    from .merge_md_cm import merge_manhattan
                except (ModuleNotFoundError, ImportError):
                    from .merge_md import merge_manhattan
                    warnings.warn("merge_md module is required for manhattan metric, roll back to Python version.")
                
                merge_result = merge_manhattan(
                    spdata=data[self.splist_,:],
                    group_sizes=self.group_sizes_,                    
                    sort_vals_sp=sort_vals[self.splist_],
                    agg_labels_sp=np.arange(len(self.splist_)),
                    radius=self.radius,
                    mergeScale=self.mergeScale_,
                    minPts=self.minPts,
                    mergeTinyGroups=self.__mergeTinyGroups
                )
                
                group2cluster = merge_result['group_cluster_labels']
                labels_sorted = group2cluster[self.groups_]

                self.labels_ = labels_sorted[np.argsort(self.ind)]
                self.inverse_ind = np.argsort(self.ind)
                self.Adj = merge_result['Adj']
                
                if self.__verbose:
                    print(f"Manhattan merging completed: {len(np.unique(self.labels_))} clusters")

                self.sp_data_pts = data[self.splist_,:]
                self._update_minpts_base_state()

            elif self.metric == 'tanimoto':
                try:
                    from .merge_td_cm import merge_tanimoto
                except (ModuleNotFoundError, ImportError):
                    from .merge_td import merge_tanimoto
                    warnings.warn("merge_td module is required for tanimoto metric, roll back to Python version.")
                
                merge_result = merge_tanimoto(
                    spdata=data[self.splist_,:],
                    group_sizes=self.group_sizes_,
                    sort_vals_sp=sort_vals[self.splist_],
                    agg_labels_sp=np.arange(len(self.splist_)),
                    radius=self.radius,
                    mergeScale=self.mergeScale_,
                    minPts=self.minPts,
                    mergeTinyGroups=self.__mergeTinyGroups
                )

                # same with manhattan distance
                group2cluster = merge_result['group_cluster_labels']
                labels_sorted = group2cluster[self.groups_]

                self.labels_ = labels_sorted[np.argsort(self.ind)]
                self.inverse_ind = np.argsort(self.ind)
                self.Adj = merge_result['Adj']
                if self.__verbose:
                    print(f"Tanimoto merging completed: {len(np.unique(self.labels_))} clusters")

                self.sp_data_pts = data[self.splist_,:]
                self._update_minpts_base_state()

        self.t3_merge = time() - self.t3_merge
        
        self.__fit__ = True
        return self



    def fit_transform(self, data):
        """Fit CLASSIX and return cluster labels.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features) or (n_samples,)
            Training data.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster label assigned to each sample.
            
        """
        
        return self.fit(data).labels_

    
        
    def predict(self, data): # This method only consistent when the clustering is well-separated / correctly clustered
        """Predict cluster labels for new samples.

        New samples are assigned to the cluster of the nearest fitted group
        starting point. This nearest-representative rule is most reliable when
        the fitted clusters are well separated.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features) or (n_features,)
            Samples to assign.
    
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted cluster labels.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        """
        if not hasattr(self, '__fit__'):
            raise NotFittedError("Please use .fit() method first.")
        
        # Lazy build label_change mapping (group id -> cluster label)
        if not hasattr(self, 'label_change'):
            if not hasattr(self, 'inverse_ind'):
                self.inverse_ind = np.argsort(self.ind)
            groups = np.asarray(self.groups_)
            self.label_change = dict(zip(groups[self.inverse_ind], self.labels_))
        
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.dtype != 'float64':
            data = data.astype('float64')
        
        n_new = data.shape[0]
        labels = np.zeros(n_new, dtype=int)
        
        # Preprocess new data according to the fitted metric
        if self.metric == 'euclidean':
            processed_data = self.preprocessing(data)
            # Euclidean splist_ is 2D, use first column
            group_centers = self.sp_data_pts  # precomputed during fit
            dists = distance.cdist(group_centers, processed_data, metric='euclidean')
        
        elif self.metric == 'manhattan':
            # Manhattan preprocessing
            mu_new = data.min(axis=0)
            processed_data = data - mu_new
            sort_vals_new = np.sum(processed_data, axis=1)
            mext_new = np.median(sort_vals_new) or 1.0
            processed_data /= mext_new
            
            # splist_ is 1D for manhattan
            group_centers = self.sp_data_pts
            dists = distance.cdist(group_centers, processed_data, metric='cityblock')
        
        elif self.metric == 'tanimoto':
            # No preprocessing for tanimoto
            processed_data = data
            
            group_centers_dense = self.sp_data_pts
            group_centers_sparse = sparse.csr_matrix(group_centers_dense)
            new_data_sparse = sparse.csr_matrix(processed_data)
            
            ips_groups = group_centers_sparse.dot(new_data_sparse.T).toarray()
            sum_groups = np.sum(group_centers_dense, axis=1, keepdims=True)
            sum_new = np.sum(processed_data, axis=1, keepdims=True).T
            denom = sum_groups + sum_new - ips_groups
            tanimoto_sim = ips_groups / np.where(denom == 0, 1e-12, denom)  # Avoid division by zero
            dists = 1 - tanimoto_sim
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Find nearest group center
        nearest_group_idx = np.argmin(dists, axis=0)
        
        # Map nearest group index to its aggregation group label, then to cluster label
        for i in range(n_new):
            group_idx = nearest_group_idx[i]
            
            # Unified way to get starting point position in sorted data
            if self.splist_.ndim == 2:  # Euclidean case
                sp_pos = self.splist_[group_idx, 0]
            else:  # Manhattan / Tanimoto case (1D)
                sp_pos = self.splist_[group_idx]
            
            # Aggregation group label at the starting point
            agg_group_label = self.groups_[sp_pos]
            
            # Map to final cluster label
            labels[i] = self.label_change.get(agg_group_label, -1)  # -1 if not found (rare)
        
        return labels
        
    
    
    def merging(self, data, agg_labels, splist, ind, sort_vals, radius=0.5, method="distance", minPts=1):
        """Merge aggregation groups into final clusters.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Preprocessed and sorted training data.
        
        agg_labels : array-like of shape (n_samples,)
            Aggregation group label for each sorted sample.
        
        splist : ndarray
            Starting-point information produced by aggregation.
        
        ind : ndarray of shape (n_samples,)
            Indices mapping sorted data back to the original input order.

        sort_vals : ndarray of shape (n_samples,)
            Sorting values associated with the sorted data.

        radius : float, default=0.5
            Aggregation radius used as the base radius for merging.
        
        method : {'distance', 'density'}, default='distance'
            Group merging strategy.

        minPts : int, default=1
            Minimum valid cluster size.


        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels in the original input order.
        """

        if method == 'density':

            agg_labels = np.asarray(agg_labels)
            labels = agg_labels.copy()
            
            self.merge_groups, self.connected_pairs_ = self._density_merge(data, np.int64(splist), 
                                                                             radius, sort_vals=sort_vals, 
                                                                             half_nrm2=self.__half_nrm2)
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
            # there are 4 clusters, the associated number of objects inside clusters are respectively of 5, 20, 25, 50. 
            # The 10th percentlie (we set percent=10, noise_mergeScale=0.1) of (5, 20, 25, 50) is 14, 
            # and we calculate threshold = 100 * noise_mergeScale =  10. Obviously, the first cluster with number of objects 5
            # satisfies both condition 5 < 14 and 5 < 10, so we classify the objects inside first cluster as outlier.
            # And then we allocate the objects inside the outlier cluster into other closest cluster.
            # This method is quite effective at solving the noise arise from small tolerance (radius).
            
            self.old_cluster_count = collections.Counter(labels)
            
            self.t4_minPts = time()
            if minPts >= 1:
                potential_noise_labels = self.outlier_filter(min_samples=minPts) # calculate the min_samples directly
                SIZE_NOISE_LABELS = len(potential_noise_labels) 
                if SIZE_NOISE_LABELS == len(np.unique(labels)):
                    warnings.warn(
                        "Setting of noise related parameters is not correct, degenerate to the method without noises detection.", 
                    DeprecationWarning)
                else:
                    for i in np.unique(potential_noise_labels):
                        labels[labels == i] = maxid # marked as noises, 
                                                    # the label number is not included in any of existing labels (maxid).
                            
                if SIZE_NOISE_LABELS > 0:
                    self.clean_index_ = labels != maxid
                    agln = agg_labels[self.clean_index_]
                    label_change = dict(zip(agln, labels[self.clean_index_])) # how object change group to cluster.
                    # allocate the outliers to the corresponding closest cluster.
                    
                    self.group_outliers_ = np.unique(agg_labels[~self.clean_index_]) # abnormal groups
                    unique_agln = np.unique(agln)
                    splist_clean = splist[unique_agln]

                    if self.__post_alloc:
                        for nsp in self.group_outliers_:
                            alloc_class = np.argmin(
                                np.linalg.norm(data[splist_clean[:, 0].astype(int)] - data[int(splist[nsp, 0])], axis=1, ord=2)
                                )
                            
                            labels[agg_labels == nsp] = label_change[unique_agln[alloc_class]]
                    else:
                        labels[np.isin(agg_labels, self.group_outliers_)] = -1

            # remove noise cluster, avoid connecting two separate to a single cluster
            # the label with the maxid is label marked noises

            self.t4_minPts = time() - self.t4_minPts
            
        else:
            self.__half_nrm2 = self.__half_nrm2[self.splist_[:, 0]]

            labels, self.old_cluster_count, SIZE_NOISE_LABELS = self._distance_merge(data=data, 
                                                                    labels=agg_labels,
                                                                    splist=np.int64(splist),
                                                                    radius=radius,
                                                                    minPts=minPts,
                                                                    scale=self.mergeScale_, 
                                                                    sort_vals=sort_vals,
                                                                    half_nrm2=self.__half_nrm2
                                                                )
            
            
        self.inverse_ind = np.argsort(ind)
        labels = labels[self.inverse_ind]
        
        if self.__verbose == 1:
            nr_old_clust_count = len(self.old_cluster_count)
            print("""CLASSIX aggregated the {datalen} data points into {num_group} groups. """.format(datalen=len(data), num_group=splist.shape[0]))
            print("""In total, {dist:.0f} distances were computed ({avg:.1f} per data point). """.format(dist=self.nrDistComp_, avg=self.nrDistComp_/len(data)))
            print("""The {num_group} groups were merged into {c_size} clusters.""".format(
                num_group=splist.shape[0], c_size=nr_old_clust_count))
            
            if nr_old_clust_count > 20:
                print("The largest 20 clusters have the following sizes:")
            else:
                print("The clusters have the following sizes:")
                
            self.pprint_format(self.old_cluster_count, truncate=self.__truncate)

            if self.minPts > 1 and SIZE_NOISE_LABELS > 0:
                print("As minPts is {minPts}, the number of clusters has been reduced to {r}.".format(
                    minPts=self.minPts, r=len(np.unique(labels))
                ))
                
            print("Use the verbose=0 parameter to suppress this info.\nUse the .explain() method to explain the clustering.")

        return labels 
    
    
    def _group_sizes_for_minpts(self):
        """Return aggregation group sizes as an integer array."""
        if self.splist_.ndim == 2:
            return self.splist_[:, 1].astype(int)
        return np.asarray(self.group_sizes_, dtype=int)


    @staticmethod
    def _renumber_labels(labels):
        """Renumber non-negative labels to consecutive integers."""
        labels = np.asarray(labels).copy()
        unique_labels = [label for label in np.unique(labels) if label >= 0]
        label_map = {old: new for new, old in enumerate(unique_labels)}
        for old, new in label_map.items():
            labels[labels == old] = new
        return labels


    def _distance_to_group_centers(self, group_id):
        """Compute distances from one group center to all group centers."""
        xi = self.sp_data_pts[group_id]
        if self.metric == 'euclidean':
            return np.linalg.norm(self.sp_data_pts - xi, axis=1, ord=2)
        if self.metric == 'manhattan':
            return np.sum(np.abs(self.sp_data_pts - xi), axis=1)
        if self.metric == 'tanimoto':
            ips = np.matmul(self.sp_data_pts, xi)
            denom = np.sum(self.sp_data_pts, axis=1) + np.sum(xi) - ips
            return 1 - ips / np.where(denom == 0, 1e-15, denom)
        raise ValueError(f"Unsupported metric: {self.metric}")


    def _compute_distance_base_labels(self, minPts=None):
        """Compute group labels after distance merging and before minPts."""
        n_groups = self.splist_.shape[0]
        labels = np.arange(n_groups)
        group_sizes = self._group_sizes_for_minpts()
        active = np.ones(n_groups, dtype=bool)
        if not self.__mergeTinyGroups and minPts is not None:
            active = group_sizes >= minPts

        adjacency = None
        if self.metric in ('manhattan', 'tanimoto'):
            adjacency = np.zeros((n_groups, n_groups), dtype=np.int8)

        merge_radius = self.mergeScale_ * self.radius
        for i in range(n_groups):
            if not active[i]:
                continue

            dists = self._distance_to_group_centers(i)
            inds = np.where(dists <= merge_radius)[0]
            inds = inds[inds >= i]
            if not self.__mergeTinyGroups:
                inds = inds[active[inds]]
            if inds.size == 0:
                continue

            if adjacency is not None:
                adjacency[i, inds] = 1
                adjacency[inds, i] = 1

            connected_labels = np.unique(labels[inds])
            min_label = np.min(connected_labels)
            for label in connected_labels:
                labels[labels == label] = min_label

        return self._renumber_labels(labels), adjacency


    def _compute_density_base_labels(self):
        """Compute group labels after density merging and before minPts."""
        n_groups = self.splist_.shape[0]
        labels = np.arange(n_groups)
        next_label = n_groups
        for component in getattr(self, 'merge_groups', []):
            component = np.asarray(component, dtype=int)
            for group_id in component:
                labels[labels == group_id] = next_label
            next_label += 1
        return labels


    def _update_minpts_base_state(self):
        """Cache the group labels that are independent of minPts changes."""
        if self.group_merging is None or str(self.group_merging).lower() == 'none':
            self._base_group_labels_ = np.arange(self.splist_.shape[0])
            self._base_group_adjacency_ = None
        elif self.group_merging == 'density':
            self._base_group_labels_ = self._compute_density_base_labels()
            self._base_group_adjacency_ = None
        elif self.__mergeTinyGroups:
            self._base_group_labels_, self._base_group_adjacency_ = self._compute_distance_base_labels()
        else:
            self._base_group_labels_, self._base_group_adjacency_ = self._compute_distance_base_labels(
                minPts=self.minPts
            )


    def _apply_minpts_to_groups(self, base_group_labels, minPts, base_adjacency=None, allow_noise=False,
                                renumber=True):
        """Apply the minPts reassignment phase to pre-merged group labels."""
        group_sizes = self._group_sizes_for_minpts()
        base_group_labels = np.asarray(base_group_labels, dtype=int)
        final_group_labels = base_group_labels.copy()
        adjacency = None if base_adjacency is None else base_adjacency.copy()

        old_cluster_count = collections.Counter()
        for group_label, group_size in zip(base_group_labels, group_sizes):
            old_cluster_count[int(group_label)] += int(group_size)

        small_clusters = [
            label for label, size in old_cluster_count.items()
            if size < minPts
        ] if minPts >= 1 else []
        size_noise_labels = len(small_clusters)
        self.group_outliers_ = np.array([], dtype=int)
        self.clean_index_ = np.ones(len(self.groups_), dtype=bool)

        if size_noise_labels == 0:
            if renumber:
                final_group_labels = self._renumber_labels(final_group_labels)
            return final_group_labels, old_cluster_count, 0, adjacency

        if size_noise_labels == len(old_cluster_count):
            warnings.warn(
                "Setting of noise related parameters is not correct; keeping "
                "the clustering before minPts reassignment.",
                DeprecationWarning
            )
            if renumber:
                final_group_labels = self._renumber_labels(final_group_labels)
            return final_group_labels, old_cluster_count, size_noise_labels, adjacency

        small_group_mask = np.isin(base_group_labels, small_clusters)
        self.group_outliers_ = np.nonzero(small_group_mask)[0]
        self.clean_index_ = ~np.isin(self.groups_, self.group_outliers_)

        if allow_noise:
            final_group_labels[small_group_mask] = -1
            if renumber:
                final_group_labels = self._renumber_labels(final_group_labels)
            return final_group_labels, old_cluster_count, size_noise_labels, adjacency

        valid_clusters = {
            label for label, size in old_cluster_count.items()
            if size >= minPts
        }
        frozen_group_labels = base_group_labels.copy()
        for group_id in self.group_outliers_:
            dists = self._distance_to_group_centers(group_id)
            for nearest_group_id in np.argsort(dists, kind='stable'):
                target_cluster = frozen_group_labels[nearest_group_id]
                if target_cluster in valid_clusters:
                    final_group_labels[group_id] = target_cluster
                    if adjacency is not None:
                        adjacency[group_id, nearest_group_id] = 2
                        adjacency[nearest_group_id, group_id] = 2
                    break

        if renumber:
            final_group_labels = self._renumber_labels(final_group_labels)
        return final_group_labels, old_cluster_count, size_noise_labels, adjacency


    def _labels_from_group_labels(self, group_labels):
        """Map group labels to sample labels in the original input order."""
        labels_sorted = np.asarray(group_labels)[np.asarray(self.groups_, dtype=int)]
        return labels_sorted[self.inverse_ind]


    def _invalidate_label_dependent_cache(self):
        """Remove cached values that depend on the current cluster labels."""
        for attr in ('label_change', 'sp_info', 'sp_to_c_info', 'centers', 'connected_paths'):
            if hasattr(self, attr):
                delattr(self, attr)


    def minPtsChange(self, minPts):
        """Change ``minPts`` without rerunning preprocessing or aggregation.

        This method supports fast hyperparameter tuning after :meth:`fit`.
        CLASSIX users can fit once with a chosen ``radius`` and then adjust
        ``minPts`` to dissolve and reassign small clusters. When
        ``mergeTinyGroups=True``, only the minPts phase is recomputed. When
        ``mergeTinyGroups=False``, distance-based group merging is recomputed
        because tiny groups are excluded from merge edges according to the new
        ``minPts`` value.

        Parameters
        ----------
        minPts : int or float
            New minimum valid cluster size. The value is validated with the same
            rules as the constructor and rounded to an integer.

        Returns
        -------
        self : CLASSIX
            The estimator with updated ``minPts`` and ``labels_``.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        """
        if not hasattr(self, '__fit__'):
            raise NotFittedError("Please use .fit() method first.")

        self.minPts = minPts
        self.t4_minPts = time()
        self.inverse_ind = np.argsort(self.ind)

        if self.group_merging is None or str(self.group_merging).lower() == 'none':
            self.labels_ = np.asarray(self.groups_)[self.inverse_ind]
            self.t4_minPts = time() - self.t4_minPts
            self._invalidate_label_dependent_cache()
            return self

        if self.group_merging == 'density':
            base_group_labels = getattr(self, '_base_group_labels_', None)
            if base_group_labels is None:
                base_group_labels = self._compute_density_base_labels()
            base_adjacency = None
            allow_noise = not self.__post_alloc
            renumber = False
        else:
            if self.__mergeTinyGroups:
                base_group_labels = getattr(self, '_base_group_labels_', None)
                base_adjacency = getattr(self, '_base_group_adjacency_', None)
                if base_group_labels is None:
                    base_group_labels, base_adjacency = self._compute_distance_base_labels()
                    self._base_group_labels_ = base_group_labels
                    self._base_group_adjacency_ = base_adjacency
            else:
                base_group_labels, base_adjacency = self._compute_distance_base_labels(
                    minPts=self.minPts
                )
            allow_noise = False
            renumber = True

        group_labels, self.old_cluster_count, _, adjacency = self._apply_minpts_to_groups(
            base_group_labels=base_group_labels,
            minPts=self.minPts,
            base_adjacency=base_adjacency,
            allow_noise=allow_noise,
            renumber=renumber,
        )
        self._group_cluster_labels_ = group_labels
        self.labels_ = self._labels_from_group_labels(group_labels)
        if adjacency is not None:
            self.Adj = adjacency

        self.t4_minPts = time() - self.t4_minPts
        self._invalidate_label_dependent_cache()
        return self


    
    def explain(self, data, index1=None, index2=None, cmap='jet', showalldata=False, showallgroups=False, showsplist=False, max_colwidth=None, replace_name=None, 
                plot=False, figsize=(10, 7), figstyle="default", savefig=False, bcolor="#f5f9f9", obj_color="k", width=1.5,  obj_msize=160, sp1_color='lime', sp2_color='cyan',
                sp_fcolor="tomato", sp_marker="+", sp_size=72, sp_mcolor="k", sp_alpha=0.05, sp_pad=0.5, sp_fontsize=10, sp_bbox=None, sp_cmarker="+", sp_csize=110, 
                sp_ccolor="crimson", sp_clinewidths=2.7,  dp_fcolor="white", dp_alpha=0.5, dp_pad=2, dp_fontsize=10, dp_bbox=None,  show_all_grp_circle=False,
                show_connected_grp_circle=False, show_obj_grp_circle=True, color="red", connect_color="green", alpha=0.3, cline_width=2,  add_arrow=True, 
                arrow_linestyle="--", arrow_fc="darkslategrey", arrow_ec="k", arrow_linewidth=1, arrow_shrinkA=2, arrow_shrinkB=2, directed_arrow=0, 
                axis='off', include_dist=False, show_connected_label=True, figname=None, fmt="pdf"):
        
        """Print and optionally plot an explanation of the clustering.

        Called without indices, this method summarizes the fitted clustering.
        Called with ``index1``, it explains which aggregation group and cluster
        contain that sample. Called with both ``index1`` and ``index2``, it
        explains whether the two samples are in the same cluster and, when
        available, prints a path of connected groups.
        
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Original data used to fit the estimator.

        index1 : int, str, array-like of shape (n_features,), optional
            First sample to explain. Integers refer to row positions, strings
            refer to pandas index labels when the fitted data was a DataFrame,
            and array-like values are interpreted as out-of-sample points.
        
        index2 : int, str, array-like of shape (n_features,), optional
            Second sample to compare with ``index1``.
        
        cmap : str, default='Set3'
            Matplotlib colormap used for scatter plots.
        
        showalldata : bool, default=False
            If ``False``, plots with more than 100,000 samples are subsampled.

        showallgroups : bool, default=False
            Whether to show all group starting-point markers and labels.

        showsplist : bool, default=False
            Whether to print a table of selected or all group starting points.
        
        max_colwidth : int, optional
            Maximum printed column width for pandas tables.
            
        replace_name : str or list, optional
            Display name or names used in printed explanations instead of the
            raw sample indices.

        plot : bool, default=False
            Whether to draw a visualization.
        
        figsize : tuple, default=(9, 6)
            Matplotlib figure size.

        figstyle : str, default="default"
            Matplotlib style sheet name.
        
        savefig : bool, default=False
            Whether to save the generated figure in an ``img`` directory.
        
        bcolor : str, default="#f5f9f9"
            Color for figure background.
        
        obj_color : str, default as "k"
            Color for the text of data of index1 and index2.
        
        obj_msize : float, optional:
            Size for markers for data of index1 and index2.
    
        sp_fcolor : str, default='tomato'
            The color marked for group centers text box. 
        
        sp_marker : str, default="+"
            The marker for the start points.
        
        sp_size : int, default=66
            The marker size for the start points.
        
        sp_mcolor : str, default='k'
            The color marked for startpoint points scatter marker.

        sp_alpha : float, default=0.3
            The value setting for transparency of text box for group centers. 
            
        sp_pad : int, default=2
            The size of text box for group centers. 
        
        sp_bbox : dict, optional
            Dict with properties for patches.FancyBboxPatch for group centers.
    
        sp_fontsize : int, optional
            The fontsize for text marked for group centers. 

        sp_cmarker : str, default="+"
            The marker for the connected group centers.
        
        sp_csize : int, default=100
            The marker size for the connected group centers.
        
        sp_ccolor : str, default="crimson"
            The marker color for the connected group centers.
        
        sp_clinewidths : str, default=2.5
            The marker width for the connected group centers. 

        dp_fcolor : str, default='white'
            The color marked for specified data objects text box. 
            
        dp_alpha : float, default=0.5
            The value setting for transparency of text box for specified data objects. 
            
        dp_pad : int, default=2
            The size of text box for specified data objects. 
            
        dp_fontsize : int, optional
            The fontsize for text marked for specified data objects.    
                      
        dp_bbox : dict, optional
            Dict with properties for patches.FancyBboxPatch for specified data objects.
        
        show_all_grp_circle : bool, default=False
            Whether or not to show all groups' periphery within the objects' clusters 
            (only applies to when data dimension is less than or equal to 2).
        
        show_connected_grp_circle : bool, default=False
            Whether or not to show all connected groups' periphery within the objects' clusters 
            (only applies to when data dimension is less than or equal to 2).
        
        show_obj_grp_circle : bool, default=True
            Whether or not to show the groups' periphery of the objects
            (only applies to when data dimension is less than or equal to 2).

        color : str, default='red'
            Color for text of group centers labels in visualization. 
        
        alpha : float, default=0.3
            Transparency of data points. Scalar or None. 
    
        cline_width : float, default=2
            Set the patch linewidth of circle for group centers.

        add_arrow : bool, default=False 
            Whether or not add arrows for connected paths.

        arrow_linestyle : str, default='--' 
            Linestyle for arrow.
        
        arrow_fc : str, default='darkslategrey' 
            Face color for arrow.

        arrow_ec : str, default='k'
            Edge color for arrow.

        arrow_linewidth : float, default=1
            Set the linewidth of the arrow edges.

        directed_arrow : int, default=0
            Whether or not the edges for arrows is directed.
            Values at {-1, 0, 1}, 0 refers to undirected, -1 refers to the edge direction opposite to 1.
        
        shrinkA, shrinkB : float, default=2
            Shrinking factor of the tail and head of the arrow respectively.

        axis : bool, default=True
            Whether to show plot axes.
        
        include_dist : bool, default=False
            Whether to include weighted distance information when computing
            shortest paths between groups.
            
        show_connected_label : bool, default=True
            Whether to include pandas index labels for connected path samples.
            
        figname : str, optional
            Set the figure name for the image to be saved.
            
        fmt : str, default='pdf'
            File format used when ``savefig=True``.

        Returns
        -------
        None
            The explanation is printed and, optionally, plotted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        
        """
        from scipy.sparse.linalg import svds
        self.t5_finalize = time()
        # -----------------------------second method--------------------------------
        if sp_bbox is None:
            sp_bbox = dict()
            sp_bbox['facecolor'] = sp_fcolor
            sp_bbox['alpha'] = sp_alpha
            sp_bbox['pad'] = sp_pad
       
        if dp_bbox is None:
            dp_bbox = dict()
            dp_bbox['facecolor'] = dp_fcolor
            dp_bbox['alpha'] = dp_alpha
            dp_bbox['pad'] = dp_pad


        if hasattr(self, '__fit__'):
            groups_ = np.array(self.groups_)
            groups_ = groups_[self.inverse_ind]
            if not hasattr(self, 'label_change'):
                self.label_change = dict(zip(groups_, self.labels_)) # how object change group to cluster.
        else:
            raise NotFittedError("Please use .fit() method first.")

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)    
        data_size = data.shape[0]
        feat_dim = data.shape[1]


        if not hasattr(self, 'self.sp_to_c_info'): #  ensure call PCA and form groups information table only once
            
            if feat_dim > 2:
                _U, self._s, self._V = svds(data, k=2, return_singular_vectors=True)
                self.x_pca = np.matmul(data, self._V[(-self._s).argsort()].T)
                sp_indices = self.splist_[:, 0] if self.splist_.ndim == 2 else self.splist_
                self.s_pca = self.x_pca[self.ind[sp_indices.astype(int)]]
                
            elif feat_dim == 2:
                self.x_pca = data.copy()
                self.s_pca = self.sp_data_pts

            else: # when data is one-dimensional, no PCA transform
                self.x_pca = np.ones((len(data.copy()), 2))
                self.x_pca[:, 0] = data[:, 0]
                self.s_pca = np.ones((len(self.splist_), 2))
                self.s_pca[:, 0] = self.sp_data_pts[:, 0].reshape(-1) 
                
            self.form_starting_point_clusters_table(data=data[self.ind])
            
        if index1 is None and index2 is not None:
            raise ValueError("Please enter a valid value for index1.")
        
        
        # pd.options.display.max_colwidth = colwidth
        dash_line = "--------"*5 

        
        if index1 is None: # analyze in the general way with a global view
            if plot:
                self.explain_viz(showalldata=showalldata, alpha=alpha, cmap=cmap, figsize=figsize, showallgroups=showallgroups, figstyle=figstyle, bcolor=bcolor, savefig=savefig, 
                                 fontsize=sp_fontsize, bbox=sp_bbox, sp_marker=sp_marker, sp_mcolor=sp_mcolor, width=width, axis=axis, fmt=fmt)
                

            
            print("CLASSIX clustered {length:.0f} data points with {dim:.0f} features.\n".format(length=data_size, dim=feat_dim) + 
                "The radius parameter was set to {tol:.2f} and minPts was set to {minPts:.0f}.\n".format(tol=self.radius, minPts=self.minPts) +
                "As the provided data was auto-scaled by a factor of 1/{scl:.2f},\npoints within a radius R={tol:.2f}*{scl:.2f}={tolscl:.2f} were grouped together.\n".format(scl=self.dataScale_, tol=self.radius, tolscl=self.dataScale_*self.radius) + 
                "In total, {dist:.0f} distances were computed ({avg:.1f} per data point).\n".format(dist=self.nrDistComp_, avg=self.nrDistComp_/data_size) + 
                "This resulted in {groups:.0f} groups, each with a unique group center.\n".format(groups=self.splist_.shape[0]) + 
                "These {groups:.0f} groups were subsequently merged into {num_clusters:.0f} clusters. ".format(groups=self.splist_.shape[0], num_clusters=len(np.unique(self.labels_)))
                 )
            
            if showsplist:
                print("A list of all group centers is shown below.")
                print(dash_line)
                print(self.sp_info.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                print(dash_line)       
            else:
                if plot:
                    print("\nTo explain the clustering of individual data points, use\n " + 
                          ".explain(index1) or .explain(index1,index2) with data indices.")
                else:
                    print("\nFor a visualisation of the clusters, use .explain(plot=True).\n" +
                          "To explain the clustering of individual data points, use\n" + 
                          ".explain(index1) or .explain(index1,index2) with data indices.")
            
        else: # index is not None, explain(index1)
            if isinstance(index1, numbers.Integral) or isinstance(index1, float):
                index1_id, index1 = int(index1), int(index1)
                object1 = self.x_pca[index1_id] # data has been normalized
                agg_label1 = groups_[index1_id] # get the group index for object1
            
            elif isinstance(index1, str):
                if hasattr(self, '_index_data'):
                    if index1 in self._index_data:
                        index1_id = np.where(self._index_data == index1)[0][0]
                        if len(set(self._index_data)) != len(self._index_data):
                            warnings.warn("The data index contains duplicates.") # SG: Can this even happen with dataframes?
                            object1 = self.x_pca[index1_id]
                            agg_label1 = groups_[index1_id]
                        else:
                            object1 = self.x_pca[index1_id]
                            agg_label1 = groups_[index1_id]
                            
                    else:
                        raise ValueError("Please use a valid value for index1.")
                else:
                    raise ValueError("Please use a valid value for index1.")
                    
            elif isinstance(index1, list) or isinstance(index1, np.ndarray):
                index1_id = -1
                index1 = np.array(index1)
                object1 = (index1 - self.mu_) / self.dataScale_ # allow for out-sample data
                
                if feat_dim > 2:
                    object1 = np.matmul(object1, self._V[np.argsort(self._s)].T)
                    
                agg_label1 = np.argmin(np.linalg.norm(self.s_pca - object1, axis=1, ord=2)) # get the group index for object1

                    
            else:
                raise ValueError("Please use a valid value for index1.")
                
            
            # explain one object
            if index2 is None:
                if replace_name is not None:
                    if isinstance(replace_name, list):
                        index1 = replace_name[0]
                    else:
                        index1 = replace_name
                else:
                    index1 = index1

                cluster_label1 = self.label_change[agg_label1]
                
                if plot:
                    from matplotlib import pyplot as plt

                    if self.x_pca.shape[0] > 1e5 and not showalldata:
                        print("Too many data points for plot. Randomly subsampled 1e5 points.")
                        selectInd = np.random.choice(self.x_pca.shape[0], 100000, replace=False)      
                    else:
                        selectInd = np.arange(self.x_pca.shape[0])
                     
                    plt.style.use(style=figstyle)
                    fig, ax = plt.subplots(figsize=figsize)
                    
                    ax.set_facecolor(bcolor)

                    s_pca = self.s_pca[self.sp_info.Cluster == cluster_label1]
                    
                    ax.scatter(self.x_pca[selectInd, 0], self.x_pca[selectInd, 1], s=60, marker=".", linewidth=0.0*width, 
                               cmap=cmap, alpha=alpha, c=self.labels_[selectInd]
                              )
                    
                    ax.scatter(s_pca[:, 0], s_pca[:, 1], marker=sp_marker, label='group centers in cluster #{0}'.format(cluster_label1), 
                               s=sp_size, linewidth=0.9*width, c=sp_mcolor, alpha=0.4)
                    
                    if feat_dim <= 2 and show_obj_grp_circle:
                        ax.add_patch(plt.Circle((self.s_pca[agg_label1, 0], self.s_pca[agg_label1, 1]), self.radius, fill=False, 
                                                color=sp1_color, alpha=0.5, lw=cline_width*1.5, clip_on=False))
                        
                    
                    if dp_fontsize is None:
                        ax.text(object1[0], object1[1], s=' ' + str(index1), bbox=dp_bbox, color=obj_color, zorder=1, ha='left', va='bottom')
                    else:
                        ax.text(object1[0], object1[1], s=' ' + str(index1), fontsize=dp_fontsize, bbox=dp_bbox, color=obj_color, zorder=1, ha='left', va='bottom')

                    
                    if isinstance(index1, str):
                        ax.scatter(object1[0], object1[1], marker="*", s=obj_msize, label='{} '.format(index1))
                    else:
                        ax.scatter(object1[0], object1[1], marker="*", s=obj_msize, label='data point {} '.format(index1))

                    for i in range(s_pca.shape[0]):
                        if feat_dim <= 2 and show_all_grp_circle:
                            ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False, color=color,
                                                     alpha=0.5, lw=cline_width*1.5, clip_on=False))
                        
                        if showallgroups:
                            if sp_fontsize is None:
                                ax.text(s_pca[i, 0], s_pca[i, 1],
                                        s=str(self.sp_info.Group[self.sp_info.Cluster == cluster_label1].astype(int).values[i]),
                                        bbox=sp_bbox, zorder=1, ha='left'
                                )
                            else:
                                ax.text(s_pca[i, 0], s_pca[i, 1],
                                        s=str(self.sp_info.Group[self.sp_info.Cluster == cluster_label1].astype(int).values[i]),
                                        fontsize=sp_fontsize, bbox=sp_bbox, zorder=1, ha='left'
                                )

                    ax.scatter(self.s_pca[agg_label1, 0], self.s_pca[agg_label1, 1], 
                               marker='.', s=sp_csize*0.3, c=sp1_color, linewidths=sp_clinewidths, 
                               label='group center {0}'.format(agg_label1)
                               )

                    ax.set_aspect('equal', adjustable='datalim')
                    ax.plot()

                    ax.legend(ncols=3, loc='best') # bbox_to_anchor=(0.5, -0.2)
                    
                    if axis:
                        ax.axis('on')
                        if feat_dim > 1:
                            ax.set_xlabel("1st principal component")
                            ax.set_ylabel("2nd principal component")
                        else:
                            ax.set_xlabel("1st principal component")
                    else:
                        ax.axis('off') # the axis here may not be consistent, so hide.
                    
                    ax.set_title("""{num_clusters:.0f} clusters (radius={tol:.2f}, minPts={minPts:.0f})""".format(
                        num_clusters=len(np.unique(self.labels_)),tol=self.radius, minPts=self.minPts))
                    
                    ax.spines['right'].set_color('none')
                    ax.spines['top'].set_color('none')

                    if savefig:
                        if not os.path.exists("img"):
                            os.mkdir("img")
                        if fmt == 'pdf':
                            if figname is not None:
                                fm = 'img/' + str(figname) + '.pdf'
                            else:
                                fm = 'img/sample.pdf'
                            plt.savefig(fm, bbox_inches='tight')
                        elif fmt == 'png':
                            if figname is not None:
                                fm = 'img/' + str(figname)  + '.png'
                            else:
                                fm = 'img/sample.png'
                            plt.savefig(fm, bbox_inches='tight')
                        else:
                            if figname is not None:
                                fm = 'img/' + str(figname) + '.' + fmt
                            else:
                                fm = 'img/sample' + '.' + fmt
                        
                        print("Image successfully saved as", fm)
                    
                    plt.show()
                    
                
                if showsplist:
                    select_sp_info = self.sp_info.iloc[[agg_label1]].copy(deep=True)
                    select_sp_info.loc[:, 'Label'] = str(np.round(index1,3))
                    print(dash_line)
                    print(select_sp_info.to_string(justify='center', index=False, max_colwidth=max_colwidth))
                    print(dash_line)       

                print(
                    """Data point %(index1)s is in group %(agg_id)i, which was merged into cluster #%(m_c)i."""% {
                        "index1":index1, "agg_id":agg_label1, "m_c":cluster_label1
                    }
                )
                if not plot:
                    print("Use .explain(..., plot=True) for a visual representation.")

            # explain two objects relationship
            else: 
                if isinstance(index2, numbers.Integral) or isinstance(index2, float):
                    index2_id, index2 = int(index2), int(index2)
                    object2 = self.x_pca[index2_id] # data has been normalized
                    agg_label2 = groups_[index2_id] # get the group index for object2
                    
                elif isinstance(index2, str):
                    if hasattr(self, '_index_data'):
                        if index2 in self._index_data:
                            index2_id = np.where(self._index_data == index2)[0][0]
                            if len(set(self._index_data)) != len(self._index_data):
                                warnings.warn("The data index contains duplicates.") # sg: can this even happen with dataframes?
                                object2 = self.x_pca[index2_id]
                                agg_label2 = groups_[index2_id]
                            else:
                                object2 = self.x_pca[index2_id]
                                agg_label2 = groups_[index2_id]
                        else:
                            raise ValueError("Please use a valid value for index2.")
                    else:
                        raise ValueError("Please use a valid value for index2.")

                
                elif isinstance(index2, list) or isinstance(index2, np.ndarray):
                    index2_id = -1
                    index2 = np.array(index2)
                    object2 = (index2 - self.mu_) / self.dataScale_ # allow for out-sample data
                    
                    if feat_dim > 2:
                        object2 = np.matmul(object2, self._V[np.argsort(self._s)].T)
                    
                    agg_label2 = np.argmin(np.linalg.norm(self.s_pca - object2, axis=1, ord=2)) # get the group index for object2
                
                else:
                    raise ValueError("Please use a valid value for index2.")

                if showsplist:
                    
                    select_sp_info = self.sp_info.iloc[[agg_label1, agg_label2]].copy(deep=True)
                    if isinstance(index1, int) or isinstance(index1, str):
                        select_sp_info.loc[:, 'Label'] = [index1, index2]
                    else:
                        select_sp_info.loc[:, 'Label'] = [str(np.round(index1, 3)), str(np.round(index2, 3))]
                        
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

                if agg_label1 == agg_label2: # when ind1 & ind2 are in the same group
                    connected_paths = [agg_label1]
                else:
                    if self.metric in ('manhattan', 'tanimoto') and hasattr(self, 'Adj'):
                        if self.metric == 'manhattan':
                            from .merge_md import bfs_shortest_path
                        else:
                            from .merge_td import bfs_shortest_path
                        
                        distm = pairwise_distances(self.sp_data_pts)
                        
                        if cluster_label1 == cluster_label2:
                            connected_paths = bfs_shortest_path(self.Adj, agg_label1, agg_label2)
                            
                            if connected_paths is None or len(connected_paths) < 1:
                                connected_paths_vis = None
                                connected_paths = []
                            else:
                                path_parts = [str(connected_paths[0])]
                                for k in range(len(connected_paths) - 1):
                                    if self.Adj[connected_paths[k], connected_paths[k+1]] == 2:
                                        path_parts.append("(minPts) <-> " + str(connected_paths[k+1]))
                                    else:
                                        path_parts.append("<-> " + str(connected_paths[k+1]))
                                connected_paths_vis = " ".join(path_parts)
                        else:
                            connected_paths = []
                    else:
                        from scipy.sparse import csr_matrix
                        
                        distm = pairwise_distances(self.sp_data_pts)
                        distmf = (distm <= self.radius*self.mergeScale_).astype(int)
                        csr_dist_m = csr_matrix(distmf)
                            
                        if cluster_label1 == cluster_label2:
                            connected_paths = find_shortest_dist_path(agg_label1, csr_dist_m, agg_label2, unweighted=not include_dist)
                            
                            connected_paths.reverse()
                            
                            if len(connected_paths)<1:
                                connected_paths_vis = None
                            else:    
                                connected_paths_vis = " <-> ".join([str(group) for group in connected_paths]) 
                            
                        else: 
                            connected_paths = []
                        
                if plot:
                    from matplotlib import pyplot as plt

                    if self.x_pca.shape[0] > 1e5 and not showalldata:
                        print("Too many data points for plot. Randomly subsampled 1e5 points.")
                        selectInd = np.random.choice(self.x_pca.shape[0], 100000, replace=False)      
                    else:
                        selectInd = np.arange(self.x_pca.shape[0])
                        
                    plt.style.use(style=figstyle)
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.set_facecolor(bcolor)
                    
                    # select indices
                    union_ind = np.where((self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2))[0]
                    s_pca = self.s_pca[union_ind]
                    
                    ax.scatter(self.x_pca[selectInd, 0], self.x_pca[selectInd, 1], s=60, marker=".", c=self.labels_[selectInd], linewidth=0*width, cmap=cmap, alpha=alpha)
                    ax.scatter(s_pca[:,0], s_pca[:,1], label='group centers', marker=sp_marker, s=sp_size, c=sp_mcolor, linewidth=0.9*width, alpha=0.4)

                    
                    if feat_dim <= 2 and show_obj_grp_circle:
                        ax.add_patch(plt.Circle((self.s_pca[agg_label1, 0], self.s_pca[agg_label1, 1]), self.radius, fill=False,
                                        color=sp1_color, alpha=0.5, lw=cline_width*1.5, clip_on=False))
                        
                        ax.add_patch(plt.Circle((self.s_pca[agg_label2, 0], self.s_pca[agg_label2, 1]), self.radius, fill=False,
                                        color=sp2_color, alpha=0.5, lw=cline_width*1.5, clip_on=False))
                                        
                    if isinstance(index1, int) or isinstance(index1, str):
                        if dp_fontsize is None:
                            ax.text(object1[0], object1[1], s=' '+str(index1), ha='left', va='bottom', zorder=1, bbox=dp_bbox, color=obj_color)
                            ax.text(object2[0], object2[1], s=' '+str(index2), ha='left', va='bottom', zorder=1, bbox=dp_bbox, color=obj_color)
                        else:
                            ax.text(object1[0], object1[1], s=' '+str(index1), ha='left', va='bottom', zorder=1, fontsize=dp_fontsize, bbox=dp_bbox, color=obj_color)
                            ax.text(object2[0], object2[1], s=' '+str(index2), ha='left', va='bottom', zorder=1, fontsize=dp_fontsize, bbox=dp_bbox, color=obj_color)
                    else:
                        if dp_fontsize is None:
                            ax.text(object1[0], object1[1], s=' '+'index 1', ha='left', va='bottom', zorder=1, bbox=dp_bbox, color=obj_color)
                            ax.text(object2[0], object2[1], s=' '+'index 2', ha='left', va='bottom', zorder=1, bbox=dp_bbox, color=obj_color)
                        else:
                            ax.text(object1[0], object1[1], s=' '+'index 1', ha='left', va='bottom', zorder=1, fontsize=dp_fontsize, bbox=dp_bbox, color=obj_color)
                            ax.text(object2[0], object2[1], s=' '+'index 2', ha='left', va='bottom', zorder=1, fontsize=dp_fontsize, bbox=dp_bbox, color=obj_color)
                    
                    if isinstance(index1, str):
                        ax.scatter(object1[0], object1[1], marker="*", s=obj_msize, 
                               label='{} '.format(index1)+'(cluster #{0})'.format(
                                   cluster_label1)
                            )
                    else:
                        ax.scatter(object1[0], object1[1], marker="*", s=obj_msize, 
                               label='data point {} '.format(index1)+'(cluster #{0})'.format(
                                   cluster_label1)
                            )
                    
                    if isinstance(index2, str):
                        ax.scatter(object2[0], object2[1], marker="*", s=obj_msize, 
                               label='{} '.format(index2)+'(cluster #{0})'.format(
                                   cluster_label1)
                            )
                    else:
                        ax.scatter(object2[0], object2[1], marker="*", s=obj_msize,
                                label='data point {} '.format(index2)+'(cluster #{0})'.format(
                                    cluster_label2)
                            )

                    for i in range(s_pca.shape[0]):
                        if feat_dim <= 2 and show_all_grp_circle:
                                ax.add_patch(plt.Circle((s_pca[i, 0], s_pca[i, 1]), self.radius, fill=False,
                                                    color=color, alpha=0.5, lw=cline_width*1.5, clip_on=False)
                                                    )

                        if showallgroups:
                            if sp_fontsize is None:
                                ax.text(s_pca[i, 0], s_pca[i, 1], 
                                        s=self.sp_info.Group[
                                            (self.sp_info.Cluster == cluster_label1) | (self.sp_info.Cluster == cluster_label2)
                                            ].values[i].astype(int).astype(str),
                                        zorder=1, ha='left', bbox=sp_bbox
                                )

                            else:
                                ax.text(s_pca[i, 0], s_pca[i, 1], 
                                        s=self.sp_info.Group[union_ind].values[i].astype(int).astype(str),
                                        fontsize=sp_fontsize, ha='left', bbox=sp_bbox
                                )
                                
                    for i in connected_paths:
                        # draw circle for connected group centers or not, 
                        # and also determine the marker of the connected group centers.
                        if i == connected_paths[0]: 
                            ax.scatter(self.s_pca[i,0], self.s_pca[i,1], marker=sp_cmarker, s=sp_csize, 
                                   label='connected groups', c=sp_ccolor, linewidths=sp_clinewidths)
                        else:
                            ax.scatter(self.s_pca[i,0], self.s_pca[i,1], marker=sp_cmarker, s=sp_csize, c=sp_ccolor, 
                                       linewidths=sp_clinewidths)

                        if feat_dim <= 2 and show_connected_grp_circle:
                            ax.add_patch(plt.Circle((self.s_pca[i, 0], self.s_pca[i, 1]), self.radius, fill=False,
                                            color=connect_color, alpha=0.5, lw=cline_width*1.5, clip_on=False))
                                
                
                    ax.scatter(self.s_pca[agg_label1, 0], self.s_pca[agg_label1, 1], 
                            marker='.', s=sp_csize*0.3, c=sp1_color, linewidths=sp_clinewidths, 
                            label='group center {0}'.format(agg_label1)
                            )

                    ax.scatter(self.s_pca[agg_label2, 0], self.s_pca[agg_label2, 1], 
                            marker='.', s=sp_csize*0.3, c=sp2_color, linewidths=sp_clinewidths, 
                            label='group center {0}'.format(agg_label2)
                            )
                    
                    nr_cps = len(connected_paths)

                    if add_arrow:
                        for i in range(nr_cps-1):
                            arrowStart=(self.s_pca[connected_paths[i], 0], self.s_pca[connected_paths[i], 1])
                            arrowStop=(self.s_pca[connected_paths[i+1], 0], self.s_pca[connected_paths[i+1], 1])

                            if directed_arrow == 0:
                                ax.annotate("", arrowStop, 
                                            xytext=arrowStart, 
                                            arrowprops=dict(arrowstyle="-|>",
                                                            shrinkA=arrow_shrinkA, 
                                                            shrinkB=arrow_shrinkB, 
                                                            edgecolor=arrow_fc,
                                                            facecolor=arrow_ec,
                                                            linestyle=arrow_linestyle,
                                                            linewidth=arrow_linewidth
                                                            )
                                            )
                                
                                ax.annotate("", arrowStart,
                                            xytext=arrowStop, 
                                            arrowprops=dict(arrowstyle="-|>",
                                                            shrinkA=arrow_shrinkA, 
                                                            shrinkB=arrow_shrinkB, 
                                                            edgecolor=arrow_fc,
                                                            facecolor=arrow_ec,
                                                            linestyle=arrow_linestyle,
                                                            linewidth=arrow_linewidth
                                                            )
                                            )
                                    
                            elif directed_arrow == 1:
                                ax.annotate("", arrowStop,
                                            xytext=arrowStart,
                                            arrowprops=dict(arrowstyle="-|>",
                                                            shrinkA=arrow_shrinkA, 
                                                            shrinkB=arrow_shrinkB, 
                                                            edgecolor=arrow_fc,
                                                            facecolor=arrow_ec,
                                                            linestyle=arrow_linestyle,
                                                            linewidth=arrow_linewidth
                                                            )
                                            )

                            else:
                                ax.annotate("", arrowStart,
                                            xytext=arrowStop, 
                                            arrowprops=dict(arrowstyle="-|>",
                                                            shrinkA=arrow_shrinkA, 
                                                            shrinkB=arrow_shrinkB, 
                                                            edgecolor=arrow_fc,
                                                            facecolor=arrow_ec,
                                                            linestyle=arrow_linestyle,
                                                            linewidth=arrow_linewidth
                                                            )
                                            )
                                

                    if cluster_label1 == cluster_label2 and len(connected_paths) > 1: # change the order of legend
                        handles, lg_labels = ax.get_legend_handles_labels()
                        lg_labels = [lg_labels[i] for i in [0,3,1,2,4,5]]
                        handles = [handles[i] for i in [0,3,1,2,4,5]]
                        ax.legend(handles, lg_labels, ncols=3, loc='best')

                    else:
                        ax.legend(ncols=3, loc='best')

                    ax.set_aspect('equal', adjustable='datalim')
                    ax.set_title("""{num_clusters:.0f} clusters (radius={tol:.2f}, minPts={minPts:.0f})""".format(
                        num_clusters=len(np.unique(self.labels_)),tol=self.radius, minPts=self.minPts))
                    
                    ax.spines['right'].set_color('none')
                    ax.spines['top'].set_color('none')

                    if axis:
                        ax.axis('on')
                        if feat_dim > 1:
                            ax.set_xlabel("1st principal component")
                            ax.set_ylabel("2nd principal component")
                        else:
                            ax.set_xlabel("1st principal component")
                    else:
                        ax.axis('off') # the axis here may not be consistent, so hide.

                    ax.plot()
                    if savefig:
                        if not os.path.exists("img"):
                            os.mkdir("img")
                            
                        if fmt == 'pdf':
                            if figname is not None:
                                fm = 'img/' + str(figname) + '.pdf'
                            else:
                                fm = 'img/sample.pdf'
                            plt.savefig(fm, bbox_inches='tight')
                        elif fmt == 'png':
                            if figname is not None:
                                fm = 'img/' + str(figname)  + '.png'
                            else:
                                fm = 'img/sample.png'
                            plt.savefig(fm, bbox_inches='tight')
                        else:
                            if figname is not None:
                                fm = 'img/' + str(figname) + '.' + fmt
                            else:
                                fm = 'img/sample' + '.' + fmt
          
                            plt.savefig(fm, bbox_inches='tight')
                            
                        print("image successfully save as", fm)
                        
                    plt.show()

                if agg_label1 == agg_label2: # when ind1 & ind2 are in the same group
                    print("The data points %(index1)s and %(index2)s are in the same group %(agg_id)i, hence were merged into the same cluster #%(m_c)i"%{
                        "index1":index1, "index2":index2, "agg_id":agg_label1, "m_c":cluster_label1}
                    )
                else:
                    if cluster_label1 == cluster_label2:
                        print(
                        """Data point %(index1)s is in group %(agg_id1)s.\nData point %(index2)s is in group %(agg_id2)s.\n"""
                            """Both groups were merged into cluster #%(cluster)i. """% {
                            "index1":index1, "index2":index2, "cluster":cluster_label1, "agg_id1":agg_label1, "agg_id2":agg_label2}
                        )

                        if connected_paths_vis is None:
                            print('No path from group {0} to group {1} with step size <=1.5*R={2:3.2f}.'.format(agg_label1, agg_label2, self.radius*self.mergeScale_))
                            print('This is because at least one of the groups was reassigned due to the minPts condition.')
                        else:
                            print("""\nThe two groups are connected via groups\n %(connected)s.""" % {
                                "connected":connected_paths_vis}
                            )

                            
                            if  hasattr(self, '_index_data') and show_connected_label:
                                show_connected_df = pd.DataFrame(columns=["Index", "Distance", "Group", "Label"])
                                show_connected_df["Index"] = np.insert(self.gcIndices(connected_paths), [0, len(connected_paths)], [index1_id, index2_id])
                                consecutive_distances = [distance.euclidean(data[index1_id], data[show_connected_df["Index"].iloc[1]])] + [distm[connected_paths[i], 
                                                                            connected_paths[i+1]] for i in range(len(connected_paths)-1)] + [distance.euclidean(
                                                                                data[show_connected_df["Index"].iloc[-2]], data[index2_id])]
                                consecutive_distances = ["{0:5.2f}".format(i*self.dataScale_) for i in consecutive_distances]
                                show_connected_df.loc[1:, "Distance"] = consecutive_distances
                                show_connected_df.loc[0, "Distance"] = '--'
                                show_connected_df["Group"] = [agg_label1] + connected_paths + [agg_label2]
                                
                                if isinstance(index1, int):
                                    table_index1 = self._index_data[index1]
                                else:
                                    table_index1 = index1
                                    
                                if isinstance(index2, int):
                                    table_index2 = self._index_data[index2]
                                else:
                                    table_index2 = index2
                                    
                                show_connected_df["Label"] = [table_index1] + self._index_data[self.gcIndices(connected_paths).astype(int)].tolist() + [table_index2] 
                                
                            else:
                                show_connected_df = pd.DataFrame(columns=["Index", "Distance", "Group"])
                                show_connected_df["Index"] = np.insert(self.gcIndices(connected_paths), [0, len(connected_paths)], [index1_id, index2_id])
                                consecutive_distances = [distance.euclidean(data[index1_id], data[show_connected_df["Index"].iloc[1]])] + [distm[connected_paths[i], 
                                                                            connected_paths[i+1]] for i in range(len(connected_paths)-1)] + [distance.euclidean(
                                                                                data[show_connected_df["Index"].iloc[-2]], data[index2_id])]
                                consecutive_distances = ["{0:5.2f}".format(i*self.dataScale_) for i in consecutive_distances]
                                show_connected_df.loc[1:, "Distance"] = consecutive_distances
                                show_connected_df.loc[0, "Distance"] = '--'
                                show_connected_df["Group"] = [agg_label1] + connected_paths + [agg_label2]

                            print('\nHere is a list of connected data points with\ntheir global data indices and group numbers:\n\n', show_connected_df.to_string(index=False), '\n')

                            print("""The distance between consecutive data points is at most R={0:0.3n}. """.format(self.radius*self.dataScale_*self.mergeScale_, width=0))
                            print("""Here, R={0:0.3n}*{1:0.3n}*{2:0.3n}, where {3:0.3n} is the chosen radius parameter, """.format(self.radius, self.dataScale_, self.mergeScale_, self.radius,  align='<', width=0))
                            print("""dataScale_={0:0.3n} is a data scaling factor determined by CLASSIX, """.format(self.dataScale_, width=0))
                            if self.mergeScale_ == 1.5:
                                print("""and mergeScale_={0:0.3n} (the default value).""".format(self.mergeScale_))
                            else:
                                print("""and mergeScale_={0:0.3n}.""".format(self.mergeScale_))

                            if not plot:
                                print("Use .explain(..., plot=True) for a visual representation.")

                    else: 
                        connected_paths = []
                        print("""Data point %(index1)s is in group %(agg_id1)i, which was merged into cluster %(c_id1)s.""" % {
                            "index1":index1, "agg_id1":agg_label1, "c_id1":cluster_label1})

                        print("""Data point %(index2)s is in group %(agg_id2)i, which was merged into cluster %(c_id2)s.""" % {
                            "index2":index2, "agg_id2":agg_label2, "c_id2":cluster_label2})   

                        print("""There is no path of overlapping groups between these clusters.""")

                self.connected_paths = connected_paths

        self.t5_finalize = self.t5_finalize - time()
        return 
    


    def explain_viz(self, showalldata=False, alpha=0.5, cmap='Set3', figsize=(10, 7), showallgroups=False, figstyle="default", bcolor="white", width=0.5, sp_marker="+", sp_mcolor="k", 
                    savefig=False, fontsize=None, bbox=None, axis="off", fmt="pdf"):
        """Visualize fitted samples and group starting points.

        Parameters
        ----------
        showalldata : bool, default=False
            If ``False``, plots with more than 100,000 samples are subsampled.

        alpha : float, default=0.5
            Marker transparency for data points.

        cmap : str, default='Set3'
            Matplotlib colormap for cluster labels.

        figsize : tuple of int, default=(10, 7)
            Matplotlib figure size.

        showallgroups : bool, default=False
            Whether to draw and label all group starting points.

        figstyle : str, default='default'
            Matplotlib style sheet name.

        bcolor : str, default='white'
            Axes background color.

        width : float, default=0.5
            Marker linewidth scaling factor.

        sp_marker : str, default='+'
            Marker used for starting points.

        sp_mcolor : str, default='k'
            Starting-point marker color.

        savefig : bool, default=False
            Whether to save the figure in the ``img`` directory.

        fontsize : int, optional
            Font size for group labels.

        bbox : dict, optional
            Matplotlib text bounding-box properties.

        axis : bool or {'on', 'off'}, default='off'
            Whether to show plot axes.

        fmt : str, default='pdf'
            File format used when ``savefig=True``.

        Returns
        -------
        None
            The visualization is shown and optionally written to disk.
        """
        
        from matplotlib import pyplot as plt

        if self.x_pca.shape[0] > 1e5 and not showalldata:
            print("Too many data points for plot. Randomly subsampled 1e5 points.")
            selectInd = np.random.choice(self.x_pca.shape[0], 100000, replace=False)      
        else:
            selectInd = np.arange(self.x_pca.shape[0])
        
        plt.style.use(style=figstyle)
        plt.figure(figsize=figsize)
        plt.rcParams['axes.facecolor'] = bcolor

        plt.scatter(self.x_pca[selectInd,0], self.x_pca[selectInd,1], s=60, marker=".", linewidth=0*width, c=self.labels_[selectInd], cmap=cmap, alpha=alpha)

        if showallgroups:
            for j in range(self.s_pca.shape[0]):
                if fontsize is None:
                    plt.text(self.s_pca[j, 0], self.s_pca[j, 1], str(j), zorder=1, ha='left', bbox=bbox)
                else:
                    plt.text(self.s_pca[j, 0], self.s_pca[j, 1], str(j), zorder=1, ha='left', fontsize=fontsize, bbox=bbox)

        if showallgroups:
            plt.scatter(self.s_pca[:,0], self.s_pca[:,1], label='group centers', 
                        marker=sp_marker, linewidth=0.9*width, c=sp_mcolor)

        plt.axis('equal')
        plt.title("""{num_clusters:.0f} clusters (radius={tol:.2f}, minPts={minPts:.0f})""".format(
                                             num_clusters=len(np.unique(self.labels_)),tol=self.radius, minPts=self.minPts))

        if axis:
            plt.axis('on')
            if self.s_pca.shape[1] > 1:
                plt.xlabel("1st principal component")
                plt.ylabel("2nd principal component")
            else:
                plt.xlabel("1st principal component")
        else:
            plt.axis('off') # the axis here may not be consistent, so hide.

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')

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
                
            print("image successfully save as", fm)
            
        plt.show()
            
        return
    
        

    def timing(self):
        """Print timing information for the fitted clustering run.

        The reported phases include preprocessing, aggregation, merging, and
        optional explanation finalization.

        Returns
        -------
        None
            Timing values are printed to standard output.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        """
        if hasattr(self, '__fit__'):
            print("t1_prepare:", self.t1_prepare)
            print("t2_aggregate:", self.t2_aggregate)
            print("t3_merge:", self.t3_merge)
            print("t3_merge time:", self.t3_merge)
            if hasattr(self, 't5_finalize'):
                print("t5_finalize time:", self.t5_finalize)
        else:
            raise NotFittedError("Please use .fit() method first.")



    def getPath(self, index1, index2, include_dist=False):
        """Return a representative path between two samples.
        
        Parameters
        ----------
        index1 : int
            Row index of the first sample.
        
        index2 : int
            Row index of the second sample.

        include_dist : bool, default=False
            Whether to use weighted distances when computing the shortest path.
            
        Returns
        -------
        connected_points : ndarray of shape (n_path_points,)
            Original input indices of samples along the connected path. If no
            path exists, an empty array is returned.
            
        """
        from scipy.sparse import csr_matrix
        
        if hasattr(self, '__fit__'):
            groups_ = np.array(self.groups_)
            groups_ = groups_[self.inverse_ind]
        else:
            raise NotFittedError("Please use .fit() method first.")
            
        if index1 == index2:
            return np.array([index1, index2])
        
        agg_label1 = groups_[index1] 
        agg_label2 = groups_[index2] 
        
        if not include_dist and hasattr(self, 'connected_pairs_'): # precomputed distance
            num_nodes = self.splist_.shape[0]
            distm = np.full((num_nodes, num_nodes), 0, dtype=int)
            for i in range(num_nodes):
                distm[i, i] = 0
                
            pairs = np.asarray(self.connected_pairs_, dtype=int)
            for pair in pairs:
                distm[pair[0], pair[1]] = distm[pair[1], pair[0]] = 1
                
            csr_dist_m = csr_matrix(distm)
            connected_paths = find_shortest_dist_path(agg_label1, csr_dist_m, agg_label2, unweighted=include_dist)
            connected_paths.reverse()
                
        else:
            distm = pairwise_distances(self.sp_data_pts)
            distm = (distm <= self.radius*self.mergeScale_).astype(int)
            csr_dist_m = csr_matrix(distm)
            connected_paths = find_shortest_dist_path(agg_label1, csr_dist_m, agg_label2, unweighted=not include_dist)
            connected_paths.reverse()
        
        if len(connected_paths) >= 1:
            connected_points = np.insert(self.gcIndices(connected_paths), [0, len(connected_paths)], [index1, index2])
            return connected_points
        else:
            return np.array([])
            
        
        
    def form_starting_point_clusters_table(self, data, aggregate=False):
        """Build the table used to explain starting points and clusters.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Data in sorted order.

        aggregate : bool, default=False
            If ``True``, coordinates are taken from the low-dimensional
            visualization coordinates. Otherwise, original sorted coordinates are
            shown.

        Returns
        -------
        None
            The resulting table is stored as ``sp_info``.
        """
        
        # won't change the original order of self.splist_
        cols = ["Group", "NrPts"]
        coord = list()
        
        if aggregate:
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


        else:
            sp_indices = self.splist_[:, 0] if self.splist_.ndim == 2 else self.splist_
            for i in sp_indices:
                fill = ""
                sp_item = np.around(data[int(i), :], 2).tolist()
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
        if self.splist_.ndim == 2:
            self.sp_info["NrPts"] = self.splist_[:, 1].astype(int)
        else:
            self.sp_info["NrPts"] = np.array(self.group_sizes_).astype(int)
        self.sp_info["Cluster"] = [self.label_change[i] for i in range(self.splist_.shape[0])]
        self.sp_info["Coordinates"] = coord 
        self.sp_to_c_info = True 
        return
        
        
    
    def visualize_linkage(self, data, scale=1.5, figsize=(10,7), labelsize=24, markersize=320, plot_boundary=False, bound_color='red', path='.', fmt='pdf'):
        
        """Visualize the distance-merging graph between starting points.
        
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Original data used for clustering.
            
        scale : float, default=1.5
            Multiplicative factor applied to ``radius`` when drawing linkage
            edges.

        figsize : tuple of int, default=(10, 7)
            Matplotlib figure size.
        
        labelsize : int, default=24
            Tick-label font size.
            
        markersize : int, default=320
            Marker size for group starting points.
            
        plot_boundary : bool, default=False
            Whether to draw each group's radius boundary when data is
            two-dimensional.
            
        bound_color : str, default='red'
            Boundary-circle color.
            
        path : str, default='.'
            Output directory for the saved figure.
            
        fmt : str, default='pdf'
            Output format. ``'pdf'`` writes a PDF; any other value writes a PNG.

        Returns
        -------
        None
            The figure is saved to ``path``.
        
        """
        from scipy.sparse import csr_matrix
        from matplotlib import pyplot as plt

        if not hasattr(self, '__fit__'):
            raise NotFittedError("Please use .fit() method first.")
            
        distm, n_components, labels = visualize_connections(data[self.ind], self.splist_, radius=self.radius, scale=round(scale,2))
        plt.rcParams['axes.facecolor'] = 'white'

        P = self.sp_data_pts
        link_list = return_csr_matrix_indices(csr_matrix(distm))
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(self.splist_.shape[0]):
            ax.scatter(P[i,0], P[i,1], s=markersize, c='k', marker='.')
            if plot_boundary and data.shape[1] <= 2:
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
            fig.savefig(path + '/linkage_mergeScale_'+str(round(scale,2))+'_tol_'+str(round(self.radius,2))+'.pdf', bbox_inches='tight')
        else:
            fig.savefig(path + '/linkage_mergeScale_'+str(round(scale,2))+'_tol_'+str(round(self.radius,2))+'.png', bbox_inches='tight')
        


    def preprocessing(self, data):
        """Apply the fitted preprocessing transform to new data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        transformed : ndarray of shape (n_samples, n_features)
            Data shifted and scaled using the fitted Euclidean preprocessing
            parameters.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.
        """

        if hasattr(self, '__fit__'):
            return (data - self.mu_) / self.dataScale_ 
        else:
            raise NotFittedError("Please use .fit() method first.")
        
    
    
    @property
    def groupCenters_(self):
        """ndarray of shape (n_groups,): Original indices of group centers."""
        if hasattr(self, '__fit__'):
            return self._gcIndices(np.arange(self.splist_.shape[0]))
        else:
            raise NotFittedError("Please use .fit() method first.")
            
    
    
    @property
    def clusterSizes_(self):
        """ndarray of shape (n_clusters,): Number of samples per cluster."""
        if hasattr(self, '__fit__'):
            counter = collections.Counter(self.labels_)
            return np.array(list(counter.values()))[np.argsort(list(counter.keys()))]
        else:
            raise NotFittedError("Please use .fit() method first.")

    
    
    def gcIndices(self, ids):
        """Map group ids to original input indices of their starting points.

        Parameters
        ----------
        ids : array-like of int
            Group identifiers.

        Returns
        -------
        indices : ndarray
            Original input row indices of the selected group starting points.
        """
        return self._gcIndices(ids)


        
    def gc2ind(self, spid):
        """Map one group id to the original index of its starting point.

        Parameters
        ----------
        spid : int
            Group identifier.

        Returns
        -------
        index : int
            Original input row index of the group's starting point.
        """
        if self.splist_.ndim == 2:
            return self.ind[self.splist_[spid, 0]]
        else:
            return self.ind[self.splist_[spid]]


    
    def load_group_centers(self, data):
        """Return the mean center of each aggregation group.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Original data used for fitting.

        Returns
        -------
        centers : ndarray of shape (n_groups, n_features)
            Mean center of each aggregation group.
        """
        
        if not hasattr(self, '__fit__'):
            raise NotFittedError("Please use .fit() method first.")
            
        if not hasattr(self, 'grp_centers'):
            self.grp_centers = calculate_cluster_centers(data[self.ind], self.groups_)
            return self.grp_centers
        else:
            return self.grp_centers
        
        

    def load_cluster_centers(self, data):
        """Return the mean center of each final cluster.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Original data used for fitting.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Mean center of each final cluster.
        """
            
        if not hasattr(self, '__fit__'):
            raise NotFittedError("Please use .fit() method first.")
            
        if not hasattr(self, 'centers'):
            self.centers = calculate_cluster_centers(data, self.labels_)
            return self.centers
        else:
            return self.centers
        
        

    def outlier_filter(self, min_samples=None, min_samples_rate=0.1): # percent
        """Return labels of clusters smaller than a size threshold.

        Parameters
        ----------
        min_samples : int or float, optional
            Absolute minimum cluster size. If ``None``, the threshold is
            computed from ``min_samples_rate``.

        min_samples_rate : float, default=0.1
            Fraction of fitted samples used as the threshold when
            ``min_samples`` is ``None``.

        Returns
        -------
        outlier_labels : list of int
            Cluster labels whose sizes are below the threshold.
        """
        
        if min_samples == None:
            min_samples = min_samples_rate*sum(self.old_cluster_count.values())
            
        return [i[0] for i in self.old_cluster_count.items() if i[1] < min_samples]
    


    def pprint_format(self, items, truncate=True):
        """Print cluster sizes in compact form.

        Parameters
        ----------
        items : collections.abc.Mapping
            Mapping from cluster labels to cluster sizes.

        truncate : bool, default=True
            If ``True``, print only the 20 largest sizes.

        Returns
        -------
        None
            The formatted sizes are printed to standard output.
        """
        
        cluster_sizes = [str(value) for key, value in sorted(items.items(), key=lambda x: x[1], reverse=True)]

        dotstr = '.'
        if truncate:
            if len(cluster_sizes) > 20: 
                dotstr = ',...'
                cluster_sizes = cluster_sizes[:20]
            
        print(" ", ",".join(cluster_sizes) + dotstr)
                
        return 
            

            
    def __repr__(self):
        """Return the estimator representation."""
        _name = "CLASSIX(radius={0.radius!r}, minPts={0.minPts!r}, group_merging={0.group_merging!r})".format(self)
        return _name 

    
    
    def __str__(self):
        """Return the estimator string representation."""
        _name = 'CLASSIX(radius={0.radius!r}, minPts={0.minPts!r}, group_merging={0.group_merging!r})'.format(self)
        return _name
    
    
    
    @property
    def radius(self):
        """float: Aggregation radius."""
        return self._radius
    
    
    
    @radius.setter
    def radius(self, value):
        """Set the aggregation radius."""
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type')
        if value <= 0:
            raise ValueError(
                "Please feed an correct value (>0) for tolerance.")
 
        self._radius = value
    
    
        
    @property
    def sorting(self):
        """str or None: Sorting strategy used during aggregation."""
        return self._sorting
    
    
    
    @sorting.setter
    def sorting(self, value):
        """Set the sorting strategy."""
        if not isinstance(value, str) and not isinstance(value, type(None)):
            raise TypeError('Expected a string type')
        if value not in ['pca', 'norm-mean', 'norm-orthant', 'sum', 'popcount'] and value != None:
            raise ValueError(
                "Please refer to an correct sorting way, namely 'pca', 'norm-mean' and 'norm-orthant'.")
        self._sorting = value

        
    
    @property
    def group_merging(self):
        """str or None: Strategy used to merge aggregation groups."""
        return self._group_merging
    
    
    
    @group_merging.setter
    def group_merging(self, value):
        """Set the group-merging strategy."""
        if not isinstance(value, str) and not isinstance(value, type(None)):
            raise TypeError('Expected a string type or None.')
        if value not in ['density', 
                         'distance'
                        ] and value is not None: # 'mst-distance', 'scc-distance', 'trivial-distance', 'trivial-density'
            if value.lower()!='none':
                raise ValueError(
                "Please refer to an correct sorting way, namely 'density' and 'distance' or None."
                ) # 'scc-distance' and 'mst-distance'
        self._group_merging = value
        

    
    @property
    def minPts(self):
        """int: Minimum valid cluster size."""
        return self._minPts
    
    
    
    @minPts.setter
    def minPts(self, value):
        """Set the minimum valid cluster size."""
        if isinstance(value, str):
            raise TypeError('Expected a float or int type.')
        
        if isinstance(value, bool):
            raise TypeError('Expected a float or int type.')
        
        if isinstance(value, dict):
            raise TypeError('Expected a float or int type.')
        
        if hasattr(value, "__len__"):
            raise TypeError('Expected a scalar.')
        
        if value < 0 or (0 < value & value < 1):
            raise ValueError('Noise_mergeScale must be 0 or greater than 1.')
        
        self._minPts = int(round(value))
    
    




def pairwise_distances(X):
    """Compute the pairwise Euclidean distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise Euclidean distances.
    """
    
    return distance.squareform(distance.pdist(X))



def visualize_connections(data, splist, radius=0.5, scale=1.5):
    """Build the group-center graph used for distance merging.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Sorted input data.

    splist : ndarray of shape (n_groups, 2)
        Starting-point information from Euclidean aggregation.

    radius : float, default=0.5
        Base aggregation radius.

    scale : float, default=1.5
        Multiplicative factor for the linkage radius.

    Returns
    -------
    distm : ndarray of shape (n_groups, n_groups)
        Binary adjacency matrix for linked starting points.

    n_components : int
        Number of connected components.

    labels : ndarray of shape (n_groups,)
        Connected-component label for each group.
    """

    from scipy.sparse.csgraph import connected_components
    
    distm = pairwise_distances(data[splist[:,0].astype(int)])
    tol = radius*scale
    distm = (distm <= tol).astype(int)
    n_components, labels = connected_components(csgraph=distm, directed=False, return_labels=True)
    return distm, n_components, labels
    
    
    
def preprocessing(data, base):
    """Preprocess data using one of CLASSIX's Euclidean sorting schemes.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Input data.

    base : {'norm-mean', 'pca', 'norm-orthant', None}
        Preprocessing and sorting basis.

    Returns
    -------
    ndata : ndarray of shape (n_samples, n_features)
        Shifted and scaled data.

    params : tuple
        Tuple ``(mu, dataScale)`` containing the shift and scale.
    """
    if base == "norm-mean":
        _mu = data.mean(axis=0)
        ndata = data - _mu
        dataScale = ndata.std()
        ndata = ndata / dataScale

    elif base == "pca":
        _mu = data.mean(axis=0)
        ndata = data - _mu # mean center
        rds = norm(ndata, axis=1) # distance of each data point from 0
        dataScale = np.median(rds) # 50% of data points are within that radius
        ndata = ndata / dataScale # now 50% of data are in unit ball 

    elif base == "norm-orthant":
        _mu = data.min(axis=0)
        ndata = data - _mu
        dataScale = ndata.std()
        ndata = ndata / dataScale

    else:
        _mu, dataScale = 0, 1 # no preprocessing
        ndata = (data - _mu) / dataScale
    return ndata, (_mu, dataScale)



def calculate_cluster_centers(data, labels):
    """Compute mean centers for labeled clusters.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Input data.

    labels : array-like of shape (n_samples,)
        Integer cluster labels. Labels are expected to be consecutive and
        zero-based.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Mean feature vector for each cluster.
    """
    classes = np.unique(labels)
    centers = np.zeros((len(classes), data.shape[1]))
    for c in classes:
        centers[c] = np.mean(data[labels==c,:], axis=0)
    return centers




# ##########################################################################################################
# **************<!-- the independent functions of finding shortest path between two objects ***************
# ##########################################################################################################


def find_shortest_dist_path(source_node=None, graph=None, target_node=None, unweighted=True):
    """Return the shortest path between two nodes in a sparse graph.
    
    Parameters
    ----------
    source_node : int
        Source vertex.
    
    graph : scipy.sparse._csr.csr_matrix
        Sparse adjacency matrix.
        
    target_node : int, default=None
        Target vertex.
    
    unweighted : bool, default=True
        If ``True``, minimize the number of edges. If ``False``, use edge
        weights stored in ``graph``.
        
    Returns
    -------
    shortest_path_to_target : list of int
        Path from ``target_node`` back to ``source_node`` as returned by scipy's
        predecessor traversal. An empty list is returned when no path exists.
        
    """
    from scipy.sparse.csgraph import shortest_path
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, unweighted=unweighted, indices=source_node, return_predecessors=True)

    if predecessors[target_node] != -9999:
        shortest_path_to_target = []
        shortest_path_to_target.append(target_node)
        predecessor = predecessors[target_node] 
        while predecessor != -9999:
            shortest_path_to_target.append(predecessor)
            predecessor = predecessors[predecessor] 
            
        return shortest_path_to_target
    else:
        return []



    
def return_csr_matrix_indices(csr_mat): 
    """Return row and column indices of nonzero CSR entries.

    Parameters
    ----------
    csr_mat : scipy.sparse.csr_matrix
        Sparse matrix.

    Returns
    -------
    indices : ndarray of shape (n_nonzero, 2)
        Row and column index pairs for nonzero entries.
    """

    from scipy.sparse import _sparsetools
    
    shape_dim1, shape_dim2 = csr_mat.shape
    length_range = csr_mat.indices
    indices = np.empty(len(length_range), dtype=csr_mat.indices.dtype)
    _sparsetools.expandptr(shape_dim1, csr_mat.indptr, indices)
    return np.array(list(zip(indices, length_range)))




def euclid(xxt, X, v):
    """Compute squared Euclidean distances from rows of ``X`` to ``v``.

    Parameters
    ----------
    xxt : ndarray of shape (n_samples,)
        Precomputed squared norms for rows of ``X``.

    X : ndarray of shape (n_samples, n_features)
        Input data.

    v : ndarray of shape (n_features,)
        Reference vector.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Squared Euclidean distances.
    """
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)







