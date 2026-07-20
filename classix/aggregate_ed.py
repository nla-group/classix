# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2026 Stefan Güttel, Xinye Chen


# Python implementation for aggregation


import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh


def _pca_sort_values(data):
    """Return PCA sorting values, with a stable fallback for zero data."""
    len_ind, fdim = data.shape
    if fdim == 1:
        return data[:, 0]
    if not np.any(data):
        return np.zeros(len_ind)

    if fdim <= 3: # memory inefficient
        gemm = get_blas_funcs("gemm", [data.T, data])
        _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
        sort_vals = data@U1.reshape(-1)
    else:
        U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
        sort_vals = U1[:,0]*s1[0]

    return sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output


def pca_aggregate(data, sorting='pca', tol=0.5):
    """Aggregate Euclidean data sorted by the first principal component.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed input data.

    sorting : str, default='pca'
        Sorting method. This implementation is specialized for PCA sorting and
        keeps the parameter for API compatibility with other aggregation
        functions.
    
    tol : float, default=0.5
        Maximum Euclidean distance from a starting point for assigning a sample
        to the corresponding aggregation group.
    
    
    Returns
    -------
    labels : list of int
        Aggregation group labels in sorted-data order.
    
    splist : list of tuple
        Pairs ``(start_index, group_size)`` for group starting points in sorted
        data.
    
    nr_dist : int
        Number of pairwise distances evaluated.

    ind : ndarray of shape (n_samples,)
        Permutation that maps sorted rows to original rows.

    sort_vals : ndarray of shape (n_samples,)
        Sorted PCA scores.
    
    data : ndarray of shape (n_samples, n_features)
        Sorted data.
    
    half_nrm2 : ndarray of shape (n_samples,)
        Half squared Euclidean norm for each sorted row.
    """
    
    len_ind = data.shape[0]
    

    # get sorting values
    sort_vals = _pca_sort_values(data)
    ind = np.argsort(sort_vals)
    data = data[ind,:] # sort data
    sort_vals = sort_vals[ind] 
 
    half_r2 = 0.5*tol**2
    half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5 # precomputation

    lab = 0
    labels = [-1] * len_ind
    nr_dist = 0 
    splist = list()

    for i in range(len_ind): 
        if labels[i] >= 0:
            continue

        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1

        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
        last_j = np.searchsorted(sort_vals, tol + sort_vals[i], side='right')
        ips = np.matmul(data[i+1:last_j,:], clustc.T)
        
        for j in range(i+1, last_j):
            if labels[j] >= 0:
                continue

            nr_dist += 1
            if half_nrm2[j] - ips[j-i-1] <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))
        lab += 1

    return labels, splist, nr_dist, ind, sort_vals, data, half_nrm2



def general_aggregate(data, sorting="pca", tol=0.5): 
    """Aggregate Euclidean data using the selected sorting strategy.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed input data.
    
    sorting : {'pca', 'norm-mean', 'norm-orthant', None}, default='pca'
        Sorting strategy used to order candidate starting points.
    
    tol : float, default=0.5
        Maximum Euclidean distance from a starting point for assigning a sample
        to the corresponding aggregation group.
    
    
    Returns
    -------
    labels : list of int
        Aggregation group labels in sorted-data order.
    
    splist : list of tuple
        Pairs ``(start_index, group_size)`` for group starting points in sorted
        data.
    
    nr_dist : int
        Number of pairwise distances evaluated.

    ind : ndarray of shape (n_samples,)
        Permutation that maps sorted rows to original rows.

    sort_vals : ndarray of shape (n_samples,)
        Sorted scalar values used for pruning comparisons.
    
    data : ndarray of shape (n_samples, n_features)
        Sorted data.
    
    half_nrm2 : ndarray of shape (n_samples,)
        Half squared Euclidean norm for each sorted row.
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    
    if sorting == "norm-mean" or sorting == "norm-orthant": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        

    elif sorting == "pca":
        # change to svd
        sort_vals = _pca_sort_values(data)
        

    else: # no sorting
        sort_vals = np.zeros(len_ind) 
        
    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind]

    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 

    
    half_r2 = tol**2 * 0.5
    half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5 # precomputation
    
    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        else:
            clustc = data[i,:] 
            labels[i] = lab
            num_group = 1
        
        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.

        for j in range(i+1, len_ind):
            if labels[j] >= 0:
                continue

            if (sort_vals[j] - sort_vals[i] > tol):
                break       

            nr_dist += 1
            dataj = data[j]

            if half_nrm2[j] - np.inner(clustc, dataj) <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  
        lab += 1

    return labels, splist, nr_dist, ind, sort_vals, data, half_nrm2





def lm_aggregate(data, sorting="pca", tol=0.5): 
    """Aggregate Euclidean data with a lower-memory implementation.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed input data.

    sorting : {'pca', 'norm-mean', 'norm-orthant', None}, default='pca'
        Sorting strategy used to order candidate starting points.

    tol : float, default=0.5
        Maximum Euclidean distance from a starting point for assigning a sample
        to the corresponding aggregation group.

    Returns
    -------
    labels : list of int
        Aggregation group labels in sorted-data order.
    
    splist : list of tuple
        Pairs ``(start_index, group_size)`` for group starting points in sorted
        data.
    
    nr_dist : int
        Number of pairwise distances evaluated.

    ind : ndarray of shape (n_samples,)
        Permutation that maps sorted rows to original rows.

    sort_vals : ndarray of shape (n_samples,)
        Sorted scalar values used for pruning comparisons.
    
    data : ndarray of shape (n_samples, n_features)
        Sorted data.
    
    half_nrm2 : None
        Placeholder returned for compatibility with the higher-memory
        aggregation functions.
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    
    if sorting == "norm-mean" or sorting == "norm-orthant": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        
    elif sorting == "pca":
        # change to svd
        sort_vals = _pca_sort_values(data)
        

    else: # no sorting
        sort_vals = np.zeros(len_ind) 

    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind]

    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        else:
            clustc = data[i,:] 
            labels[i] = lab
            num_group = 1

        for j in range(i+1, len_ind):
            if labels[j] >= 0:
                continue

            if (sort_vals[j] - sort_vals[i] > tol):
                break       

            dat = clustc - data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  

        lab += 1

    return labels, splist, nr_dist, ind, sort_vals, data, None
