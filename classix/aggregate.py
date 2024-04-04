# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2024 Stefan Güttel, Xinye Chen


# Python implementation for aggregation


import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh


def pca_aggregate(data, sort_vals, half_nrm2, len_ind, sorting="pca", tol=0.5):
    """Aggregate the data with PCA using precomputation

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    tol : float
        The tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  
    
    
    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    half_nrm2 (numpy.ndarray):
        Precomputed values for distance computation.

    """
 
    half_r2 = 0.5*tol**2
    
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

    return labels, splist, nr_dist, ind, sort_vals, data



def general_aggregate(data, sort_vals, half_nrm2, len_ind, sorting="pca", tol=0.5): 
    """Aggregate the data using precomputation

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    sorting : str
        The sorting way referred for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.
    
    tol : float
        The tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  
    
    
    Returns
    -------
    labels (list) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.

    ind (numpy.ndarray):
        Array storing Sorting indices.

    sort_vals (numpy.ndarray):
        Sorting values.
    
    data (numpy.ndarray):
        Sorted data.
    
    half_nrm2 (numpy.ndarray):
        Precomputed values for distance computation.

    """

    splist = list() # store the starting points
    half_r2 = tol**2 * 0.5
    
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

    return labels, splist, nr_dist, ind, sort_vals, data



