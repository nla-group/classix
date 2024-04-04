# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2024 Stefan Guettel, Xinye Chen


#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True

# Cython implementation of CLASSIX's aggregation phase

cimport cython
import numpy as np
cimport numpy as np 
from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

cpdef pca_aggregate(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=1] sort_vals, 
                    np.ndarray[np.float64_t, ndim=1] half_nrm2, int len_ind, 
                    str sorting='pca', double tol=0.5):
    """Aggregate the data with PCA using precomputation

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).

    sorting : str
        The sorting method for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.

    tol : float
        The tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

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
    
    cdef int num_group
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension

    cdef int nr_dist = 0 
    cdef int lab = 0 
    cdef list labels = [-1]*len_ind
    cdef list splist = list() 
    cdef np.ndarray[np.float64_t, ndim=1] clustc
    cdef np.ndarray[np.int64_t, ndim=1] ind
    cdef Py_ssize_t i, j

    cdef double[:] ips
    cdef double half_r2 = tol**2 * 0.5
    cdef double rhs

    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind] 

    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        
        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1
            
        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
        last_j = np.searchsorted(sort_vals, tol + sort_vals[i], side='right')
        ips = np.matmul(data[i+1:last_j,:], clustc)

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


cpdef general_aggregate(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=1] sort_vals, 
                    np.ndarray[np.float64_t, ndim=1] half_nrm2, int len_ind, 
                    str sorting='pca', double tol=0.5):
    """Aggregate the data using precomputation

    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).

    sorting : str
        The sorting method for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.

    tol : float
        The tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

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
    
    cdef int num_group
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef int nr_dist = 0 
    cdef int lab = 0 
    cdef double dist
    cdef list labels = [-1]*len_ind
    cdef list splist = list() 
    cdef np.ndarray[np.float64_t, ndim=1] clustc
    cdef np.ndarray[np.int64_t, ndim=1] ind
    cdef Py_ssize_t i, j, coord
    
    cdef double half_r2 = tol**2 * 0.5
    
    cdef np.ndarray[np.float64_t, ndim=1] dataj
    cdef double rhs

    ind = np.argsort(sort_vals)
    data = data[ind]
    sort_vals = sort_vals[ind] 
    
    for i in range(len_ind):
        if labels[i] >= 0:
            continue
        
        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1

        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.

        for j in range(i+1, len_ind): 
            
            if labels[j] >= 0:
                continue
            
            if sort_vals[j] - sort_vals[i] > tol:
                break       
            
            dist = 0
            dataj = data[j,:] 

            for coord in range(fdim):
                dist += clustc[coord] * dataj[coord]
                
            nr_dist += 1
            
            if half_nrm2[j] - dist <= rhs:
                num_group = num_group + 1
                labels[j] = lab

        splist.append((i, num_group)) 

        lab += 1

    return labels, splist, nr_dist, ind, sort_vals, data



