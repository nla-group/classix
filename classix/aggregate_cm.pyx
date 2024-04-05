# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2024 Stefan Güttel, Xinye Chen

# Cython implementation for aggregation


cimport cython
import numpy as np
cimport numpy as np 
from scipy.sparse.linalg import svds
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

cpdef pca_aggregate(double[:,:] data, str sorting='pca', double tol=0.5):
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

    cdef Py_ssize_t len_ind, fdim, last_j
    len_ind, fdim = data.base.shape

    cdef double[:] sort_vals
    cdef double[:, :] U1, _
    cdef long long[:] ind
    cdef unsigned int lab=0, nr_dist=0, num_group
    cdef double[:] clustc 
    

    cdef list labels = [-1]*len_ind
    cdef list splist = list() 
    cdef Py_ssize_t i, j
    
    cdef double rhs

    if fdim > 1:
        U1, s1, _ = svds(np.asarray(data), k=1, return_singular_vectors=True)
        sort_vals = U1[:,0]*s1[0]
    else:
        sort_vals = data[:,0]
        
    sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        
    ind = np.argsort(sort_vals)
    
    data = data.base[ind]
    sort_vals = sort_vals.base[ind] 
    cdef double half_r2 = tol**2 * 0.5
    cdef double[:] half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5
    cdef double[:] ips
    
    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        
        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1
            
        rhs = half_r2 - half_nrm2[i] # right-hand side of norm ineq.
        last_j = np.searchsorted(sort_vals, tol + sort_vals[i], side='right')
        ips = np.matmul(data.base[i+1:last_j,:], clustc)

        for j in range(i+1, last_j):
                    
            if labels[j] >= 0:
                continue

            nr_dist += 1
            if half_nrm2[j] - ips[j-i-1] <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))
        lab += 1

    return labels, splist, nr_dist, np.asarray(ind), np.asarray(sort_vals), np.asarray(data), np.asarray(half_nrm2)




cpdef general_aggregate(double[:,:] data, str sorting, double tol=0.5):
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

    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef unsigned int lab=0, nr_dist=0, num_group
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef list labels = [-1]*len_ind
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, j, coord
    cdef double[:] half_nrm2
    cdef double half_r2 = tol**2 * 0.5
    cdef double[:] dataj
    cdef double rhs


    if sorting == "norm-mean" or sorting == "norm-orthant":
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
    
    elif sorting == "pca":
        if fdim > 1:
            U1, s1, _ = svds(np.asarray(data), k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
        else:
            sort_vals = data[:,0]
            
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
    
    else: # no sorting
        sort_vals = np.zeros(len_ind)
        
    ind = np.argsort(sort_vals)
    data = data.base[ind]
    sort_vals = sort_vals.base[ind] 

    half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5

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

            dataj = data[j]
            for coord in range(fdim):
                dist += clustc[coord] * dataj[coord]
            
            nr_dist += 1
            
            if half_nrm2[j] - dist <= rhs:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  
        # list of [ starting point index of current group, sorting key, and number of group elements ]

        lab += 1
  
    return labels, splist, nr_dist, np.asarray(ind), np.asarray(sort_vals), np.asarray(data), np.asarray(half_nrm2)




cpdef lm_aggregate(double[:,:] data, str sorting, double tol=0.5):
    """Aggregate the data in low memory need (Linux)
    
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
    
    """

    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef unsigned int lab=0, nr_dist=0, num_group
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef list labels = [-1]*len_ind
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, j, coord
    
    
    if sorting == "norm-mean" or sorting == "norm-orthant":
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
    
    elif sorting == "pca":
        if fdim > 1:
            U1, s1, _ = svds(np.asarray(data), k=1, return_singular_vectors=True)
            sort_vals = U1[:,0]*s1[0]
        else:
            sort_vals = data[:,0]
            
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
    
    else: # no sorting
        sort_vals = np.zeros(len_ind)

    ind = np.argsort(sort_vals)
    data = data.base[ind]
    sort_vals = sort_vals.base[ind] 

    for i in range(len_ind): 
        if labels[i] >= 0:
            continue
        
        clustc = data[i,:] 
        labels[i] = lab
        num_group = 1
            
        for j in range(i+1, len_ind): 
                    
            if labels[j] >= 0:
                continue
                
            if sort_vals[j] - sort_vals[i] > tol:
                break       
            
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - data[j,coord])**2
            
            nr_dist += 1
            
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append((i, num_group))  
        # list of [ starting point index of current group, sorting key, and number of group elements ]

        lab += 1
  
    return labels, splist, nr_dist, np.asarray(ind), np.asarray(sort_vals), np.asarray(data), None
