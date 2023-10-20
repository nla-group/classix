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

#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True

# Cython implementation for aggregation


cimport cython
import numpy as np
cimport numpy as np 
from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

cpdef precompute_aggregate(np.ndarray[np.float64_t, ndim=2] data, str sorting="pca", double tol=0.5):
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
    labels (numpy.ndarray) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.
    """
    
    cdef int num_group
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef np.ndarray[np.float64_t, ndim=2] U1
    cdef int sp 
    cdef int nr_dist = 0 
    cdef int lab = 0 
    cdef double dist
    cdef np.ndarray[np.int64_t, ndim=1] labels = np.full(len_ind, -1, dtype=int)
    cdef list splist = list() 
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals
    cdef np.ndarray[np.float64_t, ndim=1] clustc
    cdef np.ndarray[np.int64_t, ndim=1] ind
    cdef Py_ssize_t i, j, coord
    
    cdef double half_r2 = tol**2 * 0.5
    cdef np.ndarray[np.float64_t, ndim=1] half_nrm2 = np.einsum('ij,ij->i', data, data) * 0.5
    
    cdef np.ndarray[np.float64_t, ndim=1] dataj
    cdef double rhs
    cdef int ii

    if sorting == "norm-mean" or sorting == "norm-orthant": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    elif sorting == "pca":
        if data.shape[1]>1:
            if fdim <= 3: # memory inefficient
                gemm = get_blas_funcs("gemm", [data.T, data])
                _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
                sort_vals = data@U1.reshape(-1)
            else:
                U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
                sort_vals = U1[:,0]*s1[0]

        else:
            sort_vals = data[:,0]
            
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    else: # no sorting
        sort_vals = np.zeros(len_ind)
        ind = np.arange(len_ind)
    
    for i in range(len_ind):
        
        sp = ind[i] 
        if labels[sp] >= 0:
            continue
        
        clustc = data[sp,:] 
        labels[sp] = lab
        num_group = 1

        rhs = half_r2 - half_nrm2[sp] # right-hand side of norm ineq.

        for ii in range(i+1, len_ind): 
            j = ind[ii]
            
            if labels[j] >= 0:
                continue
            
            if sort_vals[j] - sort_vals[sp] > tol:
                break       
            
            dist = 0
            dataj = data[j,:] 

            for coord in range(fdim):
                dist += clustc[coord] * dataj[coord]
                
            nr_dist += 1
            
            if half_nrm2[j] - dist <= rhs:
                num_group = num_group + 1
                labels[j] = lab

        splist.append((sp, sort_vals[sp], num_group)) 

        lab += 1

    return labels, splist, nr_dist, ind



cpdef aggregate(np.ndarray[np.float64_t, ndim=2] data, str sorting="pca", float tol=0.5):
    """Aggregate the data

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
    labels (numpy.ndarray) : 
        The group categories of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.
    """
    
    cdef int num_group
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef np.ndarray[np.float64_t, ndim=2] U1
    cdef int sp # sp: starting point
    cdef int nr_dist = 0 # nr_dist:if necessary, count the distance computation
    cdef int lab = 0 # lab: class
    cdef double dist # distance 
    cdef np.ndarray[np.int64_t, ndim=1] labels = np.full(len_ind, -1, dtype=int)
    cdef list splist = list() # store the starting points
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals
    cdef np.ndarray[np.float64_t, ndim=1] clustc
    cdef np.ndarray[np.int64_t, ndim=1] ind
    cdef Py_ssize_t i, j, coord
    
    
    if sorting == "norm-mean" or sorting == "norm-orthant": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    elif sorting == "pca":
        if data.shape[1]>1:
            if fdim <= 3: # memory inefficient
                gemm = get_blas_funcs("gemm", [data.T, data])
                _, U1 = eigh(gemm(1, data.T, data), subset_by_index=[fdim-1, fdim-1])
                sort_vals = data@U1.reshape(-1)
            else:
                U1, s1, _ = svds(data, k=1, return_singular_vectors=True)
                sort_vals = U1[:,0]*s1[0]

        else:
            sort_vals = data[:,0]
            
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    else: # no sorting
        sort_vals = np.zeros(len_ind)
        ind = np.arange(len_ind)
    
    for i in range(len_ind):
        sp = ind[i] 

        if labels[sp] >= 0:
            continue
        
        clustc = data[sp,:] 
        labels[sp] = lab
        num_group = 1

        for j in ind[i+1:]:
            if labels[j] >= 0:
                continue
            
            if sort_vals[j] - sort_vals[sp] > tol:
                break       
            
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - data[j,coord])**2
            
            nr_dist += 1
            
            if dist <= tol**2:
                num_group = num_group + 1
                labels[j] = lab

        splist.append((sp, sort_vals[sp], num_group)) 
        lab += 1

    return labels, splist, nr_dist, ind


