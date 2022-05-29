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

# Cython implementation for aggregation


cimport cython
import numpy as np
cimport numpy as np 
# from cython.parallel import prange
# from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
# from libc.string cimport strcmp
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

cpdef aggregate(double[:,:] data, str sorting, double tol=0.5):
    """Aggregate the data
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    sorting : str
        The sorting way refered for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.
    tol : float
        The tolerance to control the aggregation, if the distance between the starting point 
        and the object is less than or equal than the tolerance,
        the object should allocated to the group which starting point belongs to.  
    Returns
    -------
    labels (numpy.ndarray) : 
        The group category of the data after aggregation.
    
    splist (list) : 
        The list of the starting points.
    
    nr_dist (int) :
        The number of pairwise distance calculations.
    """
    cdef Py_ssize_t fdim = data.shape[1] # feature dimension
    cdef Py_ssize_t len_ind = data.shape[0] # size of data
    cdef double[:] sort_vals
    cdef double[:, :] c_data = np.empty((len_ind, fdim), dtype=np.float64)
    cdef double[:, :] U1, _  # = np.empty((len_ind, ), dtype=float)
    cdef long long[:] ind # = np.empty((len_ind, ), dtype=int)
    cdef Py_ssize_t sp # starting point index
    cdef unsigned int lab=0, nr_dist=0, num_group
    cdef double[:] clustc # starting point coordinates
    cdef double dist
    cdef long[:] labels = np.full(len_ind, -1, dtype=int) # np.zeros(, dtype=int) - 1
    cdef list splist = list() # list of starting points
    cdef Py_ssize_t i, ii, j, coord
    
    if sorting == "norm-mean" or sorting == "norm-orthant":
        c_data[...] = data
        sort_vals = np.linalg.norm(c_data, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "pca":
        # pca = PCA(n_components=1) 
        # sort_vals = pca.fit_transform(data_memview).reshape(-1)
        # ind = np.argsort(sort_vals)
        
        # change to svd 
        # data = data - data.mean(axis=0) -- already done in the clustering.fit_transform
        c_data = data - np.mean(data, axis=0)
        if data.shape[1]>1:
            U1, s1, _ = svds(c_data, k=1, return_singular_vectors="u")
            sort_vals = U1[:,0]*s1[0]
        else:
            sort_vals = c_data[:,0]
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)
    else: # no sorting
        sort_vals = np.zeros(len_ind)
        ind = np.arange(len_ind)

    for i in range(len_ind): 
        sp = ind[i] # starting point
        
        if labels[sp] >= 0:
            continue
        
        clustc = c_data[sp,:] 
        labels[sp] = lab
        num_group = 1
            
        for ii in range(i, len_ind): 
            j = ind[ii]
                    
            if labels[j] != -1:
                continue
                
            if (sort_vals[j] - sort_vals[sp] > tol):
                break       
            
            dist = 0
            for coord in range(fdim):
                dist += (clustc[coord] - c_data[j,coord])**2
            nr_dist += 1
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, sort_vals[sp], num_group])  # list of [ starting point index of current group, sorting key, and number of group elements ]

        lab += 1
  
    return np.asarray(labels), splist, nr_dist



# move to lite_func.py
# cpdef merge_pairs(list pairs):
#     """Transform connected pairs to connected groups (list)"""
# 
#     cdef list labels = list()
#     cdef list sub = list()
#     cdef Py_ssize_t i, j, maxid = 0
#     cdef Py_ssize_t len_p = len(pairs)
#     
#     cdef np.ndarray[np.int64_t, ndim=1] ulabels = np.full(len_p, -1, dtype=int) # np.zeros(len(pairs), dtype=int) - 1
#     cdef np.ndarray[np.int64_t, ndim=1] distinct_ulabels = np.unique(ulabels)
#     cdef np.ndarray[np.int64_t, ndim=1] select_arr
#     
#     for i in range(len_p):
#         if ulabels[i] == -1:
#             sub = pairs[i]
#             ulabels[i] = maxid
# 
#             for j in range(i+1, len_p):
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
#     for i in distinct_ulabels:
#         sub = list()
#         select_arr = np.where(ulabels == i)[0]
#         for j in select_arr:
#             sub = sub + pairs[j]
#         labels.append(list(set(sub)))
#         
#     return labels
# cdef check_if_intersect(list g1, list g2):
#     return set(g1).intersection(g2) != set()
