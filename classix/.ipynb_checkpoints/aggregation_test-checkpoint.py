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

# Python implementation for aggregation


import numpy as np
# from sklearn.decomposition import PCA
# from scipy.sparse.linalg import svds
from scipy.linalg import get_blas_funcs, eigh

# python implementation for aggregation
def aggregate(data, sorting="pca", tol=0.5, early_stopping=False): # , verbose=1
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
    
    splist = list() # store the starting points
    len_ind = data.shape[0]
    fdim = data.shape[1]
    
    if sorting == "norm-mean" or sorting == "norm-orthant": 
        c_data = data.copy()
        sort_vals = np.linalg.norm(c_data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    elif sorting == "pca":
        c_data = data - data.mean(axis=0)
        if data.shape[1]>1:
            gemm = get_blas_funcs("gemm", [c_data.T, c_data])
            _, U1 = eigh(gemm(1, c_data.T, c_data), subset_by_index=[fdim-1, fdim-1])
            sort_vals = c_data@U1.reshape(-1)
            # print( U1, s1, _)
        else:
            sort_vals = c_data[:,0]
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    else: # no sorting
        sort_vals = np.zeros(len_ind) # useless, blank
        ind = np.arange(len_ind)
        
    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): 
        sp = ind[i] 
        if labels[sp] >= 0:
            continue
        else:
            clustc = c_data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            if early_stopping:
                if (sort_vals[j] - sort_vals[sp] > tol):
                    break       

            dat = clustc - c_data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, lab] + [num_group] + list(data[sp,:]) ) 
        lab += 1

    # if verbose == 1:
    #    print("aggregate {} groups".format(len(np.unique(labels))))

    return np.array(labels), splist, nr_dist

