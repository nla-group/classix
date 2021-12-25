# -*- coding: utf-8 -*-

# CLASSIX: Fast and explainable clustering based on sorting

# License: BSD 3 clause

# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Python implementation for aggregation


import numpy as np
from tqdm import tqdm 
# from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds

# python implementation for aggregation
def aggregate(data, sorting="pca", tol=0.5): # , verbose=1
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting method for aggregation, default='pca', other options: 'norm-mean', 'norm-orthant'.

    tol : float
        the tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    Returns
    -------
    labels (numpy.ndarray) : 
        the group categories of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    nr_dist (int) :
        number of pairwise distance calculations
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]

    if sorting == "norm-mean" or sorting == "norm-orthant": 
        c_data = data.copy()
        sort_vals = np.linalg.norm(c_data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    elif sorting == "pca":
        # pca = PCA(n_components=1) 
        # sort_vals = pca.fit_transform(data_memview).reshape(-1)
        # ind = np.argsort(sort_vals)
        
        # change to svd 
        # data = data - data.mean(axis=0) -- already done in the clustering.fit_transform
        c_data = data - data.mean(axis=0)
        if data.shape[1]>1:
            U1, s1, _ = svds(c_data, k=1, return_singular_vectors="u")
            sort_vals = U1[:,0]*s1[0]
            # print( U1, s1, _)
        else:
            sort_vals = c_data[:,0]
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    else: # no sorting
        sort_vals = np.zeros(len_ind) # useless, blankss
        ind = np.arange(len_ind)
        
    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            clustc = c_data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            # sort_val_c = sort_vals[sp]
            # sort_val_j = sort_vals[j]

            if (sort_vals[j] - sort_vals[sp] > tol):
                break       

            # dist = np.sum((clustc - data[j,:])**2)    # slow

            dat = clustc - c_data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, sort_vals[sp], num_group] + list(data[sp,:]) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        lab += 1

    # if verbose == 1:
    #    print("aggregate {} groups".format(len(np.unique(labels))))

    return np.array(labels), splist, nr_dist





# def calculate_group_centers(data, labels):
#     agg_centers = list() 
#     for c in set(labels):
#         # indc = [i for i in range(data.shape[0]) if labels[i] == c]
#         indc = np.where(labels==c)
#         center = [-1, c] + np.mean(data[indc,:], axis=0).tolist()
#         agg_centers.append( center )
#     return agg_centers