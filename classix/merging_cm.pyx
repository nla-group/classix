#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True

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

# Python implementation for agglomeration

cimport cython
import numpy as np
cimport numpy as np
from scipy.special import betainc, gamma
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, _sparsetools
from scipy.sparse.csgraph import connected_components
ctypedef np.uint8_t uint8
np.import_array()
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)

# Disjoint set union
cpdef fast_agglomerate(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] splist, double radius, str method="distance", double scale=1.5):
    # cdef list connected_pairs = list()
    cdef list connected_pairs = [SET(i) for i in range(splist.shape[0])]
    cdef list connected_pairs_store = list()
    cdef double volume, den1, den2, cid
    cdef unsigned int i, j, internum
    cdef np.ndarray[np.float64_t, ndim=2] neigbor_sp
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals
    cdef np.ndarray[long, ndim=1] select_stps
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_overlap, c1, c2
    if method == "density":
        volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1) + np.finfo(float).eps
    else:
        volume = 0.0 # no need for distance-based method
    
    for i in range(splist.shape[0]):
        sp1 = splist[i, 3:]
        neigbor_sp = splist[i+1:, 3:]
        
        select_stps = np.arange(i+1, splist.shape[0], dtype=int)
        sort_vals = splist[i:, 1]
        
        if method == "density":
            # index_overlap = np.linalg.norm(neigbor_sp[:,2:] - sp1, ord=2, axis=-1) <= 2*radius 
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, 2*radius)
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap):
                continue
            # neigbor_sp = neigbor_sp[index_overlap]
            c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
            den1 = np.count_nonzero(c1) / volume
            for j in select_stps:
                sp2 = splist[j, 3:]
                c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
                den2 = np.count_nonzero(c2) / volume
                if check_if_overlap(sp1, sp2, radius=radius): 
                    internum = np.count_nonzero(c1 & c2)
                    cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                    if cid >= den1 or cid >= den2: 
                        # connected_pairs.append([i,j]) 
                        merge(connected_pairs[i], connected_pairs[j])
                        connected_pairs_store.append([i, j])
        else:
            # index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= scale*radius  
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, scale*radius)
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap):
                continue

            # neigbor_sp = neigbor_sp[index_overlap] 
            for j in select_stps:
                # connected_pairs.append([i,j]) 
                merge(connected_pairs[i], connected_pairs[j])
                connected_pairs_store.append([i, j])
                
    cdef SET obj = SET(0)
    cdef list labels_set = list()
    cdef list labels = [findParent(obj).data for obj in connected_pairs]
    cdef int lab
    for lab in np.unique(labels):
        labels_set.append([i for i in range(len(labels)) if labels[i] == lab])
    return labels_set, connected_pairs_store # merge_pairs(connected_pairs)



cpdef fast_query(double[:,:] data, double[:] starting_point, double[:] sort_vals, double tol):
    cdef int len_ind = data.shape[0]
    cdef double[:] candidate
    cdef double dist
    cdef Py_ssize_t i, coord
    cdef fdim = data.shape[1] # remove the first three elements that not included in the features
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_query=np.full(len_ind, False, dtype=bool)
    for i in range(len_ind):
        candidate = data[i]
        
        if (sort_vals[i+1] - sort_vals[0] > tol):
            break

        dist = 0
        for coord in range(fdim):
            dist += (candidate[coord] - starting_point[coord])**2
        
        if dist <= tol**2:
            index_query[i] = True
            
    return index_query


cdef class SET: # disjoint union
    cdef public int data
    cdef public object parent
    def __init__(self, data):
        self.data = data
        self.parent = self

        
        
cpdef findParent(s):
    if (s.data != s.parent.data) :
        s.parent = findParent((s.parent))
    return s.parent



cpdef merge(s1, s2):
    parent_of_s1 = findParent(s1)
    parent_of_s2 = findParent(s2)

    if (parent_of_s1.data != parent_of_s2.data) :
        findParent(s1).parent = parent_of_s2
   



        
# Strongly connected components finding algorithm 
cpdef scc_agglomerate(np.ndarray[np.float64_t, ndim=2] splist, double radius=0.5, double scale=1.5, int n_jobs=-1): # limited to distance-based method
    cdef list index_set = list()

    cdef np.ndarray[long, ndim=2] distm = (
        pairwise_distances(splist[:,3:], Y=None, metric='euclidean', n_jobs=n_jobs) <= radius*scale
    ).astype(int)
    
    cdef int n_components
    cdef np.ndarray[int, ndim=1] labels
    cdef list labels_set = list()
    n_components, labels = connected_components(csgraph=csr_matrix(distm), directed=False, return_labels=True)
    
    cdef int lab, i
    for lab in np.unique(labels):
        labels_set.append([i for i in range(len(labels)) if labels[i] == lab])
    return labels_set



# Deprecated function
cpdef agglomerate_trivial(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] splist, double radius, str method="distance", double scale=1.5):
    cdef list connected_pairs = list()
    cdef double volume, den1, den2, cid
    cdef unsigned int i, j, internum
    cdef np.ndarray[np.float64_t, ndim=2] neigbor_sp
    cdef np.ndarray[long, ndim=1] select_stps
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_overlap, c1, c2
    if method == "density":
        volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1) + np.finfo(float).eps
    else:
        volume = 0.0 # no need for distance-based method
    
    for i in range(splist.shape[0]):
        sp1 = splist[i, 3:]
        neigbor_sp = splist[i+1:, 3:]
        
        select_stps = np.arange(i+1, splist.shape[0], dtype=int)
        sort_vals = splist[i:, 1]
        
        if method == "density":
            index_overlap = np.linalg.norm(neigbor_sp[:,2:] - sp1, ord=2, axis=-1) <= 2*radius 
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap):
                continue
            # neigbor_sp = neigbor_sp[index_overlap]
            c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
            den1 = np.count_nonzero(c1) / volume
            for j in select_stps:
                sp2 = splist[j, 3:]
                c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
                den2 = np.count_nonzero(c2) / volume
                if check_if_overlap(sp1, sp2, radius=radius): 
                    internum = np.count_nonzero(c1 & c2)
                    cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                    if cid >= den1 or cid >= den2: 
                        connected_pairs.append([i, j])
        else:
            index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= scale*radius  
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap):
                continue

            # neigbor_sp = neigbor_sp[index_overlap] 
            for j in select_stps:
                connected_pairs.append([i, j])
    return connected_pairs



cpdef merge_pairs_dr(list pairs):
    """Transform connected pairs to connected groups (list)"""
    
    cdef unsigned int len_p = len(pairs)
    cdef unsigned int maxid = 0
    cdef long long[:] ulabels = np.full(len_p, -1, dtype=int)
    cdef list labels = list()
    cdef list sub = list()
    cdef list com = list()
    cdef Py_ssize_t i, j, ind
    
    for i in range(len_p):
        if ulabels[i] == -1:
            sub = pairs[i]
            ulabels[i] = maxid

            for j in range(i+1, len_p):
                com = pairs[j]
                if not set(sub).isdisjoint(com):
                    sub = sub + com
                    if ulabels[j] == -1:
                        ulabels[j] = maxid
                    else:
                        for ind in range(len_p):
                            if ulabels[ind] == maxid:
                                ulabels[ind] = ulabels[j]
            maxid = maxid + 1
    
    for i in np.unique(ulabels):
        sub = list()
        for j in [ind for ind in range(len_p) if ulabels[ind] == i]:
            sub = sub + pairs[int(j)]
        labels.append(list(set(sub)))
    return labels





        

# cpdef merge_pairs(list pairs):
#     """Transform connected pairs to connected groups (list)"""
#     
#     cdef unsigned int len_p = len(pairs)
#     cdef unsigned int maxid = 0
#     cdef long long[:] ulabels = np.full(len_p, -1, dtype=int)
#     cdef list labels = list()
#     cdef list sub = list()
#     cdef list com = list()
#     cdef Py_ssize_t i, j, ind
#     
#     for i in range(len_p):
#         if ulabels[i] == -1:
#             sub = pairs[i]
#             ulabels[i] = maxid
# 
#             for j in range(i+1, len_p):
#                 com = pairs[j]
#                 if not set(sub).isdisjoint(com):
#                     sub = sub + com
#                     if ulabels[j] == -1:
#                         ulabels[j] = maxid
#                     else:
#                         for ind in range(len_p):
#                             if ulabels[ind] == maxid:
#                                 ulabels[ind] = ulabels[j]
#             maxid = maxid + 1
#     
#     for i in np.unique(ulabels):
#         sub = list()
#         for j in [ind for ind in range(len_p) if ulabels[ind] == i]:
#             sub = sub + pairs[int(j)]
#         labels.append(list(set(sub)))
#     return labels



# deprecated (24/07/2021)
# def density(num, volume):
#     ''' Calculate the density
#     num: number of objects inside the cluster
#     volume: the area of the cluster
#     '''
#     return num / volume



# cpdef check_if_intersect(list g1, list g2):
#     """Check if two list have the same elements."""
#     return not set(g1).isdisjoint(g2) # set(g1).intersection(g2) != set()



cpdef check_if_overlap(double[:] starting_point, double[:] spo, double radius, int scale = 1):
    """Check if two groups formed by aggregation overlap
    """
    cdef Py_ssize_t dim
    cdef double[:] subs = starting_point.copy()
    for dim in range(starting_point.shape[0]):
        subs[dim] = starting_point[dim] - spo[dim]
    return np.linalg.norm(subs, ord=2, axis=-1) <= 2 * scale * radius


    

cpdef cal_inter_density(double[:] starting_point, double[:] spo, double radius, int num):
    """Calculate the density of intersection (lens)
    """
    cdef double in_volume = cal_inter_volume(starting_point, spo, radius)
    return num / in_volume




cpdef cal_inter_volume(double[:] starting_point, double[:] spo, double radius):
    """
    Returns the volume of the intersection of two spheres in n-dimensional space.
    The radius of the two spheres is r and the distance of their centers is d.
    For d=0 the function returns the volume of full sphere.
    Reference: https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres

    """
    
    cdef Py_ssize_t dim
    cdef unsigned int fdim = starting_point.shape[0]
    cdef double[:] subs = starting_point.copy()
    for dim in range(fdim):
        subs[dim] = starting_point[dim] - spo[dim]
    cdef double dist = np.linalg.norm(subs, ord=2, axis=-1) # the distance between the two groups
    if dist > 2*radius:
        return 0
    cdef double c = dist / 2
    return np.pi**(fdim/2)/gamma(fdim/2 + 1)*(radius**fdim)*betainc((fdim + 1)/2, 1/2, 1 - c**2/radius**2) + np.finfo(float).eps




