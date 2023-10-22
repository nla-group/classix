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

# Cython implementation for group merging

#!python
#cython: language_level=3
#cython: profile=True
#cython: linetrace=True


cimport cython
import collections
import numpy as np
cimport numpy as np
from scipy.special import betainc, gamma
ctypedef np.uint8_t uint8
np.import_array()
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)



cpdef bf_distance_agglomerate(double[:, :] data, np.ndarray[np.int32_t, ndim=1] labels,
                                double[:, :] splist, 
                                double radius, int minPts=0, double scale=1.5):

    """
    Implement CLASSIX's merging with brute force computation
    
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    labels : numpy.ndarray
        aggregation labels

    splist : numpy.ndarray
        Represent the list of starting points information formed in the aggregation. 
        list of [ starting point index of current group, sorting values, and number of group elements ].

    ind : numpy.ndarray
        Sort values.

    radius : float
        The tolerance to control the aggregation. If the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group. For details, we refer users to [1].

    minPts : int, default=0
        The threshold, in the range of [0, infity] to determine the noise degree.
        When assgin it 0, algorithm won't check noises.

    scale : float, default 1.5
        Design for distance-clustering, when distance between the two starting points 
        associated with two distinct groups smaller than scale*radius, then the two groups merge.

        
    Returns
    -------

    References
    ----------
    [1] X. Chen and S. Güttel. Fast and explainable sorted based clustering, 2022

    """

    cdef long[:] splist_indices = np.int32(splist[:, 0])
    cdef double[:, :] spdata = data.base[splist_indices]
    cdef double[:] xxt = np.einsum('ij,ij->i', spdata, spdata)
    cdef np.ndarray[np.int32_t, ndim=1] sp_cluster_label = labels[splist_indices]  
    cdef np.ndarray[np.int32_t, ndim=1] copy_sp_cluster_label
    cdef long[:] spl, ii
    cdef Py_ssize_t fdim =  splist.shape[0]
    cdef Py_ssize_t i, iii, j, ell
    cdef double[:] xi
    cdef long[:] merge_ind
    cdef long minlab
    cdef np.ndarray[np.float64_t, ndim=1] dist


    for i in range(fdim):
        xi = spdata[i, :]
        dist = euclid(xxt, spdata, xi)
        spl = np.unique( sp_cluster_label[dist <= (scale*radius)**2] )
        minlab = np.min(spl)

        for ell in spl:
            sp_cluster_label[sp_cluster_label==ell] = minlab


    cdef long[:] ul = np.unique(sp_cluster_label)
    cdef Py_ssize_t nr_u = len(ul)

    cdef long[:] cs = np.zeros(nr_u, dtype=np.int32)
    
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] cid
    cdef np.ndarray[np.int32_t, ndim=1] grp_sizes = np.int32(splist[:, 2])

    for i in range(nr_u):
        cid = sp_cluster_label==ul[i]
        sp_cluster_label[cid] = i
        cs[i] = np.sum(grp_sizes[cid])


    old_cluster_count = collections.Counter(sp_cluster_label[labels])
    cdef long[:] ncid = np.int32(np.nonzero(cs.base < minPts)[0])

    cdef Py_ssize_t SIZE_NOISE_LABELS = ncid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_label = sp_cluster_label.copy()

        for i in ncid:
            ii = np.int32(np.nonzero(copy_sp_cluster_label==i)[0])
            
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group
                
                dist = euclid(xxt, spdata, xi)
                merge_ind = np.int32(np.argsort(dist))
                for j in merge_ind:
                    if cs[copy_sp_cluster_label[j]] >= minPts:
                        sp_cluster_label[iii] = copy_sp_cluster_label[j]
                        break

        ul = np.unique(sp_cluster_label)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=np.int32)

        for i in range(nr_u):
            cid = sp_cluster_label==ul[i]
            sp_cluster_label[cid] = i
            cs[i] = np.int32(np.sum(grp_sizes[cid]))

    labels = sp_cluster_label[labels]

    return labels, old_cluster_count, SIZE_NOISE_LABELS



# Disjoint set union
cpdef agglomerate(double[:, :] data, 
                    double[:, :] splist, 
                    double radius, str method='distance', double scale=1.5):
    """
    Implement CLASSIX's merging with disjoint-set data structure, default choice for the merging.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    splist : numpy.ndarray
        List of starting points formed in the aggregation.

    radius : float
        The tolerance to control the aggregation. If the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group. For details, we refer users to [1].

    method : str, default='distance'
        Method for group merging, Available options are:
        
        - 'density': two groups are merged if the density of data points in their intersection 
           is at least as high the smaller density of both groups. This option uses the disjoint 
           set structure to speedup agglomerate.
           
        - 'distance': two groups are merged if the distance of their starting points is at 
           most scale*radius (the parameter above). This option uses the disjoint 
           set structure to speedup agglomerate.
    
    scale : float
        Design for distance-clustering, when distance between the two starting points 
        associated with two distinct groups smaller than scale*radius, then the two groups merge.

    Returns
    -------
    labels_set : list
        Connected components of graph with groups as vertices.
    
    connected_pairs_store : list
        List for connected group labels.


    References
    ----------
    [1] X. Chen and S. Güttel. Fast and explainable sorted based clustering, 2022

    """
    
    # cdef list connected_pairs = list()
    cdef list connected_pairs = [SET(i) for i in range(splist.shape[0])]
    cdef list connected_pairs_store = list()
    cdef double volume, den1, den2, cid
    cdef unsigned int i, j, internum
    cdef int ndim = data.shape[1]
    cdef int nsize_stp = splist.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] splist_indices = np.int32(splist[:, 0])

    cdef double[:, :] neigbor_sp
    cdef double[:] sort_vals
    cdef long[:] select_stps
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_overlap, c1, c2
    
    
    if method == "density":
        volume = np.pi**(ndim/2) * radius** ndim / gamma( ndim/2+1 ) + np.finfo(float).eps
    else:
        volume = 0.0 # no need for distance-based method
    
    for i in range(nsize_stp):
        sp1 = data.base[splist_indices[i]] 
        select_stps = splist_indices[i+1:]
        neigbor_sp = data.base[select_stps, :] 
        
        select_stps = np.arange(i+1, nsize_stp, dtype=int)
        sort_vals = splist.base[i:, 1]
        
        if method == "density":
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, 2*radius)
            select_stps = select_stps.base[index_overlap]
            if not np.any(index_overlap):
                continue
            
            c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
            den1 = np.count_nonzero(c1) / volume

            for j in select_stps:
                sp2 = data.base[int(splist[j, 0])] 
                c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
                den2 = np.count_nonzero(c2) / volume
                
                if check_if_overlap(sp1, sp2, radius=radius): 
                    internum = np.count_nonzero(c1 & c2)
                    cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                    if cid >= den1 or cid >= den2: 
                        merge(connected_pairs[i], connected_pairs[j])
                        connected_pairs_store.append([i, j])
        else:
            # index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= scale*radius  
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, scale*radius)
            select_stps = select_stps.base[index_overlap]
            if not np.any(index_overlap):
                continue
 
            for j in select_stps:
                merge(connected_pairs[i], connected_pairs[j])
                connected_pairs_store.append((i, j))
                
    cdef SET obj = SET(0)
    cdef list labels_set = list()
    cdef list labels = [findParent(obj).data for obj in connected_pairs]
    cdef int lab

    for lab in np.unique(labels):
        labels_set.append([i for i in range(len(labels)) if labels[i] == lab])
    return labels_set, connected_pairs_store 




cpdef fast_query(double[:,:] data, double[:] starting_point, double[:] sort_vals, double tol):
    """Fast query with built in early stopping strategy for merging."""
    cdef int len_ind = data.shape[0]
    cdef double[:] candidate
    cdef double dist
    cdef Py_ssize_t i, coord
    cdef ndim = data.shape[1] # remove the first three elements that not included in the features
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_query=np.full(len_ind, False, dtype=bool)
    for i in range(len_ind):
        candidate = data[i]
        
        if (sort_vals[i+1] - sort_vals[0] > tol):
            break

        dist = 0
        for coord in range(ndim):
            dist += (candidate[coord] - starting_point[coord])**2

        if dist <= tol**2:
            index_query[i] = True
            
    return index_query



cpdef euclid(double[:] xxt, double[:, :] X, double[:] v):
    return (xxt + np.inner(v,v).ravel() -2*X.base.dot(v)).astype(float)



cdef class SET: 
    """Disjoint-set data structure."""
    cdef public int data
    cdef public object parent
    def __init__(self, data):
        self.data = data
        self.parent = self

        
        
cpdef findParent(s):
    """Find parent of node."""
    if (s.data != s.parent.data) :
        s.parent = findParent((s.parent))
    return s.parent



cpdef merge(s1, s2):
    """Merge the roots of two node."""
    parent_of_s1 = findParent(s1)
    parent_of_s2 = findParent(s2)

    if (parent_of_s1.data != parent_of_s2.data) :
        findParent(s1).parent = parent_of_s2
   



cpdef check_if_overlap(double[:] starting_point, double[:] spo, double radius, int scale = 1):
    """Check if two groups formed by aggregation overlap.
    """
    cdef Py_ssize_t dim
    cdef double[:] subs = starting_point.copy()
    for dim in range(starting_point.shape[0]):
        subs[dim] = starting_point[dim] - spo[dim]
    return np.linalg.norm(subs, ord=2, axis=-1) <= 2 * scale * radius


    

cpdef cal_inter_density(double[:] starting_point, double[:] spo, double radius, int num):
    """Calculate the density of intersection (lens).
    """
    cdef double in_volume = cal_inter_volume(starting_point, spo, radius)
    return num / in_volume




cpdef cal_inter_volume(double[:] starting_point, double[:] spo, double radius):
    """
    Returns the volume of the intersection of two spheres in n-dimensional space.
    The radius of the two spheres is r and the distance of their centers is d.
    For d=0 the function returns the volume of full sphere.
    Reference: https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres.

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


