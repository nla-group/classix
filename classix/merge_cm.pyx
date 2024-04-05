# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2024 Stefan Güttel, Xinye Chen

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


cpdef distance_merge_mtg(double[:, :] data, list labels,
                       long[:, :] splist,  double radius, int minPts, double scale, 
                       double[:] sort_vals, double[:] half_nrm2):

    """
    Implement CLASSIX's merging without merging tiny groups
    
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    labels : list
        aggregation labels

    splist : numpy.ndarray
        Represent the list of starting points information formed in the aggregation. 
        list of [ starting point index of current group, sorting values, and number of group elements ].

    radius : float
        The tolerance to control the aggregation. If the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group. For details, we refer users to [1].

    minPts : int, default=0
        The threshold, in the range of [0, infity] to determine the noise degree.
        When assign it 0, algorithm won't check noises.

    scale : float, default 1.5
        Design for distance-clustering, when distance between the two starting points 
        associated with two distinct groups smaller than scale*radius, then the two groups merge.

    sort_vals : numpy.ndarray
        Sorting values.
        
    half_nrm2 : numpy.ndarray
        Precomputed values for distance computation.

        
    Returns
    -------
    labels : numpy.ndarray
        The merging labels.
    
    old_cluster_count : int
        The number of clusters without outliers elimination.
    
    SIZE_NOISE_LABELS : int
        The number of clusters marked as outliers.
    
    References
    ----------
    [1] X. Chen and S. Güttel. Fast and explainable sorted based clustering, 2022

    """

    cdef long[:] splist_indices = splist[:, 0]
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] gp_nr = splist.base[:, 1] >= minPts
    cdef np.ndarray[np.int64_t, ndim=1] arr_labels = np.array(labels)
    cdef double[:, :] spdata = data.base[splist_indices]
    cdef np.ndarray[np.int64_t, ndim=1] sp_cluster_labels = arr_labels[splist_indices]   
    cdef double[:] sort_vals_sp = sort_vals.base[splist_indices]
    cdef np.ndarray[np.float64_t, ndim=1] eucl
    cdef np.ndarray[np.int64_t, ndim=1] copy_sp_cluster_labels, inds
    cdef long[:] spl, ii
    cdef Py_ssize_t fdim =  splist.shape[0]
    cdef Py_ssize_t i, iii, j, ell, last_j
    cdef double[:] xi
    cdef long[:] merge_ind
    cdef int minlab
    cdef np.ndarray[np.float64_t, ndim=1] dist
    
    radius = scale*radius
    cdef double scaled_radius = (radius)**2/2

    for i in range(fdim):
        if not gp_nr[i]:    # tiny groups can not take over large ones
            continue
        
        xi = spdata[i, :]

        last_j = np.searchsorted(sort_vals_sp, radius + sort_vals_sp[i], side='right')
        eucl = half_nrm2.base[i:last_j] - np.matmul(spdata.base[i:last_j,:], xi)
        inds = i + np.where((eucl <= scaled_radius - half_nrm2[i]) & gp_nr[i:last_j])[0]

        spl = np.unique(sp_cluster_labels[inds])
        minlab = np.min(spl)

        for ell in spl:
            sp_cluster_labels[sp_cluster_labels==ell] = minlab


    cdef long[:] ul = np.unique(sp_cluster_labels)
    cdef Py_ssize_t nr_u = len(ul)

    cdef long[:] cs = np.zeros(nr_u, dtype=int)
    
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] cid
    cdef np.ndarray[np.int64_t, ndim=1] grp_sizes = np.int64(splist[:, 1])

    for i in range(nr_u):
        cid = sp_cluster_labels==ul[i]
        sp_cluster_labels[cid] = i
        cs[i] = np.sum(grp_sizes[cid])

    old_cluster_count = collections.Counter(sp_cluster_labels[arr_labels])
    cdef long[:] ncid = np.nonzero(cs.base < minPts)[0]

    cdef Py_ssize_t SIZE_NOISE_LABELS = ncid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_labels = sp_cluster_labels.copy()

        for i in ncid:
            ii = np.nonzero(copy_sp_cluster_labels==i)[0]
            
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group
                
                dist = half_nrm2 - np.matmul(spdata, xi) + half_nrm2[iii]
                merge_ind = np.argsort(dist)
                for j in merge_ind:
                    if cs[copy_sp_cluster_labels[j]] >= minPts:
                        sp_cluster_labels[iii] = copy_sp_cluster_labels[j]
                        break

        ul = np.unique(sp_cluster_labels)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=int)

        for i in range(nr_u):
            cid = sp_cluster_labels==ul[i]
            sp_cluster_labels[cid] = i
            cs[i] = np.sum(grp_sizes[cid])

    arr_labels = sp_cluster_labels[arr_labels]

    return arr_labels, old_cluster_count, SIZE_NOISE_LABELS



cpdef distance_merge(double[:, :] data, list labels,
                       long[:, :] splist,  double radius, int minPts, double scale, 
                       double[:] sort_vals, double[:] half_nrm2):

    """
    Implement CLASSIX's merging with early stopping and BLAS routines
    
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    labels : list
        aggregation labels

    splist : numpy.ndarray
        Represent the list of starting points information formed in the aggregation. 
        list of [ starting point index of current group, sorting values, and number of group elements ].

    radius : float
        The tolerance to control the aggregation. If the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group. For details, we refer users to [1].

    minPts : int, default=0
        The threshold, in the range of [0, infity] to determine the noise degree.
        When assign it 0, algorithm won't check noises.

    scale : float, default 1.5
        Design for distance-clustering, when distance between the two starting points 
        associated with two distinct groups smaller than scale*radius, then the two groups merge.

    sort_vals : numpy.ndarray
        Sorting values.
        
    half_nrm2 : numpy.ndarray
        Precomputed values for distance computation.

        
    Returns
    -------
    labels : numpy.ndarray
        The merging labels.
    
    old_cluster_count : int
        The number of clusters without outliers elimination.
    
    SIZE_NOISE_LABELS : int
        The number of clusters marked as outliers.
    
    References
    ----------
    [1] X. Chen and S. Güttel. Fast and explainable sorted based clustering, 2022

    """

    cdef long[:] splist_indices = splist[:, 0]
    cdef np.ndarray[np.int64_t, ndim=1] arr_labels = np.array(labels)
    cdef double[:, :] spdata = data.base[splist_indices]
    cdef np.ndarray[np.int64_t, ndim=1] sp_cluster_labels = arr_labels[splist_indices]   
    cdef double[:] sort_vals_sp = sort_vals.base[splist_indices]
    cdef np.ndarray[np.float64_t, ndim=1] eucl
    cdef np.ndarray[np.int64_t, ndim=1] copy_sp_cluster_labels, inds
    cdef long[:] spl, ii
    cdef Py_ssize_t fdim =  splist.shape[0]
    cdef Py_ssize_t i, iii, j, ell, last_j
    cdef double[:] xi
    cdef long[:] merge_ind
    cdef int minlab
    cdef np.ndarray[np.float64_t, ndim=1] dist
    
    radius = scale*radius
    cdef double radius_2 = (radius)**2/2

    for i in range(fdim):
        xi = spdata[i, :]

        last_j = np.searchsorted(sort_vals_sp, radius + sort_vals_sp[i], side='right')
        eucl = half_nrm2.base[i:last_j] - np.matmul(spdata.base[i:last_j,:], xi)
        inds = i + np.where(eucl <= radius_2 - half_nrm2[i])[0]

        spl = np.unique(sp_cluster_labels[inds])
        minlab = np.min(spl)

        for ell in spl:
            sp_cluster_labels[sp_cluster_labels==ell] = minlab


    cdef long[:] ul = np.unique(sp_cluster_labels)
    cdef Py_ssize_t nr_u = len(ul)

    cdef long[:] cs = np.zeros(nr_u, dtype=int)
    
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] cid
    cdef np.ndarray[np.int64_t, ndim=1] grp_sizes = np.int64(splist[:, 1])

    for i in range(nr_u):
        cid = sp_cluster_labels==ul[i]
        sp_cluster_labels[cid] = i
        cs[i] = np.sum(grp_sizes[cid])

    old_cluster_count = collections.Counter(sp_cluster_labels[arr_labels])
    cdef long[:] ncid = np.nonzero(cs.base < minPts)[0]

    cdef Py_ssize_t SIZE_NOISE_LABELS = ncid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_labels = sp_cluster_labels.copy()

        for i in ncid:
            ii = np.nonzero(copy_sp_cluster_labels==i)[0]
            
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group
                
                dist = half_nrm2 - np.matmul(spdata, xi) + half_nrm2[iii]
                merge_ind = np.argsort(dist)
                for j in merge_ind:
                    if cs[copy_sp_cluster_labels[j]] >= minPts:
                        sp_cluster_labels[iii] = copy_sp_cluster_labels[j]
                        break

        ul = np.unique(sp_cluster_labels)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=int)

        for i in range(nr_u):
            cid = sp_cluster_labels==ul[i]
            sp_cluster_labels[cid] = i
            cs[i] = np.sum(grp_sizes[cid])

    arr_labels = sp_cluster_labels[arr_labels]

    return arr_labels, old_cluster_count, SIZE_NOISE_LABELS






# Disjoint set union
cpdef density_merge(double[:, :] data, long[:, :] splist, double radius, double[:] sort_vals, double[:] half_nrm2):
    """
    Implement CLASSIX's merging with disjoint-set data structure, default choice for the merging.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input that is array-like of shape (n_samples,).
    
    splist : numpy.ndarray
        Represent the list of starting points information formed in the aggregation. 
        list of [ starting point index of current group, sorting values, and number of group elements ]

    radius : float
        The tolerance to control the aggregation. If the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group. For details, we refer users to [1].

    sort_vals : numpy.ndarray
        Sorting values.

    half_nrm2 : numpy.ndarray
        Precomputed values for distance computation.

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
    
    cdef double volume, den1, den2, cid, radius_2, radius_22
    cdef unsigned int i, j, internum
    cdef int ndim = data.shape[1]
    cdef int last_j
    
    cdef int len_sp = splist.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] splist_indices = np.int64(splist[:, 0])
    cdef double[:] sort_vals_sp = sort_vals.base[splist_indices]
    cdef double[:] half_nrm2_sp = half_nrm2.base[splist_indices]

    cdef double[:, :] spdata = data.base[splist_indices]

    cdef double[:, :] neigbor_sp
    cdef long[:] select_stps
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] index_overlap, c1, c2
    
    volume = np.pi**(ndim/2) * radius** ndim / gamma( ndim/2+1 ) + np.finfo(float).eps
    radius_2 = (2*radius)**2 / 2
    radius_22 = radius**2
    
    for i in range(len_sp):
        sp1 = spdata[i]

        last_j = np.searchsorted(sort_vals_sp, 2*radius + sort_vals_sp[i], side='right')
        neigbor_sp = spdata.base[i+1:last_j]
        
        index_overlap = half_nrm2_sp.base[i+1:last_j] - np.matmul(neigbor_sp, sp1) <= radius_2 - half_nrm2_sp[i]

        select_stps = i+1 + np.where(index_overlap)[0]

        if not np.any(index_overlap):
            continue

        c1 = euclid(2*half_nrm2.base, data, sp1) <= radius_22
        den1 = np.count_nonzero(c1) / volume

        for j in select_stps:
            sp2 = spdata[j]

            c2 = euclid(2*half_nrm2.base, data, sp2) <= radius_22
            den2 = np.count_nonzero(c2) / volume
            
            if check_if_overlap(sp1, sp2, radius=radius): 
                internum = np.count_nonzero(c1 & c2)
                cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                if cid >= den1 or cid >= den2: 
                    merge(connected_pairs[i], connected_pairs[j])
                    connected_pairs_store.append([i, j])

    cdef SET obj = SET(0)
    cdef list labels_set = list()
    cdef list labels = [findParent(obj).data for obj in connected_pairs]
    cdef int lab

    for lab in np.unique(labels):
        labels_set.append([i for i in range(len(labels)) if labels[i] == lab])
    
    return labels_set, connected_pairs_store 




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

