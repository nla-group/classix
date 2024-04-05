# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2024 Stefan Güttel, Xinye Chen

# Python implementation for group merging

import collections
import numpy as np
from scipy.special import betainc, gamma


def distance_merge_mtg(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2):
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
    
    splist_indices = splist[:, 0]
    gp_nr = splist[:, 1] >= minPts
    spdata = data[splist_indices]
    sort_vals_sp = sort_vals[splist_indices]

    labels = np.asarray(labels)
    sp_cluster_labels = labels[splist_indices]

    radius = scale*radius
    scaled_radius = 0.5*(radius)**2

    for i in range(splist.shape[0]):
        if not gp_nr[i]:    # tiny groups can not take over large ones
            continue

        xi = spdata[i, :]
        rhs = scaled_radius - half_nrm2[i]
        last_j = np.searchsorted(sort_vals_sp, radius + sort_vals_sp[i], side='right')

        inds = (half_nrm2[i:last_j] - np.matmul(spdata[i:last_j], xi) <= rhs)
        inds = i + np.where(inds&gp_nr[i:last_j])[0]

        spl = np.unique(sp_cluster_labels[inds])
        minlab = np.min(spl)
        
        for label in spl:
            sp_cluster_labels[sp_cluster_labels==label] = minlab

    # rename labels to be 1,2,3,... and determine cluster sizes
    ul = np.unique(sp_cluster_labels)
    nr_u = len(ul)
    
    cs = np.zeros(nr_u, dtype=int)
    grp_sizes = splist[:, 1].astype(int)

    for i in range(nr_u):
        cid = sp_cluster_labels==ul[i]
        sp_cluster_labels[cid] = i
        cs[i] = np.sum(grp_sizes[cid])

    old_cluster_count = collections.Counter(sp_cluster_labels[labels])
    cid = np.nonzero(cs < minPts)[0]
    SIZE_NOISE_LABELS = cid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_label = sp_cluster_labels.copy()
        
        for i in cid:
            ii = np.nonzero(copy_sp_cluster_label==i)[0] # find all tiny groups with that label
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group

                dist = half_nrm2 - np.matmul(spdata, xi) + half_nrm2[iii]
                merge_ind = np.argsort(dist)
                for j in merge_ind:
                    if cs[copy_sp_cluster_label[j]] >= minPts:
                        sp_cluster_labels[iii] = copy_sp_cluster_label[j]
                        break

        ul = np.unique(sp_cluster_labels)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=int)

        for i in range(nr_u):
            cid = sp_cluster_labels==ul[i]
            sp_cluster_labels[cid] = i
            cs[i] = np.sum(grp_sizes[cid])

    labels = sp_cluster_labels[labels]

    return labels, old_cluster_count, SIZE_NOISE_LABELS



def distance_merge(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2):
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

    minPts : int
        The threshold, in the range of [0, infity] to determine the noise degree.
        When assign it 0, algorithm won't check noises.

    scale : float
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
    
    splist_indices = splist[:, 0]
    
    sort_vals_sp = sort_vals[splist_indices]
    spdata = data[splist_indices]
    
    labels = np.asarray(labels)
    sp_cluster_labels = labels[splist_indices]
    radius = scale*radius
    radius_2 = (radius)**2/2

    for i in range(splist.shape[0]):
        xi = spdata[i, :]

        last_j = np.searchsorted(sort_vals_sp, radius + sort_vals_sp[i], side='right')
        eucl = half_nrm2[i:last_j] - np.matmul(spdata[i:last_j,:], xi)
        inds = np.where(eucl <= radius_2 - half_nrm2[i]) 

        inds = i + inds[0]
        spl = np.unique(sp_cluster_labels[inds])
        minlab = np.min(spl)
        
        for label in spl:
            sp_cluster_labels[sp_cluster_labels==label] = minlab

    # rename labels to be 1,2,3,... and determine cluster sizes
    ul = np.unique(sp_cluster_labels)
    nr_u = len(ul)
    
    cs = np.zeros(nr_u, dtype=int)
    grp_sizes = splist[:, 1].astype(int)

    for i in range(nr_u):
        cid = sp_cluster_labels==ul[i]
        sp_cluster_labels[cid] = i
        cs[i] = np.sum(grp_sizes[cid])

    old_cluster_count = collections.Counter(sp_cluster_labels[labels])
    cid = np.nonzero(cs < minPts)[0]
    SIZE_NOISE_LABELS = cid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_label = sp_cluster_labels.copy()
        
        for i in cid:
            ii = np.nonzero(copy_sp_cluster_label==i)[0] # find all tiny groups with that label
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group

                dist = half_nrm2 - np.matmul(spdata, xi) + half_nrm2[iii]
                merge_ind = np.argsort(dist)
                for j in merge_ind:
                    if cs[copy_sp_cluster_label[j]] >= minPts:
                        sp_cluster_labels[iii] = copy_sp_cluster_label[j]
                        break

        ul = np.unique(sp_cluster_labels)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=int)

        for i in range(nr_u):
            cid = sp_cluster_labels==ul[i]
            sp_cluster_labels[cid] = i
            cs[i] = np.sum(grp_sizes[cid])

    labels = sp_cluster_labels[labels]

    return labels, old_cluster_count, SIZE_NOISE_LABELS




def density_merge(data, splist, radius, sort_vals, half_nrm2):
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
    
    connected_pairs = [SET(i) for i in range(splist.shape[0])]
    connected_pairs_store = []
    splist_indices = splist[:, 0]
    sort_vals_sp = sort_vals[splist_indices]
    spdata = data[splist_indices]
    half_nrm2_sp = half_nrm2[splist_indices]

    volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1)

    radius_2 = (2*radius)**2
    radius_22 = radius**2

    for i in range(splist.shape[0]):
        sp1 = spdata[i]
        last_j = np.searchsorted(sort_vals_sp, 2*radius + sort_vals_sp[i], side='right')
        
        neigbor_sp = spdata[i+1:last_j]
        
        # THIS PART WE OMIT THE FAST QUERY WITH EARLY STOPPING CAUSE WE DON'T THINK EARLY STOPPING IN PYTHON CODE CAN BE FASTER THAN 
        # PYTHON BROADCAST, BUT IN THE CYTHON CODE, WE IMPLEMENT FAST QUERY WITH EARLY STOPPING BY LEVERAGING THE ADVANTAGE OF SORTED 
        # AGGREGATION.
        # index_overlap = euclid(2*half_nrm2_sp[i+1:last_j], neigbor_sp, sp1) <= radius_2 # 2*distance_scale*radius

        index_overlap = half_nrm2_sp[i+1:last_j] - np.matmul(neigbor_sp, sp1) <= radius_2 - half_nrm2_sp[i]
        
        select_stps = i+1 + np.where(index_overlap)[0]
        # calculate their neighbors ...1)
        # later decide whether sps should merge with neighbors
        if not np.any(index_overlap):
            continue

        c1 = euclid(2*half_nrm2, data, sp1) <= radius_22
        den1 = np.count_nonzero(c1) / volume
        
        for j in select_stps:
            sp2 = spdata[j]

            c2 = euclid(2*half_nrm2, data, sp2) <= radius_22
            den2 = np.count_nonzero(c2) / volume

            if check_if_overlap(sp1, sp2, radius=radius): 
                internum = np.count_nonzero(c1 & c2)
                cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                if cid >= den1 or cid >= den2: 
                    merge(connected_pairs[i], connected_pairs[j])
                    connected_pairs_store.append([i, j])

        
    labels = [findParent(i).data for i in connected_pairs]
    labels_set = list()
    for i in np.unique(labels):
        labels_set.append(np.where(labels == i)[0].tolist())

    return labels_set, connected_pairs_store 




def euclid(xxt, X, v):
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)
    


class SET:
    """Disjoint-set data structure."""
    def __init__( self, data):
        self.data = data
        self.parent = self

        
        
def findParent(s):
    """Find parent of node."""
    if (s.data != s.parent.data) :
        s.parent = findParent((s.parent))
    return s.parent



def merge(s1, s2):
    """Merge the roots of two node."""
    parent_of_s1 = findParent(s1)
    parent_of_s2 = findParent(s2)

    if (parent_of_s1.data != parent_of_s2.data) :
        findParent(s1).parent = parent_of_s2
        
        

def check_if_overlap(starting_point, spo, radius, scale = 1):
    """Check if two groups formed by aggregation overlap
    """
    return np.linalg.norm(starting_point - spo, ord=2, axis=-1) <= 2 * scale * radius

    

def cal_inter_density(starting_point, spo, radius, num):
    """Calculate the density of intersection (lens)
    """
    in_volume = cal_inter_volume(starting_point, spo, radius)
    return num / in_volume



def cal_inter_volume(starting_point, spo, radius):
    """
    Returns the volume of the intersection of two spheres in n-dimensional space.
    The radius of the two spheres is r and the distance of their centers is d.
    For d=0 the function returns the volume of full sphere.
    Reference: https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres

    """
    
    dim =starting_point.shape[0]
    dist = np.linalg.norm(starting_point - spo, ord=2, axis=-1) # the distance between the two groups
    
    if dist > 2*radius:
        return 0
    c = dist / 2
    return np.pi**(dim/2)/gamma(dim/2 + 1)*(radius**dim)*betainc((dim + 1)/2, 1/2, 1 - c**2/radius**2)

