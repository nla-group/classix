# -*- coding: utf-8 -*-
#
# CLASSIX: Fast and explainable clustering based on sorting
#
# MIT License
#
# Copyright (c) 2026 Stefan Güttel, Xinye Chen

# Python implementation for group merging

import collections
import numpy as np
from scipy.special import betainc, gamma


def distance_merge_mtg(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2):
    """Merge Euclidean groups while preventing tiny groups from initiating edges.
    
    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed and sorted training data.
    
    labels : array-like of shape (n_samples,)
        Aggregation group labels in sorted-data order.

    splist : ndarray of shape (n_groups, 2)
        Starting-point index and group size for each aggregation group.

    radius : float
        Base aggregation radius.

    minPts : int
        Minimum valid cluster size.

    scale : float
        Multiplicative factor for the merge radius.

    sort_vals : ndarray of shape (n_samples,)
        Sorting values in sorted-data order.
        
    half_nrm2 : ndarray of shape (n_groups,)
        Half squared norms of the group starting points.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels in sorted-data order.
    
    old_cluster_count : collections.Counter
        Cluster sizes before small-cluster reassignment.
    
    SIZE_NOISE_LABELS : int
        Number of clusters smaller than ``minPts`` before reassignment.
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
    """Merge Euclidean groups by distance between starting points.
    
    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed and sorted training data.
    
    labels : array-like of shape (n_samples,)
        Aggregation group labels in sorted-data order.

    splist : ndarray of shape (n_groups, 2)
        Starting-point index and group size for each aggregation group.

    radius : float
        Base aggregation radius.

    minPts : int
        Minimum valid cluster size.

    scale : float
        Multiplicative factor for the merge radius.

    sort_vals : ndarray of shape (n_samples,)
        Sorting values in sorted-data order.
        
    half_nrm2 : ndarray of shape (n_groups,)
        Half squared norms of the group starting points.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels in sorted-data order.
    
    old_cluster_count : collections.Counter
        Cluster sizes before small-cluster reassignment.
    
    SIZE_NOISE_LABELS : int
        Number of clusters smaller than ``minPts`` before reassignment.
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
    """Merge Euclidean groups with the density-overlap criterion.
    
    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Preprocessed and sorted training data.
    
    splist : ndarray of shape (n_groups, 2)
        Starting-point index and group size for each aggregation group.

    radius : float
        Base aggregation radius.

    sort_vals : ndarray of shape (n_samples,)
        Sorting values in sorted-data order.

    half_nrm2 : ndarray of shape (n_samples,)
        Half squared norms of sorted samples.

        
    Returns
    -------
    labels_set : list of list of int
        Connected components of the group-overlap graph.
    
    connected_pairs_store : list of list of int
        Pairwise group connections used to build explanations.
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
    """Compute squared Euclidean distances from rows of ``X`` to ``v``.

    Parameters
    ----------
    xxt : ndarray of shape (n_samples,)
        Precomputed squared norms for rows of ``X``.

    X : ndarray of shape (n_samples, n_features)
        Input data.

    v : ndarray of shape (n_features,)
        Reference vector.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Squared Euclidean distances.
    """
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)
    


class SET:
    """Node in the disjoint-set data structure used for group merging.

    Parameters
    ----------
    data : int
        Group identifier stored by this node.
    """
    def __init__( self, data):
        """Initialize a disjoint-set node."""
        self.data = data
        self.parent = self

        
        
def findParent(s):
    """Return the root node of a disjoint-set component.

    Parameters
    ----------
    s : SET
        Node whose component root is requested.

    Returns
    -------
    parent : SET
        Root node after path compression.
    """
    if (s.data != s.parent.data) :
        s.parent = findParent((s.parent))
    return s.parent



def merge(s1, s2):
    """Merge two disjoint-set components.

    Parameters
    ----------
    s1, s2 : SET
        Nodes whose components should be merged.

    Returns
    -------
    None
        The parent pointer of one component is updated in place.
    """
    parent_of_s1 = findParent(s1)
    parent_of_s2 = findParent(s2)

    if (parent_of_s1.data != parent_of_s2.data) :
        findParent(s1).parent = parent_of_s2
        
        

def check_if_overlap(starting_point, spo, radius, scale = 1):
    """Return whether two aggregation balls overlap.

    Parameters
    ----------
    starting_point : ndarray of shape (n_features,)
        Center of the first aggregation ball.

    spo : ndarray of shape (n_features,)
        Center of the second aggregation ball.

    radius : float
        Ball radius.

    scale : float, default=1
        Additional multiplicative scale.

    Returns
    -------
    overlap : bool
        ``True`` if the centers are at most ``2 * scale * radius`` apart.
    """
    return np.linalg.norm(starting_point - spo, ord=2, axis=-1) <= 2 * scale * radius

    

def cal_inter_density(starting_point, spo, radius, num):
    """Calculate sample density in the intersection of two balls.

    Parameters
    ----------
    starting_point : ndarray of shape (n_features,)
        Center of the first ball.

    spo : ndarray of shape (n_features,)
        Center of the second ball.

    radius : float
        Common ball radius.

    num : int
        Number of samples in the intersection.

    Returns
    -------
    density : float
        ``num`` divided by the intersection volume.
    """
    in_volume = cal_inter_volume(starting_point, spo, radius)
    return num / in_volume



def cal_inter_volume(starting_point, spo, radius):
    """Return the intersection volume of two equal-radius hyperspheres.

    Parameters
    ----------
    starting_point : ndarray of shape (n_features,)
        Center of the first hypersphere.

    spo : ndarray of shape (n_features,)
        Center of the second hypersphere.

    radius : float
        Common hypersphere radius.

    Returns
    -------
    volume : float
        Volume of the intersection. If the centers are farther than
        ``2 * radius`` apart, the volume is zero.

    Notes
    -----
    The formula follows the standard hypersphere lens-volume expression using
    the regularized incomplete beta function.
    """
    
    dim =starting_point.shape[0]
    dist = np.linalg.norm(starting_point - spo, ord=2, axis=-1) # the distance between the two groups
    
    if dist > 2*radius:
        return 0
    c = dist / 2
    return np.pi**(dim/2)/gamma(dim/2 + 1)*(radius**dim)*betainc((dim + 1)/2, 1/2, 1 - c**2/radius**2)

