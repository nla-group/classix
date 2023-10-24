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

# Python implementation for group merging

import collections
import numpy as np
from scipy.special import betainc, gamma



def bf_distance_agglomerate(data, labels, splist, radius, minPts=0, scale=1.5):
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
    
    splist_indices = splist[:, 0].astype(int)

    spdata = data[splist_indices]
    xxt = np.einsum('ij,ij->i', spdata, spdata) 
    # np.einsum('ij,ij->i', data, data) <-> np.sum(data * data, axis=1)

    sp_cluster_label = labels[splist_indices]

    for i in range(splist.shape[0]):
        xi = spdata[i, :]
        spl = np.unique( sp_cluster_label[euclid(xxt, spdata, xi) <= (scale*radius)**2] )
        minlab = np.min(spl)

        for label in spl:
            sp_cluster_label[sp_cluster_label==label] = minlab

    # rename labels to be 1,2,3,... and determine cluster sizes
    ul = np.unique(sp_cluster_label)
    nr_u = len(ul)
    
    cs = np.zeros(nr_u, dtype=int)
    grp_sizes = splist[:, 2].astype(int)

    for i in range(nr_u):
        cid = sp_cluster_label==ul[i]
        sp_cluster_label[cid] = i
        cs[i] = np.sum(grp_sizes[cid])

    old_cluster_count = collections.Counter(sp_cluster_label[labels])
    cid = np.nonzero(cs < minPts)[0]
    SIZE_NOISE_LABELS = cid.size

    if SIZE_NOISE_LABELS > 0:
        copy_sp_cluster_label = sp_cluster_label.copy()
        
        for i in cid:
            ii = np.nonzero(copy_sp_cluster_label==i)[0] # find all tiny groups with that label
            for iii in ii:
                xi = spdata[iii, :]    # starting point of one tiny group

                dist = euclid(xxt, spdata, xi)
                merge_ind = np.argsort(dist)
                for j in merge_ind:
                    if cs[copy_sp_cluster_label[j]] >= minPts:
                        sp_cluster_label[iii] = copy_sp_cluster_label[j]
                        break

        ul = np.unique(sp_cluster_label)
        nr_u = len(ul)
        cs = np.zeros(nr_u, dtype=int)

        for i in range(nr_u):
            cid = sp_cluster_label==ul[i]
            sp_cluster_label[cid] = i
            cs[i] = np.sum(grp_sizes[cid])

    labels = sp_cluster_label[labels]

    return labels, old_cluster_count, SIZE_NOISE_LABELS



def agglomerate(data, splist, radius, method='distance', scale=1.5):
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
    
    connected_pairs = [SET(i) for i in range(splist.shape[0])]
    connected_pairs_store = []
    splist_indices = splist[:, 0].astype(int)


    if method == "density":
        volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1)
    else:
        volume = None
        
    for i in range(splist.shape[0]):
        sp1 =  data[splist_indices[i]]
        neigbor_sp = data[splist_indices[i+1:]] 
        select_stps = np.arange(i+1, splist.shape[0], dtype=int)
        sort_vals = splist[i:, 1]
        
        if method == "density":                    # calculate the density
            # THIS PART WE OMIT THE FAST QUERY WITH EARLY STOPPING CAUSE WE DON'T THINK EARLY STOPPING IN PYTHON CODE CAN BE FASTER THAN 
            # PYTHON BROADCAST, BUT IN THE CYTHON CODE, WE IMPLEMENT FAST QUERY WITH EARLY STOPPING BY LEVERAGING THE ADVANTAGE OF SORTED 
            # AGGREGATION.
            index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= 2*radius # 2*distance_scale*radius
            select_stps = select_stps[index_overlap]
            # calculate their neighbors ...1)
            # later decide whether sps should merge with neighbors
            if not np.any(index_overlap):
                continue

            c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
            den1 = np.count_nonzero(c1) / volume
            
            for j in select_stps.astype(int):
                sp2 = data[int(splist[j, 0])] # splist[int(j), 3:]

                c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
                den2 = np.count_nonzero(c2) / volume

                if check_if_overlap(sp1, sp2, radius=radius): 
                    internum = np.count_nonzero(c1 & c2)
                    cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                    if cid >= den1 or cid >= den2: 
                        merge(connected_pairs[i], connected_pairs[j])
                        connected_pairs_store.append([i, j])

        else: 
            # THIS PART WE OMIT THE FAST QUERY WITH EARLY STOPPING CAUSE WE DON'T THINK EARLY STOPPING IN PYTHON CODE CAN BE FASTER THAN 
            # PYTHON BROADCAST, BUT IN THE CYTHON CODE, WE IMPLEMENT FAST QUERY WITH EARLY STOPPING BY LEVERAGING THE ADVANTAGE OF SORTED 
            # AGGREGATION.
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, scale*radius)
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap): # calculate their neighbors ...1)
                continue
    
            for j in select_stps:
                # two groups merge when their starting points distance is smaller than 1.5*radius
                # connected_pairs.append([i,j]) # store connected groups
                merge(connected_pairs[i], connected_pairs[j])
                connected_pairs_store.append((i, j))
        
    labels = [findParent(i).data for i in connected_pairs]
    labels_set = list()
    for i in np.unique(labels):
        labels_set.append(np.where(labels == i)[0].tolist())
    return labels_set, connected_pairs_store  # merge_pairs(connected_pairs)






def fast_query(data, starting_point, sort_vals, tol):
    """Fast query with built in early stopping strategy for merging."""
    len_ind = data.shape[0]
    fdim = data.shape[1] # remove the first three elements that not included in the features
    index_query=np.full(len_ind, False, dtype=bool)
    for i in range(len_ind):
        candidate = data[i]
        
        if (sort_vals[i+1] - sort_vals[0] > tol):
            break
            
        dat = candidate - starting_point
        dist = np.inner(dat, dat)
        if dist <= tol**2:
            index_query[i] = True
            
    return index_query



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

