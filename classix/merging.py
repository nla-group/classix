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


import numpy as np
# from tqdm import tqdm 
from scipy.special import betainc, gamma
# from sklearn.metrics import pairwise_distances # for scc algorithm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, _sparsetools
# from scipy.sparse.csgraph import minimum_spanning_tree


def fast_agglomerate(data, splist, radius, method='distance', scale=1.5):
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
    # connected_pairs = list()
    connected_pairs = [SET(i) for i in range(splist.shape[0])]
    connected_pairs_store = []
    if method == "density":
        volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1)
    else:
        volume = None
        
    for i in range(splist.shape[0]):
        sp1 =  data[int(splist[i, 0])] # splist[int(i), 3:]
        neigbor_sp = data[splist[i+1:, 0].astype(int)] # splist[i+1:, 3:] # deprecated: splist[i+1:, :] 
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

            # if len(index_overlap[0]) <= 0:  # -> [for np.where, removed],  make sure the neighbors is not empty
            #     continue

            # neigbor_sp = neigbor_sp[index_overlap,:] # calculate their neighbor ...2)
            c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
            den1 = np.count_nonzero(c1) / volume
            # den1 = splist[int(i), 2] / volume # density(splist[int(i), 2], volume = volume) 
            for j in select_stps.astype(int):
                sp2 = data[int(splist[j, 0])] # splist[int(j), 3:]

                c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
                den2 = np.count_nonzero(c2) / volume
                #den2 = splist[int(j), 2] / volume # density(splist[int(j), 2], volume = volume)
                if check_if_overlap(sp1, sp2, radius=radius): # , scale=distance_scale
                    # interdata = np.linalg.norm(data[agg_labels==i] - sp2, ord=2, axis=-1)
                    # internum = len(interdata[interdata <= 2*radius])
                    internum = np.count_nonzero(c1 & c2)
                    cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
                    if cid >= den1 or cid >= den2: # eta*cid >= den1 or eta*cid >= den2:
                        # connected_pairs.append([i,j]) # store connected groups
                        merge(connected_pairs[i], connected_pairs[j])
                        connected_pairs_store.append([i, j])
        else: # "distance": 
            # THIS PART WE OMIT THE FAST QUERY WITH EARLY STOPPING CAUSE WE DON'T THINK EARLY STOPPING IN PYTHON CODE CAN BE FASTER THAN 
            # PYTHON BROADCAST, BUT IN THE CYTHON CODE, WE IMPLEMENT FAST QUERY WITH EARLY STOPPING BY LEVERAGING THE ADVANTAGE OF SORTED 
            # AGGREGATION.
            # index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= scale*radius  # 2*distance_scale*radius
            index_overlap = fast_query(neigbor_sp, sp1, sort_vals, scale*radius)
            select_stps = select_stps[index_overlap]
            if not np.any(index_overlap): # calculate their neighbors ...1)
                continue
    
            # neigbor_sp = neigbor_sp[index_overlap] # calculate their neighbor ...2)
            for j in select_stps:
                # two groups merge when their starting points distance is smaller than 1.5*radius
                # connected_pairs.append([i,j]) # store connected groups
                merge(connected_pairs[i], connected_pairs[j])
                connected_pairs_store.append([i, j])
        # else:
        #     raise ValueError(
        #         f"The method '{method}' does not exist,"
        #         f"please assign a new value")
        
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



# def scc_agglomerate(splist, radius=0.5, scale=1.5, n_jobs=-1): # limited to distance-based method
#     """
#     Implement CLASSIX's merging with strongly connected components finding algorithm in undirected graph.
#     """
# 
#     distm = pairwise_distances(splist[:,3:], Y=None, metric='euclidean', n_jobs=n_jobs)
#     index_set = []
#     distm = (distm <= radius*scale).astype(int)
# 
#     n_components, labels = connected_components(csgraph=csr_matrix(distm), directed=False, return_labels=True)
#     labels_set = list()
#     for i in np.unique(labels):
#         labels_set.append(np.where(labels == i)[0].tolist())
#     return labels_set



# def minimum_spanning_tree_agglomerate(splist, radius=0.5, scale=1.5):
#     """
#     Implement CLASSIX's merging with minimum spanning tree clustering.
#     """
#     # distm = pairwise_distances(splist[:,3:], Y=None, metric='euclidean', n_jobs=-1)
#     # min_span = minimum_spanning_tree(csr_matrix(distm))
#     # print(min_span)
#     # min_span = csr_matrix((min_span.todense()<=radius*scale).astype(int))
#     # link_list = [list(pair)for pair in csr_matrix_indices(min_span)]
#     # return link_list
#     from mst_clustering import MSTClustering
# 
#     # predict the labels with the MST algorithm
#     model = MSTClustering(cutoff_scale=radius*scale)
#     labels = model.fit_predict(splist[:, 3:])
#     labels_set = list()
#     for i in np.unique(labels):
#         labels_set.append(np.where(labels == i)[0].tolist())
#     return labels_set




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
        
        
        
# def merge_pairs(pairs):
#     """Transform connected pairs to connected groups (list)"""
#     
#     len_p = len(pairs)
#     ulabels = np.full(len_p, -1, dtype=int)
#     labels = list()
#     maxid = 0
#     for i in range(len_p):
#         if ulabels[i] == -1:
#             sub = pairs[i]
#             ulabels[i] = maxid
# 
#             for j in range(i+1, len_p):
#                 com = pairs[j]
#                 if set(sub).intersection(com) != set():
#                     sub = sub + com
#                     if ulabels[j] == -1:
#                         ulabels[j] = maxid
#                     else:
#                         ulabels[ulabels == maxid] = ulabels[j]
# 
#             maxid = maxid + 1
# 
#     for i in np.unique(ulabels):
#         sub = list()
#         for j in [p for p in range(len_p) if ulabels[p] == i]: # np.where(ulabels == i)[0] 
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


# deprecated (24/07/2021)
# def csr_matrix_indices(S):
#     """
#     Return a list of the indices of nonzero entries of a csr_matrix S
#     """
#     major_dim, minor_dim = S.shape
#     minor_indices = S.indices
# 
#     major_indices = np.empty(len(minor_indices), dtype=S.indices.dtype)
#     _sparsetools.expandptr(major_dim, S.indptr, major_indices)
# 
#     return zip(major_indices, minor_indices)




def check_if_overlap(starting_point, spo, radius, scale = 1):
    """Check if two groups formed by aggregation overlap
    """
    return np.linalg.norm(starting_point - spo, ord=2, axis=-1) <= 2 * scale * radius

    

def cal_inter_density(starting_point, spo, radius, num):
    """Calculate the density of intersection (lens)
    """
    in_volume = cal_inter_volume(starting_point, spo, radius)
    return num / in_volume



# deprecated (24/07/2021)
# def cal_inter_area(starting_point, spo, radius):
#     """Calculate the area of intersection (lens)
#     """
#     
#     area = 2 * (radius**2) * np.arccos(dist / (2 * radius)) - dist * np.sqrt(4*radius**2 - dist**2) / 2
#     return area    



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



#====================================================================================================================#
# ------------------------------------------------------------------------------------------------------------------

# Deprecated function
# def agglomerate_trivial(data, splist, radius, method="distance", scale=1.5):
#     connected_pairs = list()
#     for i in range(splist.shape[0]):
#         sp1 = splist[int(i), 3:]
#         neigbor_sp = splist[i+1:, 3:]
#         select_stps = np.arange(i+1, splist.shape[0], dtype=int)
#         if method == "density":                    # calculate the density
#             volume = np.pi**(data.shape[1]/2) * radius**data.shape[1] / gamma(data.shape[1]/2+1)
#             index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= radius # 2*distance_scale*radius
#             # index_overlap = np.where(np.linalg.norm(splist[:,3:] - sp1, ord=2, axis=-1) <= 2*radius)  # -> [for np.where, removed], 2*distance_scale*radius
#             select_stps = select_stps[index_overlap]
#             neigbor_sp = neigbor_sp[index_overlap,:] # calculate their neighbor ...2)
#             # calculate their neighbors ...1)
#             # later decide whether sps should merge with neighbors
#             if not np.any(index_overlap):
#                 continue
#             # neigbor_sp = neigbor_sp[index_overlap]

#             c1 = np.linalg.norm(data-sp1, ord=2, axis=-1) <= radius
#             den1 = np.count_nonzero(c1) / volume
#             # den1 = splist[int(i), 2] / volume # density(splist[int(i), 2], volume = volume) 
#             for j in select_stps:
#                 sp2 = splist[int(j), 3:]
# 
#                 c2 = np.linalg.norm(data-sp2, ord=2, axis=-1) <= radius
#                 den2 = np.count_nonzero(c2) / volume
#                 #den2 = splist[int(j), 2] / volume # density(splist[int(j), 2], volume = volume)
#                 if check_if_overlap(sp1, sp2, radius=radius): # , scale=distance_scale
#                     # interdata = np.linalg.norm(data[agg_labels==i] - sp2, ord=2, axis=-1)
#                     # internum = len(interdata[interdata <= 2*radius])
#                     internum = np.count_nonzero(c1 & c2)
#                     cid = cal_inter_density(sp1, sp2, radius=radius, num=internum)
#                     if cid >= den1 or cid >= den2: # eta*cid >= den1 or eta*cid >= den2:
#                         connected_pairs.append([i,j]) # store connected groups

#         else:#  "distance": 
#             index_overlap = np.linalg.norm(neigbor_sp - sp1, ord=2, axis=-1) <= scale*radius  # 2*distance_scale*radius
#             select_stps = select_stps[index_overlap]
#             if not np.any(index_overlap): # calculate their neighbors ...1)
#                 continue

#             # neigbor_sp = neigbor_sp[index_overlap] # calculate their neighbor ...2)
#             for j in select_stps:
#                 # two groups merge when their starting points distance is smaller than 1.5*radius
#                 connected_pairs.append([i,j]) # store connected groups
#         
#     return connected_pairs



# Deprecated function
# def merge_pairs_dr(pairs):
#     """Transform connected pairs to connected groups (list)"""
#     
#     len_p = len(pairs)
#     ulabels = np.full(len_p, -1, dtype=int)
#     labels = list()
#     maxid = 0
#     for i in range(len_p):
#         if ulabels[i] == -1:
#             sub = pairs[i]
#             ulabels[i] = maxid
# 
#             for j in range(i+1, len_p):
#                 com = pairs[j]
#                 if set(sub).intersection(com) != set():
#                     sub = sub + com
#                     if ulabels[j] == -1:
#                         ulabels[j] = maxid
#                     else:
#                         ulabels[ulabels == maxid] = ulabels[j]
# 
#             maxid = maxid + 1
# 
#     for i in np.unique(ulabels):
#         sub = list()
#         for j in [p for p in range(len_p) if ulabels[p] == i]: # np.where(ulabels == i)[0] 
#             sub = sub + list(pairs[int(j)])
#         labels.append(list(set(sub)))
#     return labels

# ------------------------------------------------------------------------------------------------------------------
#====================================================================================================================#
