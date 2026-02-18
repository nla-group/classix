import numpy as np
import scipy.sparse as sparse
from copy import deepcopy
from collections import deque
from spmv import spsubmatxvec

def merge_tanimoto(spdata, group_sizes, sort_vals_sp, agg_labels_sp, radius, mergeScale, minPts, mergeTinyGroups):
    """
    Perform Tanimoto-based group merging for CLASSIX clustering.

    This function merges aggregation groups (starting points) using Tanimoto distance
    (1 - Tanimoto similarity) with a scaled radius threshold. It first builds a graph
    of connected groups, performs union-find style merging (assigning to the smallest
    label), then redistributes points from small clusters (size < minPts) to the
    nearest valid cluster using full Tanimoto distance.

    Parameters
    ----------
    spdata : ndarray of shape (n_groups, n_features)
        Coordinates of the group starting points (aggregation representatives).
        
    group_sizes : array-like of shape (n_groups,)
        Number of points in each aggregation group.

    sort_vals_sp : ndarray of shape (n_groups,)
        Precomputed sorting values (e.g., L1 norms) for the starting points,
        used to define search windows.

    agg_labels_sp : ndarray of shape (n_groups,)
        Initial group labels from aggregation phase (usually 0 to n_groups-1).

    radius : float
        Base radius parameter from CLASSIX.

    mergeScale : float
        Scaling factor for the merging radius (effective radius = mergeScale * radius).

    minPts : int
        Minimum number of points required for a cluster to be valid.
        Groups/clusters smaller than minPts are redistributed.

    mergeTinyGroups : bool
        If False, tiny groups (size < minPts) are ignored when building edges in
        the merging graph (they do not initiate connections).

    Returns
    -------
    dict
        Dictionary containing:
        - 'group_cluster_labels' : ndarray of shape (n_groups,)
            Final cluster label for each starting point (0, 1, 2, ..., n_clusters-1).
        - 'Adj' : ndarray of shape (n_groups, n_groups), dtype=int8
            Adjacency matrix of the merging graph.
            1 = connected by Tanimoto distance <= mergeScale * radius
            2 = connection created during small-cluster redistribution

    """
    n_groups = len(spdata)
    label_sp = agg_labels_sp.copy()
    
    spdatas = sparse.csr_matrix(spdata)
    spdatas_data = spdatas.data.astype(np.float64)
    spdatas_indices = spdatas.indices.astype(np.int32)
    spdatas_indptr = spdatas.indptr.astype(np.int32)
    
    Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    
    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue
            
        xi = spdata[i, :].astype(np.float64)
        search_radius = sort_vals_sp[i] / (1 - mergeScale * radius)
        last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')
        
        window_start = i
        window_end = min(last_j, n_groups)
        window_size = window_end - window_start
        
        if window_size > 0:
            ips = np.zeros(window_size, dtype=np.float64)
            spsubmatxvec(
                spdatas_data, spdatas_indptr, spdatas_indices,
                window_start, window_end, xi, ips
            )
            
            denom = sort_vals_sp[i] + sort_vals_sp[window_start:window_end] - ips
            denom[denom == 0] = 1e-15
            tanimoto_dist = 1 - ips / denom
            
            rel_inds = np.where(tanimoto_dist <= mergeScale * radius)[0]
            inds = i + rel_inds
            
            if not mergeTinyGroups:
                valid = np.array(group_sizes)[inds] >= minPts
                inds = inds[valid]
            
            Adj[i, inds] = 1
            Adj[inds, i] = 1
            
            connected_labels = np.unique(label_sp[inds])
            if len(connected_labels) > 1:
                minlab = np.min(connected_labels)
                for lbl in connected_labels:
                    label_sp[label_sp == lbl] = minlab

        
    ul = np.unique(label_sp)
    group_sizes_arr = np.array(group_sizes)
    cluster_sizes = {lbl: np.sum(group_sizes_arr[label_sp == lbl]) for lbl in ul}
    
    small_clusters = [lbl for lbl, size in cluster_sizes.items() if size < minPts]
    
    if small_clusters:
        label_sp_fixed = label_sp.copy()
        
        for cluster_id in small_clusters:
            group_ids = np.where(label_sp_fixed == cluster_id)[0]
            for gid in group_ids:
                xi = spdata[gid, :].astype(np.float64)
                ips = np.zeros(n_groups, dtype=np.float64)
                
                spsubmatxvec(
                    spdatas_data, spdatas_indptr, spdatas_indices,
                    0, n_groups, xi, ips
                )
                
                denom = sort_vals_sp[gid] + sort_vals_sp - ips
                denom[denom == 0] = 1e-15
                dist = 1 - ips / denom
                
                sorted_indices = np.argsort(dist, kind='stable')
                for target_gid in sorted_indices:
                    target_cluster = label_sp_fixed[target_gid]
                    if cluster_sizes[target_cluster] >= minPts:
                        label_sp[gid] = target_cluster
                        Adj[gid, target_gid] = 2 
                        Adj[target_gid, gid] = 2
                        break

    final_ul = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(final_ul)}
    label_sp = np.array([final_map[l] for l in label_sp])

    return {
        'group_cluster_labels': label_sp,
        'Adj': Adj
    }


def bfs_shortest_path(adj_matrix, start, goal):
    """Find shortest path between two nodes using BFS on an adjacency matrix.
    
    Parameters
    ----------
    adj_matrix : ndarray of shape (n, n)
        Adjacency matrix where nonzero entries indicate connections.
        Value 2 indicates a connection created during minPts redistribution.
    
    start : int
        Start node index (0-based).
    
    goal : int
        Goal node index (0-based).
    
    Returns
    -------
    list or None
        List of node indices forming the shortest path, or None if no path exists.
    """
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current_node, path = queue.popleft()
        if current_node == goal:
            return path
        for neighbor, connected in enumerate(adj_matrix[current_node]):
            if connected and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None