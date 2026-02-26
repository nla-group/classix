import numpy as np
from collections import deque

def merge_manhattan(
    spdata,              
    group_sizes,         
    sort_vals_sp,        
    agg_labels_sp,       
    radius,
    mergeScale,
    minPts,
    mergeTinyGroups
):
    """
    Perform Manhattan distance-based group merging for CLASSIX clustering.

    This function merges aggregation groups (starting points) using Manhattan (L1)
    distance with a scaled radius threshold. It builds a graph of connected groups,
    performs union-find style merging (assigning to the smallest label), then
    redistributes points from small clusters (size < minPts) to the nearest valid
    cluster using full Manhattan distance.

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
            1 = connected by Manhattan distance <= mergeScale * radius
            2 = connection created during small-cluster redistribution

    """
    n_groups = len(spdata)
    Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    label_sp = agg_labels_sp.copy()  

    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue

        xi = spdata[i]
        search_radius = mergeScale * radius + sort_vals_sp[i]
        last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')

        if last_j > i:
            dists = np.sum(np.abs(spdata[i:last_j] - xi), axis=1)
            inds_rel = np.where(dists <= mergeScale * radius)[0]
            inds = i + inds_rel

            if not mergeTinyGroups:
                valid = group_sizes[inds] >= minPts
                inds = inds[valid]

            Adj[i, inds] = 1
            Adj[inds, i] = 1

            connected_labels = np.unique(label_sp[inds])
            if len(connected_labels) > 1:
                minlab = np.min(connected_labels)
                for lbl in connected_labels:
                    label_sp[label_sp == lbl] = minlab

    ul = np.unique(label_sp)
    cs = np.zeros(len(ul), dtype=int)
    group_sizes = np.array(group_sizes)

    new_label_map = {old: new for new, old in enumerate(ul)}
    for old, new in new_label_map.items():
        mask = (label_sp == old)
        cs[new] = np.sum(group_sizes[mask])
        label_sp[mask] = new

    small_clusters = np.where(cs < minPts)[0]

    label_sp_copy = label_sp.copy()

    for cluster_id in small_clusters:
        group_ids = np.where(label_sp_copy == cluster_id)[0]
        for gid in group_ids:
            xi = spdata[gid]
            dists = np.sum(np.abs(spdata - xi), axis=1) 
            order = np.argsort(dists)

            for nearest_gid in order:
                target_cluster = label_sp_copy[nearest_gid]
                if cs[target_cluster] >= minPts:
                    label_sp[gid] = target_cluster
                    Adj[gid, nearest_gid] = 2
                    Adj[nearest_gid, gid] = 2
                    break

    ul_final = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(ul_final)}
    label_sp = np.array([final_map[l] for l in label_sp])

    return {
        'group_cluster_labels': label_sp, 
        'Adj': Adj,
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