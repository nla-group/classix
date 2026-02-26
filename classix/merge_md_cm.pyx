# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

ctypedef cnp.int64_t INT64_t
ctypedef cnp.int32_t INT32_t
ctypedef cnp.int8_t INT8_t
ctypedef cnp.float64_t FLOAT64_t

# Union-Find
cdef inline INT32_t find_root(INT32_t[:] parent, INT32_t i) noexcept nogil:
    """Find the root of the node and perform path compression. """
    cdef INT32_t root = i
    while root != parent[root]:
        root = parent[root]
    
    cdef INT32_t curr = i
    cdef INT32_t temp
    while curr != root:
        temp = parent[curr]
        parent[curr] = root
        curr = temp
    return root

cdef inline void union_sets(INT32_t[:] parent, INT32_t i, INT32_t j) noexcept nogil:
    """Merge two sets, always pointing the larger root to the smaller root to simulate the ‘minimum label’ logic."""
    cdef INT32_t root_i = find_root(parent, i)
    cdef INT32_t root_j = find_root(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

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
    
    cdef FLOAT64_t[:, :] spdata_mv = np.asarray(spdata, dtype=np.float64)
    cdef INT64_t[:] group_sizes_mv = np.asarray(group_sizes, dtype=np.int64)
    cdef FLOAT64_t[:] sort_vals_mv = np.asarray(sort_vals_sp, dtype=np.float64)
    cdef INT64_t[:] agg_labels_mv = np.asarray(agg_labels_sp, dtype=np.int64)
    
    cdef Py_ssize_t n_groups = spdata_mv.shape[0]
    cdef Py_ssize_t n_features = spdata_mv.shape[1]
    
    cdef INT8_t[:, :] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    cdef INT32_t[:] parent = np.arange(n_groups, dtype=np.int32)
    
    cdef double dist_threshold = mergeScale * radius
    cdef double search_radius, dist, diff
    cdef Py_ssize_t i, j, k, last_j, left, right, mid
    
    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes_mv[i] < minPts:
            continue

        search_radius = dist_threshold + sort_vals_mv[i]
        
        left = i + 1
        right = n_groups
        while left < right:
            mid = (left + right) // 2
            if sort_vals_mv[mid] <= search_radius:
                left = mid + 1
            else:
                right = mid
        last_j = left

        if last_j > i + 1:
            for j in range(i + 1, last_j):
                if not mergeTinyGroups and group_sizes_mv[j] < minPts:
                    continue
                
                # Compute L1 Distance with Early Exit
                dist = 0.0
                for k in range(n_features):
                    diff = spdata_mv[i, k] - spdata_mv[j, k]
                    dist += fabs(diff)
                    if dist > dist_threshold:
                        break
                
                if dist <= dist_threshold:
                    Adj[i, j] = 1
                    Adj[j, i] = 1
                    union_sets(parent, i, j)

    
    cdef INT64_t[:] final_labels = np.empty(n_groups, dtype=np.int64)
    cdef INT64_t[:] root_to_min_label = np.full(n_groups, 9223372036854775807, dtype=np.int64) # Max INT64
    cdef INT32_t root
    
    for i in range(n_groups):
        root = find_root(parent, i)
        if agg_labels_mv[i] < root_to_min_label[root]:
            root_to_min_label[root] = agg_labels_mv[i]
            
    for i in range(n_groups):
        root = find_root(parent, i)
        final_labels[i] = root_to_min_label[root]

    cdef INT64_t[:] ul = np.unique(final_labels)
    cdef INT64_t[:] cs = np.zeros(ul.shape[0], dtype=np.int64)
    cdef INT64_t[:] label_map = np.zeros(np.max(ul) + 1, dtype=np.int64) # Map original value -> 0..K
    
    for i in range(ul.shape[0]):
        label_map[ul[i]] = i
        
    for i in range(n_groups):
        final_labels[i] = label_map[final_labels[i]]
        cs[final_labels[i]] += group_sizes_mv[i]

    cdef INT64_t[:] label_sp_copy = final_labels.copy()
    cdef double min_dist
    cdef Py_ssize_t best_neighbor
    cdef INT64_t target_cluster
    
    for i in range(n_groups):
        # Check if current point belongs to a small cluster
        if cs[label_sp_copy[i]] < minPts:
            
            min_dist = 1.7976931348623157e+308 # DBL_MAX
            best_neighbor = -1
            
            # Linear scan for nearest valid neighbor (replacing argsort)
            for j in range(n_groups):
                target_cluster = label_sp_copy[j]
                
                # Check validity first to skip distance calc
                if cs[target_cluster] >= minPts:
                    # Self-check usually redundant if size < minPts but good for safety
                    if i == j: continue 
                    
                    dist = 0.0
                    for k in range(n_features):
                        diff = spdata_mv[i, k] - spdata_mv[j, k]
                        dist += fabs(diff)
                        if dist >= min_dist: 
                            break
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor = j
            
            if best_neighbor != -1:
                target_cluster = label_sp_copy[best_neighbor]
                final_labels[i] = target_cluster
                Adj[i, best_neighbor] = 2
                Adj[best_neighbor, i] = 2

    cdef INT64_t[:] ul_final = np.unique(final_labels)
    cdef INT64_t[:] final_map = np.full(np.max(ul_final) + 1, -1, dtype=np.int64)
    
    for i in range(ul_final.shape[0]):
        final_map[ul_final[i]] = i
        
    for i in range(n_groups):
        final_labels[i] = final_map[final_labels[i]]

    return {
        'group_cluster_labels': np.asarray(final_labels), 
        'Adj': np.asarray(Adj),
    }