# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix
from spmv import spsubmatxvec

def merge_tanimoto(
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
    spdata = np.ascontiguousarray(spdata, dtype=np.float64)
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    sort_vals_sp = np.ascontiguousarray(sort_vals_sp, dtype=np.float64)
    agg_labels_sp = np.asarray(agg_labels_sp, dtype=np.int64)

    cdef int n_groups = spdata.shape[0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] label_sp = agg_labels_sp.copy()

    spdatas = csr_matrix(spdata)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] spdatas_data = spdatas.data.astype(np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] spdatas_indices = spdatas.indices.astype(np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] spdatas_indptr = spdatas.indptr.astype(np.int32)

    cdef cnp.ndarray[cnp.int8_t, ndim=2] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] ips = np.zeros(n_groups, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] denom = np.empty(n_groups, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tanimoto_dist = np.empty(n_groups, dtype=np.float64)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] ips_slice
    cdef cnp.ndarray[cnp.float64_t, ndim=1] denom_slice
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tanimoto_slice

    cdef cnp.ndarray[cnp.int64_t, ndim=1] rel_inds
    cdef cnp.ndarray[cnp.int64_t, ndim=1] inds
    cdef cnp.ndarray[cnp.int64_t, ndim=1] connected_labels

    cdef int i, window_start, window_end, window_size
    cdef double search_radius
    cdef long minlab

    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue

        search_radius = sort_vals_sp[i] / (1.0 - mergeScale * radius)
        window_end = np.searchsorted(sort_vals_sp, search_radius, side='right')
        window_start = i
        window_size = min(window_end, n_groups) - window_start

        if window_size > 0:
            ips_slice = ips[:window_size]

            # Compute inner products only for the sorted search window
            spsubmatxvec(
                spdatas_data, spdatas_indptr, spdatas_indices,
                window_start, window_start + window_size, spdata[i], ips_slice
            )

            denom_slice = denom[:window_size]
            denom_slice[:] = sort_vals_sp[i] + sort_vals_sp[window_start:window_start + window_size] - ips_slice
            denom_slice[denom_slice == 0.0] = 1e-15

            tanimoto_slice = tanimoto_dist[:window_size]
            tanimoto_slice[:] = 1.0 - ips_slice / denom_slice

            rel_inds = np.nonzero(tanimoto_slice <= mergeScale * radius)[0]

            if rel_inds.size > 0:
                inds = window_start + rel_inds

                if not mergeTinyGroups:
                    inds = inds[group_sizes[inds] >= minPts]

                if inds.size > 0:
                    # Symmetric adjacency edges
                    Adj[i, inds] = 1
                    Adj[inds, i] = 1

                    # Merge to smallest label (np.unique returns sorted array)
                    connected_labels = np.unique(label_sp[inds])
                    if connected_labels.size > 1:
                        minlab = connected_labels[0]
                        for cl in connected_labels:
                            label_sp[label_sp == cl] = minlab

    cdef cnp.ndarray[cnp.int64_t, ndim=1] unique_labels = np.unique(label_sp)
    cdef dict cluster_sizes = {}
    cdef long lbl
    for lbl in unique_labels:
        cluster_sizes[lbl] = np.sum(group_sizes[label_sp == lbl])

    small_clusters = [lbl for lbl in unique_labels if cluster_sizes[lbl] < minPts]

    cdef cnp.ndarray[cnp.int64_t, ndim=1] label_sp_fixed
    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_ids
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sorted_indices
    cdef long cluster_id, gid, target_gid, target_cluster

    if small_clusters:
        label_sp_fixed = label_sp.copy()

        for cluster_id in small_clusters:
            group_ids = np.nonzero(label_sp_fixed == cluster_id)[0]
            for gid in group_ids:
                # Full inner products with all groups
                spsubmatxvec(
                    spdatas_data, spdatas_indptr, spdatas_indices,
                    0, n_groups, spdata[gid], ips
                )

                denom[:] = sort_vals_sp[gid] + sort_vals_sp - ips
                denom[denom == 0.0] = 1e-15
                tanimoto_dist[:] = 1.0 - ips / denom

                sorted_indices = np.argsort(tanimoto_dist)

                for target_gid in sorted_indices:
                    target_cluster = label_sp_fixed[target_gid]
                    if cluster_sizes[target_cluster] >= minPts:
                        label_sp[gid] = target_cluster
                        Adj[gid, target_gid] = 2
                        Adj[target_gid, gid] = 2
                        break

    cdef cnp.ndarray[cnp.int64_t, ndim=1] final_ul = np.unique(label_sp)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_cluster_labels = np.searchsorted(final_ul, label_sp)

    return {
        'group_cluster_labels': group_cluster_labels,
        'Adj': Adj
    }