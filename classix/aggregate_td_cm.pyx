# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
import scipy.sparse as sparse
from libcpp.vector cimport vector


cdef Py_ssize_t bisect_right(double[:] a, double x, Py_ssize_t lo, Py_ssize_t hi) nogil:
    cdef Py_ssize_t mid
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def aggregate_tanimoto(
    double[:, :] data,
    double radius
):
    """Tanimoto leader-follower clustering (sphere-exclusion).
    Fully cross-platform (Windows + Linux + macOS)."""
    if radius <= 0.0 or radius >= 1.0:
        raise ValueError("radius must be in (0, 1)")

    cdef Py_ssize_t n = data.shape[0]

    cdef double[:] sort_vals = np.sum(data, axis=1)

    cdef cnp.intp_t[:] ind = np.argsort(sort_vals, kind='stable')

    cdef double[:, :] data_sorted = np.asarray(data)[ind]
    cdef double[:] sort_vals_sorted = np.asarray(sort_vals)[ind]

    datas = sparse.csr_matrix(np.asarray(data_sorted))
    cdef double[:] csr_data = datas.data
    cdef int[:] csr_indices = datas.indices
    cdef int[:] csr_indptr = datas.indptr

    cdef cnp.intp_t[:] labels = np.full(n, -1, dtype=np.intp)
    cdef vector[cnp.intp_t] splist
    cdef vector[cnp.intp_t] group_sizes

    splist.reserve(n // 10)
    group_sizes.reserve(n // 10)

    cdef Py_ssize_t i, j, k, last_j
    cdef cnp.intp_t lab = 0
    cdef Py_ssize_t nr_dist = 0
    cdef double sv_i, search_upper, dot
    cdef double multiplier = radius / (1.0 + radius)

    for i in range(n):
        if labels[i] >= 0:
            continue

        labels[i] = lab
        splist.push_back(i)
        group_sizes.push_back(1)

        sv_i = sort_vals_sorted[i]
        search_upper = sv_i / radius

        last_j = bisect_right(sort_vals_sorted, search_upper, i + 1, n)

        if last_j > i + 1:
            nr_dist += (last_j - (i + 1))

            for j in range(i + 1, last_j):
                if labels[j] != -1:
                    continue

                dot = 0.0
                for k in range(csr_indptr[j], csr_indptr[j + 1]):
                    dot += data_sorted[i, csr_indices[k]] * csr_data[k]

                if dot >= multiplier * (sv_i + sort_vals_sorted[j]):
                    labels[j] = lab
                    group_sizes[lab] += 1

        lab += 1

    # Convert C++ vectors to numpy
    cdef Py_ssize_t n_groups = splist.size()
    cdef cnp.intp_t[:] splist_np = np.empty(n_groups, dtype=np.intp)
    cdef cnp.intp_t[:] group_sizes_np = np.empty(n_groups, dtype=np.intp)

    for i in range(n_groups):
        splist_np[i] = splist[i]
        group_sizes_np[i] = group_sizes[i]

    return {
        'labels': np.asarray(labels),
        'splist': np.asarray(splist_np),
        'group_sizes': np.asarray(group_sizes_np),
        'ind': np.asarray(ind),
        'sort_vals': np.asarray(sort_vals_sorted),
        'data_sorted': np.asarray(data_sorted),
        'nr_dist': nr_dist
    }
