# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs


def aggregate_manhattan(double[:, :] data, double radius):
    """
    Cython optimized Manhattan sphere-exclusion clustering.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t dim = data.shape[1]
   
    cdef double[:] sort_vals = np.sum(data, axis=1)
    
    cdef cnp.intp_t[:] ind = np.argsort(sort_vals, kind='stable')
   
    cdef double[:, :] data_sorted = np.asarray(data)[ind]
    cdef double[:] sort_vals_sorted = np.asarray(sort_vals)[ind]

    cdef cnp.intp_t[:] labels = np.full(n, -1, dtype=np.intp)
    
    cdef list splist = []
    cdef list group_sizes = []
   
    cdef Py_ssize_t i, j, k
    cdef cnp.intp_t lab = 0
    cdef Py_ssize_t nr_dist = 0         
    cdef double d
    cdef int current_group_size

    for i in range(n):
        if labels[i] >= 0:
            continue
           
        labels[i] = lab
        splist.append(i)
       
        current_group_size = 1
        
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
           
            d = 0.0
            for k in range(dim):
                d += fabs(data_sorted[i, k] - data_sorted[j, k])
                if d > radius:
                    break
           
            nr_dist += 1
           
            if d <= radius:
                labels[j] = lab
                current_group_size += 1
       
        group_sizes.append(current_group_size)
        lab += 1
   
    return {
        'labels': np.asarray(labels),
        'splist': np.array(splist, dtype=np.intp),
        'group_sizes': np.array(group_sizes, dtype=np.intp),
        'ind': np.asarray(ind),
        'sort_vals': np.asarray(sort_vals_sorted),
        'data_sorted': np.asarray(data_sorted),
        'nr_dist': nr_dist
    }
