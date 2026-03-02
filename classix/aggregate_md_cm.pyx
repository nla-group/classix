# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport fabs

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

def aggregate_manhattan(double[:, :] data, double radius):
    """
    Cython optimized version of aggregate_manhattan.
    Inputs and Outputs match the original function exactly.
    """
    cdef int n = data.shape[0]
    cdef int dim = data.shape[1]
    
    cdef double[:] sort_vals = np.sum(data, axis=1)
    cdef long[:] ind = np.argsort(sort_vals)
    
    cdef double[:, :] data_sorted = np.empty((n, dim), dtype=np.float64)
    cdef double[:] sort_vals_sorted = np.empty(n, dtype=np.float64)
    
    cdef int i, j, k, r
    for i in range(n):
        r = ind[i]
        sort_vals_sorted[i] = sort_vals[r]
        for k in range(dim):
            data_sorted[i, k] = data[r, k]

    cdef long[:] labels = np.full(n, -1, dtype=int)
    splist = []      # Python list is fine for accumulating indices
    group_sizes = [] # Python list
    
    cdef int lab = 0
    cdef long nr_dist = 0
    cdef double limit
    cdef double d, diff
    cdef int current_group_size
    
    for i in range(n):
        if labels[i] >= 0:
            continue
            
        labels[i] = lab
        splist.append(i)
        
        current_group_size = 1
        limit = sort_vals_sorted[i] + radius
        
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
        'splist': np.array(splist),
        'group_sizes': np.array(group_sizes),
        'ind': np.asarray(ind),
        'sort_vals': np.asarray(sort_vals_sorted),
        'data_sorted': np.asarray(data_sorted),
        'nr_dist': nr_dist
    }
