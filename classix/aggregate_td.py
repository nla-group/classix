import numpy as np
import scipy.sparse as sparse
from spmv import spsubmatxvec  

def aggregate_tanimoto(data, radius):
    n, _ = data.shape
    
    sort_vals = np.sum(data, axis=1)
    
    ind = np.argsort(sort_vals, kind='stable')
    data_sorted = data[ind]
    sort_vals_sorted = sort_vals[ind]
    
    datas = sparse.csr_matrix(data_sorted)
    
    labels = np.full(n, -1, dtype=int)
    splist = []
    group_sizes = []
    lab = 0
    nr_dist = 0
    
    rhs = 1 / (1 - radius) + 1
    rhsi = 1 / rhs
    
    for i in range(n):
        if labels[i] >= 0:
            continue
        
        clustc = data_sorted[i]
        labels[i] = lab
        splist.append(i)
        group_sizes.append(1)
        
        search_radius = sort_vals_sorted[i] / (1 - radius)
        last_j = np.searchsorted(sort_vals_sorted, search_radius, side='right')
        
        if last_j > i + 1:
            n_rows = last_j - (i + 1)
            ips = np.zeros(n_rows, dtype=np.float64)
            
            spsubmatxvec(
                datas.data.astype(np.float64),
                datas.indptr.astype(np.int32),
                datas.indices.astype(np.int32),
                i + 1,
                last_j,
                clustc.astype(np.float64),
                ips
            )
            
            nr_dist += n_rows
            
            denom = sort_vals_sorted[i+1:last_j] + sort_vals_sorted[i]
            vec = ips / denom
            vec_mask = vec >= rhsi
            
            notAssigned = labels[i+1:last_j] < 0
            reassignMask = np.logical_and(notAssigned, vec_mask)
            
            if np.any(reassignMask):
                labels[i+1:last_j][reassignMask] = lab
                group_sizes[-1] += np.sum(reassignMask)
        
        lab += 1
    
    return {
        'labels': labels,                     
        'splist': np.array(splist),          
        'group_sizes': np.array(group_sizes),
        'ind': ind,                           
        'sort_vals': sort_vals_sorted,       
        'data_sorted': data_sorted,           
        'nr_dist': nr_dist
    }
