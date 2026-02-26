import numpy as np

def aggregate_manhattan(data, radius, verbose=False):
    n, _ = data.shape
    sort_vals = np.sum(data, axis=1)
    
    ind = np.argsort(sort_vals)
    data_sorted = data[ind]
    sort_vals_sorted = sort_vals[ind]
    
    labels = np.full(n, -1, dtype=int)
    splist = []
    group_sizes = []
    lab = 0
    nr_dist = 0
    
    for i in range(n):
        if labels[i] >= 0:
            continue
        
        clustc = data_sorted[i]
        labels[i] = lab
        splist.append(i)
        group_sizes.append(1)
        
        search_radius = radius + sort_vals_sorted[i]
        last_j = np.searchsorted(sort_vals_sorted, search_radius, side='right')
        
        if last_j > i + 1:
            dists = np.sum(np.abs(clustc - data_sorted[i+1:last_j]), axis=1)
            nr_dist += len(dists)
            
            mask = (dists <= radius) & (labels[i+1:last_j] < 0)
            if np.any(mask):
                labels[i+1:last_j][mask] = lab
                group_sizes[-1] += np.sum(mask)
        
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
