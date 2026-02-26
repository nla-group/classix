import numpy as np
from copy import deepcopy
from collections import deque
import scipy.sparse as sparse
from spmv import spsubmatxvec

class CLASSIX_T:
    def __init__(self, sorting="popcount", radius=0.3, minPts=1, group_merging="tanimoto_distance", norm=True, 
            mergeScale=1.4, post_alloc=True, mergeTinyGroups=True, verbose=1, short_log_form=True, use_spmv=True):
        self.group_merging = group_merging
        self.mergeScale = mergeScale
        self.mergeTinyGroups = mergeTinyGroups
        self.sorting = sorting
        self.radius = radius
        self.minPts = minPts
        self.norm = norm

        self.post_alloc = post_alloc
        self.mergeTinyGroups = mergeTinyGroups
        self.verbose = verbose
        self.truncate = short_log_form
        self.labels = None

    def __str__(self):
        return f"CLASSIX_T(sorting={self.sorting}, radius={self.radius}, minPts={self.minPts}, group_merging={self.group_merging}, mergeScale={self.mergeScale}, mergeTinyGroups={self.mergeTinyGroups})"
    
    def __repr__(self):
        return f"CLASSIX_T(sorting={self.sorting}, radius={self.radius}, minPts={self.minPts}, group_merging={self.group_merging}, mergeScale={self.mergeScale}, mergeTinyGroups={self.mergeTinyGroups})"
    

    def fit(self, data, r=None, mergeScale=None, minPts=None):
        if r is not None:
            self.radius = r
        
        if mergeScale is not None:
            self.mergeScale = mergeScale

        if minPts is not None:
            self.minPts = minPts

        n, fdim = data.shape
        sort_vals = np.sum(data, axis=1)

        self.ind = np.argsort(sort_vals, kind='stable')
        self.unsort_ind = np.argsort(self.ind, kind='stable')
        sort_vals = sort_vals[self.ind] 
        data = data[self.ind,:] 

        # print("\nOWN AGGREGATION")
        lab = 0

        self.labels = np.full(n, -1, dtype=int)
        nr_dist = 0
        self.splist = []
        rhs = 1 / (1 - self.radius) + 1
        rhsi = 1 / rhs

        self.group_sizes = []

        datas = sparse.csr_matrix(data)
        datas_data = datas.data.astype(np.float64)
        datas_indices = datas.indices.astype(np.int32)
        datas_indptr = datas.indptr.astype(np.int32)
        self.aggregation_dict = {}
        
        
        # Aggregation
        for i in range(n):
            if self.labels[i] >= 0:
                continue
                
            clustc = data[i,:].astype(np.float64)
            self.labels[i] = lab

            self.splist.append(i)
            self.group_sizes.append(1)
            
            search_radius = sort_vals[i] / (1 - self.radius)
            
            last_j = np.searchsorted(sort_vals, search_radius, side='right')
            self.aggregation_dict[i] = last_j

            window_start = i + 1
            window_end = min(last_j, n)  # 防止 last_j > n (理論上不會，但保險)
            window_size = max(0, window_end - window_start)

            if window_size > 0:
                ips = np.zeros(window_size, dtype=np.float64)
                
                spsubmatxvec(
                    datas_data,
                    datas_indptr,
                    datas_indices,
                    window_start,
                    window_end,
                    clustc,
                    ips
                )
                
                nr_dist += window_size
                
                denom = sort_vals[window_start:window_end] + sort_vals[i]
                vec = ips / denom
                vec_mask = vec >= rhsi
            else:
                vec_mask = np.empty(0, dtype=bool)
            
            # Robust padding to exactly length n
            left_pad = window_start
            right_pad = n - (window_start + window_size)
            vec_mask_padded = np.pad(vec_mask, (left_pad, right_pad), 'constant', constant_values=False)
            
            # Safety check (can remove in production)
            if len(vec_mask_padded) != n:
                raise ValueError(f"Padding error: got {len(vec_mask_padded)}, expected {n}")

            reassignMask = np.logical_and(self.labels < 0, vec_mask_padded)
            
            if np.any(reassignMask):
                self.labels[reassignMask] = lab
                self.group_sizes[-1] += np.sum(reassignMask)

            lab += 1
            
        self.groups_ = deepcopy(self.labels)
        
        # merging
        self.spdata = data[self.splist,:]
        spdatas = sparse.csr_matrix(self.spdata)
        spdatas_data = spdatas.data.astype(np.float64)
        spdatas_indices = spdatas.indices.astype(np.int32)
        spdatas_indptr = spdatas.indptr.astype(np.int32)
        spdata_len = len(self.splist)
        self.group_centers = self.spdata

        sort_vals_sp = sort_vals[self.splist]

        # merging
        label_sp = self.labels[self.splist].copy()
        scale = self.mergeScale
        self.Adj = np.zeros((len(self.splist), len(self.splist)), dtype=np.int8)
        
        minPts = self.minPts

        for i in range(len(self.splist)):
            if not self.mergeTinyGroups and self.group_sizes[i] < minPts:
                continue
                
            xi = self.spdata[i, :].astype(np.float64)
            search_radius = sort_vals_sp[i] / (1 - self.mergeScale * self.radius)

            last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')
            
            window_start = i
            window_end = min(last_j, spdata_len)
            window_size = max(0, window_end - window_start)
            
            if window_size > 0:
                ips = np.zeros(window_size, dtype=np.float64)
                
                spsubmatxvec(
                    spdatas_data,
                    spdatas_indptr,
                    spdatas_indices,
                    window_start,
                    window_end,
                    xi,
                    ips
                )
                
                denom = sort_vals_sp[i] + sort_vals_sp[window_start:window_end] - ips
                tanimoto_distance = 1 - ips / denom
            else:
                tanimoto_distance = np.empty(0, dtype=np.float64)
            
            inds_rel = np.where(tanimoto_distance <= scale * self.radius)[0]
            inds = i + inds_rel

            if not self.mergeTinyGroups:
                valid = np.array(self.group_sizes)[inds] >= minPts
                inds = inds[valid]

            self.Adj[i, inds] = 1
            self.Adj[inds, i] = 1

            connected_labels = np.unique(label_sp[inds])
            if len(connected_labels) > 1:
                minlab = np.min(connected_labels)
                for lbl in connected_labels:
                    label_sp[label_sp == lbl] = minlab

        self.initial_merging_labels = deepcopy(label_sp)

        # Cluster redistribution
        ul = np.unique(label_sp)
        cs = np.zeros(len(ul))
        group_sizes_arr = np.array(self.group_sizes)
        for new_id, old_lbl in enumerate(ul):
            mask = (label_sp == old_lbl)
            cs[new_id] = np.sum(group_sizes_arr[mask])
            label_sp[mask] = new_id

        small_clusters = np.where(cs < self.minPts)[0]

        labels_sp_copy_2 = deepcopy(label_sp)

        for cluster_id in small_clusters:
            
            group_ids = np.where(labels_sp_copy_2 == cluster_id)[0]
            for gid in group_ids:
                xi = self.spdata[gid, :].astype(np.float64)
                ips = np.zeros(spdata_len, dtype=np.float64)
                
                spsubmatxvec(
                    spdatas_data,
                    spdatas_indptr,
                    spdatas_indices,
                    0,
                    spdata_len,
                    xi,
                    ips
                )
                
                denom = sort_vals_sp[gid] + sort_vals_sp - ips
                d = 1 - ips / denom
                
                o = np.argsort(d, kind='stable')
                for nearest_gid in o:
                    
                    target_cluster = labels_sp_copy_2[nearest_gid]
                    if cs[target_cluster] >= minPts:
                        label_sp[gid] = target_cluster
                        self.Adj[gid, nearest_gid] = 2
                        self.Adj[nearest_gid, gid] = 2
                        break

        # 最終重新編號
        ul = np.unique(label_sp)
        final_map = {old: new for new, old in enumerate(ul)}
        label_sp = np.vectorize(final_map.get)(label_sp)

        # 映射回所有點
        self.labels = np.array([label_sp[group] if group != -1 else -1 for group in self.labels])

        self.labels = self.labels[self.unsort_ind]
        self.groups_ = self.groups_[self.unsort_ind]
        self.group_labels = label_sp
        self.group_centre_pts = self.spdata
        self.group_centers = self.splist
        return self

    def explain(self, ind1=None, ind2=None):
        if ind1 is None and ind2 is None:
            # If there are no specific points to explain, print the general information
            print(f"The data was clustered into {len(self.group_labels)} groups. These were further merged into {len(np.unique(self.group_labels))} clusters.")

        if ind1 is not None and ind2 is None:
            # If only one index is provided, print the cluster information for that index
            print(f"The data point at index {ind1} was assigned to cluster {self.labels[ind1]}")

        if ind1 is not None and ind2 is not None:
            # If two indices are provided, print the cluster information
            if self.labels[ind1] == self.labels[ind2]:
                # If the two data points belong to the same cluster, check if they are in the same group
                print(f"The data points at indices {ind1} and {ind2} belong to the same cluster.")
                agg_label_1 = self.aggregation_labels[ind1]
                agg_label_2 = self.aggregation_labels[ind2]
                if agg_label_1 == agg_label_2:
                    # If the two data points are in the same group, print the information
                    print(f"The data points at indices {ind1} and {ind2} are in the same group {agg_label_1}.")
                    return None

                else:
                    # If the two data points are in different groups, find the shortest connection path between the two groups
                    print(f"The data points at indices {ind1} and {ind2} are in different groups.")
                    connected_path = self.bfs_shortest_path(self.Adj, agg_label_1, agg_label_2)
                    if connected_path is not None:
                        # If a connection path is found, print the path
                        print(f'The connections between {ind1} and {ind2} are via this path: {connected_path[0]} ', end="")
                        for k in range(len(connected_path)-1):
                            if self.Adj[connected_path[k]-1, connected_path[k+1]-1] == 2:
                                print(f'(minPts) -> {connected_path[k+1]}', end="")
                            else:
                                print(f' -> {connected_path[k+1]}', end="")

                        return connected_path
                    
                    else:
                        # If no connection path is found, print that there is no connection, there must be something wrong with the code
                        print(f"No connection path found between {ind1} and {ind2}, although they belong to different groups in the same cluster. Please check the program for bugs!")
                        return None

            else:
                print(f"The data points at indices {ind1} and {ind2} belong to different clusters.")


    def bfs_shortest_path(self, adj_matrix, start, goal):
        # Convert start and goal from 1-based to 0-based indices
        start -= 1
        goal -= 1
        
        # Initialize the queue with the start node and a path containing only the start node
        queue = deque([(start, [start])])
        
        # Keep track of visited nodes to avoid cycles
        visited = set()
        
        while queue:
            # Dequeue a node and the path that led to it
            current_node, path = queue.popleft()
            
            # If the current node is the goal, return the path
            if current_node == goal:
                return [node + 1 for node in path]  # Convert back to 1-based indices
            
            # Mark the current node as visited
            visited.add(current_node)
            
            # Enqueue all unvisited neighbors
            for neighbor, connected in enumerate(adj_matrix[current_node]):
                if connected and neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None


    def bfs_shortest_path(self, adj_matrix, start, goal):
        start -= 1
        goal -= 1
        
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            current_node, path = queue.popleft()
            if current_node == goal:
                return [node + 1 for node in path]
            visited.add(current_node)
            for neighbor, connected in enumerate(adj_matrix[current_node]):
                if connected and neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None
