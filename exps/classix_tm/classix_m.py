import numpy as np
from copy import deepcopy
from collections import deque

class CLASSIX_M:
    def __init__(self, sorting="popcount", radius=0.3, minPts=1, group_merging="manhattan_distance",
                 norm=True, mergeScale=1.4, post_alloc=True, mergeTinyGroups=True, verbose=1, short_log_form=True, use_spmv=False):
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
        return f"CLASSIX_M(sorting={self.sorting}, radius={self.radius}, minPts={self.minPts}, group_merging={self.group_merging}, mergeScale={self.mergeScale}, mergeTinyGroups={self.mergeTinyGroups})"
    
    def __repr__(self):
        return f"CLASSIX_M(sorting={self.sorting}, radius={self.radius}, minPts={self.minPts}, group_merging={self.group_merging}, mergeScale={self.mergeScale}, mergeTinyGroups={self.mergeTinyGroups})"
    
    def fit(self, data, r=None, mergeScale=None, minPts=None):
        if r is not None:
            self.radius = r
        
        if mergeScale is not None:
            self.mergeScale = mergeScale

        if minPts is not None:
            self.minPts = minPts

        n, fdim = data.shape
        # Preprocessing and Normalization
        M = np.row_stack([np.min(data, axis=0)]*n)
        data = data - M
        sort_vals = np.sum(data, axis=1)
        mext = np.median(sort_vals)
        data/=mext
        sort_vals/=mext

        self.ind = np.argsort(sort_vals)
        self.unsort_ind = np.argsort(self.ind)
        sort_vals = sort_vals[self.ind] 
        data = data[self.ind,:] # sort data

        # print("\nOWN AGGREGATION")
        lab = 0

        # labels of the data points. Initially all are -1 to keep track of unallocated points
        self.labels = np.array([-1]*n)
        nr_dist = 0
        # Indices of group starting pts, equivalent of the gc list in the original code
        self.splist = []
        r = r
        self.group_sizes = [] # group sizes
        self.group_starting_pts = [] # group starting pts in the sorted array

        # Aggregation
        for i in range(n):
            
            if self.labels[i] >= 0:
                continue
                
            clustc = data[i,:]
            self.labels[i] = lab

            self.splist.append(i)
            self.group_sizes.append(1)
            search_radius = self.radius + sort_vals[i]
            
            last_j = np.searchsorted(sort_vals, search_radius, side='right')

            
            nr_dist += last_j - i - 1

            distance = np.linalg.norm(clustc - data[i+1:last_j,:], ord=1, axis=1)

            notAssigned = self.labels < 0
            vec_mask = distance <= self.radius
            vec_mask = np.pad(vec_mask, (i+1, n - last_j), 'constant', constant_values=False)
            reassignMask = np.logical_and(notAssigned, vec_mask)
            self.group_sizes[-1] += np.sum(reassignMask)
            self.labels[reassignMask] = lab

            lab += 1

        self.groups_ = deepcopy(np.array(self.labels))
        
        # merging
        self.spdata = data[self.splist,:]
        self.group_centers = self.spdata

        # Scores of the group starting points
        sort_vals_sp = sort_vals[self.splist]

        # Group labels of the group starting points
        label_sp = self.labels[self.splist]
        scale = self.mergeScale

        import scipy.sparse as sparse
        

        self.Adj = np.zeros((len(self.splist), len(self.splist)), dtype=np.int8)
        
        minPts = self.minPts

        for i in range(len(self.splist)):
            if not self.mergeTinyGroups:
                if self.group_sizes[i] < minPts:
                    continue
                
            xi = self.spdata[i, :]
            search_radius = self.mergeScale*self.radius + sort_vals_sp[i]

            last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')
            first_j = i

            distance = np.linalg.norm(self.spdata[first_j:last_j,:] - xi, ord=1, axis=1)
            inds = np.where(distance <= scale*self.radius)

            if not self.mergeTinyGroups:
                inds = inds[0][self.group_sizes[inds[0]] >= minPts]
                inds = i + inds
            
            else:
                inds = i + inds[0]

            self.Adj[i, inds] = 1
            self.Adj[inds, i] = 1

            spl = np.unique(label_sp[ inds ])

            minlab = np.min(spl)

            for label in spl:
                label_sp[label_sp==label] = minlab

        # Cluster redistribution
        print(" minPts Merging")
        ul = np.unique(label_sp)
        cs = np.zeros(len(ul))
        self.group_sizes = np.array(self.group_sizes)
        for i in range(len(ul)):
            id = np.where(label_sp==ul[i])
            label_sp[id] = i
            cs[i] = np.sum(self.group_sizes[id])

        small_clusters = np.where(cs<minPts)[0]

        print("small clusters", small_clusters)
        labels_sp_copy_2 = deepcopy(label_sp)

        for i in small_clusters:
            ii = np.where(labels_sp_copy_2==i)[0]
            for iii in ii:
                xi = self.spdata[iii, :]
                # d = Tanimoto distances to all other group centers
                d = np.linalg.norm(self.spdata - xi, ord=1, axis=1)
                
                o = np.argsort(d)
                # merge with the closest group that has more than minPts
                for j in o:
                    try:
                        if cs[labels_sp_copy_2[j]]>=minPts:
                            label_sp[iii] = labels_sp_copy_2[j]
                            # Instead of having a separate Adjanceny matrix for the points merged with minPts criteria
                            # we can just update the old Adjacency matrix here. The issue is that the distance matrix will
                            # have some weird discrepancies.
                            self.Adj[iii, j] = 2
                            self.Adj[j, iii] = 2
                            break
                    except:
                            print("minPts: ", minPts)
                            print("CS: ", cs)
                            print("label_sp_copy_2", labels_sp_copy_2)
                            print("label_sp_copy_2[j]; ", labels_sp_copy_2[j])
                            print("CS[label_sp_copy_2[j]]: ", cs[labels_sp_copy_2[j]])
                            print("#######################")

        ul = np.unique(label_sp)
        cs = np.zeros(len(ul))
        for i in range(len(ul)):
            id = np.where(label_sp==ul[i])
            label_sp[id] = i
            cs[i] = np.sum(self.group_sizes[id])

        print("final cluster sizes", cs)
        
        for idx, label in enumerate(self.labels):
            self.labels[idx] = label_sp[label]

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
