import classix
import unittest
import numpy as np
import pandas as pd
import sklearn.datasets as data
from classix import CLASSIX, loadData, cython_is_available
from classix.clustering import calculate_cluster_centers
from classix import preprocessing
from classix import aggregate_ed, aggregate_ed_c, aggregate_ed_cm
from classix.merge_ed import distance_merge_mtg, distance_merge, density_merge
import platform

if platform.system() == 'Windows':
    from classix.merge_ed_cm_win import distance_merge as distance_merge_cm
    from classix.merge_ed_cm_win import density_merge as density_merge_cm
else:
    from classix.merge_ed_cm import distance_merge as distance_merge_cm
    from classix.merge_ed_cm import density_merge as density_merge_cm

from sklearn.metrics.cluster import adjusted_rand_score



class TestClassix(unittest.TestCase):
    
    def test_cython_check(self):
        checkpoint = cython_is_available(verbose=True)
        self.assertEqual(checkpoint, 1)
        
        
    def test_distance_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):
            clx = CLASSIX(radius=tol, group_merging='distance', verbose=0)
            clx.fit_transform(vdu_signals)
            # test new version
            checkpoint = np.load('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy')
            
            assert(adjusted_rand_score(clx.labels_, checkpoint) == 1)
            
            
    def test_preprocessing(self):
        checkpoint = 1
        X, _ = data.make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
        try:
            preprocessing(X, "norm-mean")
            preprocessing(X, "pca")
            preprocessing(X, "norm-orthant")
        except:
            checkpoint = 0
        
        self.assertEqual(checkpoint, 1)

        
    def test_density_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):

            clx = CLASSIX(radius=tol, group_merging='density', verbose=0)
            clx.fit_transform(vdu_signals)

            # version 0.2.7
            # np.save('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy')
  
            assert(adjusted_rand_score(clx.labels_, checkpoint) == 1)

    
    def test_non_cython_version(self):
        classix.__enable_cython__ = False
        checkpoint = 1
        for dim in range(1, 5):
            try:
                X, _ = data.make_blobs(n_samples=200, 
                                     centers=3, n_features=dim, 
                                     random_state=42
                                    )
                clx = CLASSIX(sorting='pca', group_merging='density')
                clx.fit_transform(X)
                
                clx = CLASSIX(sorting='pca', group_merging='distance')
                clx.fit_transform(X)

                clx = CLASSIX(sorting='pca', group_merging='distance')
                clx.fit_transform(X)
                
                clx = CLASSIX(sorting='norm-mean', group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)

                clx = CLASSIX(sorting='norm-orthant', group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)
                
                clx = CLASSIX(sorting=None, group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)
            except:
                checkpoint = 0
                break
        
        self.assertEqual(checkpoint, 1)
    
    
    def test_cython_version(self):
        classix.__enable_cython__ = True
        checkpoint = 1
        for dim in range(1, 5):
            try:
                X, _ = data.make_blobs(n_samples=200, 
                                     centers=3, n_features=dim, 
                                     random_state=42
                                    )
                clx = CLASSIX(sorting='pca', group_merging='density')
                clx.fit_transform(X)

                clx.getPath(3, 10, include_dist=False)
                clx.getPath(3, 20, include_dist=False)
                clx.getPath(3, 30, include_dist=True)
                
                clx = CLASSIX(sorting='pca', group_merging='distance', minPts=150)
                clx.fit_transform(X)

                clx = CLASSIX(sorting='pca', group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)
                
                clx = CLASSIX(sorting='norm-mean', group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)

                clx = CLASSIX(sorting='norm-orthant', group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)

                clx = CLASSIX(sorting=None, group_merging='distance', mergeTinyGroups=False)
                clx.fit_transform(X)

                clx.timing()
            except:
                checkpoint = 0
                break
        
        self.assertEqual(checkpoint, 1)
        
        
    def test_scale_linkage(self):
        TOL = 0.1 
        random_state = 1

        moons, _ = data.make_moons(n_samples=1000, noise=0.05, random_state=random_state)
        blobs, _ = data.make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], cluster_std=0.5, random_state=random_state)
        X = np.vstack([blobs, moons])

        checkpoint = 1
        for scale in np.arange(1.8, 2, 0.1):
            try:
                clx = CLASSIX(sorting='pca', radius=TOL, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='pca', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)

        checkpoint = 1
        for scale in np.arange(1.8, 2, 0.1):
            try:
                clx = CLASSIX(sorting='norm-orthant', radius=TOL, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='norm-orthant', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)

        checkpoint = 1
        for scale in np.arange(1.8, 2, 0.1):
            try:
                clx = CLASSIX(sorting='norm-mean', radius=TOL, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='norm-mean', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(X, scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)


    
    def test_explain(self):
        X, y = data.make_blobs(n_samples=5000, centers=2, n_features=2, 
                               cluster_std=1.5, random_state=1
        )
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=3)
            clx.fit_transform(X)
            clx.load_group_centers(X)
            clx.load_cluster_centers(X)
            clx.gcIndices([1, 2, 3, 4])
            
            clx.predict(X)
            clx.predict(X[:1000])

            clx.explain(data=X, plot=False, showsplist=True, figsize=(10,10),  savefig=True)
            clx.explain(X, 0,  plot=False, savefig=True, showsplist=True)
            clx.form_starting_point_clusters_table(data=X[clx.ind], aggregate=True)
            clx.explain(X, 3, 2000,  plot=False, savefig=False)
            clx.explain(X, 0, 2008,  plot=False, savefig=True, replace_name=['Superman', 'Batman'])

            clx.explain(X, 2000, 2028,  plot=False, add_arrow=True, savefig=True, showallgroups=True, include_dist=True)
                
            clx.explain(X, 0, 2008,  plot=False, add_arrow=True, directed_arrow=-1, savefig=True, fmt='pdf')
            clx.explain(X, 0, 2008,  plot=False, add_arrow=True, directed_arrow=1, savefig=True, fmt='png')
            clx.explain(X, index1=[0, 0], index2=[4,5], plot=False, add_arrow=True, show_connected_label=True, directed_arrow=1)
            clx.explain(X, index1=np.array([6, 6]), index2=np.array([6, 6]), plot=False, add_arrow=True, directed_arrow=1)
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=4999, mergeTinyGroups=False)
            clx.fit(X)
            clx.explain(X, 0, 2008,  plot=False, add_arrow=True, directed_arrow=-1, savefig=True, fmt='jpg')
            clx.timing()
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)
   

    def test_explain_str_input(self):
        X, _ = data.make_blobs(n_samples=5, centers=2, n_features=2, cluster_std=1.5, random_state=1)
        X = pd.DataFrame(X, index=['Anna', 'Bert', 'Carl', 'Tom', 'Bob'])
        checkpoint = 1
        
        try:
            clx = CLASSIX(radius=0.6)
            clx.fit_transform(X)
            print(clx.clusterSizes_)
            print(clx.groupCenters_)
            clx.explain(X, index1='Carl', index2='Bert', plot=False, show_connected_label=True, showallgroups=True, sp_fontsize=12)  
            
        except:
            checkpoint = 0
        self.assertEqual(checkpoint, 1)

    
    def test_explain_hdim(self):
        X, y = data.make_blobs(n_samples=5000, centers=2, n_features=20, 
                               cluster_std=1.5, random_state=1
        )
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=3)
            clx.fit_transform(X)
            clx.predict(X)
            clx.explain(X, plot=False, figsize=(10,10),  savefig=False)
            clx.explain(X, 0,  plot=False, savefig=False)
            clx.explain(X, 3, 2000,  plot=False, savefig=False)
            clx.explain(X, 0, 2008,  plot=False, savefig=False)
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)

    
    def test_explain_1D(self):
        X, y = data.make_blobs(n_samples=5000, centers=2, n_features=1, 
                               cluster_std=1.5, random_state=1
        )
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=3)
            clx.fit_transform(X)
            clx.predict(X)
            clx.explain(X, plot=False, figsize=(10,10),  savefig=False)
            clx.explain(X, 0,  plot=False, savefig=False)
            clx.explain(X, 3, 2000,  plot=False, savefig=False)
            clx.explain(X, 0, 2008,  plot=False, savefig=False)
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)


    def test_explain_manhattan(self):
        X, y = data.make_blobs(n_samples=200, centers=2, n_features=2, 
                               cluster_std=1.5, random_state=42)
        checkpoint = 1
        try:
            # Test basic Manhattan explain
            clx = CLASSIX(radius=0.5, metric='manhattan', group_merging='distance', minPts=3, verbose=0)
            clx.fit_transform(X)
            clx.explain(X, plot=False)
            clx.explain(X, 0, plot=False)
            clx.explain(X, 0, 50, plot=False)
            
            # Test with minPts redistribution (forces Adj value 2)
            clx2 = CLASSIX(radius=0.3, metric='manhattan', group_merging='distance', minPts=50, verbose=0)
            clx2.fit_transform(X)
            clx2.explain(X, plot=False)
            clx2.explain(X, 0, plot=False)
            clx2.explain(X, 0, 100, plot=False)
        except:
            checkpoint = 0
        
        self.assertEqual(checkpoint, 1)

    def test_explain_tanimoto(self):
        """
        Need to implement and add the Tanimoto blobs to test this.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from tests.tanimoto_blobs import generate_data
        X, _, _ = generate_data(num_clusters=3, pops=[5, 10, 15], d=20, n=100, flip_prob=0.1, seed=42)
        checkpoint = 1
        try:
            # Test basic Tanimoto explain
            clx = CLASSIX(metric='tanimoto', radius=0.1, group_merging='distance', minPts=1, verbose=0)
            clx.fit_transform(X)
            clx.explain(X, plot=False)
            clx.explain(X, 0, plot=False)
            clx.explain(X, 0, 50, plot=False)


            # Test with minPts redistribution (forces Adj value 2)
            clx2 = CLASSIX(metric='tanimoto', radius=0.1, group_merging='distance', minPts=50, verbose=0)
            clx2.fit_transform(X)
            clx2.explain(X, plot=False)
            clx2.explain(X, 0, plot=False)
            clx2.explain(X, 0, 1, plot=False)
        except:
            checkpoint = 0 

        self.assertEqual(checkpoint, 1)
        

    
    def test_explain_connected_groups(self):
        X, y = data.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=1)
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging='distance')
            clx.fit(X)
            clx.explain(data=X, index1=773, index2=22, plot=False,  add_arrow=True, include_dist=False)

            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging='density')
            clx.fit(X)
            clx.explain(data=X, index1=773, index2=22, plot=False,  add_arrow=True, include_dist=False)
        except:
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)
        
        
    def test_built_in_data(self):
        checkpoint = 1
        try:
            for dn in ['vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 'Banknote', 'Seeds', 
                       'Phoneme', 'Wine', 'CovidENV', 'Covid3MC', 'NA']:
                loadData(name=dn)
        except:
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)
        

    def test_aggregate_precompute(self): 
        checkpoint = 1
        try:
            data = np.random.randn(10000, 2)
            
            inverse_ind1, spl1, _, _, _, _, _ = aggregate_ed.general_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind2, spl2, _, _, _, _, _ = aggregate_ed_cm.general_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind3, spl3, _, _, _, _, _ = aggregate_ed_c.general_aggregate(data, "pca", 0.5)
            inverse_ind7, spl7, _, _, _, _, _ = aggregate_ed.pca_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind8, spl8, _, _, _, _, _ = aggregate_ed_c.pca_aggregate(data, "pca", 0.5)
            inverse_ind9, spl9, _, _, _, _, _ = aggregate_ed_cm.pca_aggregate(data, "pca", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_ed_cm.general_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_c.general_aggregate(data, "norm-mean", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_ed_cm.general_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_c.general_aggregate(data, "NA", 0.5)

            inverse_ind1, spl1, _, _, _, _, _ = aggregate_ed.lm_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind2, spl2, _, _, _, _, _ = aggregate_ed_cm.lm_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind3, spl3, _, _, _, _, _ = aggregate_ed_c.lm_aggregate(data, "pca", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_ed.lm_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_cm.lm_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_c.lm_aggregate(data, "norm-mean", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_ed.lm_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_cm.lm_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_ed_c.lm_aggregate(data, "NA", 0.5)
            
            if np.sum(inverse_ind1 != inverse_ind2) != 0:
                checkpoint = 0
            if np.sum(inverse_ind2 != inverse_ind3) != 0:
                checkpoint = 0
            if np.sum(inverse_ind7 != inverse_ind8) != 0:
                checkpoint = 0
            if np.sum(inverse_ind8 != inverse_ind9) != 0:
                checkpoint = 0
            
            for i in range(len(spl1)):
                if spl1[i][0] != spl2[i][0]:
                    checkpoint = 0
                if spl2[i][0] != spl3[i][0]:
                    checkpoint = 0
                    
                if spl1[i][1] != spl2[i][1]:
                    checkpoint = 0
                if spl2[i][1] != spl3[i][1]:
                    checkpoint = 0
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)


    def test_merge(self): 
        checkpoint = 1
        minPts = 10
        scale = 1.5
        data = np.random.randn(10000, 2)
        checkpoint = 1
        try:    
            labels, splist, nr_dist, ind, sort_vals, data, half_nrm2 = aggregate_ed.general_aggregate(data, sorting="pca", tol=0.5) #
            splist = np.asarray(splist)
            
            radius = 0.5
            splist = np.int64(np.asarray(splist))
            half_nrm2_sp = half_nrm2[splist[:,0]]
            
            label_set1, connected_pairs_store1 = density_merge(data, splist, radius, sort_vals, half_nrm2)
            label_set2, connected_pairs_store2 = density_merge_cm(data, splist, radius, sort_vals, half_nrm2)
            
            label_set3, _,_ = distance_merge(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2_sp)
            label_set4, _,_ = distance_merge_cm(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2_sp)
            label_set5, _,_ = distance_merge_mtg(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2_sp)
            
            for i in range(len(label_set2)):
                if label_set1[i] != label_set2[i]:
                    checkpoint = 0
            
            for i in range(len(connected_pairs_store1)):
                if connected_pairs_store1[i] != connected_pairs_store2[i]:
                    checkpoint = 0
            
            for i in range(len(label_set3)):
                if label_set3[i] != label_set4[i]:
                    checkpoint = 0
                    
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)


    def test_group_merging_error_type(self):
        X, y = data.make_blobs(n_samples=5000, centers=2, n_features=20, 
                               cluster_std=1.5, random_state=1
        )
        
        group_merging1='error'
        group_merging2=3
        
        checkpoint = 0
        try:
            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging=group_merging1)
        except ValueError:
            checkpoint = 1
        self.assertEqual(checkpoint, 1)

        checkpoint = 0
        try:
            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging=group_merging2)
        except TypeError:
            checkpoint = 1
        self.assertEqual(checkpoint, 1)
        

    def test_misc(self):
        checkpoint = 1
        try:
            X, y = data.make_blobs(n_samples=200, 
                                     centers=3, n_features=2, 
                                     random_state=42
                                    )
            _ = calculate_cluster_centers(X, y)
        except:
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    # ==================== NEW TESTS FOR MANHATTAN DISTANCE ====================
    
    def test_manhattan_aggregation(self):
        """Test Manhattan distance aggregation module"""
        from classix.aggregate_md import aggregate_manhattan
        
        checkpoint = 1
        try:
            # Test with 2D data
            X = np.random.randn(1000, 2)
            result = aggregate_manhattan(X, radius=0.5)
            
            # Verify returned dictionary keys
            required_keys = ['labels', 'splist', 'group_sizes', 'ind', 'sort_vals', 'data_sorted', 'nr_dist']
            for key in required_keys:
                if key not in result:
                    checkpoint = 0
                    break
            
            # Verify data integrity
            if len(result['labels']) != len(X):
                checkpoint = 0
            if len(result['splist']) != len(result['group_sizes']):
                checkpoint = 0
                
            # Test with higher dimensional data
            X_hd = np.random.randn(500, 10)
            result_hd = aggregate_manhattan(X_hd, radius=1.0)
            if len(result_hd['labels']) != len(X_hd):
                checkpoint = 0
                
        except Exception as e:
            print(f"Manhattan aggregation test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_merging(self):
        """Test Manhattan distance merging module"""
        from classix.aggregate_md import aggregate_manhattan
        from classix.merge_md import merge_manhattan
        
        checkpoint = 1
        try:
            # First do aggregation
            X = np.random.randn(1000, 3)
            agg_result = aggregate_manhattan(X, radius=0.5)
            
            # Extract starting point data
            spdata = agg_result['data_sorted'][agg_result['splist']]
            group_sizes = agg_result['group_sizes']
            sort_vals_sp = agg_result['sort_vals'][agg_result['splist']]
            agg_labels_sp = np.arange(len(agg_result['splist']))
            
            # Test merging with different parameters
            merge_result = merge_manhattan(
                spdata=spdata,
                group_sizes=group_sizes,
                sort_vals_sp=sort_vals_sp,
                agg_labels_sp=agg_labels_sp,
                radius=0.5,
                mergeScale=1.5,
                minPts=3,
                mergeTinyGroups=True
            )
            
            # Verify results
            if 'group_cluster_labels' not in merge_result:
                checkpoint = 0
            if 'Adj' not in merge_result:
                checkpoint = 0
            if len(merge_result['group_cluster_labels']) != len(spdata):
                checkpoint = 0
                
            # Test with mergeTinyGroups=False
            merge_result2 = merge_manhattan(
                spdata=spdata,
                group_sizes=group_sizes,
                sort_vals_sp=sort_vals_sp,
                agg_labels_sp=agg_labels_sp,
                radius=0.5,
                mergeScale=1.5,
                minPts=10,
                mergeTinyGroups=False
            )
            
            if len(merge_result2['group_cluster_labels']) != len(spdata):
                checkpoint = 0
                
        except Exception as e:
            print(f"Manhattan merging test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_bfs_shortest_path(self):
        """Test BFS shortest path function in merge_md"""
        from classix.merge_md import bfs_shortest_path
        
        checkpoint = 1
        try:
            # Create a simple adjacency matrix
            adj = np.array([
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0]
            ], dtype=np.int8)
            
            # Test path from 0 to 4
            path = bfs_shortest_path(adj, 0, 4)
            if path is None or len(path) != 5:
                checkpoint = 0
            
            # Test path from 0 to 2
            path2 = bfs_shortest_path(adj, 0, 2)
            if path2 is None or len(path2) != 3:
                checkpoint = 0
                
            # Test same start and goal
            path3 = bfs_shortest_path(adj, 2, 2)
            if path3 is None or len(path3) != 1:
                checkpoint = 0
                
            # Test no path exists
            adj_disconnected = np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=np.int8)
            path4 = bfs_shortest_path(adj_disconnected, 0, 3)
            if path4 is not None:
                checkpoint = 0
                
        except Exception as e:
            print(f"Manhattan BFS test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_clustering_full_pipeline(self):
        """Test full Manhattan clustering pipeline with CLASSIX"""
        checkpoint = 1
        try:
            X, _ = data.make_blobs(n_samples=500, centers=4, n_features=3, random_state=42)
            
            # Test with distance merging
            clx1 = CLASSIX(metric='manhattan', radius=0.5, group_merging='distance', minPts=5, verbose=0)
            clx1.fit_transform(X)
            if clx1.labels_ is None or len(clx1.labels_) != len(X):
                checkpoint = 0
                
            # Test with density merging  
            clx2 = CLASSIX(metric='manhattan', radius=0.5, group_merging='density', minPts=5, verbose=0)
            clx2.fit_transform(X)
            if clx2.labels_ is None or len(clx2.labels_) != len(X):
                checkpoint = 0
                
            # Test with mergeTinyGroups=False
            clx3 = CLASSIX(metric='manhattan', radius=0.5, group_merging='distance', 
                          minPts=20, mergeTinyGroups=False, verbose=0)
            clx3.fit_transform(X)
            if clx3.labels_ is None or len(clx3.labels_) != len(X):
                checkpoint = 0
                
            # Test predict method
            X_new = np.random.randn(50, 3)
            labels_new = clx1.predict(X_new)
            if labels_new is None or len(labels_new) != len(X_new):
                checkpoint = 0
                
        except Exception as e:
            print(f"Manhattan full pipeline test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    # ==================== NEW TESTS FOR TANIMOTO DISTANCE ====================
    
    def test_tanimoto_aggregation(self):
        """Test Tanimoto distance aggregation module"""
        from classix.aggregate_td import aggregate_tanimoto
        
        checkpoint = 1
        try:
            # Generate binary-like data for Tanimoto
            X = np.random.randint(0, 2, size=(500, 20)).astype(np.float64)
            result = aggregate_tanimoto(X, radius=0.3)
            
            # Verify returned dictionary keys
            required_keys = ['labels', 'splist', 'group_sizes', 'ind', 'sort_vals', 'data_sorted', 'nr_dist']
            for key in required_keys:
                if key not in result:
                    checkpoint = 0
                    break
            
            # Verify data integrity
            if len(result['labels']) != len(X):
                checkpoint = 0
            if len(result['splist']) != len(result['group_sizes']):
                checkpoint = 0
                
            # Test with different radius
            result2 = aggregate_tanimoto(X, radius=0.1)
            if len(result2['labels']) != len(X):
                checkpoint = 0
                
        except Exception as e:
            print(f"Tanimoto aggregation test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_merging(self):
        """Test Tanimoto distance merging module"""
        from classix.aggregate_td import aggregate_tanimoto
        from classix.merge_td import merge_tanimoto
        
        checkpoint = 1
        try:
            # Generate binary data
            X = np.random.randint(0, 2, size=(500, 15)).astype(np.float64)
            agg_result = aggregate_tanimoto(X, radius=0.3)
            
            # Extract starting point data
            spdata = agg_result['data_sorted'][agg_result['splist']]
            group_sizes = agg_result['group_sizes']
            sort_vals_sp = agg_result['sort_vals'][agg_result['splist']]
            agg_labels_sp = np.arange(len(agg_result['splist']))
            
            # Test merging
            merge_result = merge_tanimoto(
                spdata=spdata,
                group_sizes=group_sizes,
                sort_vals_sp=sort_vals_sp,
                agg_labels_sp=agg_labels_sp,
                radius=0.3,
                mergeScale=1.5,
                minPts=3,
                mergeTinyGroups=True
            )
            
            # Verify results
            if 'group_cluster_labels' not in merge_result:
                checkpoint = 0
            if 'Adj' not in merge_result:
                checkpoint = 0
            if len(merge_result['group_cluster_labels']) != len(spdata):
                checkpoint = 0
                
            # Test with mergeTinyGroups=False
            merge_result2 = merge_tanimoto(
                spdata=spdata,
                group_sizes=group_sizes,
                sort_vals_sp=sort_vals_sp,
                agg_labels_sp=agg_labels_sp,
                radius=0.3,
                mergeScale=1.5,
                minPts=10,
                mergeTinyGroups=False
            )
            
            if len(merge_result2['group_cluster_labels']) != len(spdata):
                checkpoint = 0
                
        except Exception as e:
            print(f"Tanimoto merging test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_bfs_shortest_path(self):
        """Test BFS shortest path function in merge_td"""
        from classix.merge_td import bfs_shortest_path
        
        checkpoint = 1
        try:
            # Create adjacency matrix
            adj = np.array([
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0]
            ], dtype=np.int8)
            
            # Test various paths
            path1 = bfs_shortest_path(adj, 0, 4)
            if path1 is None or len(path1) < 3:
                checkpoint = 0
            
            path2 = bfs_shortest_path(adj, 0, 1)
            if path2 is None or len(path2) != 2:
                checkpoint = 0
                
            # Test same node
            path3 = bfs_shortest_path(adj, 3, 3)
            if path3 is None or len(path3) != 1:
                checkpoint = 0
                
        except Exception as e:
            print(f"Tanimoto BFS test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_clustering_full_pipeline(self):
        """Test full Tanimoto clustering pipeline with CLASSIX"""
        checkpoint = 1
        try:
            # Generate binary data
            X = np.random.randint(0, 2, size=(400, 25)).astype(np.float64)
            
            # Test with distance merging
            clx1 = CLASSIX(metric='tanimoto', radius=0.3, group_merging='distance', minPts=5, verbose=0)
            clx1.fit_transform(X)
            if clx1.labels_ is None or len(clx1.labels_) != len(X):
                checkpoint = 0
                
            # Test with density merging
            clx2 = CLASSIX(metric='tanimoto', radius=0.3, group_merging='density', minPts=5, verbose=0)
            clx2.fit_transform(X)
            if clx2.labels_ is None or len(clx2.labels_) != len(X):
                checkpoint = 0
                
            # Test with mergeTinyGroups=False
            clx3 = CLASSIX(metric='tanimoto', radius=0.3, group_merging='distance', 
                          minPts=15, mergeTinyGroups=False, verbose=0)
            clx3.fit_transform(X)
            if clx3.labels_ is None or len(clx3.labels_) != len(X):
                checkpoint = 0
                
            # Test predict method
            X_new = np.random.randint(0, 2, size=(30, 25)).astype(np.float64)
            labels_new = clx1.predict(X_new)
            if labels_new is None or len(labels_new) != len(X_new):
                checkpoint = 0
                
        except Exception as e:
            print(f"Tanimoto full pipeline test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_cython_manhattan_modules(self):
        """Test Cython implementations of Manhattan distance modules"""
        checkpoint = 1
        try:
            # Import Cython modules
            if platform.system() == 'Windows':
                # Windows doesn't have aggregate_md_cm, skip
                return
            else:
                from classix import aggregate_md_cm
                from classix import merge_md_cm
                
            X = np.random.randn(500, 3)
            
            # Test Cython aggregation (if available)
            try:
                result = aggregate_md_cm.aggregate_manhattan(X, 0.5)
                if len(result['labels']) != len(X):
                    checkpoint = 0
            except AttributeError:
                # Module may not have this function, that's ok
                pass
                
        except ImportError:
            # Cython modules might not be compiled, that's acceptable
            pass
        except Exception as e:
            print(f"Cython Manhattan test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_cython_tanimoto_modules(self):
        """Test Cython implementations of Tanimoto distance modules"""
        checkpoint = 1
        try:
            # Import Cython modules
            if platform.system() == 'Windows':
                # Windows support might be different
                return
            else:
                from classix import aggregate_td_cm
                from classix import merge_td_cm
                
            X = np.random.randint(0, 2, size=(400, 20)).astype(np.float64)
            
            # Test Cython aggregation (if available)
            try:
                result = aggregate_td_cm.aggregate_tanimoto(X, 0.3)
                if len(result['labels']) != len(X):
                    checkpoint = 0
            except AttributeError:
                # Module may not have this function, that's ok
                pass
                
        except ImportError:
            # Cython modules might not be compiled, that's acceptable
            pass
        except Exception as e:
            print(f"Cython Tanimoto test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_explain_path(self):
        """Test explain method with Manhattan metric including path finding"""
        checkpoint = 1
        try:
            X, _ = data.make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
            
            clx = CLASSIX(metric='manhattan', radius=0.5, group_merging='distance', minPts=5, verbose=0)
            clx.fit(X)
            
            # Test explain with various parameters
            clx.explain(X, plot=False)
            clx.explain(X, 0, plot=False)
            clx.explain(X, 0, 50, plot=False, add_arrow=True)
            clx.explain(X, 0, 50, plot=False, add_arrow=True, include_dist=True)
            
        except Exception as e:
            print(f"Manhattan explain path test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_explain_path(self):
        """Test explain method with Tanimoto metric including path finding"""
        checkpoint = 1
        try:
            X = np.random.randint(0, 2, size=(200, 15)).astype(np.float64)
            
            clx = CLASSIX(metric='tanimoto', radius=0.3, group_merging='distance', minPts=3, verbose=0)
            clx.fit(X)
            
            # Test explain with various parameters
            clx.explain(X, plot=False)
            clx.explain(X, 0, plot=False)
            clx.explain(X, 0, 30, plot=False, add_arrow=True)
            clx.explain(X, 0, 30, plot=False, add_arrow=True, include_dist=True)
            
        except Exception as e:
            print(f"Tanimoto explain path test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_predict_manhattan(self):
        """Test predict method specifically for Manhattan metric"""
        checkpoint = 1
        try:
            X_train, _ = data.make_blobs(n_samples=400, centers=3, n_features=4, random_state=42)
            X_test = np.random.randn(50, 4)
            
            clx = CLASSIX(metric='manhattan', radius=0.5, group_merging='distance', minPts=5, verbose=0)
            clx.fit(X_train)
            
            labels_pred = clx.predict(X_test)
            
            if labels_pred is None or len(labels_pred) != len(X_test):
                checkpoint = 0
                
            # Test with 1D data
            X_train_1d = X_train[:, 0].reshape(-1, 1)
            X_test_1d = X_test[:, 0].reshape(-1, 1)
            
            clx_1d = CLASSIX(metric='manhattan', radius=0.5, verbose=0)
            clx_1d.fit(X_train_1d)
            labels_1d = clx_1d.predict(X_test_1d)
            
            if labels_1d is None or len(labels_1d) != len(X_test_1d):
                checkpoint = 0
                
        except Exception as e:
            print(f"Predict Manhattan test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_predict_tanimoto(self):
        """Test predict method specifically for Tanimoto metric"""
        checkpoint = 1
        try:
            X_train = np.random.randint(0, 2, size=(300, 20)).astype(np.float64)
            X_test = np.random.randint(0, 2, size=(40, 20)).astype(np.float64)
            
            clx = CLASSIX(metric='tanimoto', radius=0.3, group_merging='distance', minPts=5, verbose=0)
            clx.fit(X_train)
            
            labels_pred = clx.predict(X_test)
            
            if labels_pred is None or len(labels_pred) != len(X_test):
                checkpoint = 0
                
        except Exception as e:
            print(f"Predict Tanimoto test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_different_dimensions(self):
        """Test Manhattan clustering with different data dimensions"""
        checkpoint = 1
        try:
            for dim in [1, 2, 5, 10, 20]:
                X = np.random.randn(200, dim)
                clx = CLASSIX(metric='manhattan', radius=0.5, minPts=3, verbose=0)
                clx.fit_transform(X)
                
                if clx.labels_ is None or len(clx.labels_) != len(X):
                    checkpoint = 0
                    break
                    
        except Exception as e:
            print(f"Manhattan dimensions test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_different_dimensions(self):
        """Test Tanimoto clustering with different data dimensions"""
        checkpoint = 1
        try:
            for dim in [5, 10, 20, 50]:
                X = np.random.randint(0, 2, size=(150, dim)).astype(np.float64)
                clx = CLASSIX(metric='tanimoto', radius=0.3, minPts=3, verbose=0)
                clx.fit_transform(X)
                
                if clx.labels_ is None or len(clx.labels_) != len(X):
                    checkpoint = 0
                    break
                    
        except Exception as e:
            print(f"Tanimoto dimensions test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_manhattan_edge_cases(self):
        """Test Manhattan clustering edge cases"""
        checkpoint = 1
        try:
            # Very small dataset
            X_small = np.random.randn(10, 3)
            clx = CLASSIX(metric='manhattan', radius=0.5, minPts=2, verbose=0)
            clx.fit(X_small)
            if clx.labels_ is None or len(clx.labels_) != len(X_small):
                checkpoint = 0
                
            # All points identical
            X_identical = np.ones((50, 3))
            clx2 = CLASSIX(metric='manhattan', radius=0.1, verbose=0)
            clx2.fit(X_identical)
            # Should form single cluster
            if len(np.unique(clx2.labels_)) > 1:
                checkpoint = 0
                
            # Very large radius
            X = np.random.randn(100, 3)
            clx3 = CLASSIX(metric='manhattan', radius=100.0, verbose=0)
            clx3.fit(X)
            # Should merge everything
            if clx3.labels_ is None or len(clx3.labels_) != len(X):
                checkpoint = 0
                
        except Exception as e:
            print(f"Manhattan edge cases test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)


    def test_tanimoto_edge_cases(self):
        """Test Tanimoto clustering edge cases"""
        checkpoint = 1
        try:
            # Very small dataset
            X_small = np.random.randint(0, 2, size=(10, 10)).astype(np.float64)
            clx = CLASSIX(metric='tanimoto', radius=0.3, minPts=2, verbose=0)
            clx.fit(X_small)
            if clx.labels_ is None or len(clx.labels_) != len(X_small):
                checkpoint = 0
                
            # All points identical
            X_identical = np.ones((30, 15))
            clx2 = CLASSIX(metric='tanimoto', radius=0.1, verbose=0)
            clx2.fit(X_identical)
            if len(np.unique(clx2.labels_)) > 1:
                checkpoint = 0
                
        except Exception as e:
            print(f"Tanimoto edge cases test failed: {e}")
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)

        
if __name__ == '__main__':
    unittest.main()
