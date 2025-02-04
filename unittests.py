# MIT License
#
# Copyright (c) 2024 Stefan Güttel, Xinye Chen
#


import classix
import unittest
import numpy as np
import pandas as pd
import sklearn.datasets as data
from classix import CLASSIX, loadData, cython_is_available
from classix.clustering import calculate_cluster_centers
from classix import preprocessing
from classix import aggregate, aggregate_c, aggregate_cm
from classix.merge import distance_merge_mtg, distance_merge, density_merge
import platform

if platform.system() == 'Windows':
    from classix.merge_cm_win import distance_merge as distance_merge_cm
    from classix.merge_cm_win import density_merge as density_merge_cm
else:
    from classix.merge_cm import distance_merge as distance_merge_cm
    from classix.merge_cm import density_merge as density_merge_cm

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
                clx.visualize_linkage(scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='pca', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)

        checkpoint = 1
        for scale in np.arange(1.8, 2, 0.1):
            try:
                clx = CLASSIX(sorting='norm-orthant', radius=TOL, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='norm-orthant', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)

        checkpoint = 1
        for scale in np.arange(1.8, 2, 0.1):
            try:
                clx = CLASSIX(sorting='norm-mean', radius=TOL, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(scale=scale, figsize=(8,8), labelsize=24)
            except:
                checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        checkpoint = 1
        for tol in np.arange(0.9, 1, 0.1):
            try:
                clx = CLASSIX(sorting='norm-mean', radius=tol, group_merging='distance', verbose=0)
                clx.fit_transform(X)
                clx.visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True)
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
            clx.load_group_centers()
            clx.load_cluster_centers()
            clx.gcIndices([1, 2, 3, 4])
            
            clx.predict(X)
            clx.predict(X[:1000])

            clx.explain(plot=False, showsplist=True, figsize=(10,10),  savefig=True)
            clx.explain(0,  plot=False, savefig=True, showsplist=True)
            clx.form_starting_point_clusters_table(aggregate=True)
            clx.explain(3, 2000,  plot=False, savefig=False)
            clx.explain(0, 2008,  plot=False, savefig=True, replace_name=['Superman', 'Batman'])

            clx.explain(2000, 2028,  plot=False, add_arrow=True, savefig=True, showallgroups=True, include_dist=True)
                
            clx.explain(0, 2008,  plot=False, add_arrow=True, directed_arrow=-1, savefig=True, fmt='pdf')
            clx.explain(0, 2008,  plot=False, add_arrow=True, directed_arrow=1, savefig=True, fmt='png')
            clx.explain(index1=[0, 0], index2=[4,5], plot=False, add_arrow=True, show_connected_label=True, directed_arrow=1)
            clx.explain(index1=np.array([6, 6]), index2=np.array([6, 6]), plot=False, add_arrow=True, directed_arrow=1)
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=4999, mergeTinyGroups=False)
            clx.fit(X)
            clx.explain(0, 2008,  plot=False, add_arrow=True, directed_arrow=-1, savefig=True, fmt='jpg')
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
            clx.explain(index1='Carl', index2='Bert', plot=False, show_connected_label=True, showallgroups=True, sp_fontsize=12)  
            
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
            clx.explain(plot=False, figsize=(10,10),  savefig=False)
            clx.explain(0,  plot=False, savefig=False)
            clx.explain(3, 2000,  plot=False, savefig=False)
            clx.explain(0, 2008,  plot=False, savefig=False)
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
            clx.explain(plot=False, figsize=(10,10),  savefig=False)
            clx.explain(0,  plot=False, savefig=False)
            clx.explain(3, 2000,  plot=False, savefig=False)
            clx.explain(0, 2008,  plot=False, savefig=False)
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)

    
    def test_explain_connected_groups(self):
        X, y = data.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=1)
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging='distance')
            clx.fit(X)
            clx.explain(773, 22, plot=False,  add_arrow=True, include_dist=False)

            clx = CLASSIX(radius=0.1, minPts=99, verbose=1, group_merging='density')
            clx.fit(X)
            clx.explain(773, 22, plot=False,  add_arrow=True, include_dist=False)
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
            
            inverse_ind1, spl1, _, _, _, _, _ = aggregate.general_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind2, spl2, _, _, _, _, _ = aggregate_cm.general_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind3, spl3, _, _, _, _, _ = aggregate_c.general_aggregate(data, "pca", 0.5)
            inverse_ind7, spl7, _, _, _, _, _ = aggregate.pca_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind8, spl8, _, _, _, _, _ = aggregate_c.pca_aggregate(data, "pca", 0.5)
            inverse_ind9, spl9, _, _, _, _, _ = aggregate_cm.pca_aggregate(data, "pca", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_cm.general_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_c.general_aggregate(data, "norm-mean", 0.5)
            
            _, _, _, _, _, _, _ = aggregate_cm.general_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_c.general_aggregate(data, "NA", 0.5)

            inverse_ind1, spl1, _, _, _, _, _ = aggregate.lm_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind2, spl2, _, _, _, _, _ = aggregate_cm.lm_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind3, spl3, _, _, _, _, _ = aggregate_c.lm_aggregate(data, "pca", 0.5)
            
            _, _, _, _, _, _, _ = aggregate.lm_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_cm.lm_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_c.lm_aggregate(data, "norm-mean", 0.5)
            
            _, _, _, _, _, _, _ = aggregate.lm_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_cm.lm_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregate_c.lm_aggregate(data, "NA", 0.5)
            
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
            labels, splist, nr_dist, ind, sort_vals, data, half_nrm2 = aggregate.general_aggregate(data, sorting="pca", tol=0.5) #
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
        
if __name__ == '__main__':
    unittest.main()
