# MIT License
#
# Copyright (c) 2023 Stefan Güttel, Xinye Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import classix
import unittest
import numpy as np
import pandas as pd
import sklearn.datasets as data
from classix import CLASSIX, loadData, cython_is_available
from classix.clustering import calculate_cluster_centers
from classix import novel_normalization
from classix import aggregation, aggregation_c, aggregation_cm
from classix.merging import distance_merging_mtg, distance_merging, density_merging
import platform

if platform.system() == 'Windows':
    from classix.merging_cm_win import distance_merging as distance_merging_cm
    from classix.merging_cm_win import density_merging as density_merging_cm
else:
    from classix.merging_cm import distance_merging as distance_merging_cm
    from classix.merging_cm import density_merging as density_merging_cm

from sklearn.metrics.cluster import adjusted_rand_score



class TestClassix(unittest.TestCase):
    
    def test_cython_check(self):
        checkpoint = 1
        try:
            cython_is_available()
            cython_is_available(verbose=True)
        except:
            checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        
    def test_distance_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):
            clx1 = CLASSIX(radius=tol, group_merging='distance', verbose=0, memory=True)
            clx1.fit_transform(vdu_signals)

            clx2 = CLASSIX(radius=tol, group_merging='distance', verbose=0, memory=False)
            clx2.fit_transform(vdu_signals)

            if adjusted_rand_score(clx1.labels_, clx2.labels_) != 1:
                raise ValueError("Inconsistent results.")
            # version 0.2.7
            # np.save('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy')
            
            assert(adjusted_rand_score(clx1.labels_, checkpoint) == 1)
            assert(adjusted_rand_score(clx2.labels_, checkpoint) == 1)
            
            
    def test_normalization(self):
        checkpoint = 1
        X, _ = data.make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
        try:
            novel_normalization(X, "norm-mean")
            novel_normalization(X, "pca")
            novel_normalization(X, "norm-orthant")
        except:
            checkpoint = 0
        
        self.assertEqual(checkpoint, 1)

        
    def test_density_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):
            clx1 = CLASSIX(radius=tol, group_merging='density', verbose=0, memory=True)
            clx1.fit_transform(vdu_signals)

            clx2 = CLASSIX(radius=tol, group_merging='density', verbose=0, memory=False)
            clx2.fit_transform(vdu_signals)

            if adjusted_rand_score(clx1.labels_, clx2.labels_) != 1:
                raise ValueError("Inconsistent results.")
            # version 0.2.7
            # np.save('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy')
  
            assert(adjusted_rand_score(clx1.labels_, checkpoint) == 1)
            assert(adjusted_rand_score(clx2.labels_, checkpoint) == 1)

    
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
                
                clx = CLASSIX(sorting='pca', group_merging='distance', minPts=150)
                clx.fit_transform(X)

                clx = CLASSIX(sorting='pca', group_merging='distance', memory=False)
                clx.fit_transform(X)

                clx = CLASSIX(sorting='pca', group_merging='distance', memory=True, mergeTinyGroups=False)
                clx.fit_transform(X)
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
            clx.predict(X)
            clx.predict(X, memory=True)
            clx.explain(plot=True, figsize=(10,10),  savefig=True)
            clx.explain(0,  plot=True, savefig=True)
            clx.explain(3, 2000,  plot=True, savefig=False)
            clx.explain(0, 2008,  plot=True, savefig=True)
            clx.explain(2000, 2028,  plot=True, add_arrow=True, savefig=True)
            clx.explain(0, 2008,  plot=True, add_arrow=True, directed_arrow=0, savefig=True)
            clx.explain(0, 2008,  plot=True, add_arrow=True, directed_arrow=-1, savefig=True)
            clx.explain(0, 2008,  plot=True, add_arrow=True, directed_arrow=1, savefig=True)

            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=4999, mergeTinyGroups=False)
            clx.fit(X)
            clx.explain(0, 2008,  plot=True, add_arrow=True, directed_arrow=-1, savefig=True)
            
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
            clx.explain(index1='Carl', index2='Bert', plot=True, showallgroups=True, sp_fontsize=12)        
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
            clx.explain(plot=True, figsize=(10,10),  savefig=False)
            clx.explain(0,  plot=True, savefig=False)
            clx.explain(3, 2000,  plot=True, savefig=False)
            clx.explain(0, 2008,  plot=True, savefig=False)
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)

        
    def test_built_in_data(self):
        checkpoint = 1
        try:
            for dn in ['vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 'Banknote', 'Seeds', 'Phoneme', 'Wine', 'NA']:
                loadData(name=dn)
        except:
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)
        

    def test_aggregation_precompute(self): 
        checkpoint = 1
        try:
            data = np.random.randn(10000, 2)
            
            inverse_ind1, spl1, _, _, _, _, _ = aggregation.precompute_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind2, spl2, _, _, _, _, _ = aggregation_cm.precompute_aggregate(data, sorting="pca", tol=0.5)
            inverse_ind3, spl3, _, _, _, _, _ = aggregation_c.precompute_aggregate(data, "pca", 0.5)
            inverse_ind4, spl4, _, _, _, _ = aggregation.aggregate(data, sorting="pca", tol=0.5)
            inverse_ind5, spl5, _, _, _, _ = aggregation_c.aggregate(data, "pca", 0.5)
            inverse_ind6, spl6, _, _, _, _ = aggregation_cm.aggregate(data, "pca", 0.5)
            inverse_ind7, spl7, _, _, _, _, _ = aggregation.precompute_aggregate_pca(data, sorting="pca", tol=0.5)
            inverse_ind8, spl8, _, _, _, _, _ = aggregation_c.precompute_aggregate_pca(data, "pca", 0.5)
            inverse_ind9, spl9, _, _, _, _, _ = aggregation_cm.precompute_aggregate_pca(data, "pca", 0.5)
            
            _, _, _, _, _, _, _ = aggregation_cm.precompute_aggregate(data, sorting="norm-mean", tol=0.5)
            _, _, _, _, _, _, _ = aggregation_c.precompute_aggregate(data, "norm-mean", 0.5)
            
            _, _, _, _, _, _, _ = aggregation_cm.precompute_aggregate(data, sorting="NA", tol=0.5)
            _, _, _, _, _, _, _ = aggregation_c.precompute_aggregate(data, "NA", 0.5)
            
            if np.sum(inverse_ind1 != inverse_ind2) != 0:
                checkpoint = 0
            if np.sum(inverse_ind2 != inverse_ind3) != 0:
                checkpoint = 0
            if np.sum(inverse_ind3 != inverse_ind4) != 0:
                checkpoint = 0
            if np.sum(inverse_ind5 != inverse_ind6) != 0:
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
                if spl3[i][0] != spl4[i][0]:
                    checkpoint = 0
                if spl4[i][0] != spl5[i][0]:
                    checkpoint = 0
                if spl5[i][0] != spl6[i][0]:
                    checkpoint = 0
                if spl6[i][0] != spl7[i][0]:
                    checkpoint = 0
                    
                if spl1[i][1] != spl2[i][1]:
                    checkpoint = 0
                if spl2[i][1] != spl3[i][1]:
                    checkpoint = 0
                if spl3[i][1] != spl4[i][1]:
                    checkpoint = 0
                if spl4[i][1] != spl5[i][1]:
                    checkpoint = 0
                if spl5[i][1] != spl6[i][1]:
                    checkpoint = 0
                if spl6[i][1] != spl7[i][1]:
                    checkpoint = 0
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)


    def test_merge(self): 
        checkpoint = 1
        minPts = 10
        scale = 1.5

        try:
            data = np.random.randn(10000, 2)
            checkpoint = 1
            labels, splist, nr_dist, ind, sort_vals, data, half_nrm2 = aggregation.precompute_aggregate(data, sorting="pca", tol=0.5) #
            splist = np.asarray(splist)
            
            radius = 0.5
            splist = np.asarray(splist)
            label_set1, connected_pairs_store1 = density_merging(data, splist, radius, sort_vals, half_nrm2)
            label_set2, connected_pairs_store2 = density_merging_cm(data, splist, radius, sort_vals, half_nrm2)
            
            label_set3, _,_ = distance_merging(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2)
            label_set4, _,_ = distance_merging_cm(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2)
            label_set5, _,_ = distance_merging_mtg(data, labels, splist, radius, minPts, scale, sort_vals, half_nrm2)
            
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
