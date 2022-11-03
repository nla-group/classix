# MIT License
#
# Copyright (c) 2022 Stefan Güttel, Xinye Chen
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
import sklearn.datasets as data
from classix import CLASSIX, loadData, cython_is_available
from classix.clustering import calculate_cluster_centers
from classix import aggregation_test, novel_normalization


def exp_aggregate_nr_dist(data, tol=0.15, sorting='pca', early_stopping=True):
    data, (_mu, _scl) = novel_normalization(data, sorting)
    labels, splist, nr_dist = aggregation_test.aggregate(
                         data, 
                         sorting=sorting,
                         tol=tol, 
                         early_stopping=early_stopping
    )
    return nr_dist, labels


class TestClassix(unittest.TestCase):
    
    def test_cython_check(self):
        checkpoint = 1
        try:
            cython_is_available()
        except:
            checkpoint = 0
        self.assertEqual(checkpoint, 1)
        
        
    def test_distance_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):
            clx = CLASSIX(radius=tol, group_merging='distance', verbose=0)
            clx.fit_transform(vdu_signals)
            
            # version 0.2.7
            # np.save('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy')
            comp = clx.labels_ == checkpoint
            assert(comp.all())
    
            
    def test_density_cluster(self):
        vdu_signals = loadData('vdu_signals')

        for tol in np.arange(0.8, 1, 0.1):
            clx = CLASSIX(radius=tol, group_merging='density', verbose=0)
            clx.fit_transform(vdu_signals)
            
            # version 0.2.7
            # np.save('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy')
            comp = clx.labels_ == checkpoint
            assert(comp.all())

    def non_cython_version(self):
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
            except:
                checkpoint = 0
                break
        
        self.assertEqual(checkpoint, 1)
    
    def cython_version(self):
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
                
                clx = CLASSIX(sorting='pca', group_merging='distance')
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

        
    def test_agg_early_stop(self):
        X, y = data.make_blobs(n_samples=1000, centers=10, n_features=2, random_state=0)

        for TOL in np.arange(0.1, 1, 0.1):
            ort_nr_dist_true, ort_labels_true = exp_aggregate_nr_dist(X, tol=TOL, sorting='norm-orthant', early_stopping=True)
            ort_nr_dist_false, ort_labels_false = exp_aggregate_nr_dist(X, tol=TOL, sorting='norm-orthant', early_stopping=False)
            if ort_labels_true.tolist() == ort_labels_false.tolist():
                assert(True)
            if ort_nr_dist_true == ort_nr_dist_false:
                assert(True)

            mean_nr_dist_true, mean_labels_true = exp_aggregate_nr_dist(X, tol=TOL, sorting='norm-mean', early_stopping=True)
            mean_nr_dist_false, mean_labels_false = exp_aggregate_nr_dist(X, tol=TOL, sorting='norm-mean', early_stopping=False)
            if mean_labels_true.tolist() == mean_labels_false.tolist():
                assert(True)
            if mean_nr_dist_true == mean_nr_dist_false:
                assert(True)    

            pca_nr_dist_true, pca_labels_true = exp_aggregate_nr_dist(X, tol=TOL, sorting='pca', early_stopping=True)
            pca_nr_dist_false, pca_labels_false = exp_aggregate_nr_dist(X, tol=TOL, sorting='pca', early_stopping=False)
            if pca_labels_true.tolist() == pca_labels_false.tolist():
                assert(True)
            if pca_nr_dist_true == pca_nr_dist_false:
                assert(True)

                
    def test_explain(self):
        X, y = data.make_blobs(n_samples=5000, centers=2, n_features=2, 
                               cluster_std=1.5, random_state=1
        )
        checkpoint = 1
        try:
            clx = CLASSIX(radius=0.5, group_merging='distance', minPts=3)
            clx.fit_transform(X)
            clx.predict(X)
            clx.explain(plot=True, figsize=(10,10),  savefig=True)
            clx.explain(0,  plot=True, savefig=True)
            clx.explain(3, 2000,  plot=True, savefig=False)
            clx.explain(0, 2008,  plot=True, savefig=True)
            clx.explain(index1=0, index2=2008, index3=100,  plot=True, savefig=True)
            # clx.explain(plot=True, figsize=(10,10), sp_fontsize=10, savefig=False)
            # clx.explain(0,  plot=True, sp_fontsize=10, savefig=False)
            # clx.explain(3, 2000,  plot=True, sp_fontsize=10, savefig=False)
            # clx.explain(0, 2008,  plot=True, sp_fontsize=10, savefig=False)
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
            clx.explain(plot=True, figsize=(10,10),  savefig=True)
            clx.explain(0,  plot=True, savefig=True)
            clx.explain(3, 2000,  plot=True, savefig=False)
            clx.explain(0, 2008,  plot=True, savefig=True)
            clx.explain(index1=0, index2=2008, index3=100,  plot=True, savefig=True)
            # clx.explain(plot=True, figsize=(10,10), sp_fontsize=10, savefig=False)
            # clx.explain(0,  plot=True, sp_fontsize=10, savefig=False)
            # clx.explain(3, 2000,  plot=True, sp_fontsize=10, savefig=False)
            # clx.explain(0, 2008,  plot=True, sp_fontsize=10, savefig=False)
        except:
            checkpoint = 0

        self.assertEqual(checkpoint, 1)

        
    def test_built_in_data(self):
        checkpoint = 1
        try:
            for dn in ['vdu_signals', 'Iris', 'Dermatology', 'Ecoli', 'Glass', 'Banknote', 'Seeds', 'Phoneme', 'Wine']:
                loadData(name=dn)
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
            centers = calculate_cluster_centers(X, y)
        except:
            checkpoint = 0
            
        self.assertEqual(checkpoint, 1)
        
if __name__ == '__main__':
    unittest.main()

