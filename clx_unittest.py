# A quick test for CLASSIX
#
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

import unittest
import numpy as np
import sklearn.datasets as data
from classix import CLASSIX, load_data
from classix import visualize_connections, return_csr_matrix_indices

class TestClassix(unittest.TestCase):
    def test_distance_cluster(self):
        vdu_signals = load_data('vdu_signals')

        for tol in np.arange(0.5,1.01,0.1):
            clx = CLASSIX(radius=tol, group_merging='distance', verbose=0)
            clx.fit_transform(vdu_signals)
            
            # version 0.2.7
            # np.save('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_distance_' + str(np.round(tol,2)) + '.npy')
            comp = clx.labels_ == checkpoint
            assert(comp.all())

    def test_density_cluster(self):
        vdu_signals = load_data('vdu_signals')

        for tol in np.arange(0.5,1.01,0.1):
            clx = CLASSIX(radius=tol, group_merging='density', verbose=0)
            clx.fit_transform(vdu_signals)
            
            # version 0.2.7
            # np.save('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy', clx.labels_) 
            
            # test new version
            checkpoint = np.load('classix/data/checkpoint_density_' + str(np.round(tol,2)) + '.npy')
            comp = clx.labels_ == checkpoint
            assert(comp.all())

    def rn_scale_explore(self):
        TOL = 0.1 
        random_state = 1
        
        moons, _ = data.make_moons(n_samples=1000, noise=0.05, random_state=random_state)
        blobs, _ = data.make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], cluster_std=0.5, random_state=random_state)
        X = np.vstack([blobs, moons])

        for scale in np.arange(1, 3.3, 0.1):
            classix = CLASSIX(sorting='pca', radius=TOL, group_merging='distance', verbose=0)
            classix.fit_transform(X)
            visualize_linkage(scale=scale, figsize=(8,8), labelsize=24, path='img')

        for tol in np.arange(0.1, 1.3, 0.1):
            classix = CLASSIX(sorting='pca', radius=tol, group_merging='distance', verbose=0)
            classix.fit_transform(X)
            visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True, path='img')


if __name__ == '__main__':
    unittest.main()
