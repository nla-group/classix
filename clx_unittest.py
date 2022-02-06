import unittest
import numpy as np
import sklearn.datasets as data
from classix import CLASSIX, load_data


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



if __name__ == '__main__':
    unittest.main()
