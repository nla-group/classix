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
        
        
        
if __name__ == '__main__':
    unittest.main()
