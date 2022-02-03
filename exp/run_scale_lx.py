import math
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as data
from scipy.sparse import csr_matrix, _sparsetools
from classix import CLASSIX, visualize_connections, return_csr_matrix_indices
import matplotlib.pyplot as plt

def rn_scale_explore():
    plt.style.use('bmh')

    TOL = 0.1 
    random_state = 1
    moons, _ = data.make_moons(n_samples=1000, noise=0.05, random_state=random_state)
    blobs, _ = data.make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], cluster_std=0.5, random_state=random_state)
    X = np.vstack([blobs, moons])

    for scale in np.arange(1, 3.3, 0.1):
        classix = CLASSIX(sorting='pca', radius=TOL, group_merging='distance', verbose=0)
        classix.fit_transform(X)
        classix.visualize_linkage(scale=scale, figsize=(8,8), labelsize=24, path='img')


    for tol in np.arange(0.1, 1.3, 0.1):
        classix = CLASSIX(sorting='pca', radius=tol, group_merging='distance', verbose=0)
        classix.fit_transform(X)
        classix.visualize_linkage(scale=1.5, figsize=(8,8), labelsize=24, plot_boundary=True, path='img')
    
    
