import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from classix import CLASSIX
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(0)

def rn_mcs_it(save=False):
    plt.style.use('bmh')
    plt.rcParams['axes.facecolor'] = 'white'
    random_state = 1
    moons, _ = make_moons(n_samples=1000, noise=0.05, random_state=random_state)
    blobs, _ = make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], cluster_std=0.5, random_state=random_state)
    X = np.vstack([blobs, moons])
    if save:
        np.save("data/mcs_data.npy", X)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X[:, 0], X[:,1], color='b')
    ax.set_aspect('equal', adjustable='datalim')
    # plt.show()
    TOL = 0.1

    minPts = 5
    classix = CLASSIX(sorting='pca', radius=TOL, group_merging='distance', minPts=minPts, post_alloc=False, verbose=0)
    classix.fit_transform(X)

    X_clean = X[classix.clean_index,:]

    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X_clean[classix.labels_[classix.clean_index] == 0,0], X_clean[classix.labels_[classix.clean_index] == 0,1], c="hotpink", s=20)
    ax.scatter(X_clean[classix.labels_[classix.clean_index] == 1,0], X_clean[classix.labels_[classix.clean_index] == 1,1], c="yellowgreen", s=20)
    ax.scatter(X_clean[classix.labels_[classix.clean_index] == 2,0], X_clean[classix.labels_[classix.clean_index] == 2,1], c="tomato", s=20)
    ax.scatter(X_clean[classix.labels_[classix.clean_index] == 3,0], X_clean[classix.labels_[classix.clean_index] == 3,1], c="cadetblue", s=20)
    ax.scatter(X[classix.labels_ == -1,0], X[classix.labels_ == -1,1], c="k", s=20) 
    ax.set_aspect('equal', adjustable='datalim')
    plt.tick_params(axis='both', labelsize=20)
    # plt.xticks([])
    # plt.yticks([])
    ax.grid(False)
    plt.savefig('results/exp1/X_noises.pdf', bbox_inches='tight')
    # plt.show()


    classix = CLASSIX(sorting='pca', radius=TOL, group_merging='distance', minPts=minPts, post_alloc=True, verbose=0)
    classix.fit_transform(X)

    X_clean = X[classix.clean_index,:]
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X[classix.labels_== 0,0], X[classix.labels_== 0,1], c="hotpink", s=20)
    ax.scatter(X[classix.labels_== 1,0], X[classix.labels_== 1,1], c="yellowgreen", s=20)
    ax.scatter(X[classix.labels_== 2,0], X[classix.labels_== 2,1], c="tomato", s=20)
    ax.scatter(X[classix.labels_== 3,0], X[classix.labels_== 3,1], c="cadetblue", s=20)
    # cbar = plt.colorbar() 
    # cbar.ax.tick_params(labelsize=22) 
    ax.set_aspect('equal', adjustable='datalim')
    plt.tick_params(axis='both', labelsize=20)
    # plt.xticks([])
    # plt.yticks([])
    ax.grid(False)
    plt.savefig('results/exp1/X_no_noises.pdf', bbox_inches='tight')
    # plt.show()
