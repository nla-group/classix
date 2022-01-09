import time
import random
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import metrics
from src.clustering import CLASSIX
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


def test_method_labels_blobs(X=None, y=None, method = None, _range=np.arange(0.05, 0.3, 0.005)):
    classixr = list()
    classixm = list()
    
    for i in _range:
        classix = CLASSIX(
            sorting='pca', radius=i, 
            group_merging=method, verbose=0)
        classix.fit_transform(X)
        #print('tol: {}, number of distinct labels:'.format(i), len(np.unique(dbcu.labels_)))
        classix_mi = metrics.adjusted_mutual_info_score(y, classix.labels_)
        classix_ri = metrics.adjusted_rand_score(y, classix.labels_)
        
        classixm.append(classix_mi)
        classixr.append(classix_ri)
     
    return classixm, classixr



def run_sensitivity_test_blobs(_range, dataset_sizes, fe_dim=10, n_clusters=10, cstd=1):
    for index, size in enumerate(dataset_sizes):
        # Use sklearns make_blobs to generate a random dataset with specified size
        # dimension and number of clusters
        X, y = sklearn.datasets.make_blobs(n_samples=size,
                                                   n_features=fe_dim,
                                                   centers=n_clusters, 
                                                   cluster_std=cstd)
        
        dbcur_den_ami, dbcur_den_ari = test_method_labels_blobs(X=X, y=y, method='density', _range=_range)
        dbcur_dis_ami, dbcur_dis_ari = test_method_labels_blobs(X=X, y=y, method='distance',_range=_range)
        
        # sns.set(font_scale=5)
        plt.figure(figsize=(6, 3.6))
        
        plt.rcParams['axes.facecolor'] = 'white'
        # plt.rc('font', family='serif')
        # plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(_range, dbcur_dis_ari, label='ARI - distance',
                 marker='.', markersize=10, c='salmon')  

        plt.plot(_range, dbcur_den_ari, label='ARI - density',
                 marker='o', markersize=7.2, c='darkred')

        plt.plot(_range, dbcur_dis_ami, label='AMI - distance',
                 marker='+', markersize=10, c='darkseagreen')  

        plt.plot(_range, dbcur_den_ami, label='AMI - density',
                 marker='*', markersize=7.2, c='darkolivegreen')

        plt.legend(fontsize=15, fancybox=True, loc='lower left')
        plt.ylim(0, 1)
        plt.tick_params(axis='both',  labelsize=18)

        plt.savefig('fresults/index{0}_with_size{1}'.format(index, size)+'.pdf', bbox_inches='tight')
        # plt.show()
        

def rn_tol_st():
    plt.style.use('bmh')
    seed = 0

    np.random.seed(seed)
    random.seed(seed)
    dataset_sizes = np.hstack([np.arange(1, 6) * 1000, np.arange(6,10) * 1000])
    # print("data size: " + " ".join([str(i) for i in dataset_sizes]))

    _range = np.arange(0.2, 1.005, 0.005)
    run_sensitivity_test_blobs(_range, dataset_sizes=dataset_sizes)