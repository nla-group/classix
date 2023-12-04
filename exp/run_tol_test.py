
import random
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import metrics
from classix import CLASSIX
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



def run_sensitivity_test_blobs(dataset_sizes, _range, fe_dim=10, n_clusters=10, cstd=1):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    csv_data = pd.DataFrame()
    for index, size in enumerate(dataset_sizes):
        # Use sklearns make_blobs to generate a random dataset with specified size
        # dimension and number of clusters
        X, y = sklearn.datasets.make_blobs(n_samples=size,
                                           n_features=fe_dim,
                                           centers=n_clusters, 
                                           cluster_std=cstd,
                                           random_state=0)
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        dbcur_den_ami, dbcur_den_ari = test_method_labels_blobs(X=X, y=y, method='density', _range=_range)
        dbcur_dis_ami, dbcur_dis_ari = test_method_labels_blobs(X=X, y=y, method='distance',_range=_range)
        
        csv_data['dbcur_den_ami'] = dbcur_den_ami
        csv_data['dbcur_den_ari'] = dbcur_den_ari
        csv_data['dbcur_dis_ami'] = dbcur_dis_ami
        csv_data['dbcur_dis_ari'] = dbcur_dis_ari
        
        csv_data.to_csv("results/exp2/index{0}_with_size{1}".format(index, size)+'.csv', index=False)


def plot_sensitivity(dataset_sizes, _range):
    plt.style.use('bmh')
    
    for index, size in enumerate(dataset_sizes):
        csv_data = pd.read_csv("results/exp2/index{0}_with_size{1}".format(index, size)+'.csv')
        
        # sns.set(font_scale=5)
        plt.figure(figsize=(6, 3.6))
        
        plt.rcParams['axes.facecolor'] = 'white'
        # plt.rc('font', family='serif')
        # plt.rcParams['axes.facecolor'] = 'white'
        plt.plot(_range, csv_data['dbcur_dis_ari'].values, label='ARI - distance',
                 marker='.', markersize=10, c='salmon')  

        plt.plot(_range, csv_data['dbcur_den_ari'].values, label='ARI - density',
                 marker='o', markersize=7.2, c='darkred')

        plt.plot(_range, csv_data['dbcur_dis_ami'].values, label='AMI - distance',
                 marker='+', markersize=10, c='darkseagreen')  

        plt.plot(_range, csv_data['dbcur_den_ami'].values, label='AMI - density',
                 marker='*', markersize=7.2, c='darkolivegreen')

        plt.legend(fontsize=18, fancybox=True, loc='lower left')
        plt.ylim(-0.02, 1.02)
        plt.xlim(0.08, 1.02)
        plt.xticks([0.1, 0.4, 0.7, 1])
        plt.tick_params(axis='both',  labelsize=18)

        plt.savefig('results/exp2/index{0}_with_size{1}'.format(index, size)+'.pdf', bbox_inches='tight')
        # plt.show()
