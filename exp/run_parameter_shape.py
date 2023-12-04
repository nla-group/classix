import random
import numpy as np
import pandas as pd
import hdbscan
from sklearn import metrics
from classix import CLASSIX
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from quickshift.QuickshiftPP import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('bmh')
seed = 0
np.random.seed(seed)
random.seed(seed)


def load_data(file='data/Shape sets/Aggregation.txt'):
    data = pd.read_csv(file,sep="\\s+", header = None)
    return data

def test_meanshift_labels(X=None, y=None,  _range=np.arange(0.05, 0.505, 0.005)):
    ar = list()
    am = list()
    
    for i in _range:
        meanshift = MeanShift(bandwidth=i).fit(X)
        meanshift.fit(X)
        mi = metrics.adjusted_mutual_info_score(y, meanshift.labels_)
        ri = metrics.adjusted_rand_score(y, meanshift.labels_)
        
        ar.append(ri)
        am.append(mi)
     
    return ar, am

def test_classix_labels(X=None, y=None, minPts=2, method = None, scale=1.5, _range=np.arange(0.05, 0.3, 0.005)):
    ar = list()
    am = list()
    
    for i in _range:
        classix = CLASSIX(
            sorting='pca', radius=i, minPts=minPts, scale=scale, post_alloc=True,
            group_merging=method, verbose=0)
        classix.fit(X)
        mi = metrics.adjusted_mutual_info_score(y, classix.labels_)
        ri = metrics.adjusted_rand_score(y, classix.labels_)
        
        am.append(mi)
        ar.append(ri)
     
    return ar, am

def test_hdbscan_labels(X=None, y=None, _range=np.arange(2, 21, 1)):
    ar = list()
    am = list()
    for i in _range:
        _hdbscan = hdbscan.HDBSCAN(min_cluster_size=int(i), algorithm='best', core_dist_n_jobs=1)
        
        _hdbscan.fit(X)
        mi = metrics.adjusted_mutual_info_score(y, _hdbscan.labels_)
        ri = metrics.adjusted_rand_score(y, _hdbscan.labels_)
        
        ar.append(ri)
        am.append(mi)
     
    return ar, am

def test_dbscan_labels(X=None, y=None, minPts=5, _range=np.arange(0.05, 0.505, 0.005)):
    ar = list()
    am = list()
    
    for i in _range:
        dbscan = DBSCAN(eps=i, min_samples=minPts, n_jobs=1)
        
        dbscan.fit(X)
        mi = metrics.adjusted_mutual_info_score(y, dbscan.labels_)
        ri = metrics.adjusted_rand_score(y, dbscan.labels_)
        
        ar.append(ri)
        am.append(mi)
     
    return ar, am

def test_quickshiftpp_labels(X=None, y=None, k=10, _range=np.arange(0.05, 0.505, 0.005)):
    ar = list()
    am = list()
    
    for i in _range:
        quicks = QuickshiftPP(k=k, beta=i)
        quicks.fit(X.copy(order='C'))
        mi = metrics.adjusted_mutual_info_score(y, quicks.memberships)
        ri = metrics.adjusted_rand_score(y, quicks.memberships)
        
        ar.append(ri)
        am.append(mi)
     
    return ar, am

# restrict to 2D data
def run_sensitivity_test(_range, loc, files, clustering='CLASSIX (Density)', fix_k=5, scale=1.5, label_files=None):
    for k in range(len(files)):
        data = load_data(loc + files[k] + '.txt')
        if label_files != None:
            X = data.values
            y = load_data(loc + label_files[k])[0].values
        else:
            X = data[[0,1]].values; y = data[2].values # specific situation, might change
        
        if clustering == 'CLASSIX (density)':
            # X = (X - X.mean(axis=0)) / X.std(axis=0)
            dbcur_ami, dbcur_ars = test_classix_labels(X=X, y=y, method='density', minPts=fix_k, _range=_range)
        elif clustering == 'CLASSIX (distance)':
            dbcur_ami, dbcur_ars = test_classix_labels(X=X, y=y, method='distance', minPts=fix_k, scale=scale, _range=_range)
        elif clustering == 'HDBSCAN':
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            dbcur_ami, dbcur_ars = test_hdbscan_labels(X=X, y=y, _range=_range)
        elif clustering == 'DBSCAN':
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            dbcur_ami, dbcur_ars = test_dbscan_labels(X=X, y=y, minPts=fix_k, _range=_range)
        elif clustering == 'Quickshift++':
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            dbcur_ami, dbcur_ars = test_quickshiftpp_labels(X=X, y=y, k=fix_k, _range=_range)
        elif clustering == 'Mean shift':
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            dbcur_ami, dbcur_ars = test_meanshift_labels(X=X, y=y, _range=_range)
        else:
            raise ValueError('Specify a concrete clustering algorithms.')
        

        # sns.set(font_scale=5)
        plt.figure(figsize=(35, 3.6))
        plt.rcParams['axes.facecolor'] = 'white'
        # plt.rc('font', family='serif')
        plt.plot(_range, dbcur_ars, label='ARI',
                 marker='o', markersize=10, c='indigo')  

        plt.plot(_range, dbcur_ami, label='AMI',
                 marker='*', markersize=8, c='cornflowerblue')
    
        plt.xticks(_range)
        plt.legend(fontsize=15, fancybox=True, loc='best')
        plt.ylim(-.05, 1.05)
        plt.tick_params(axis='both',  labelsize=18)

        plt.savefig('results/exp4/{}'.format(files[k])+clustering+'.pdf', bbox_inches='tight')
        # plt.show()



def main():
    shape_sets = ['Aggregation', 'Compound', 
                  'D31', 'Flame', 
                  'Jain', 'Pathbased', 
                  'R15', 'Spiral'
                 ]

    loc = "data/Shape sets/"
    
    np.random.seed(seed)
    _range = np.arange(0.1, 0.85, 0.025)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, fix_k=0, clustering='Mean shift', label_files=None)
    
    np.random.seed(seed)
    _range = np.arange(0.1, 0.85, 0.025)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, fix_k=5, clustering='DBSCAN', label_files=None)

    np.random.seed(seed)
    _range = np.arange(2, 32, 1)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, clustering='HDBSCAN', label_files=None)
    
    np.random.seed(seed)
    _range = np.arange(0.1, 0.85, 0.025)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, fix_k=10, clustering='Quickshift++', label_files=None)

    np.random.seed(seed)
    _range = np.arange(0.025, 0.525, 0.025)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, fix_k=0, clustering='CLASSIX (distance)', scale=1.5, label_files=None)

    np.random.seed(seed)
    _range = np.arange(0.025, 0.525, 0.025)
    run_sensitivity_test(_range, loc=loc, files=shape_sets, fix_k=0, clustering='CLASSIX (density)', label_files=None)