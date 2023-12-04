import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from aggregation_test import aggregate
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.metrics.cluster import adjusted_rand_score as ari


def shift_scale(X, sorting):
    X_copy = X.copy()
    if sorting == "norm-mean":
        _mu = X_copy.mean(axis=0)
        X_copy = X_copy - _mu
        _scl = X_copy.std()
        X_copy = X_copy / _scl
        
    elif sorting == "norm-orthant":
        _mu = X_copy.min(axis=0)
        X_copy = X_copy - _mu
        _scl = X_copy.std()
        X_copy = X_copy / _scl     
        
    elif sorting == "pca":
        _mu = X_copy.mean(axis=0)
        X_copy = X_copy - _mu # mean center
        rds = norm(X_copy, axis=1) # distance of each data point from 0
        _scl = np.median(rds) # 50% of data points are within that radius
        X_copy = X_copy / _scl # now 50% of data are in unit ball
     
    return X_copy

        
    
def count_distance():
    np.random.seed(0)
    NUM = 1000
    FDIM = 100
    # early stopping
    nr_dist_centers_num = np.zeros((NUM, 3))
    centers_num_ln = np.zeros((NUM, 3))
    centers_num_ami = np.zeros((NUM, 3))
    centers_num_ari = np.zeros((NUM, 3))
    
    nr_dist_fdim_num = np.zeros((NUM, 3))
    fdim_num_ln = np.zeros((NUM, 3))
    fdim_num_ami = np.zeros((NUM, 3))
    fdim_num_ari = np.zeros((NUM, 3))
    
    # no early stopping
    nnr_dist_centers_num = np.zeros((NUM, 3))
    ncenters_num_ln = np.zeros((NUM, 3))
    ncenters_num_ami = np.zeros((NUM, 3))
    ncenters_num_ari = np.zeros((NUM, 3))
    
    nnr_dist_fdim_num = np.zeros((NUM, 3))
    nfdim_num_ln = np.zeros((NUM, 3))
    nfdim_num_ami = np.zeros((NUM, 3))
    nfdim_num_ari = np.zeros((NUM, 3))
    
    for i in range(NUM):
        X, y = make_blobs(n_samples=1000, centers=i+1, n_features=2,
                          random_state=0)
        # X = (X - X.mean())/X.std()
        X_orthant = shift_scale(X, "norm-orthant")
        X_mean = shift_scale(X, "norm-mean")
        X_pca = shift_scale(X, "pca")
        labels_no,sp_no, nr_dist_no, _ = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=True)
        labels_nm, sp_nm, nr_dist_nm, _ = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=True)
        labels_pca, sp_pca, nr_dist_pca, _ = aggregate(X_pca, tol=0.5, sorting='pca', early_stopping=True)

        nlabels_no, nsp_no, nnr_dist_no, _ = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=False)
        nlabels_nm, nsp_nm, nnr_dist_nm, _ = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=False)
        nlabels_pca, nsp_pca, nnr_dist_pca, _ = aggregate(X_pca, tol=0.5, sorting='pca', early_stopping=False)
        
        nr_dist_centers_num[i] = [nr_dist_no, nr_dist_nm, nr_dist_pca]
        centers_num_ln[i] = [len(sp_no), len(sp_nm), len(sp_pca)]
        centers_num_ami[i] =  np.array([ami(labels_no, y), ami(labels_nm, y), ami(labels_pca, y)])
        centers_num_ari[i] =  np.array([ari(labels_no, y), ari(labels_nm, y), ari(labels_pca, y)])
        
        nnr_dist_centers_num[i] = [nnr_dist_no, nnr_dist_nm, nnr_dist_pca]
        ncenters_num_ln[i] = [len(nsp_no), len(nsp_nm), len(nsp_pca)]
        ncenters_num_ami[i] =  np.array([ami(nlabels_no, y), ami(nlabels_nm, y), ami(nlabels_pca, y)])
        ncenters_num_ari[i] =  np.array([ari(nlabels_no, y), ari(nlabels_nm, y), ari(nlabels_pca, y)])
        
    for num in range(FDIM):
        X, y = make_blobs(n_samples=1000, centers=10, n_features=num+1, random_state=0)     
        # X = (X - X.mean())/X.std()
        
        X_orthant = shift_scale(X, "norm-orthant")
        X_mean = shift_scale(X, "norm-mean")
        X_pca = shift_scale(X, "pca")
        
        labels_no,sp_no, nr_dist_no, _ = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=True)
        labels_nm, sp_nm, nr_dist_nm, _ = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=True)
        labels_pca, sp_pca, nr_dist_pca, _ = aggregate(X_pca, tol=0.5, sorting='pca', early_stopping=True)
        
        nlabels_no, nsp_no, nnr_dist_no, _ = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=False)
        nlabels_nm, nsp_nm, nnr_dist_nm, _ = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=False)
        nlabels_pca, nsp_pca, nnr_dist_pca, _ = aggregate(X_pca, tol=0.5, sorting='pca', early_stopping=False)
        
        nr_dist_fdim_num[num] = np.array([nr_dist_no, nr_dist_nm, nr_dist_pca])
        fdim_num_ln[num] = np.array([len(sp_no), len(sp_nm), len(sp_pca)])
        fdim_num_ami[num] =  np.array([ami(labels_no, y), ami(labels_nm, y), ami(labels_pca, y)])
        fdim_num_ari[num] =  np.array([ari(labels_no, y), ari(labels_nm, y), ari(labels_pca, y)])
        
        nnr_dist_fdim_num[num] = np.array([nnr_dist_no, nnr_dist_nm, nnr_dist_pca])
        nfdim_num_ln[num] = np.array([len(nsp_no), len(nsp_nm), len(nsp_pca)])
        nfdim_num_ami[num] =  np.array([ami(nlabels_no, y), ami(nlabels_nm, y), ami(nlabels_pca, y)])
        nfdim_num_ari[num] =  np.array([ari(nlabels_no, y), ari(nlabels_nm, y), ari(nlabels_pca, y)])
    
    concat_list = [nr_dist_centers_num, centers_num_ln, centers_num_ami, centers_num_ari,\
                   nr_dist_fdim_num, fdim_num_ln, fdim_num_ami, fdim_num_ari, \
                   nnr_dist_centers_num, ncenters_num_ln, ncenters_num_ami, centers_num_ari,\
                   nnr_dist_fdim_num, nfdim_num_ln, nfdim_num_ami, fdim_num_ari]
    
    fname = ['results/nr_dist_centers_num.csv', 'results/centers_num_ln.csv',
             'results/centers_num_ami.csv', 'results/centers_num_ari.csv', 
             'results/nr_dist_fdim_num.csv', 'results/fdim_num_ln.csv', 
             'results/fdim_num_ami.csv', 'results/fdim_num_ari.csv',
             'results/nnr_dist_centers_num.csv', 'results/ncenters_num_ln.csv',
             'results/ncenters_num_ami.csv', 'results/ncenters_num_ari.csv', 
             'results/nnr_dist_fdim_num.csv', 'results/nfdim_num_ln.csv', 
             'results/nfdim_num_ami.csv', 'results/nfdim_num_ari.csv']
    
    columns = ["Norm-orthant","Norm-mean","PCA"]
    for i in range(len(fname)):
        pd.DataFrame(concat_list[i], columns=columns).to_csv(fname[i], index=False)



def exp_aggregate_nr_dist(data, tol=0.15, sorting='pca', early_stopping=True):
    data = shift_scale(data, sorting)
    labels, splist, nr_dist, _ = aggregate(data, sorting=sorting, tol=tol, early_stopping=early_stopping)
    return nr_dist, labels


def rn_sort_early_stp():
    # data = np.load("datasets/artificial samples/test_X.npy")
    # y = np.load("datasets/artificial samples/test_y.npy")
    data, y = make_blobs(n_samples=1000, centers=10, n_features=2, random_state=0)

    norm_orthant_true = list()
    norm_orthant_false = list()
    norm_mean_true = list()

    norm_mean_false = list()
    pca_dist_true = list()
    pca_dist_false = list()

    for TOL in np.arange(0.1,1.01, 0.01):
        ort_nr_dist_true, ort_labels_true = exp_aggregate_nr_dist(data, tol=TOL, sorting='norm-orthant', early_stopping=True)
        ort_nr_dist_false, ort_labels_false = exp_aggregate_nr_dist(data, tol=TOL, sorting='norm-orthant', early_stopping=False)
        if ort_labels_true.tolist() != ort_labels_false.tolist():
            # print(np.round(TOL,2), "norm-orthant pass.")
            raise ValueError("Early stopping ruins the aggregation.")
            # to test if early stopping is effective and maintain the same aggregation.
        
        mean_nr_dist_true, mean_labels_true = exp_aggregate_nr_dist(data, tol=TOL, sorting='norm-mean', early_stopping=True)
        mean_nr_dist_false, mean_labels_false = exp_aggregate_nr_dist(data, tol=TOL, sorting='norm-mean', early_stopping=False)
        if mean_labels_true.tolist() != mean_labels_false.tolist():
            # print(np.round(TOL,2), "norm-mean pass.")
            raise ValueError("Early stopping ruins the aggregation.")
            
        pca_nr_dist_true, pca_labels_true = exp_aggregate_nr_dist(data, tol=TOL, sorting='pca', early_stopping=True)
        pca_nr_dist_false, pca_labels_false = exp_aggregate_nr_dist(data, tol=TOL, sorting='pca', early_stopping=False)
        if pca_labels_true.tolist() != pca_labels_false.tolist():
            # print(np.round(TOL,2), "pca pass.")
            raise ValueError("Early stopping ruins the aggregation.")
            
        norm_orthant_true.append(ort_nr_dist_true)
        norm_orthant_false.append(ort_nr_dist_false)
        norm_mean_true.append(mean_nr_dist_true)
        norm_mean_false.append(mean_nr_dist_false)
        pca_dist_true.append(pca_nr_dist_true)
        pca_dist_false.append(pca_nr_dist_false)
    
    np.save('results/norm_orthant_true.npy', norm_orthant_true)
    np.save('results/norm_orthant_false.npy', norm_orthant_false)
    np.save('results/norm_mean_true.npy', norm_mean_true)
    np.save('results/norm_mean_false.npy', norm_mean_false)
    np.save('results/pca_dist_true.npy', pca_dist_true)
    np.save('results/pca_dist_false.npy', pca_dist_false)
    
    
    
def rn_sort_plot1():
    norm_orthant_true = np.load('results/norm_orthant_true.npy', allow_pickle=True)
    norm_orthant_false = np.load('results/norm_orthant_false.npy', allow_pickle=True)
    norm_mean_true = np.load('results/norm_mean_true.npy', allow_pickle=True)
    norm_mean_false = np.load('results/norm_mean_false.npy', allow_pickle=True)
    pca_dist_true = np.load('results/pca_dist_true.npy', allow_pickle=True)
    pca_dist_false = np.load('results/pca_dist_false.npy', allow_pickle=True)
    
    data = pd.DataFrame()
    data['PCA'] = pca_dist_false
    data['PCA - early stopping'] = pca_dist_true

    data['Norm-orthant'] = norm_orthant_false
    data['Norm-orthant - early stopping'] = norm_orthant_true

    data['Norm-mean'] = norm_mean_false
    data['Norm-mean - early stopping'] = norm_mean_true

    plt.figure(figsize=(8,6))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.plot(np.arange(0.1, 1.01, 0.05), data['PCA'],  
             marker='p', markersize=6, markerfacecolor='none', linewidth=3, linestyle=":", label='PCA')
    plt.plot(np.arange(0.1, 1.01, 0.05), data['PCA - early stopping'], 
             marker='p', markersize=6, markerfacecolor='none', linewidth=3, linestyle="-", label='PCA - early stopping')
    plt.plot(np.arange(0.1, 1.01, 0.05), data['Norm-orthant'],  
             marker='*', markersize=6, markerfacecolor='none', linewidth=3, linestyle=":", label='Norm-orthant')
    plt.plot(np.arange(0.1, 1.01, 0.05), data['Norm-orthant - early stopping'], 
             marker='*', markersize=6, markerfacecolor='none', linewidth=3, linestyle="-", label='Norm-orthant - early stopping')
    plt.plot(np.arange(0.1, 1.01, 0.05), data['Norm-mean'], 
             marker='P', markersize=6, markerfacecolor='none', linewidth=3, linestyle=":", label='Norm-mean')
    plt.plot(np.arange(0.1, 1.01, 0.05), data['Norm-mean - early stopping'], 
              marker='P', markersize=6, markerfacecolor='none', linewidth=3, linestyle="-", label='Norm-mean - early stopping')
    plt.legend(fontsize=17, bbox_to_anchor=(1.75, 1))
    plt.tick_params(axis='both', which='major', width=3, labelsize=17)
    plt.savefig('results/sorting_comp_1.pdf', bbox_inches='tight')
    # plt.show()
    
    
    
def rn_sort_plot2():
    nr_dist_centers_num = pd.read_csv("results/nr_dist_centers_num.csv").iloc[:500]
    nnr_dist_centers_num = pd.read_csv("results/nnr_dist_centers_num.csv").iloc[:500]
    plt.figure(figsize=(8,6))
    plt.rcParams['axes.facecolor'] = 'white'

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nnr_dist_centers_num['PCA'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle=":", label='PCA')

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nr_dist_centers_num['PCA'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle="-", label='PCA - early stopping')

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nnr_dist_centers_num['Norm-orthant'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle=":", label='Norm-orthant')

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nr_dist_centers_num['Norm-orthant'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle="-", label='Norm-orthant - early stopping')

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nnr_dist_centers_num['Norm-mean'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle=":", label='Norm-mean')

    plt.plot(np.arange(1, len(nr_dist_centers_num)+1, 1), nr_dist_centers_num['Norm-mean'],  
             marker='p', markersize=3, markerfacecolor='none', linewidth=6, linestyle="-", label='Norm-mean - early stopping')

    plt.tick_params(axis='both', which='major', width=3, labelsize=17)
    plt.grid(True)
    plt.xlim(-3,len(nr_dist_centers_num)+3)
    plt.legend(fontsize=17, bbox_to_anchor=(1.01, 1))
    plt.savefig('results/sorting_comp_2.pdf', bbox_inches='tight')
    # plt.show()
