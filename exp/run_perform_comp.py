# The test is referenced from https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
import time
import hdbscan
import warnings
import sklearn.cluster
import scipy.cluster
import sklearn.datasets
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
from classix.aggregation_test import aggregate
from classix import CLASSIX
from quickshift.QuickshiftPP import *
from sklearn import metrics
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
np.random.seed(0)


def benchmark_algorithm_tdim(dataset_dimensions, cluster_function, function_args, function_kwds,
                        dataset_size=10000, dataset_n_clusters=10, max_time=45, sample_size=10, algorithm=None):

    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result_time = np.nan * np.ones((len(dataset_dimensions), sample_size))
    result_ar = np.nan * np.ones((len(dataset_dimensions), sample_size))
    result_ami = np.nan * np.ones((len(dataset_dimensions), sample_size))
    
    for index, dimension in enumerate(dataset_dimensions):
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            # set cluster_std=0.1 to ensure clustering rely less on tuning parameters.
            data, labels = sklearn.datasets.make_blobs(n_samples=dataset_size,
                                                       n_features=dimension,
                                                       centers=dataset_n_clusters, 
                                                       cluster_std=1) 

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function.fit(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time
            if algorithm == "Quickshift++":
                preds = cluster_function.memberships
            else:
                preds = cluster_function.labels_
            # print("labels num:", len(np.unique(preds))) 
            ar = metrics.adjusted_rand_score(labels, preds)
            ami = metrics.adjusted_mutual_info_score(labels, preds)
            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time: # Luckily, it won't happens in our experiment.
                result_time[index, s] = time_taken
                result_ar[index, s] = ar
                result_ami[index, s] = ami
                return pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_time.flatten()]).T, columns=['x','y']), \
                       pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_ar.flatten()]).T, columns=['x','y']), \
                       pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_ami.flatten()]).T, columns=['x','y'])
            else:
                result_time[index, s] = time_taken
                result_ar[index, s] = ar
                result_ami[index, s] = ami

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_time.flatten()]).T, columns=['x','y']), \
           pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_ar.flatten()]).T, columns=['x','y']), \
           pd.DataFrame(np.vstack([dataset_dimensions.repeat(sample_size), result_ami.flatten()]).T, columns=['x','y'])



def benchmark_algorithm_tsize(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=10, algorithm=None):

    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result_time = np.nan * np.ones((len(dataset_sizes), sample_size))
    result_ar = np.nan * np.ones((len(dataset_sizes), sample_size))
    result_ami = np.nan * np.ones((len(dataset_sizes), sample_size))
    
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            # set cluster_std=0.1 to ensure clustering rely less on tuning parameters.
            data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters, 
                                                       cluster_std=1) 

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function.fit(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time
            if algorithm == "Quickshift++":
                preds = cluster_function.memberships
            else:
                preds = cluster_function.labels_
            # print("labels num:", len(np.unique(preds))) 
            ar = metrics.adjusted_rand_score(labels, preds)
            ami = metrics.adjusted_mutual_info_score(labels, preds)
            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time: # Luckily, it won't happens in our experiment.
                result_time[index, s] = time_taken
                result_ar[index, s] = ar
                result_ami[index, s] = ami
                return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_time.flatten()]).T, columns=['x','y']), \
                       pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_ar.flatten()]).T, columns=['x','y']), \
                       pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_ami.flatten()]).T, columns=['x','y'])
            else:
                result_time[index, s] = time_taken
                result_ar[index, s] = ar
                result_ami[index, s] = ami

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_time.flatten()]).T, columns=['x','y']), \
           pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_ar.flatten()]).T, columns=['x','y']), \
           pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size), result_ami.flatten()]).T, columns=['x','y'])



def rn_gaussian_dim():
    warnings.filterwarnings("ignore")
    sns.set_context('poster')
    sns.set_palette('Paired', 10)
    sns.set_color_codes()
    dataset_dimensions = np.hstack([np.arange(1, 11) * 10])

    np.random.seed(0)
    with threadpool_limits(limits=1, user_api='blas'):
        k_means = sklearn.cluster.KMeans(n_clusters=10, init='k-means++')
        k_means_time, k_means_ar, k_means_ami = benchmark_algorithm_tdim(dataset_dimensions, k_means, (), {})

        dbscan = sklearn.cluster.DBSCAN(eps=10, min_samples=1, n_jobs=1, algorithm='ball_tree')
        dbscan_btree_time, dbscan_btree_ar, dbscan_btree_ami = benchmark_algorithm_tdim(dataset_dimensions, dbscan, (), {})

        dbscan = sklearn.cluster.DBSCAN(eps=10, min_samples=1, n_jobs=1, algorithm='kd_tree')
        dbscan_kdtree_time, dbscan_kdtree_ar, dbscan_kdtree_ami = benchmark_algorithm_tdim(dataset_dimensions, dbscan, (), {})

        hdbscan_ = hdbscan.HDBSCAN(algorithm='best', core_dist_n_jobs=1)
        hdbscan_time, hdbscan_ar, hdbscan_ami = benchmark_algorithm_tdim(dataset_dimensions, hdbscan_, (), {})

        classix = CLASSIX(sorting='pca', radius=0.3, minPts=5, group_merging='distance', verbose=0) 
        classix_time, classix_ar, classix_ami = benchmark_algorithm_tdim(dataset_dimensions, classix, (), {})
        
        quicks = QuickshiftPP(k=20, beta=0.7)
        quicks_time, quicks_ar, quicks_ami = benchmark_algorithm_tdim(dataset_dimensions, quicks, (), {}, algorithm='Quickshift++')

    k_means_time.to_csv("results/exp1/gd_kmeans_time.csv",index=False)
    dbscan_kdtree_time.to_csv("results/exp1/gd_dbscan_kdtree_time.csv",index=False)
    dbscan_btree_time.to_csv("results/exp1/gd_dbscan_btree_time.csv",index=False)
    hdbscan_time.to_csv("results/exp1/gd_hdbscan_time.csv",index=False)
    classix_time.to_csv("results/exp1/gd_classix_time.csv",index=False)
    quicks_time.to_csv("results/exp1/gd_quicks_time.csv",index=False)
    
    k_means_ar.to_csv("results/exp1/gd_kmeans_ar.csv",index=False)
    dbscan_kdtree_ar.to_csv("results/exp1/gd_dbscan_kdtree_ar.csv",index=False)
    dbscan_btree_ar.to_csv("results/exp1/gd_dbscan_btree_ar.csv",index=False)
    hdbscan_ar.to_csv("results/exp1/gd_hdbscan_ar.csv",index=False)
    classix_ar.to_csv("results/exp1/gd_classix_ar.csv",index=False)
    quicks_ar.to_csv("results/exp1/gd_quicks_ar.csv",index=False)

    

def rn_gaussian_size():
    warnings.filterwarnings("ignore")
    sns.set_context('poster')
    sns.set_palette('Paired', 10)
    sns.set_color_codes()
    np.random.seed(0)
    dataset_sizes = np.hstack([np.arange(1, 11) * 5000])

    np.random.seed(0)
    with threadpool_limits(limits=1, user_api='blas'):
        k_means = sklearn.cluster.KMeans(n_clusters=10, init='k-means++')
        k_means_time, k_means_ar, k_means_ami = benchmark_algorithm_tsize(dataset_sizes, k_means, (), {})

        dbscan = sklearn.cluster.DBSCAN(eps=3, min_samples=1, n_jobs=1, algorithm='ball_tree')
        dbscan_btree_time, dbscan_btree_ar, dbscan_btree_ami = benchmark_algorithm_tsize(dataset_sizes, dbscan, (), {})

        dbscan = sklearn.cluster.DBSCAN(eps=3, min_samples=1, n_jobs=1, algorithm='kd_tree')
        dbscan_kdtree_time, dbscan_kdtree_ar, dbscan_kdtree_ami = benchmark_algorithm_tsize(dataset_sizes, dbscan, (), {})

        hdbscan_ = hdbscan.HDBSCAN(algorithm='best', core_dist_n_jobs=1)
        hdbscan_time, hdbscan_ar, hdbscan_ami = benchmark_algorithm_tsize(dataset_sizes, hdbscan_, (), {})

        classix = CLASSIX(sorting='pca', radius=0.3, minPts=5, group_merging='distance', verbose=0) 
        classix_time, classix_ar, classix_ami = benchmark_algorithm_tsize(dataset_sizes, classix, (), {})
        
        quicks = QuickshiftPP(k=20, beta=0.7)
        quicks_time, quicks_ar, quicks_ami = benchmark_algorithm_tsize(dataset_sizes, quicks, (), {}, algorithm='Quickshift++')


    k_means_time.to_csv("results/exp1/gs_kmeans_time.csv",index=False)
    dbscan_kdtree_time.to_csv("results/exp1/gs_dbscan_kdtree_time.csv",index=False)
    dbscan_btree_time.to_csv("results/exp1/gs_dbscan_btree_time.csv",index=False)
    hdbscan_time.to_csv("results/exp1/gs_hdbscan_time.csv",index=False)
    classix_time.to_csv("results/exp1/gs_classix_time.csv",index=False)
    quicks_time.to_csv("results/exp1/gs_quicks_time.csv",index=False)
    
    k_means_ar.to_csv("results/exp1/gs_kmeans_ar.csv",index=False)
    dbscan_kdtree_ar.to_csv("results/exp1/gs_dbscan_kdtree_ar.csv",index=False)
    dbscan_btree_ar.to_csv("results/exp1/gs_dbscan_btree_ar.csv",index=False)
    hdbscan_ar.to_csv("results/exp1/gs_hdbscan_ar.csv",index=False)
    classix_ar.to_csv("results/exp1/gs_classix_ar.csv",index=False)
    quicks_ar.to_csv("results/exp1/gs_quicks_ar.csv",index=False)
    


def run_gassian_plot():
    # -------------------------------dim
    k_means_time = pd.read_csv("results/exp1/gd_kmeans_time.csv")
    dbscan_kdtree_time = pd.read_csv("results/exp1/gd_dbscan_kdtree_time.csv")
    dbscan_btree_time = pd.read_csv("results/exp1/gd_dbscan_btree_time.csv")
    hdbscan_time = pd.read_csv("results/exp1/gd_hdbscan_time.csv")
    classix_time = pd.read_csv("results/exp1/gd_classix_time.csv")
    quicks_time = pd.read_csv("results/exp1/gd_quicks_time.csv")
    
    k_means_ar = pd.read_csv("results/exp1/gd_kmeans_ar.csv")
    dbscan_kdtree_ar = pd.read_csv("results/exp1/gd_dbscan_kdtree_ar.csv")
    dbscan_btree_ar = pd.read_csv("results/exp1/gd_dbscan_btree_ar.csv")
    hdbscan_ar = pd.read_csv("results/exp1/gd_hdbscan_ar.csv")
    classix_ar = pd.read_csv("results/exp1/gd_classix_ar.csv")
    quicks_ar = pd.read_csv("results/exp1/gd_quicks_ar.csv")
    
    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'
    # plt.rc('font', family='serif')

    ax = sns.lineplot(data=k_means_time, x="x", y="y", marker='v', markersize=13, label='k-means++', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=dbscan_kdtree_time, x="x", y="y", marker='s', markersize=13, label='DBSCAN (kd-tree)', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=dbscan_btree_time, x="x", y="y", marker='o', markersize=13, label='DBSCAN (ball tree)', linestyle=":", linewidth=6)
    ax = sns.lineplot(data=hdbscan_time, x="x", y="y", marker='<', markersize=13, label='HDBSCAN', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=classix_time, x="x", y="y", marker='*', markersize=17, label='CLASSIX', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=quicks_time, x="x", y="y", marker='p', markersize=17, label='Quickshift++', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=6)
    
    ax.set(xlabel='dimensions', ylabel='time (s)', title="Gaussian blobs (n=10000)")
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/gaussian_dim_time.pdf', bbox_inches='tight')

    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'
    # plt.rc('font', family='serif')

    ax = sns.lineplot(data=k_means_ar, x="x", y="y", marker='v', markersize=13, label='k-means++', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=dbscan_kdtree_ar, x="x", y="y", marker='s', markersize=13, label='DBSCAN (kd-tree)', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=dbscan_btree_ar, x="x", y="y", marker='o', markersize=13, label='DBSCAN (ball tree)', linestyle=":", linewidth=6)
    ax = sns.lineplot(data=hdbscan_ar, x="x", y="y", marker='<', markersize=13, label='HDBSCAN', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=classix_ar, x="x", y="y", marker='*', markersize=17, label='CLASSIX', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=quicks_ar, x="x", y="y", marker='p', markersize=17, label='Quickshift++', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=6)
    
    ax.set(xlabel='dimensions', ylabel='adjusted Rand index', title="Gaussian blobs (n=10000)")
    ax.set(ylim=(-.1, 1.1))
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/gaussian_dim_ar.pdf', bbox_inches='tight')
    
    # -------------------------------size
    k_means_time = pd.read_csv("results/exp1/gs_kmeans_time.csv")
    dbscan_kdtree_time = pd.read_csv("results/exp1/gs_dbscan_kdtree_time.csv")
    dbscan_btree_time = pd.read_csv("results/exp1/gs_dbscan_btree_time.csv")
    hdbscan_time = pd.read_csv("results/exp1/gs_hdbscan_time.csv")
    classix_time = pd.read_csv("results/exp1/gs_classix_time.csv")
    quicks_time = pd.read_csv("results/exp1/gs_quicks_time.csv")
    
    k_means_ar = pd.read_csv("results/exp1/gs_kmeans_ar.csv")
    dbscan_kdtree_ar = pd.read_csv("results/exp1/gs_dbscan_kdtree_ar.csv")
    dbscan_btree_ar = pd.read_csv("results/exp1/gs_dbscan_btree_ar.csv")
    hdbscan_ar = pd.read_csv("results/exp1/gs_hdbscan_ar.csv")
    classix_ar = pd.read_csv("results/exp1/gs_classix_ar.csv")
    quicks_ar = pd.read_csv("results/exp1/gs_quicks_ar.csv")
    
    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'

    ax = sns.lineplot(data=k_means_time, x="x", y="y", marker='v', markersize=13, label='k-means++', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=dbscan_kdtree_time, x="x", y="y", marker='s', markersize=13, label='DBSCAN (kd-tree)', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=dbscan_btree_time, x="x", y="y", marker='o', markersize=13, label='DBSCAN (ball tree)', linestyle=":", linewidth=6)
    ax = sns.lineplot(data=hdbscan_time, x="x", y="y", marker='<', markersize=13, label='HDBSCAN', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=classix_time, x="x", y="y", marker='*', markersize=17, label='CLASSIX', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=quicks_time, x="x", y="y", marker='p', markersize=17, label='Quickshift++', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=6)
    
    ax.set(xlabel='data size', ylabel='time (s)', title="Gaussian blobs (dim=10)")
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/gaussian_size_time.pdf', bbox_inches='tight')

    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'

    ax = sns.lineplot(data=k_means_ar, x="x", y="y", marker='v', markersize=13, label='k-means++', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=dbscan_kdtree_ar, x="x", y="y", marker='s', markersize=13, label='DBSCAN (kd-tree)', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=dbscan_btree_ar, x="x", y="y", marker='o', markersize=13, label='DBSCAN (ball tree)', linestyle=":", linewidth=6)
    ax = sns.lineplot(data=hdbscan_ar, x="x", y="y", marker='<', markersize=13, label='HDBSCAN', linestyle="-", linewidth=6)
    ax = sns.lineplot(data=classix_ar, x="x", y="y", marker='*', markersize=17, label='CLASSIX', linestyle="--", linewidth=6)
    ax = sns.lineplot(data=quicks_ar, x="x", y="y", marker='p', markersize=17, label='Quickshift++', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=6)
    
    ax.set(xlabel='data size', ylabel='adjusted Rand index', title="Gaussian blobs (dim=10)")
    ax.set(ylim=(0, 1.1))
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/gaussian_size_ar.pdf', bbox_inches='tight')
    
    
    
def run_comp_sort():

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

        size_range = np.arange(5000, 51000, 5000)
        dim_range = np.hstack([np.arange(1, 11) * 10])
        # early stopping

        nr_dist_size = []
        nr_dist_size_n = []
        nr_dist_dim = []
        nr_dist_dim_n = []
        # no early stopping

        for i in size_range:
            X, y = sklearn.datasets.make_blobs(n_samples=i, centers=10, n_features=10,
                              random_state=0)
            # X = (X - X.mean())/X.std()
            #X_orthant = shift_scale(X, "norm-orthant")
            #X_mean = shift_scale(X, "norm-mean")
            X_pca = shift_scale(X, "pca")
            #labels_no,sp_no, nr_dist_no = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=True)
            #labels_nm, sp_nm, nr_dist_nm = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=True)
            labels_pca, sp_pca, nr_dist_pca = aggregate(X_pca, tol=0.3, sorting='pca', early_stopping=True)

            #nlabels_no, nsp_no, nnr_dist_no = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=False)
            #nlabels_nm, nsp_nm, nnr_dist_nm = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=False)
            nlabels_pca, nsp_pca, nnr_dist_pca = aggregate(X_pca, tol=0.3, sorting='pca', early_stopping=False)
            nr_dist_size.append(nr_dist_pca/i)
            nr_dist_size_n.append(nnr_dist_pca/i)

        for i in dim_range:
            X, y = sklearn.datasets.make_blobs(n_samples=10000, centers=10, n_features=i,
                              random_state=0)
            # X = (X - X.mean())/X.std()
            #X_orthant = shift_scale(X, "norm-orthant")
            #X_mean = shift_scale(X, "norm-mean")
            X_pca = shift_scale(X, "pca")
            #labels_no,sp_no, nr_dist_no = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=True)
            #labels_nm, sp_nm, nr_dist_nm = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=True)
            labels_pca, sp_pca, nr_dist_pca = aggregate(X_pca, tol=0.3, sorting='pca', early_stopping=True)

            #nlabels_no, nsp_no, nnr_dist_no = aggregate(X_orthant, tol=0.5, sorting='norm-orthant', early_stopping=False)
            #nlabels_nm, nsp_nm, nnr_dist_nm = aggregate(X_mean, tol=0.5, sorting='norm-mean', early_stopping=False)
            nlabels_pca, nsp_pca, nnr_dist_pca = aggregate(X_pca, tol=0.3, sorting='pca', early_stopping=False)
            nr_dist_dim.append(nr_dist_pca/10000)
            nr_dist_dim_n.append(nnr_dist_pca/10000)        
        return nr_dist_size, nr_dist_size_n, nr_dist_dim, nr_dist_dim_n

    nr_dist_size, nr_dist_size_n, nr_dist_dim, nr_dist_dim_n = count_distance()
    
    et = pd.DataFrame()
    et['x'], et['y'] = np.arange(5000, 51000, 5000), nr_dist_size

    net = pd.DataFrame()
    net['x'], net['y'] = np.arange(5000, 51000, 5000), nr_dist_size_n
    
    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'

    ax = sns.lineplot(data=et, x='x', y='y', marker='o', markersize=13, label='with early termination', linestyle="-", color='indigo', linewidth=6)
    ax = sns.lineplot(data=net, x='x', y='y', marker='s', markersize=13, label='without early termination', linestyle="--", color='purple',linewidth=6)
    plt.ylim(0, 110)
    ax.set(xlabel='data size', ylabel='avg dist pp', title="Gaussian blobs (dim=10)")
    
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/comp_sort_size.pdf', bbox_inches='tight')

    et = pd.DataFrame()
    et['x'], et['y'] = np.hstack([np.arange(1, 11) * 10]), nr_dist_dim

    net = pd.DataFrame()
    net['x'], net['y'] = np.hstack([np.arange(1, 11) * 10]), nr_dist_dim_n
    
    plt.figure(figsize=(12,8))
    plt.style.use('bmh')
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'
    
    ax = sns.lineplot(data=et, x='x', y='y', marker='o', markersize=13, label='with early termination', linestyle="-", color='indigo', linewidth=6)
    ax = sns.lineplot(data=net, x='x', y='y', marker='s', markersize=13, label='without early termination', linestyle="--", color='purple',linewidth=6)

    ax.set(xlabel='dimensions', ylabel='avg dist pp', title="Gaussian blobs (n=10000)")
    
    plt.tick_params(axis='both',  labelsize=22)
    plt.savefig('results/exp1/comp_sort_dim.pdf', bbox_inches='tight')
