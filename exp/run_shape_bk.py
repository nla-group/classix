import time
import random
import hdbscan
import numpy as np
import pandas as pd
from sklearn import metrics
from classix import CLASSIX
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from quickshift.QuickshiftPP import *
from sklearn.cluster import MeanShift
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

shape_sets = ['Aggregation.txt', 'Compound.txt', 
              'D31.txt', 'Flame.txt', 
              'Jain.txt', 'Pathbased.txt', 
              'R15.txt', 'Spiral.txt']

def load_data(file):
    data = pd.read_csv(file,sep="\\s+", header = None)
    return data


def rn_cluster_shape():
    plt.style.use('bmh')
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    # a higher Silhouette Coefficient score relates to a model with better defined clusters.
    # a lower Davies-Bouldin index relates to a model with better separation between the clusters.
    # a higher Calinski-Harabasz score relates to a model with better defined clusters.
    # a higher adjusted rand score relates to a better model
    # the dim sets do not provide groud truth label. so some of the metric can not apply

    classix_den_results = list()
    classix_dist_results = list()
    kmeans_results = list()
    dbscan_results = list()
    hdbscan_results = list()
    quicks_results = list()
    meanshift_results = list()

    classix_den_nr_dist = list()
    classix_dist_nr_dist = list()

    # classix_den_sh_scores = list()
    # classix_dist_sh_scores = list()
    # kmeans_sh_scores = list()
    # dbscan_sh_scores = list()
    # hdbscan_sh_scores = list()
    # quicks_sh_scores = list()
    # meanshift_sh_scores = list()

    # classix_den_db_scores = list()
    # classix_dist_db_scores = list()
    # kmeans_db_scores = list()
    # dbscan_db_scores = list()
    # hdbscan_db_scores = list()
    # quicks_db_scores = list()
    # meanshift_db_scores = list()
    
    # classix_den_ch_scores = list()
    # classix_dist_ch_scores = list()
    # kmeans_ch_scores = list()
    # dbscan_ch_scores = list()
    # hdbscan_ch_scores = list()
    # quicks_ch_scores = list()
    # meanshift_ch_scores = list()
    
    classix_den_ri_scores = list()
    classix_dist_ri_scores = list()
    kmeans_ri_scores = list()
    dbscan_ri_scores = list()
    hdbscan_ri_scores = list()
    quicks_ri_scores = list()
    meanshift_ri_scores = list()
    
    classix_den_mi_scores = list()
    classix_dist_mi_scores = list()
    kmeans_mi_scores = list()
    dbscan_mi_scores = list()
    hdbscan_mi_scores = list()
    quicks_mi_scores = list()
    meanshift_mi_scores = list()

    classix_den_time_scores = list()
    classix_dist_time_scores = list()
    kmeans_time_scores = list()
    dbscan_time_scores = list()
    hdbscan_time_scores = list()
    quicks_time_scores = list()
    meanshift_time_scores = list()

    den_tols = [0.25, 0.125, 0.05, 0.2, 0.425, 0.25, 0.135, 0.325]
    den_minPts = [6, 6, 10, 3, 0, 9, 4, 0]
    
    dist_tols = [0.1, 0.1, 0.025, 0.2, 0.3, 0.15, 0.15, 0.25]
    dist_minPts = [0, 2, 21, 9, 0, 10, 4, 5]
    
    eps = [0.125, 0.2 ,0.08, 0.275, 0.3, 0.225, 0.1, 0.325]
    min_samples = [5, 6, 3, 5, 4, 8, 5, 5]
    
    min_cluster_size = [12, 4, 6, 3, 16, 15, 5, 2]

    quicks_k = [10, 12, 10, 10, 10, 10, 10, 8]
    quicks_beta = [.325, .625, .625, .575, .650, .425, .5, .8]
    
    bandwidth = [0.475, 0.625, 0.2, 0.825, 0.7, 0.7, 0.25, 0.475]
    
    sample_size = 10

    with threadpool_limits(limits=1, user_api='blas'):
        for i in range(len(shape_sets)):
            data = load_data("data/Shape sets/" + shape_sets[i])

            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                # classix will normalize during the clustering, so we arrange the z-score nomarlization to others after that.
                classix_dist = CLASSIX(sorting='pca', radius=dist_tols[i], group_merging='distance', minPts=dist_minPts[i], verbose=0, post_alloc=True)
                classix_dist.fit_transform(data[[0,1]])
                et = time.time()
                unit_time = unit_time + et - st
            classix_dist_time_scores.append(unit_time/sample_size)
            
            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                # classix will normalize during the clustering, so it is not necessarily apply z-score normalization 
                # we arrange the z-score nomarlization to other algorithm after that.
                classix_den = CLASSIX(sorting='pca', radius=den_tols[i], group_merging='density', minPts=den_minPts[i], verbose=0, post_alloc=True)
                classix_den.fit_transform(data[[0,1]])
                et = time.time()
                unit_time = unit_time + et - st
            classix_den_time_scores.append(unit_time/sample_size)

            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                _data = (data[[0,1]] - data[[0,1]].mean(axis=0)) / data[[0,1]].std(axis=0)
                kmeans = KMeans(n_clusters=len(np.unique(data[2])), random_state=0)
                kmeans.fit(_data)
                et = time.time()
                unit_time = unit_time + et - st
            kmeans_time_scores.append(unit_time/sample_size)

            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                _data = (data[[0,1]] - data[[0,1]].mean(axis=0)) / data[[0,1]].std(axis=0)
                dbscan = DBSCAN(eps=eps[i], min_samples=min_samples[i])
                dbscan.fit(_data)
                et = time.time()
                unit_time = unit_time + et - st
            dbscan_time_scores.append(unit_time/sample_size)
            
            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                _data = (data[[0,1]] - data[[0,1]].mean(axis=0)) / data[[0,1]].std(axis=0)
                _hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size[i], core_dist_n_jobs=1)
                hdbscan_labels = _hdbscan.fit_predict(_data)
                et = time.time()
                unit_time = unit_time + et - st
            hdbscan_time_scores.append(unit_time/sample_size)
            
            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                _data = (data[[0,1]] - data[[0,1]].mean(axis=0)) / data[[0,1]].std(axis=0)
                quicks = QuickshiftPP(k=quicks_k[i], beta=quicks_beta[i])
                quicks.fit(_data.values.copy(order='C'))
                quicks_labels = quicks.memberships
                et = time.time()
                unit_time = unit_time + et - st
            quicks_time_scores.append(unit_time/sample_size)
            
            unit_time = 0
            for _iter in range(sample_size):
                np.random.seed(_iter)
                st = time.time()
                _data = (data[[0,1]] - data[[0,1]].mean(axis=0)) / data[[0,1]].std(axis=0)
                meanshift = MeanShift(bandwidth=bandwidth[i])
                meanshift.fit(_data)
                et = time.time()
                unit_time = unit_time + et - st
            meanshift_time_scores.append(unit_time/sample_size)
            
            # classix_dist_sh = metrics.silhouette_score(data[[0,1]], classix_dist.labels_)
            # classix_dist_db = metrics.davies_bouldin_score(data[[0,1]], classix_dist.labels_)
            # classix_dist_ch = metrics.calinski_harabasz_score(data[[0,1]], classix_dist.labels_)
            classix_dist_ri = metrics.adjusted_rand_score(data[2], classix_dist.labels_) 
            classix_dist_mi = metrics.adjusted_mutual_info_score(data[2], classix_dist.labels_) 
            classix_dist_results.append(classix_dist.labels_)
            classix_dist_nr_dist.append(classix_dist.dist_nr)
            
            # classix_den_sh = metrics.silhouette_score(data[[0,1]], classix_den.labels_)
            # classix_den_db = metrics.davies_bouldin_score(data[[0,1]], classix_den.labels_)
            # classix_den_ch = metrics.calinski_harabasz_score(data[[0,1]], classix_den.labels_)
            classix_den_ri = metrics.adjusted_rand_score(data[2], classix_den.labels_) 
            classix_den_mi = metrics.adjusted_mutual_info_score(data[2], classix_den.labels_) 
            classix_den_results.append(classix_den.labels_)
            classix_den_nr_dist.append(classix_den.dist_nr)

            # kmeans_sh = metrics.silhouette_score(data[[0,1]], kmeans.labels_)
            # kmeans_db = metrics.davies_bouldin_score(data[[0,1]], kmeans.labels_)
            # kmeans_ch = metrics.calinski_harabasz_score(data[[0,1]], kmeans.labels_)
            kmeans_ri = metrics.adjusted_rand_score(data[2], kmeans.labels_)
            kmeans_mi = metrics.adjusted_mutual_info_score(data[2], kmeans.labels_) 
            kmeans_results.append(kmeans.labels_)

            # dbscan_sh = metrics.silhouette_score(data[[0,1]], dbscan.labels_)
            # dbscan_db= metrics.davies_bouldin_score(data[[0,1]], dbscan.labels_)
            # dbscan_ch = metrics.calinski_harabasz_score(data[[0,1]], dbscan.labels_)
            dbscan_ri = metrics.adjusted_rand_score(data[2], dbscan.labels_)
            dbscan_mi = metrics.adjusted_mutual_info_score(data[2], dbscan.labels_) 
            dbscan_results.append(dbscan.labels_)

            # hdbscan_sh = metrics.silhouette_score(data[[0,1]], hdbscan_labels)
            # hdbscan_db= metrics.davies_bouldin_score(data[[0,1]], hdbscan_labels)
            # hdbscan_ch = metrics.calinski_harabasz_score(data[[0,1]], hdbscan_labels)
            hdbscan_ri = metrics.adjusted_rand_score(data[2], hdbscan_labels)
            hdbscan_mi = metrics.adjusted_mutual_info_score(data[2], hdbscan_labels) 
            hdbscan_results.append(hdbscan_labels)
            
            # quicks_sh = metrics.silhouette_score(data[[0,1]], quicks_labels)
            # quicks_db= metrics.davies_bouldin_score(data[[0,1]], quicks_labels)
            # quicks_ch = metrics.calinski_harabasz_score(data[[0,1]], quicks_labels)
            quicks_ri = metrics.adjusted_rand_score(data[2], quicks_labels)
            quicks_mi = metrics.adjusted_mutual_info_score(data[2], quicks_labels) 
            quicks_results.append(quicks_labels)
            
            # meanshift_sh = metrics.silhouette_score(data[[0,1]], meanshift.labels_)
            # meanshift_db= metrics.davies_bouldin_score(data[[0,1]], meanshift.labels_)
            # meanshift_ch = metrics.calinski_harabasz_score(data[[0,1]], meanshift.labels_)
            meanshift_ri = metrics.adjusted_rand_score(data[2], meanshift.labels_)
            meanshift_mi = metrics.adjusted_mutual_info_score(data[2], meanshift.labels_) 
            meanshift_results.append(meanshift.labels_)
            
            # classix_den_sh_scores.append(classix_den_sh)
            # classix_dist_sh_scores.append(classix_dist_sh)
            # kmeans_sh_scores.append(kmeans_sh)
            # dbscan_sh_scores.append(dbscan_sh)
            # hdbscan_sh_scores.append(hdbscan_sh)
            # quicks_sh_scores.append(quicks_sh)
            # meanshift_sh_scores.append(meanshift_sh)

            # classix_den_db_scores.append(classix_den_db)
            # classix_dist_db_scores.append(classix_dist_db)
            # kmeans_db_scores.append(kmeans_db)
            # dbscan_db_scores.append(dbscan_db)
            # hdbscan_db_scores.append(hdbscan_db)
            # quicks_db_scores.append(quicks_db)
            # meanshift_db_scores.append(meanshift_db)

            # classix_den_ch_scores.append(classix_den_ch)
            # classix_dist_ch_scores.append(classix_dist_ch)
            # kmeans_ch_scores.append(kmeans_ch)
            # dbscan_ch_scores.append(dbscan_ch)
            # hdbscan_ch_scores.append(hdbscan_ch)
            # quicks_ch_scores.append(quicks_ch)
            # meanshift_ch_scores.append(meanshift_ch)

            classix_den_ri_scores.append(classix_den_ri)
            classix_dist_ri_scores.append(classix_dist_ri)
            kmeans_ri_scores.append(kmeans_ri)
            dbscan_ri_scores.append(dbscan_ri)
            hdbscan_ri_scores.append(hdbscan_ri)
            quicks_ri_scores.append(quicks_ri)
            meanshift_ri_scores.append(meanshift_ri)

            classix_den_mi_scores.append(classix_den_mi)
            classix_dist_mi_scores.append(classix_dist_mi)
            kmeans_mi_scores.append(kmeans_mi)
            dbscan_mi_scores.append(dbscan_mi)
            hdbscan_mi_scores.append(hdbscan_mi)
            quicks_mi_scores.append(quicks_mi)
            meanshift_mi_scores.append(meanshift_mi)

    plt.figure(figsize=(28, 32))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    plot_num=1
    
    for i_dataset in range(len(shape_sets)):
        data = load_data("data/Shape sets/" + shape_sets[i_dataset]).values[:,:2]
        data = (data - data.mean(axis=0))/data.std(axis=0)

        clustering_algorithms_results = (
            ('k-means++', kmeans_results),
            ('Mean shift', meanshift_results),
            ('DBSCAN', dbscan_results),
            ('HDBSCAN', hdbscan_results),
            ('Quickshift++', quicks_results),
            ('CLASSIX - distance', classix_dist_results),
            ('CLASSIX - density', classix_den_results)
        )

        clustering_algorithms_time = {
            'k-means++': kmeans_time_scores,
            'Mean shift': meanshift_time_scores,
            'DBSCAN': dbscan_time_scores,
            'HDBSCAN': hdbscan_time_scores,
            'Quickshift++': quicks_time_scores,
            'CLASSIX - distance': classix_dist_time_scores,
            'CLASSIX - density': classix_den_time_scores
        }

        for name, results in clustering_algorithms_results:
            plt.subplot(len(shape_sets), len(clustering_algorithms_results), plot_num)
            # plt.rc('font', family='serif')
            
            prediction = results[i_dataset]
            plt.scatter(data[prediction!=-1, 0], data[prediction!=-1, 1], c=prediction[prediction!=-1], cmap='tab20') # alternative: 'Set1', 'Set2', 'tab10', 'tab20'
            plt.scatter(data[prediction==-1, 0], data[prediction==-1, 1], color='k')
            plt.xlim(-2.5, 2.5)
            plt.ylim(-3, 3)
            plt.xticks(())
            plt.yticks(())
            
            if name == "CLASSIX - distance":
                plt.text(.01, .01, ('%.2f' % np.round(classix_dist_nr_dist[i_dataset]/len(data),2)), #.lstrip('0'),
                         transform=plt.gca().transAxes, size=28,
                         horizontalalignment='left')
            elif name == "CLASSIX - density":
                plt.text(.01, .01, ('%.2f' % np.round(classix_den_nr_dist[i_dataset]/len(data),2)), #.lstrip('0'),
                         transform=plt.gca().transAxes, size=28,
                         horizontalalignment='left')

            if i_dataset == 0:
                plt.title(name, size=27)
                
            plt.text(.99, .01, ('%.2fs' % np.round(clustering_algorithms_time[name][i_dataset],2)), #.lstrip('0'),
                     transform=plt.gca().transAxes, size=28,
                     horizontalalignment='right')

            plot_num = plot_num + 1
            
    plt.savefig('results/exp4/shape_sets_general'+'.pdf', bbox_inches='tight')
    # plt.show()
    
    stack_data_n = ['Aggregation', 'Compound', 
                  'D31', 'Flame', 
                  'Jain', 'Pathbased', 
                  'R15', 'Spiral']*7
    
    cluster_n = ['k-means++']*8 + ['Mean shift']*8 + ['DBSCAN']*8 + ['HDBSCAN']*8 + ['Quickshift++']*8 + ['CLASSIX - distance']*8 + ['CLASSIX - density']*8
    stack_ri = np.hstack([kmeans_ri_scores, meanshift_ri_scores, dbscan_ri_scores, hdbscan_ri_scores, quicks_ri_scores, classix_dist_ri_scores, classix_den_ri_scores])
    stack_mi = np.hstack([kmeans_mi_scores, meanshift_mi_scores, dbscan_mi_scores, hdbscan_mi_scores, quicks_ri_scores, classix_dist_mi_scores, classix_den_mi_scores])
    stack_time = np.hstack([kmeans_time_scores, meanshift_time_scores, dbscan_time_scores, hdbscan_time_scores, quicks_ri_scores, classix_dist_time_scores, classix_den_time_scores])
    SCORE_STORE = pd.DataFrame()
    SCORE_STORE['Dataset'] = stack_data_n
    SCORE_STORE['Clustering'] = cluster_n
    SCORE_STORE['ARI'] = stack_ri
    SCORE_STORE['AMI'] = stack_mi
    SCORE_STORE['Time'] = stack_time
    SCORE_STORE.to_csv('results/exp4/shape_r.csv', index=False)
    
    ai_data = SCORE_STORE[['Dataset','Clustering', 'ARI']]
    dname = ai_data.Dataset.unique()
    cname = ai_data.Clustering.unique()
    index_data = ai_data.groupby(['Dataset', 'Clustering']).mean()['ARI']
    sdata = pd.DataFrame()
    for i in range(len(dname)):
        sdata[dname[i]] =  index_data[dname[i]].values

    sdata.index = index_data[dname[0]].index
    sdata = sdata.T[cname]
    sdata.to_csv('results/exp4/shape_ari.csv')
    
    ai_data = SCORE_STORE[['Dataset','Clustering', 'AMI']]
    dname = ai_data.Dataset.unique()
    cname = ai_data.Clustering.unique()
    index_data = ai_data.groupby(['Dataset', 'Clustering']).mean()['AMI']
    sdata = pd.DataFrame()
    for i in range(len(dname)):
        sdata[dname[i]] =  index_data[dname[i]].values

    sdata.index = index_data[dname[0]].index
    sdata = sdata.T[cname]
    sdata.to_csv('results/exp4/shape_ami.csv')

    
    
    
    
def shape_index_plot():
    data = pd.read_csv("results/exp4/shape_r.csv")
    plt.figure(figsize=(15,12))
    sns.set(font_scale=2, style="whitegrid")
    plt.rcParams['axes.facecolor'] = 'white'
    g = sns.barplot(data=data, x='Dataset',y='ARI', hue='Clustering')
    plt.ylim(0,1.01)
    g.legend(loc='center right', bbox_to_anchor=(0.98, -0.18), ncol=3)
    plt.tight_layout()
    plt.savefig("results/exp4/shape_sets_ri.pdf", bbox_inches='tight')
    # plt.show()

    # data['Dataset'] = shape_sets_name*5
    plt.figure(figsize=(15,12))
    sns.set(font_scale=2, style="whitegrid")
    sns.despine()
    plt.rcParams['axes.facecolor'] = 'white'
    g = sns.barplot(data=data, x='Dataset',y='AMI', hue='Clustering')
    plt.ylim(0,1.01)
    g.legend(loc='center right', bbox_to_anchor=(0.98, -0.18), ncol=3)
    plt.tight_layout()
    plt.savefig("results/exp4/shape_sets_mi.pdf", bbox_inches='tight')
    # plt.show()
    


def shape_pred_test():
    np.random.seed(0)
    data = pd.read_csv("data/Shape sets/complex9.txt", header=None)
    X = data[[0,1]].values
    y = data[[2]].values

    N = len(X)
    seed = np.arange(0, N)
    np.random.shuffle(seed)

    split = 0.9
    X_train, y_train = X[seed[:int(round(N*split))]], y[seed[:int(round(N*split))]]
    X_test, y_test = X[seed[int(round(N*split)):]], y[seed[int(round(N*split)):]]

    clx = CLASSIX(sorting='norm-orthant', radius=0.12, verbose=0,  group_merging='density', minPts=31)
    clx.fit(X_train)
    prediction = clx.predict(X_test)
    
    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='tab10')
    plt.tick_params(axis='both', labelsize=24)
    plt.xticks(np.arange(0, 701, 100))
    plt.yticks(np.arange(0, 501, 100))
    plt.grid(False)
    plt.savefig('results/exp4/complex9_train_groudt.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='tab10')
    plt.tick_params(axis='both', labelsize=24)
    plt.xticks(np.arange(0, 701, 100))
    plt.yticks(np.arange(0, 501, 100))
    plt.grid(False)
    plt.savefig('results/exp4/complex9_test_groudt.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(X_train[:,0], X_train[:,1], c=clx.labels_, cmap='tab10')
    plt.tick_params(axis='both', labelsize=24)
    plt.xticks(np.arange(0, 701, 100))
    plt.yticks(np.arange(0, 501, 100))
    plt.grid(False)
    plt.savefig('results/exp4/complex9_train_classix.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(X_test[:,0], X_test[:,1], c=prediction, cmap='tab10')
    plt.tick_params(axis='both', labelsize=24)
    plt.xticks(np.arange(0, 701, 100))
    plt.yticks(np.arange(0, 501, 100))
    plt.grid(False)
    plt.savefig('results/exp4/complex9_test_classix.pdf', bbox_inches='tight')
    # plt.show()
