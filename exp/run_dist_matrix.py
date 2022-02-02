import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns; sns.set_theme()
seed = 0
np.random.seed(seed)

"""We test without normalization"""


def normalize(data, shift = 'z-score'): 
    if shift not in ['mean', 'min', 'z-score', 'pca']:
        raise ValueError("please enter a correct shift parameter.")

    if shift == 'min':
        _mu = data.min(axis=0)
        _scl = data.std()
        cdata = data / _scl

    elif shift == 'mean':
        _mu = data.mean(axis=0)
        cdata = data - _mu
        _scl = cdata.std()
        cdata = cdata / _scl

    elif shift == 'pca':
        _mu = data.mean(axis=0)
        cdata = data - _mu # mean center
        rds = norm(cdata - _mu, axis=1) # distance of each data point from 0
        _scl = np.median(rds) # 50% of data points are within that radius
        cdata = cdata / _scl

    else: #  shift == 'z-score':
        _mu = data.mean(axis=0)
        _scl = data.std(axis=0)
        cdata = (data - _mu) / _scl

    return cdata, (_mu, _scl)



def sorting(data, sorting='pca'):
    if sorting=='norm-mean':
        data, parameters = normalize(data, shift='mean')
        size = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(size)
    if sorting=='norm-orthant':
        data, parameters = normalize(data, shift='min')
        size = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(size)
    if sorting=='pca':
        # data, parameters = normalize(data, shift='pca')
        pca = PCA(n_components=1)
        size = pca.fit_transform(data).reshape(-1)
        ind = np.argsort(size)

    return data[ind], size



def rn_wine_dataset():
    plt.style.use('ggplot')
    data = pd.read_csv("data/Real_data/Wine.csv")
    X = data.drop(['14'],axis=1).values
    font_scale = 3
    dist_corr = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            dist_corr[j,i] = dist_corr[i,j] = np.linalg.norm(X[i]-X[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.yticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.savefig('results/original_wine.pdf', bbox_inches='tight')
    # plt.show()


    ndata, size_pca = sorting(X, sorting='pca')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.yticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.savefig('results/pca_wine.pdf', bbox_inches='tight')
    # plt.show()

    ndata, size_no = sorting(X, sorting='norm-orthant')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.yticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.savefig('results/norm-orthant_wine.pdf', bbox_inches='tight')
    # plt.show()


    ndata, size_nm = sorting(X, sorting='norm-mean')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.yticks([0,  25,  50,  75, 100, 125, 150, 175])
    plt.savefig('results/norm-mean_wine.pdf', bbox_inches='tight')
    # plt.show()

    sorting_df = pd.DataFrame()
    sorting_df['PCA'] = size_pca
    sorting_df['Norm-mean'] = size_nm
    sorting_df['Norm-orthant'] = size_no

    sns.set(style='ticks', color_codes=True, font_scale=3)
    g = sns.pairplot(sorting_df, corner=True, height=4.2, aspect=1)

    plt.savefig('results/sort_pair_plot_wine.pdf', bbox_inches='tight')
    # plt.show()
    
    
    
def rn_iris_dataset():
    plt.style.use('ggplot')
    data = pd.read_csv("data/Real_data/Iris.csv")
    le = preprocessing.LabelEncoder()
    data['Species'] = le.fit_transform(data['Species'])
    X = data.drop(['Species','Id'],axis=1).values
    y = data['Species'].values
    
    font_scale = 3
    dist_corr = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            dist_corr[j,i] = dist_corr[i,j] = np.linalg.norm(X[i]-X[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.yticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.savefig('results/original_iris.pdf', bbox_inches='tight')
    # plt.show()
    
    ndata, size_pca = sorting(X, sorting='pca')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.yticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.savefig('results/pca_iris.pdf', bbox_inches='tight')
    # plt.show()
    
    ndata, size_no = sorting(X, sorting='norm-orthant')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.yticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.savefig('results/norm-orthant_iris.pdf', bbox_inches='tight')
    # plt.show()
    

    ndata, size_nm = sorting(X, sorting='norm-mean')
    dist_corr_sort = np.zeros((len(ndata), len(ndata)))
    for i in range(len(ndata)):
        for j in range(i, len(ndata)):
            dist_corr_sort[j,i] = dist_corr_sort[i,j] = np.linalg.norm(ndata[i]-ndata[j], ord=2, axis=0)

    sns.set(rc={'figure.figsize':(12,10)}, font_scale=font_scale)
    fig, ax = plt.subplots()
    im = ax.imshow(dist_corr_sort, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.xticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.yticks([0,  20,  40,  60,  80, 100, 120, 140])
    plt.savefig('results/norm-mean_iris.pdf', bbox_inches='tight')
    # plt.show()
    
    
    sorting_df = pd.DataFrame()
    sorting_df['PCA'] = size_pca
    sorting_df['Norm-mean'] = size_nm
    sorting_df['Norm-orthant'] = size_no

    sns.set(style='ticks', color_codes=True, font_scale=3)
    g = sns.pairplot(sorting_df, corner=True, height=4.2, aspect=1)
    plt.savefig('results/sort_pair_plot_iris.pdf', bbox_inches='tight')
    # plt.show()