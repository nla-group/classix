import cv2
import os
import copy
import numpy as np
import pandas as pd
from classix import CLASSIX
import matplotlib.pyplot as plt
from collections import OrderedDict

def order_pics(figs):
    images = list()
    labels = list()
    for i in range(40):
        num = i + 1
        for img in figs:
            try:
                if int(img.split('_')[1].replace('.jpg','')) == num:
                    images.append(img)
                    labels.append(num)
            except:
                pass
    return images, labels
    
def load_images(folder, shape=(100, 100)):
    images = list()
    figs = os.listdir(folder)
    figs, targets= order_pics(figs)
    for filename in figs:
        img = cv2.imread(os.path.join(folder,filename)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # transform to grayscale
        img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA) # resize to 80x80
        if img is not None:
            images.append(img)
    images, targets = np.array(images), np.array(targets) - 1
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
    return images, targets

def reassign_labels(labels, fix=None):
    if fix != None:
        value_count = pd.Series(labels[labels != fix]).value_counts()
    else:
        value_count = pd.Series(labels).value_counts()
    change = dict(zip(value_count.index, np.arange(value_count.shape[0])))
    change[fix] = value_count.shape[0]
    clabels = copy.deepcopy(labels)
    for i in range(len(labels)):
        labels[i] = change[labels[i]]
    return labels


def rn_facial_cluster():
    clear_cmaps =  [ 'viridis', 'cividis', 'pink', 'inferno', 'winter', 
                    'copper','magma', 'autumn', 'summer', 'hot', 'plasma', 'Wistia',  
                    'afmhot','spring']

    cmaps = OrderedDict()

    cmaps['Sequential'] = [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    cmaps['Diverging'] = [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    cmaps['Sequential_2'] = [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper']

    cmaps['Perceptually Uniform Sequential'] = [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

    cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                            'Dark2', 'Set1', 'Set2', 'Set3',
                            'tab10', 'tab20', 'tab20b', 'tab20c']

    cmaps['Miscellaneous'] = [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
                'gist_ncar']

    folder = 'data/OlivettiFaces'
    data, targets = load_images(folder)

    test = data[targets < 10]  
    test_targets = targets[targets < 10]  
    classix = CLASSIX(sorting='pca', radius=0.6, minPts=7, verbose=0, group_merging='distance')
    classix.fit_transform(test)

    labels = np.array(classix.labels_)
    outlier_label = max(labels) + 1
    color_outlier = 'gray'
    # print("outliers color and label:{}, {}".format(color_outlier, outlier_label))
    outliers = [i for i in range(len(classix.agg_labels_)) if classix.agg_labels_[i] in classix.group_outliers_]
    labels[outliers] = max(labels) + 1
    # for i in range(10):
    #     print([labels[i*10 + j] for j in range(10)])

    image_shape = (100, 100)
    fig, axs = plt.subplots(10, 10, figsize=(15,15))
    for i in range(10):
        for j in range(10):
            axs[i,j].axis("off")
            if labels[i*10 + j] != outlier_label:
                axs[i,j].imshow(test[i*10 + j].reshape(image_shape),
                       cmap=clear_cmaps[labels[i*10 + j]],
                       interpolation="nearest")
            else:
                axs[i,j].imshow(test[i*10 + j].reshape(image_shape),
                       cmap=color_outlier,
                       interpolation="nearest")

    plt.savefig('results/exp6/faces1.pdf', bbox_inches='tight')
    
    image_shape = (100, 100)
    fig, axs = plt.subplots(5, 20, figsize=(30,7.5))
    for i in range(5):
        for j in range(20):
            axs[i,j].axis("off")
            if labels[i*20 + j] != outlier_label:
                axs[i,j].imshow(test[i*20 + j].reshape(image_shape),
                       cmap=clear_cmaps[labels[i*20 + j]],
                       interpolation="nearest")
            else:
                axs[i,j].imshow(test[i*20 + j].reshape(image_shape),
                       cmap=color_outlier,
                       interpolation="nearest")

    plt.savefig('results/exp6/faces1_rectangle.pdf', bbox_inches='tight')
    # plt.show()
