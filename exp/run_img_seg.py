import cv2
import os
import time
import hdbscan
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from classix import CLASSIX
from quickshift.QuickshiftPP import *
# from classix.aggregation_cm import aggregate
from classix import calculate_cluster_centers, novel_normalization

imagePaths1 = [
    'data/COCO2017/000000002473.jpg',
    'data/COCO2017/000000000009.jpg',
    'data/COCO2017/000000001332.jpg',
    'data/COCO2017/000000000247.jpg',
    'data/COCO2017/000000000034.jpg',
    'data/COCO2017/000000067406.jpg',
    'data/COCO2017/000000000092.jpg',
    'data/COCO2017/000000003255.jpg',
    'data/COCO2017/000000082180.jpg',
    'data/COCO2017/000000000074.jpg',
    'data/COCO2017/000000001451.jpg',
    'data/COCO2017/000000006012.jpg'
]

imagePaths2 = [
    'data/ImageNet/n01496331_9611.jpg', 
    'data/ImageNet/ILSVRC2012_val_00000017.jpg', 
    'data/ImageNet/ILSVRC2012_val_00000029.jpg',
    'data/ImageNet/ILSVRC2012_val_00011129.jpg',
    'data/ImageNet/ILSVRC2012_val_00005026.jpg',
    'data/ImageNet/n01592084_2950.jpg'
]

params1 = {
 'radius': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
 'minPts': [80, 120, 150, 80, 150, 200, 50, 50, 50, 50, 100, 100],
 'eps': [3, 3.5, 2, 2.5, 4, 3, 3, 2, 1.5, 1.5, 1.5, 1],
 'min_samples': [10, 20, 10, 15, 10, 5, 15, 8, 8, 12, 12, 6],
 'min_cluster_size': [200, 60, 125, 120, 35, 50, 80, 50, 50, 50, 15, 50],
 'quicks_k': [450, 100, 215, 220, 150, 70, 95, 125, 135, 85, 95, 75],
 'quicks_beta': [0.3, 0.7, 0.5, 0.4, 0.7, 0.5, 0.9, 0.7, 0.8, 0.7, 0.6, 0.5]
}

params2 = {
'eps': [2, 5, 5, 4, 3, 5],
'min_samples': [15, 5, 10, 9, 8, 6],
'min_cluster_size': [35, 18, 38, 33, 48, 50],
'quicks_k': [60, 50, 180, 470, 60, 70],
'quicks_beta': [0.3, 0.3, 0.5, 0.4, 0.6, 0.5],
'radius': [0.1, 0.2, 0.2, 0.1, 0.2, 0.2],
'minPts': [100, 100, 150, 50, 100, 100]
}

# def rn_img_anime_minpts():
#     file ='data/Saitama_serious_profile.png'
#     image=cv2.imread(file)
#     img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (120,120), interpolation = cv2.INTER_AREA)
#     vectorized = np.float32(img.reshape((-1,3)))
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
#     classix1 = CLASSIX(sorting='pca', radius=0.01, minPts=20, verbose=0, group_merging='distance')
#     classix1.fit_transform(vectorized)
#     classix_centers = np.uint8(classix1.load_cluster_centers())
#     classix_res = classix_centers[classix1.labels_]
#     classix_result_image1 = classix_res.reshape((img.shape))

#     classix2 = CLASSIX(sorting='pca', radius=0.01, minPts=35, verbose=0, group_merging='distance')
#     classix2.fit_transform(vectorized)
#     classix_centers = np.uint8(classix2.load_cluster_centers())
#     classix_res = classix_centers[classix2.labels_]
#     classix_result_image2 = classix_res.reshape((img.shape))

#     classix3 = CLASSIX(sorting='pca', radius=0.01, minPts=50, verbose=0, group_merging='distance')
#     classix3.fit_transform(vectorized)
#     classix_centers = np.uint8(classix3.load_cluster_centers())
#     classix_res = classix_centers[classix3.labels_]
#     classix_result_image3 = classix_res.reshape((img.shape))

#     figure_size = 15
#     plt.figure(figsize=(figure_size,figure_size))
#     plt.subplot(2,3,1),plt.imshow(classix_result_image1)
#     plt.title('clusters = %i' % len(set(classix1.labels_)), fontsize=22), plt.xticks([]), plt.yticks([])
#     plt.subplot(2,3,2),plt.imshow(classix_result_image2)
#     plt.title('clusters = %i' % len(set(classix2.labels_)), fontsize=22), plt.xticks([]), plt.yticks([])
#     plt.subplot(2,3,3),plt.imshow(classix_result_image3)
#     plt.title('clusters = %i' % len(set(classix3.labels_)), fontsize=22), plt.xticks([]), plt.yticks([])
#     plt.savefig('fresults/OPM_denoise_.pdf', bbox_inches='tight')
#     plt.show()

    

# def rn_img_real_tol():
#     files = ['n01496331_9611.jpg', 
#              'ILSVRC2012_val_00000017.jpg', 
#              'ILSVRC2012_val_00000029.jpg',
#              'ILSVRC2012_val_00011129.jpg',
#              'ILSVRC2012_val_00005026.jpg',
#              'n01592084_2950.jpg']
#     image = cv2.imread('data/ImageNet/'+files[0])
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
#     plt.axis('on')
#     plt.imshow(img)
#     # plt.show()
#     
#     tols = [0.5, 0.4, 0.3, 0.2, 0.1]
#     figure_size = 10
#     for file in files:
#         image=cv2.imread('data/ImageNet/'+file)
#         img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (250,200), interpolation = cv2.INTER_AREA)
#         vectorized = np.float32(img.reshape((-1,3)))
# 
#         plt.figure(figsize=(figure_size,figure_size))
#         plt.imshow(img)
#         plt.title('Original Image', fontsize=30), plt.xticks([]), plt.yticks([])
#         plt.savefig('fresults/IMG'+str(file)+'.pdf', bbox_inches='tight')
#         # plt.show()
# 
#         # nvectorized = (vectorized - vectorized.mean(axis=0))/vectorized.std(axis=0)
#         nvectorized, factor = novel_normalization(vectorized, "pca")
# 
#         for i in range(len(tols)):
#             classix = CLASSIX(sorting='pca', radius=tols[i], minPts=50, verbose=0, group_merging='distance', scale=1.05)
#             classix.fit_transform(vectorized)
#             classix_centers = np.uint8(classix.load_cluster_centers())
#             classix_res = classix_centers[classix.labels_]
#             classix_result_image = classix_res.reshape((img.shape))
# 
#             plt.figure(figsize=(figure_size,figure_size))
#             plt.imshow(classix_result_image)
#             plt.title('clusters = %i' % len(set(classix.labels_)), fontsize=30), plt.xticks([]), plt.yticks([])
#             plt.savefig('fresults/exp7/CLU_IMG'+str(file) + str(i) + '.pdf', bbox_inches='tight')
#             # plt.show()
    
    

def load_images(imagePaths):
    data = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (250, 200), interpolation = cv2.INTER_AREA) 
        data.append(image)
    return np.array(data)



def rn_img_real_comp(imagePaths, params, sample_size=10):
    data = load_images(imagePaths) 
    
    radius = params['radius']
    eps = params['eps']
    min_samples = params['min_samples']
    minPts = params['minPts']
    min_cluster_size = params['min_cluster_size']
    
    quicks_k = params['quicks_k'] 
    quicks_beta = params['quicks_beta']
    
    org_images = list()
    dbscan_images = list()
    hdbscan_images = list()
    quicks_images = list()
    classix_images = list()
    
    dbscan_labels = list()
    hdbscan_labels = list()
    quicks_labels = list()
    classix_labels = list()
    
    dbscan_times = list()
    hdbscan_times = list()
    quicks_times = list()
    classix_times = list()
    
    classix_distances = list()
    plot_num = 1
    for i in range(len(imagePaths)):
        img = data[i]
        vectorized = np.float64(img.reshape((-1,3)))

        dbscan_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            dbscan = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(vectorized)
        dbscan_time = time.time() - dbscan_time
        # print("DBSCAN time:", dbscan_time/sample_size)
        dbscan_centers = calculate_cluster_centers(vectorized, dbscan.labels_)
        dbscan_centers = np.uint8(dbscan_centers)
        dbscan_res = dbscan_centers[dbscan.labels_]
        dbscan_image = dbscan_res.reshape((img.shape))

        hdbscan_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            hdbscan_ = hdbscan.HDBSCAN(algorithm='best', core_dist_n_jobs=1, min_cluster_size=min_cluster_size[i]).fit(vectorized)
        hdbscan_time = time.time() - hdbscan_time
        # print("HDBSCAN time:", hdbscan_time/sample_size)
        hdbscan_centers = calculate_cluster_centers(vectorized, hdbscan_.labels_)
        hdbscan_centers = np.uint8(hdbscan_centers)
        hdbscan_res = hdbscan_centers[hdbscan_.labels_]
        hdbscan_image = hdbscan_res.reshape((img.shape))

        quicks_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            quicks = QuickshiftPP(k=quicks_k[i], beta=quicks_beta[i])
            quicks.fit(vectorized.copy(order='C'))
            quicks_labels_ = quicks.memberships
        quicks_time = time.time() - quicks_time
        # print("Quickshift++ time:", quicks_time/sample_size)
        quicks_centers = calculate_cluster_centers(vectorized, quicks_labels_)
        quicks_centers = np.uint8(quicks_centers)
        quicks_res = quicks_centers[quicks_labels_]
        quicks_image = quicks_res.reshape((img.shape))
        
        dist = 0
        classix_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            classix = CLASSIX(sorting='pca',
                              radius=radius[i],
                              minPts=minPts[i]+1, 
                              verbose=0, 
                              group_merging='distance', 
                              scale=1.05).fit(vectorized)
            
            dist = dist + classix.dist_nr
        classix_time = time.time() - classix_time
        classix_distances.append(dist/(len(vectorized)*sample_size))
        # print("CLASSIX time:", classix_time/sample_size)
        classix_centers = np.uint8(classix.load_cluster_centers())
        classix_res = classix_centers[classix.labels_]
        classix_image = classix_res.reshape((img.shape))
        
        org_images.append(img)
        dbscan_images.append(dbscan_image)
        hdbscan_images.append(hdbscan_image)
        quicks_images.append(quicks_image)
        classix_images.append(classix_image)
        
        dbscan_labels.append(dbscan.labels_)
        hdbscan_labels.append(hdbscan_.labels_)
        quicks_labels.append(quicks_labels_)
        classix_labels.append(classix.labels_)

        dbscan_times.append(dbscan_time/sample_size)
        hdbscan_times.append(hdbscan_time/sample_size)
        quicks_times.append(quicks_time/sample_size)
        classix_times.append(classix_time/sample_size)
        
    clustering_results = { 
        'Origin': org_images,
        'DBSCAN': dbscan_images,
        'HDBSCAN': hdbscan_images,
        'Quickshift++': quicks_images,
        'CLASSIX': classix_images
    }
    
    clustering_labels = {
        'Origin': [None]*len(imagePaths),
        'DBSCAN': dbscan_labels,
        'HDBSCAN': hdbscan_labels,
        'Quickshift++': quicks_labels,
        'CLASSIX': classix_labels
    }

    clustering_times = {
        'Origin' : [None]*len(imagePaths), 
        'DBSCAN': dbscan_times,
        'HDBSCAN': hdbscan_times,
        'Quickshift++': quicks_times,
        'CLASSIX': classix_times
    }
    
    return clustering_results, clustering_labels, clustering_times, classix_distances


def img_plot(imagePaths, clustering_results, clustering_labels, clustering_times, classix_distances, fontsize = 55, maxlen=6, savefile=None):
    plot_num = 1
    plt.figure(figsize=(10*len(clustering_results), 9*len(imagePaths)))
    for i_dataset in range(len(imagePaths)):  
        if i_dataset >= maxlen:
            break
        for name in clustering_results:
            plt.subplot(len(imagePaths), len(clustering_results), plot_num)
            plt.imshow(clustering_results[name][i_dataset])

            if name != 'Origin':
                plt.title(name+' (%i clusters)' % len(set(clustering_labels[name][i_dataset])),fontsize=int(1.27*fontsize));

            plt.axis('off');

            if name != 'Origin':
                plt.text(.99, .02, ('%.2fs' % np.round(clustering_times[name][i_dataset], 2)),
                         transform=plt.gca().transAxes, size=1.5*fontsize,
                         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5)
                        );
                
            if name == "CLASSIX":
                plt.text(.01, .02, ('%.2f' % np.round(classix_distances[i_dataset], 2)),
                         transform=plt.gca().transAxes, size=1.5*fontsize,
                         horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5)
                        );

            plot_num = plot_num + 1
            
            plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99, wspace=0.00001, hspace=0.00001)
    plt.tight_layout()
    plt.savefig('results/exp7/img_' + str(savefile) + '.pdf', bbox_inches='tight')
    # plt.show()
    
    
