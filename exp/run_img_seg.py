import cv2
import os
import time
import hdbscan
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.clustering import CLASSIX
from quickshift.QuickshiftPP import *
from src.aggregation_cm import aggregate
from src.clustering import calculate_cluster_centers, novel_normalization



def rn_img_anime_minpts():
    file ='data/Saitama_serious_profile.png'
    image=cv2.imread(file)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (120,120), interpolation = cv2.INTER_AREA)
    vectorized = np.float32(img.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    classix1 = CLASSIX(sorting='pca', radius=0.01, minPts=20, verbose=0, group_merging='distance')
    classix1.fit_transform(vectorized)
    classix_centers = np.uint8(classix1.load_cluster_centers())
    classix_res = classix_centers[classix1.labels_]
    classix_result_image1 = classix_res.reshape((img.shape))

    classix2 = CLASSIX(sorting='pca', radius=0.01, minPts=35, verbose=0, group_merging='distance')
    classix2.fit_transform(vectorized)
    classix_centers = np.uint8(classix2.load_cluster_centers())
    classix_res = classix_centers[classix2.labels_]
    classix_result_image2 = classix_res.reshape((img.shape))

    classix3 = CLASSIX(sorting='pca', radius=0.01, minPts=50, verbose=0, group_merging='distance')
    classix3.fit_transform(vectorized)
    classix_centers = np.uint8(classix3.load_cluster_centers())
    classix_res = classix_centers[classix3.labels_]
    classix_result_image3 = classix_res.reshape((img.shape))

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(2,3,1),plt.imshow(classix_result_image1)
    plt.title('Segmented image with clusters = %i' % len(set(classix1.labels_)), fontsize=12), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,2),plt.imshow(classix_result_image2)
    plt.title('Segmented image with clusters = %i' % len(set(classix2.labels_)), fontsize=12), plt.xticks([]), plt.yticks([])
    plt.subplot(2,3,3),plt.imshow(classix_result_image3)
    plt.title('Segmented image with clusters = %i' % len(set(classix3.labels_)), fontsize=12), plt.xticks([]), plt.yticks([])
    plt.savefig('fresults/OPM_denoise_.pdf', bbox_inches='tight')
    plt.show()

    

def rn_img_real_tol():
    files = ['n01496331_9611.jpg', 
             'ILSVRC2012_val_00000017.jpg', 
             'ILSVRC2012_val_00000029.jpg',
             'ILSVRC2012_val_00011129.jpg',
             'ILSVRC2012_val_00005026.jpg',
             'n01592084_2950.jpg']
    image = cv2.imread('data/ImageNet/'+files[0])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
    plt.axis('on')
    plt.imshow(img)
    plt.show()
    
    tols = [0.5, 0.4, 0.3, 0.2, 0.1]
    figure_size = 10
    for file in files:
        image=cv2.imread('data/ImageNet/'+file)
        img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (250,200), interpolation = cv2.INTER_AREA)
        vectorized = np.float32(img.reshape((-1,3)))

        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(img)
        plt.title('Original Image', fontsize=30), plt.xticks([]), plt.yticks([])
        plt.savefig('fresults/IMG'+str(file)+'.pdf', bbox_inches='tight')
        plt.show()

        # nvectorized = (vectorized - vectorized.mean(axis=0))/vectorized.std(axis=0)
        nvectorized, factor = novel_normalization(vectorized, "pca")

        for i in range(len(tols)):
            classix = CLASSIX(sorting='pca', radius=tols[i], minPts=50, verbose=0, group_merging='distance', scale=1.05)
            classix.fit_transform(vectorized)
            classix_centers = np.uint8(classix.load_cluster_centers())
            classix_res = classix_centers[classix.labels_]
            classix_result_image = classix_res.reshape((img.shape))

            plt.figure(figsize=(figure_size,figure_size))
            plt.imshow(classix_result_image)
            plt.title('Segmented image with clusters = %i' % len(set(classix.labels_)), fontsize=30), plt.xticks([]), plt.yticks([])
            plt.savefig('fresults/CLU_IMG'+str(file) + str(i) + '.pdf', bbox_inches='tight')
            plt.show()
    
    
    

    
def rn_img_real_comp():
    imagePaths = [
     'datasets/COCO2017/000000002473.jpg',
     'datasets/COCO2017/000000000009.jpg',
     'datasets/COCO2017/000000006012.jpg',
     'datasets/COCO2017/000000000247.jpg',
     'datasets/COCO2017/000000000034.jpg',
     'datasets/COCO2017/000000067406.jpg',
     'datasets/COCO2017/000000000092.jpg',
     'datasets/COCO2017/000000003255.jpg',
     'datasets/COCO2017/000000082180.jpg',
     'datasets/COCO2017/000000000074.jpg',
     'datasets/COCO2017/000000001451.jpg',
     'datasets/COCO2017/000000001332.jpg'
    ]
    
    data = []
    target = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (210, 170), interpolation = cv2.INTER_AREA) 
        data.append(image)

    data = np.array(data) 
    figure_size = 3.6
    fontsize = 18
    eps =  [3, 2, 2, 2, 3, 1, 3, 2, 1.5, 1.5, 1.5, 1]
    min_samples = [10, 12, 10, 15, 10, 10, 15, 8, 8, 12, 12, 6]
    minPts = [100, 100, 150, 50, 100, 50, 50, 50, 50, 50, 100, 100]
    min_cluster_size = [100, 30, 25, 50, 10, 30, 80, 50, 50, 50, 15, 50]
    sample_size = 10
    quicks_k = [360, 60, 55, 170, 105, 85, 95, 125, 135, 85, 95, 75]
    quicks_beta = [0.3, 0.7, 0.5, 0.4, 0.7, 0.5, 0.9, 0.7, 0.8, 0.7, 0.6, 0.5]
    
    for i in range(len(data)):
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
        dbscan_result_image = dbscan_res.reshape((img.shape))


        classix_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            classix = CLASSIX(sorting='pca', radius=0.15, minPts=minPts[i], verbose=0, group_merging='distance', scale=1.05).fit(vectorized)
        classix_time = time.time() - classix_time
        # print("CLASSIX time:", classix_time/sample_size)
        classix_centers = np.uint8(classix.load_cluster_centers())
        classix_res = classix_centers[classix.labels_]
        classix_result_image = classix_res.reshape((img.shape))


        hdbscan_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            hdbscan_ = hdbscan.HDBSCAN(algorithm='best', core_dist_n_jobs=1, min_cluster_size=min_cluster_size[i]).fit(vectorized)
        hdbscan_time = time.time() - hdbscan_time
        # print("HDBSCAN time:", hdbscan_time/sample_size)
        hdbscan_centers = calculate_cluster_centers(vectorized, hdbscan_.labels_)
        hdbscan_centers = np.uint8(hdbscan_centers)
        hdbscan_res = hdbscan_centers[hdbscan_.labels_]
        hdbscan_result_image = hdbscan_res.reshape((img.shape))

        quicks_time = time.time()
        for _iter in range(sample_size):
            np.random.seed(_iter)
            quicks = QuickshiftPP(k=quicks_k[i], beta=quicks_beta[i])
            quicks.fit(vectorized.copy(order='C'))
            quicks_labels = quicks.memberships
        quicks_time = time.time() - quicks_time
        # print("Quickshift++ time:", quicks_time/sample_size)
        quicks_centers = calculate_cluster_centers(vectorized, quicks_labels)
        quicks_centers = np.uint8(quicks_centers)
        quicks_res = quicks_centers[quicks_labels]
        quicks_result_image = quicks_res.reshape((img.shape))
        
        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(img)
        plt.title('Original Image',fontsize=fontsize), plt.xticks([]), plt.yticks([])
        plt.savefig('fresults/segmentation_org_' + str(i) + '.pdf', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(dbscan_result_image)
        plt.title('DBSCAN (%i clusters)' % len(set(dbscan.labels_)),fontsize=fontsize), plt.xticks([]), plt.yticks([])
        plt.text(.99, .01, ('%.2fs' % (dbscan_time/sample_size)).lstrip('0'), transform=plt.gca().transAxes, size=20, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig('fresults/segmentation_dbscan_' + str(i) + '.pdf', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(hdbscan_result_image)
        plt.title('HDBSCAN (%i clusters)' % len(set(hdbscan_.labels_)),fontsize=fontsize), plt.xticks([]), plt.yticks([])
        plt.text(.99, .01, ('%.2fs' % (hdbscan_time/sample_size)).lstrip('0'), transform=plt.gca().transAxes, size=20, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig('fresults/segmentation_hdbscan_' + str(i) + '.pdf', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(classix_result_image)
        plt.title('CLASSIX (%i clusters)' % len(set(classix.labels_)),fontsize=fontsize), plt.xticks([]), plt.yticks([])
        plt.text(.99, .01, ('%.2fs' % (classix_time/sample_size)).lstrip('0'), transform=plt.gca().transAxes, size=20, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig('fresults/segmentation_classix_' + str(i) + '.pdf', bbox_inches='tight')

        plt.figure(figsize=(figure_size,figure_size))
        plt.imshow(quicks_result_image)
        plt.title('Quickshift++ (%i clusters)' % len(set(quicks_labels)),fontsize=fontsize), plt.xticks([]), plt.yticks([])
        plt.text(.99, .01, ('%.2fs' % (quicks_time/sample_size)).lstrip('0'), transform=plt.gca().transAxes, size=20, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig('fresults/segmentation_quicks_' + str(i) + '.pdf', bbox_inches='tight')
        
        plt.show()