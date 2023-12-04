# This script will reproduce all of experimental results in the paper.
# Running all experiments is time consuming, we recommend users just select some of them for test.

# import sys 
import os
import pickle
import numpy as np
import run_dist_matrix
import run_sort_cp
import run_mcs_noises
import run_tol_test
import run_scale_lx
import run_perform_comp
import run_shape_bk
import run_parameter_shape
import run_sklearn_bk
import run_real_world
import run_facial_cluster
import run_img_seg
from tqdm import tqdm 

    
store_dir = "results"
if not os.path.isdir(store_dir):
    os.makedirs(store_dir)
    

RUNEXP1 = True
RUNEXP2 = True
RUNEXP3 = True
RUNEXP4 = True
RUNEXP5 = True
RUNEXP6 = True
RUNEXP7 = True

pbar = tqdm(total=sum([RUNEXP1, RUNEXP2, RUNEXP3, RUNEXP4, RUNEXP5, RUNEXP6, RUNEXP7]))
# sys.stdout.write("Progress: [ %s" % ("" * 9))
# sys.stdout.flush()
# sys.stdout.write("\b" * (9+1)) 
       
# ============================ analysis part ===================================
# if RUNANALYSIS1_1:
#     run_dist_matrix.rn_wine_dataset()
#     run_dist_matrix.rn_iris_dataset()
#     # sys.stdout.write("=")
#     # sys.stdout.flush()
#     pbar.update(1)
        
# if RUNANALYSIS1_2:
#     run_sort_cp.rn_sort_early_stp()
#     run_sort_cp.count_distance()
#     run_sort_cp.rn_sort_plot1()
#     run_sort_cp.rn_sort_plot2()
#     # sys.stdout.write("=")
#     # sys.stdout.flush()
#     pbar.update(1)

# if RUNANALYSIS1_3:
#     run_scale_lx.rn_scale_explore()
#     # sys.stdout.write("=")
#     # sys.stdout.flush()
#     pbar.update(1)
    
    
# ============================ experiment part ===================================
#---------------------------------------------------------------------------------------------------------   
# scalability -- Gassuain blobs [exp 1]  
if RUNEXP1: 
    store_dir = "results/exp1"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    
    run_mcs_noises.rn_mcs_it(save=True)
    run_perform_comp.rn_gaussian_size()
    run_perform_comp.rn_gaussian_dim()
    run_perform_comp.run_gassian_plot()
    run_perform_comp.run_comp_sort()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------    
# tolerance sensitivity  [exp 2]
if RUNEXP2:
    store_dir = "results/exp2"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        
    dataset_sizes = np.hstack([np.arange(1, 6) * 1000, np.arange(6,10) * 1000])
    # print("data size: " + " ".join([str(i) for i in dataset_sizes]))

    _range = np.arange(0.1, 1.005, 0.05)
    run_tol_test.run_sensitivity_test_blobs(dataset_sizes, _range)
    run_tol_test.plot_sensitivity(dataset_sizes, _range)
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------   
# sklearn synthetic benchmark [exp 3]
if RUNEXP3:
    store_dir = "results/exp3"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        
    run_sklearn_bk.rn_sklearn_benchmark()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------   
# shape benchmark [exp 4]
if RUNEXP4:
    store_dir = "results/exp4"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        
    run_parameter_shape.main() # search the best parameter 
    run_shape_bk.rn_cluster_shape()
    run_shape_bk.shape_index_plot()
    run_shape_bk.shape_pred_test()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------  
# UCI benchmark [exp 5]
if RUNEXP5:
    store_dir = "results/exp5"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        
    run_real_world.params_search()
    run_real_world.visualize_params_global()
    # run_real_world.visualize_params_search()
    run_real_world.compare_best_params()
    # run_real_world.kamil_industry_test()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------  
# facical clustering experiment [exp 6]
if RUNEXP6:
    store_dir = "results/exp6"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
        
    run_facial_cluster.rn_facial_cluster()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------  
# image segmentation [exp 7]
if RUNEXP7:
    store_dir = "results/exp7"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    clust_r, clust_l, clust_t, clust_d = run_img_seg.rn_img_real_comp(
        run_img_seg.imagePaths1, run_img_seg.params1
    )

    with open('results/exp7/clust_r.pkl', 'wb') as f:
        pickle.dump(clust_r, f)

    with open('results/exp7/clust_l.pkl', 'wb') as f:
        pickle.dump(clust_l, f)

    with open('results/exp7/clust_t.pkl', 'wb') as f:
        pickle.dump(clust_t, f)

    np.save('results/exp7/clust_d.npy', clust_d)

    run_img_seg.img_plot(
             run_img_seg.imagePaths1,
             clust_r, clust_l, clust_t, clust_d,
             fontsize=45, maxlen=6,
             savefile='seg1'
    )
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)

    
pbar.close()
# sys.stdout.write(" ]\n") 