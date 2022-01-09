# This script will reproduce all of experimental results in the paper.
# Running all experiments is time consuming, we recommend users just select some of them for test.

# import sys 
import run_dist_matrix
import run_sort_cp
import run_mcs_noises
import run_tol_test
import run_scale_lx
import run_perform_comp
import run_shape_bk
import run_parameter_shape
import run_synthetic_bk
import run_real_world
import run_facial_cluster
import run_img_seg
from tqdm import tqdm 

RUNANALYSIS1 = True
RUNANALYSIS2 = True
RUNANALYSIS3 = True
RUNANALYSIS4 = True

RUNEXP1 = True
RUNEXP2 = True
RUNEXP3 = True
RUNEXP4 = True
RUNEXP5 = True
RUNEXP6 = True
RUNEXP7 = True

pbar = tqdm(total=sum([RUNANALYSIS1, RUNANALYSIS2, RUNANALYSIS3, RUNANALYSIS4, 
                       RUNEXP1, RUNEXP2, RUNEXP3, RUNEXP4, RUNEXP5, RUNEXP6, RUNEXP7]))
# sys.stdout.write("Progress: [ %s" % ("" * 9))
# sys.stdout.flush()
# sys.stdout.write("\b" * (9+1)) 
       
# ============================ analysis part ===================================
if RUNANALYSIS1:
    run_dist_matrix.rn_wine_dataset()
    run_dist_matrix.rn_iris_dataset()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
        
if RUNANALYSIS2:
    run_sort_cp.rn_sort_early_stp()
    run_sort_cp.count_distance()
    run_sort_cp.rn_sort_plot1()
    run_sort_cp.rn_sort_plot2()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
if RUNANALYSIS3:
    run_mcs_noises.rn_mcs_it(save=True)
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
if RUNANALYSIS4:
    run_scale_lx.rn_scale_explore()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
# ============================ experiment part ===================================

#---------------------------------------------------------------------------------------------------------   
# scalability [exp 1]
if RUNEXP1:
    run_perform_comp.rn_gaussian_size()
    run_perform_comp.rn_gaussian_dim()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------    
# tolerance sensitivity  [exp 2]
if RUNEXP2:
    run_tol_test.rn_tol_st()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------   
# sklearn synthetic benchmark [exp 3]
if RUNEXP3:
    run_synthetic_bk.rn_sklearn_benchmark()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------   
# shape benchmark [exp 4]
if RUNEXP4:
    run_parameter_shape.main()
    run_shape_bk.rn_cluster_shape()
    run_shape_bk.shape_index_plot()
    run_shape_bk.shape_predict_test()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
    
# UCI benchmark [exp 5]
if RUNEXP5:
    run_real_world.params_search()
    run_real_world.compare_best_params()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------  
# facical clustering experiment [exp 6]
if RUNEXP6:
    run_facial_cluster.rn_facial_cluster()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)
    
#---------------------------------------------------------------------------------------------------------  
# image segmentation [exp 7]
if RUNEXP7:
    run_img_seg.rn_img_anime_minpts()
    run_img_seg.rn_img_real_tol()
    run_img_seg.rn_img_real_comp()
    # sys.stdout.write("=")
    # sys.stdout.flush()
    pbar.update(1)

    
pbar.close()
# sys.stdout.write(" ]\n") 