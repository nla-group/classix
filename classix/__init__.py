try:
    from .aggregation_cm import aggregate
    from .merging_cm import fast_agglomerate as agglomerate 
except ModuleNotFoundError:
    from .aggregation import aggregate
    from .merging import fast_agglomerate as agglomerate 
    print("Cython fail.")
    
from .clustering import CLASSIX
from .clustering import load_data
from .clustering import calculate_cluster_centers
from .clustering import novel_normalization
