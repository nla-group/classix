import warnings
try:
    # %load_ext Cython
    # !python3 setup.py build_ext --inplace
    import scipy, numpy
    if scipy.__version__ == '1.8.0' or numpy.__version__ < '1.22.0':
        from .aggregation_c import aggregate 
        # cython without memory view, solve the error from scipy ``TypeError: type not understood``
    else:
        from .aggregation_cm import aggregate
        # cython with memory view
    from .merging_cm import fast_agglomerate as agglomerate 
except (ModuleNotFoundError, ValueError):
    from .aggregation import aggregate
    from .merging import fast_agglomerate as agglomerate 
    # warnings.warn("This CLASSIX installation is not using Cython.")
    
from .clustering import CLASSIX
from .clustering import load_data
from .clustering import calculate_cluster_centers
from .clustering import novel_normalization
