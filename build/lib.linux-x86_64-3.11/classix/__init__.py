
__version__ = '1.1.2'                  
__enable_cython__ = True 


from .clustering import CLASSIX
from .clustering import loadData
from .clustering import cython_is_available
from .clustering import calculate_cluster_centers
from .clustering import normalization


"""For Matlab's users to check the CLASSIX's version."""
version = __version__
