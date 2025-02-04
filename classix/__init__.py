
__version__ = '1.2.8'                  
__enable_cython__ = True 
__debug_token__ = "asdasdaskj1.2.8"
from .clustering import CLASSIX
from .clustering import loadData
from .clustering import cython_is_available
from .clustering import calculate_cluster_centers
from .clustering import preprocessing


"""For Matlab's users to check the CLASSIX's version."""
version = __version__
