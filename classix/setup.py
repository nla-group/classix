from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["aggregation_c.pyx", "aggregation_cm.pyx", "merging_cm.pyx"]),
)