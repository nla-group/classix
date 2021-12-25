import setuptools
from Cython.Build import cythonize
import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="ClassixClustering",
    packages=["classix"],
    version="0.1.1",
    setup_requires=["cython>=0.29.4", "numpy>=1.20.0", "scipy>1.6.0", "matplotlib"],
    install_requires=["numpy>=1.20.0", "pandas", "matplotlib"],
    ext_modules=cythonize(["classix/*.pyx"], include_path=["classix"]),
    package_data={"classix": ["aggregation_c.pyx", "aggregation_cm.pyx", "merging_cm.pyx"]},
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="Stefan Guettel, Xinye Chen",
    author_email="stefan.guettel@manchester.ac.uk, xinye.chen@manchester.ac.uk",
    description="Fast and explainable clustering based on sorting",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/CLASSIX.git",
    license='BSD 3-Clause'
)
