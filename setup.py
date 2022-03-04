import setuptools
from Cython.Build import cythonize
import numpy

with open("README.rst", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="ClassixClustering",
    packages=["classix"],
    version="0.4.5",
    setup_requires=["cython", "numpy>=1.20.0", "scipy>=1.2.1", "matplotlib"],
    install_requires=["numpy>=1.20.0", "pandas", "matplotlib"],
    ext_modules=cythonize(["classix/*.pyx"], include_path=["classix"]),
    package_data={"classix": ["aggregation_c.pyx",
                              "aggregation_cm.pyx", 
                              "merging_cm.pyx"]
                 },
    include_dirs=[numpy.get_include()],
    classifiers=["Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10"
            ],
    long_description=long_description,
    author="Xinye Chen, Stefan Güttel",
    author_email="xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
    description="Fast and explainable clustering based on sorting",
    long_description_content_type='text/x-rst',
    url="https://github.com/nla-group/CLASSIX.git",
    license='MIT License'
)


