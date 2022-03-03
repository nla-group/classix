import setuptools
from Cython.Build import cythonize
import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="ClassixClustering",
    packages=["classix"],
    version="0.4.1",
    setup_requires=["cython>=0.29.4", "numpy>=1.22.0", "scipy>1.6.0", "matplotlib"],
    install_requires=["numpy>=1.22.0", "pandas", "matplotlib"],
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
            ],
    long_description=long_description,
    author="Xinye Chen, Stefan Güttel",
    author_email="xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
    description="Fast and explainable clustering based on sorting",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/CLASSIX.git",
    license='MIT License'
)


