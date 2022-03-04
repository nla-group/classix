import numpy
import logging
import setuptools
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

_version="0.5.0"
logging.basicConfig()
log = logging.getLogger(__file__)

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)


with open("README.rst", 'r') as f:
    long_description = f.read()

try:
    from Cython.Build import cythonize
    
    setuptools.setup(
        name="ClassixClustering",
        packages=["classix"],
        version=_version,
        setup_requires=["cython", "numpy>=1.3.0"],
        install_requires=["numpy>=1.3.0", "scipy>=0.7.0", "pandas", "matplotlib", "requests"],
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


except ext_errors as ex:
    log.warn(ex)
    log.warn("The C extension could not be compiled")

    setuptools.setup(
        name="ClassixClustering",
        packages=["classix"],
        version=_version,
        setup_requires=["numpy>=1.3.0"],
        install_requires=["numpy>=1.3.0", "scipy>=0.7.0", "pandas", "matplotlib", "requests"],
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

    log.info("Plain-Python installation succeeded.")