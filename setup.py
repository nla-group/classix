import warnings
import logging
import setuptools
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import Extension

try:
    from Cython.Distutils import build_ext
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools.command.build_ext import build_ext
    
_version="1.2.6"
logging.basicConfig()
log = logging.getLogger(__file__)

ext_errors = (CCompilerError, ModuleNotFoundError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)

with open("README.rst", 'r') as f:
    long_description = f.read()

def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]
    
class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)
        
        
setup_args = {'name':"classixclustering",
        'packages':["classix"],
        'version':_version,
        'install_requires':["cython>=0.27",
                             "numpy>=1.17.3",
                             "scipy>=1.7.0",
                             "pandas",
                             "matplotlib>=3.5",
                             "requests"], # requirements()

        'packages': ['classix'],
        'cmdclass': {'build_ext': CustomBuildExtCommand},
        'classifiers':["Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS',
                "Programming Language :: Python :: 3",
                ],
        'long_description':long_description,
        'author':"Xinye Chen, Stefan Güttel",
        'author_email':"xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
        'description':"Fast and explainable clustering based on sorting",
        'long_description_content_type':'text/x-rst',
        'url':"https://github.com/nla-group/CLASSIX.git",
        'license':'MIT License'
}

aggregation_c = Extension('classix.aggregate_c',
                        sources=['classix/aggregate_c.pyx'])

aggregation_cm = Extension('classix.aggregate_cm',
                        sources=['classix/aggregate_cm.pyx'])

merging_cm = Extension('classix.merge_cm',
                        sources=['classix/merge_cm.pyx'])

merging_cm_win = Extension('classix.merge_cm_win',
                        sources=['classix/merge_cm_win.pyx'])

try:
    # from Cython.Build import cythonize
    
    setuptools.setup(
        setup_requires=["cython", "numpy>=1.17.3"],
        # ext_modules=cythonize(["classix/*.pyx"], include_path=["classix"]),
        ext_modules=[aggregation_c,
                     aggregation_cm,
                     merging_cm,
                     merging_cm_win
                    ],
        **setup_args
    )


except ext_errors as ext:
    log.warning(ext)
    log.warning("The C extension could not be compiled.")

    setuptools.setup(setup_requires=["numpy>=1.17.3"], **setup_args)
    log.info("Plain-Python installation succeeded.")
    
    
