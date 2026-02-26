from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class build_ext_with_numpy(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        self.include_dirs.append(numpy.get_include())

spmv_module = Extension(
    'spmv',
    sources=['spmv.cpp'],
    extra_compile_args=['-O3', '-march=native'],
)

setup(
    name='spmv',
    version='2.0',
    description='Sparse submatrix-vector multiplication for CLASSIX_T',
    ext_modules=[spmv_module],
    setup_requires=['numpy'],
    cmdclass={'build_ext': build_ext_with_numpy},
)
