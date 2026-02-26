import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import os

class CustomBuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy as np
        self.include_dirs.append(np.get_include())

define_macros = []
if os.environ.get("CYTHON_TRACE"):
    define_macros.append(("CYTHON_TRACE", "1"))

if sys.platform == "win32":
    c_args = ["/O2"]
else:
    c_args = ["-O3"]
    

pyx_extensions = [
    Extension("classix.aggregate_ed_c", ["classix/aggregate_ed_c.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.aggregate_ed_cm", ["classix/aggregate_ed_cm.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.merge_ed_cm", ["classix/merge_ed_cm.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.merge_ed_cm_win", ["classix/merge_ed_cm_win.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.aggregate_md_cm", ["classix/aggregate_md_cm.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.merge_md_cm", ["classix/merge_md_cm.pyx"], extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.aggregate_td_cm", ["classix/aggregate_td_cm.pyx"], language="c++", extra_compile_args=c_args, define_macros=define_macros),
    Extension("classix.merge_td_cm", ["classix/merge_td_cm.pyx"], extra_compile_args=c_args, define_macros=define_macros),
]

c_extensions = [
    Extension("spmv", ["classix/spmv.cpp"], language="c++", extra_compile_args=c_args),
]

setup(
    ext_modules=cythonize(
        pyx_extensions,
        compiler_directives={"language_level": "3", "linetrace": bool(os.environ.get("CYTHON_TRACE"))},
        annotate=False,
    ) + c_extensions,
    cmdclass={"build_ext": CustomBuildExt},
)
