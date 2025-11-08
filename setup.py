# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

class CustomBuildExt(build_ext):
    def run(self):
        self.include_dirs.append(np.get_include())
        super().run()

ext_modules = [
    Extension("classix.aggregate_c", ["classix/aggregate_c.pyx"]),
    Extension("classix.aggregate_cm", ["classix/aggregate_cm.pyx"]),
    Extension("classix.merge_cm", ["classix/merge_cm.pyx"]),
    Extension("classix.merge_cm_win", ["classix/merge_cm_win.pyx"]),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)