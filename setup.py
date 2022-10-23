from setuptools import setup
from Cython.build import cythonize

setup(
    ext_modules=cythonize("colour.pyx")
)