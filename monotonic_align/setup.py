from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="monotonic_align.core",
        sources=["core.pyx"],
        include_dirs=[numpy.get_include()],
    )
]
