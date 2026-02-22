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

setup(
    name="monotonic_align",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}  # important for modern Python
    ),
)