from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

os.environ["CPPFLAGS"] = "-fopenmp"     # Adding it as extra_compile_args didn't work :(

__version__ = "0.0.4"

ext_modules = [
    Pybind11Extension("monotonic_alignment_cpp", ["main.cpp"])
]

setup(
    name="monotonic_alignment_cpp",
    version=__version__,
    author="Dmytro Balaban",
    author_email="dimabalabanokda@gmail.com",
    description="Monotonic alignment for VITS written in C++",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)