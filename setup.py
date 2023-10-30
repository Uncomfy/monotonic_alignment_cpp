import sys
import subprocess

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except:
    assert subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"]) == 0
    from pybind11.setup_helpers import Pybind11Extension, build_ext

from setuptools import setup

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
    install_requires=[
        "numpy"
    ],
)