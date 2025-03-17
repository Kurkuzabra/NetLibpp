# setup.py
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# Define the C++ extension
ext_modules = [
    Extension(
        "netlibpp_cpy",
        sources=["netlibpp/src/include/graph_func.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-lpthread", "-shared", "-std=c++2a", "-O4", "-fPIC"],
        extra_link_args=["-std=c++2a", "-lpthread"]
    ),
]


setup(
    name="netlibpp",
    version="0.0.1",

    packages=["netlibpp", "netlibpp/src/include"],
    ext_modules=ext_modules,
    # cmdclass={"build_ext": CMakeBuild},
    # install_requires=["pybind11>=2.6.0"],
    install_requires=[],
    python_requires=">=3.6",
)
