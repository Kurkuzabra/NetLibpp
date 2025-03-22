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
        name="netlibpp_cpy",
        sources=["netlibpp/src/include/graph_func.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-lpthread", "-shared", "-std=c++2a", "-O4", "-fPIC"],
        extra_link_args=["-std=c++2a", "-lpthread"]
    ),
]


setup(
    name="netlibpp",
    # version="0.0.3",

    packages=["netlibpp"],
    ext_modules=ext_modules,
    install_requires=[],
    python_requires=">=3.6",
    package_data={
        'netlibpp': ['*.pyi'],
    },
    exclude_package_data={
        "": ["*.c", "*.cpp", "*.h", "*.hpp", "*.cc", "*.hh", "*.o"],
    },
    include_package_data=True,
)
