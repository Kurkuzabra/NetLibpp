# setup.py
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


base_compile_args = ["-lpthread", "-shared", "-std=c++2a", "-O4", "-fPIC"]
compile_args = {
    'win32': [],
    'linux': [],
    'darwin': ['-I $(brew --prefix libomp)/include', '-L $(brew --prefix libomp)/lib'] 
}

base_link_args = ["-std=c++2a", "-lpthread"]
link_args = {
    'win32': [],
    'linux': [],
    'darwin': []
}

# Choose args based on the current platform
current_platform = sys.platform
extra_compile_args = base_compile_args + compile_args.get(current_platform, [])
extra_link_args = base_link_args + link_args.get(current_platform, [])



# Define the C++ extension
ext_modules = [
    Extension(
        name="netlibpp_cpy",
        sources=["netlibpp/src/include/graph_func.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
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
