# setup.py
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess

# Get Homebrew's OpenMP paths (macOS-specific)
def get_brew_path(lib_name):
    try:
        return subprocess.check_output(f"brew --prefix {lib_name}", shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return None

base_compile_args = ["-lpthread", "-shared", "-std=c++2a", "-O4", "-fPIC"]
compile_args = {
    'win32': [],
    'linux': ['-fopenmp'],
    'darwin': [] 
}

base_link_args = ["-std=c++2a", "-lpthread"]
link_args = {
    'win32': [],
    'linux': [],
    'darwin': ['-l omp']
}

# Choose args based on the current platform
current_platform = sys.platform
extra_compile_args = base_compile_args + compile_args.get(current_platform, [])
extra_link_args = base_link_args + link_args.get(current_platform, [])

if current_platform == 'darwin':
    omp_include = f"{get_brew_path('libomp')}/include"
    omp_lib = f"{get_brew_path('libomp')}/lib"
    if omp_include and omp_lib:
        extra_compile_args.extend([
            f'-I{omp_include}',  # Headers
            '-Xpreprocessor',     # Required for Clang + OpenMP
            '-fopenmp'           # Enables OpenMP
        ])
        extra_link_args.extend([
            f'-L{omp_lib}',      # Library path
            '-lomp'              # Link OpenMP
        ])
    else:
        print("Warning: libomp not found via Homebrew!")

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
