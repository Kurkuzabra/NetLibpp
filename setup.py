# setup.py
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess
from pathlib import Path

# Get Homebrew's OpenMP paths (macOS-specific)
def get_brew_path(lib_name):
    try:
        return subprocess.check_output(f"brew --prefix {lib_name}", shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return None

class QhullBuild(build_ext):
    def run(self):
        # Build Qhull statically
        subprocess.run(["python", "build_qhull.py"], check=True)
        super().run()

base_compile_args = ["-O4"]
compile_args = {
    'win32': ["/O2", "/openmp", "/std:c++20", "/MD",  r"/Inetlibp\src\extern\qhull\src"],
    'linux': ["-fopenmp", "-std=c++20",  "-fPIC", "-Inetlibpp/src/extern/qhull/src", "-Inetlibpp/src/extern/qhull/src/libqhull_r"],
    'darwin': ["-std=c++20", "-fPIC", "-Inetlibpp/src/extern/qhull/src"] 
}

base_link_args = []
link_args = {
    'win32': ["-lstdc++", "-shared",  r"/link netlibpp\src\extern\qhull\build\Release\qhull_r.lib", "libqhullstatic_r.lib"],
    'linux': ["-lpthread", "-fopenmp", "-Lnetlibpp/src/extern/qhull/build", "-lqhullstatic_r"],
    'darwin': ["-lpthread", "-Lnetlibpp/src/extern/qhull/build", "-lqhullstatic_r"]
}

define_macros = {
    "windows": ("qh_QHULL_dllimport", ""),
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
            '-Xpreprocessor',    # Required for Clang + OpenMP
            '-fopenmp'
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
        include_dirs=[
            pybind11.get_include(),
            os.path.join("netlibpp", "src", "extern", "qhull", "src")
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        # extra_objects=[os.path.join("netlibpp", "src", "extern", "qhull", "build", "libqhullstatic_r.a")],
        define_macros=[("qh_QHpointer_1", "1")] + define_macros.get(current_platform, [])
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
    cmdclass={"build_ext": QhullBuild},
)
