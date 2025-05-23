import os
import subprocess
import sys
from pathlib import Path

def build_qhull(qhull_dir: str) -> None:
    qhull_path = Path(qhull_dir)
    build_dir = qhull_path / "build"

    cmake_options = [
        "cmake",
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_POLICY_DEFAULT_CMP0000=NEW",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    ]

    # relocatable lib
    if sys.platform == "win32":
        cmake_options.extend([
            "-A", "x64"
        ])
    else:
        cmake_options.extend([
            "-DCMAKE_CXX_FLAGS=-fPIC", 
            "-DCMAKE_C_FLAGS=-fPIC"
        ])

    cmake_options.append(str(qhull_path))
    subprocess.run(cmake_options, check=True)
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--config", "Release"
    ]
    
    # enable arallel build
    if sys.platform != "win32":
        build_cmd.extend(["--", f"-j{os.cpu_count()}"])
        
    subprocess.run(build_cmd, check=True)
    if sys.platform == "win32":
        subprocess.run(r"dir \"netlibpp\src\extern\qhull\build\Release\"")
        subprocess.run(r"dir \"netlibpp\src\extern\qhull\build\Release\qhullstatic_r.lib\"")

if __name__ == "__main__":
    qhull_source_dir = os.path.join("netlibpp", "src", "extern", "qhull")
    build_qhull(qhull_source_dir)