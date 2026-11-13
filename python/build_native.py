"""Compile C++/CUDA without installing compilers or modifying the system."""

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def build(cpu_only=False):
    folder = ROOT / "build"
    folder.mkdir(exist_ok=True)
    binary = folder / ("benchmark.exe" if os.name == "nt" else "benchmark")

    if not cpu_only:
        compiler = shutil.which("nvcc")
        if not compiler:
            raise RuntimeError("nvcc not found. Install CUDA on a machine with an NVIDIA GPU or use --cpu-only.")
        if os.name == "nt" and not shutil.which("cl"):
            raise RuntimeError("CUDA on Windows requires MSVC. Open the Visual Studio x64 Native Tools terminal with C++ support.")

        command = [compiler, "-std=c++17", "-O3", "-arch=native", "-DHAVE_CUDA=1", "native/benchmark.cpp", "native/cpu.cpp", "native/gpu.cu", "-o", str(binary)]
        # The host compiler enables optimization and parallelism when available.
        command += ["-Xcompiler", "/O2,/openmp" if os.name == "nt" else "-O3,-fopenmp"]
        if os.name != "nt":
            # nvcc uses the host linker but does not forward -fopenmp.
            command.append("-lgomp")
    else:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler:
            command = [compiler, "-std=c++17", "-O3", "native/benchmark.cpp", "native/cpu.cpp", "-o", str(binary)]
            if Path(compiler).name.startswith("g++"):
                command.append("-fopenmp")
        elif importlib.util.find_spec("ziglang"):
            # An installed Zig compiler can validate C++ without MSVC on this PC.
            command = [sys.executable, "-m", "ziglang", "c++", "-std=c++17", "-O3", "native/benchmark.cpp", "native/cpu.cpp", "-o", str(binary)]
        else:
            raise RuntimeError("No C++ compiler found. Install GCC, Clang, MSVC, or Zig.")

    environment = os.environ.copy()
    environment["ZIG_GLOBAL_CACHE_DIR"] = str(folder / "zig-global")
    environment["ZIG_LOCAL_CACHE_DIR"] = str(folder / "zig-local")
    print("Compiling:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print("Executable:", binary)
    return binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    arguments = parser.parse_args()
    build(arguments.cpu_only)
