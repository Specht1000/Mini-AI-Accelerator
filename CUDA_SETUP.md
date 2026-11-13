# NVIDIA CUDA C++ setup on Windows

For a beginner walkthrough covering downloads, installation, a first CUDA program, and Google Colab, start with [CUDA C++ from scratch](CUDA_ZERO.md).

The Python virtual environment isolates NumPy, Matplotlib, and Jupyter. The NVIDIA driver, CUDA Toolkit, and C++ compiler are system tools, installed separately from pip.

## Requirements

1. A CUDA-compatible NVIDIA GPU and an installed NVIDIA driver. Run `nvidia-smi` to inspect the GPU and driver. Installing the Toolkit alone does not establish that a compatible GPU is present.
2. NVIDIA CUDA Toolkit. Run `nvcc --version` to check the compiler. This project currently detects Toolkit 13.4 on this machine.
3. A Visual Studio C++ toolchain supported by your installed Toolkit. In Visual Studio Installer, install **Desktop development with C++**, including the x64/x86 MSVC tools and a Windows SDK. Check NVIDIA's compiler compatibility table for your Toolkit version. During setup, neither `cl.exe` in the current terminal nor a Visual Studio installation with the required C++ tools was detected by `vswhere`.

Open **x64 Native Tools Command Prompt for Visual Studio** after installation. This configures the compiler, linker, headers, and libraries; adding `cl.exe` alone to PATH is insufficient.

```bat
cd /d "C:\Projetos\LIA\Mini AI Accelerator"
where cl
where nvcc
nvidia-smi
.venv\Scripts\python.exe compare.py
```

This compiles the CPU and CUDA sources, runs both implementations, validates numerical outputs, and generates the report. If you use a Developer PowerShell terminal instead, use `Set-Location` to enter the project folder.

## How C++ and CUDA are linked

`python/build_native.py` invokes nvcc with `native/benchmark.cpp`, `native/cpu.cpp`, and `native/gpu.cu` in one build. nvcc compiles device code, delegates host C++ compilation to MSVC on Windows, and links the CUDA runtime automatically. No Python CUDA package is needed for these native kernels.

Equivalent explicit command, from the configured x64 tools terminal:

```bat
nvcc -std=c++17 -O3 -arch=native -DHAVE_CUDA=1 native/benchmark.cpp native/cpu.cpp native/gpu.cu -Xcompiler /O2,/openmp -o build/benchmark.exe
build\benchmark.exe --self-test --require-gpu
.venv\Scripts\python.exe compare.py --binary build/benchmark.exe
```

`-arch=native` targets the visible local GPU. Build on a machine with the target NVIDIA GPU. For cross-compilation or deployment to another architecture, choose a supported explicit architecture for that GPU instead.

For a different C++ application, the corresponding minimal CMake integration is:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyCudaApp LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(my_app main.cpp kernels.cu)
target_compile_features(my_app PRIVATE cxx_std_17)
set_target_properties(my_app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(my_app PRIVATE CUDA::cudart)
```

`CUDA::cudart` supplies the CUDA runtime dependency and its usage requirements. This project's CMake build already enables CUDA and includes `native/gpu.cu`; its nvcc-based helper is the simplest local path.

## Virtual environment

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
python compare.py
```

Activation is optional. If PowerShell blocks activation, use the interpreter directly without changing execution policy:

```powershell
.\.venv\Scripts\python.exe compare.py
```

## References

- [NVIDIA CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA nvcc compiler and linking documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [CMake FindCUDAToolkit imported targets](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)
