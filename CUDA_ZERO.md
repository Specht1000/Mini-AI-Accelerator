# CUDA C++ from scratch

This guide explains what to download, how to configure Windows, and how to run your first CUDA C++ program. It also covers Google Colab for computers without an NVIDIA GPU.

## 1. What is CUDA C++?

CUDA is NVIDIA's platform for GPU computing. CUDA C++ extends C++ with features for launching functions called **kernels** on an NVIDIA GPU. CUDA source files normally use the `.cu` extension.

You obtain the development tools by downloading the **CUDA Toolkit**. Its compiler is called **nvcc**. Installing a Python package is not required to compile this project's native CUDA kernels.

| Component | Purpose | Installation location |
|---|---|---|
| NVIDIA GPU | Executes the device code | Physical local computer or cloud machine |
| NVIDIA driver | Lets the operating system communicate with the GPU | System |
| CUDA Toolkit | Provides nvcc, headers, runtime, and development libraries | System |
| MSVC C++ tools | Compile and link the Windows host code | System |
| Python `.venv` | Isolates NumPy, Matplotlib, and Jupyter dependencies | Project directory |

The virtual environment does not provide a GPU, driver, or Windows C++ compiler.

## 2. Choose the right route

| Your situation | Recommended route |
|---|---|
| No NVIDIA GPU, including Intel-only or AMD-only graphics | Use Google Colab: section 3 |
| Compatible NVIDIA GPU on Windows | Install the local tools: section 4 |
| Only want this project's CPU measurements | Use section 7's CPU-only command |

**For the computer used in this project, which has no NVIDIA GPU, start with Colab.** Installing CUDA locally does not enable CUDA execution on Intel or AMD graphics. You do not need to install MSVC or NVIDIA drivers locally to run a remote Colab notebook.

You can check the graphics model in Windows **Device Manager → Display adapters**. Compare NVIDIA models against the [CUDA GPU list](https://developer.nvidia.com/cuda-gpus) and your chosen Toolkit's supported architectures. Brand alone does not establish support for every Toolkit release.

## 3. Use a GPU through Google Colab

### Prepare the project package

The repository already includes:

- [notebooks/CUDA_Colab.ipynb](notebooks/CUDA_Colab.ipynb): the remote workflow.
- `dist/cpu_gpu_cuda.zip`: source code, model data, and documentation to upload.

After changing project files, regenerate the package from PowerShell:

```powershell
Set-Location "C:\Projetos\LIA\Mini AI Accelerator"
.\.venv\Scripts\python.exe python/prepare_demo.py
```

### Open the notebook and select a GPU

1. Open [Google Colab](https://colab.research.google.com/) and sign in.
2. Choose **File → Upload notebook**, then select `notebooks/CUDA_Colab.ipynb` from your computer.
3. In the top menu, choose **Runtime → Change runtime type**.
4. Select an available **GPU**, such as T4, and save.
5. Click **Run all**. If prompted, connect to the hosted runtime.
6. When the upload button appears, select `dist/cpu_gpu_cuda.zip`. Keep it zipped.
7. Wait for compilation and measurement. The final cells show charts and download results.

For a Portuguese Colab interface, these menu labels are **Arquivo → Fazer upload de notebook**, **Ambiente de execução → Alterar tipo de ambiente de execução**, and **Executar tudo**. Upload wording may vary slightly with the interface.

The first code cell runs `nvidia-smi` and `nvcc --version`. It should print the remote NVIDIA GPU and compiler information. If `nvidia-smi` is missing, verify that you selected a GPU runtime and are connected to it, then rerun the cell.

GPU access is subject to availability and usage limits; a particular GPU is not guaranteed. The benchmark compares **the CPU and GPU of the Colab machine**, not your local CPU against a remote GPU. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 4. Install CUDA C++ locally on Windows

Continue here only if you have access to a compatible NVIDIA GPU for local execution.

### Step 1 — Install the NVIDIA driver

Download the driver for your GPU and Windows version from [NVIDIA Drivers](https://www.nvidia.com/drivers/). Follow the installer and restart if requested. The driver is separate from the development toolkit; do not assume that installing the Toolkit also installed a working driver.

Open a new PowerShell terminal and run:

```powershell
nvidia-smi
```

A successful result includes the GPU model and driver version. Its displayed **CUDA Version** indicates driver compatibility, not the installed nvcc version. Use `nvcc --version` later to inspect the Toolkit compiler.

### Step 2 — Install the C++ compiler

Download [Visual Studio Build Tools for C++](https://visualstudio.microsoft.com/visual-cpp-build-tools/), or modify an existing Visual Studio installation.

In **Visual Studio Installer**, select **Desktop development with C++**, including the x64/x86 MSVC compiler tools and a Windows SDK. Choose versions supported by the CUDA Toolkit you plan to install; consult its [Windows compatibility table](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Build Tools provide command-line compilation without requiring the complete IDE. **Visual Studio Code is an editor and does not itself install MSVC.**

### Step 3 — Download the CUDA Toolkit

1. Open [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads).
2. Select Windows, the matching architecture and operating system version, and an installer.
3. Download and run the installer.
4. Follow the installation prompts for the development tools.
5. Open a new terminal after installation.

If you need a specific version for compiler or GPU compatibility, use the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Follow the installation guide for that version rather than assuming the newest release supports every older GPU.

The project previously detected Toolkit 13.4 locally. This establishes that nvcc was installed, not that a local CUDA GPU exists.

### Step 4 — Open a configured development terminal

From the Windows Start menu, open **Developer PowerShell for Visual Studio**, configured for x64. This sets up compiler, linker, header, and library paths. Simply adding the folder containing `cl.exe` to PATH is insufficient.

Run these **PowerShell** commands:

```powershell
Set-Location "C:\Projetos\LIA\Mini AI Accelerator"
Get-Command cl,nvcc,nvidia-smi
nvcc --version
nvidia-smi
```

All three tools should be found. `cl` is the Windows C++ compiler, `nvcc` is the CUDA compiler driver, and `nvidia-smi` reports GPU/driver information.

If you prefer **x64 Native Tools Command Prompt for Visual Studio**, the equivalent **Command Prompt** commands are:

```bat
cd /d "C:\Projetos\LIA\Mini AI Accelerator"
where.exe cl
where.exe nvcc
where.exe nvidia-smi
```

Do not copy `cd /d` into PowerShell: it is Command Prompt syntax. In PowerShell, use `Set-Location`. Use `Get-Command` or `where.exe` to locate programs; plain `where` is a PowerShell alias with a different meaning.

## 5. Compile your first CUDA C++ program

On the NVIDIA-equipped machine, create a file named `hello_cuda.cu` with the following contents:

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::fprintf(stderr, "No usable CUDA device: %s (count=%d)\n",
                     cudaGetErrorString(status), count);
        return 1;
    }

    hello<<<1, 4>>>();
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return 0;
}
```

From the configured Developer PowerShell terminal, in the folder containing that file:

```powershell
nvcc -std=c++17 -arch=native hello_cuda.cu -o hello_cuda.exe
.\hello_cuda.exe
```

`__global__` declares a kernel. `<<<1, 4>>>` launches one block with four threads. `cudaDeviceSynchronize()` waits for GPU completion and exposes execution errors. Successful output contains one greeting per thread; their order is not guaranteed.

`-arch=native` targets the visible GPU on the build machine. Cross-compiling for a different machine requires an appropriate explicit GPU architecture instead. See the [nvcc documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## 6. How C++ and CUDA are linked

For this project, nvcc receives both `.cpp` and `.cu` sources. It compiles the GPU code, delegates Windows host compilation to MSVC, and links the CUDA runtime. You do not manually connect a physical GPU to Python.

The project helper performs the build:

```powershell
.\.venv\Scripts\python.exe python/build_native.py
.\build\benchmark.exe --self-test --require-gpu
```

For the equivalent explicit nvcc command and a CMake example using `CUDA::cudart`, see [CUDA_SETUP.md](CUDA_SETUP.md).

## 7. Run this project's AI benchmark

If the Python environment has not yet been created:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

From a configured development terminal on an NVIDIA-equipped machine:

```powershell
.\.venv\Scripts\python.exe compare.py
```

This builds the native code, trains the model if needed, executes CPU and CUDA inference, validates outputs, and generates the analysis in `results/comparison/`.

Without an NVIDIA GPU, use Colab or run the existing CPU executable locally:

```powershell
.\.venv\Scripts\python.exe compare.py --cpu-only --binary build/benchmark.exe
```

If the executable is missing, `compare.py --cpu-only` can build one using a supported installed CPU compiler. A CPU-only run does not produce a CPU/GPU winner.

## 8. Common errors

| Error or symptom | What to check |
|---|---|
| `CUDA on Windows requires MSVC` | Install C++ tools and use an x64 Visual Studio development terminal. |
| `cl` not found | MSVC is missing or its environment is not loaded. Activating `.venv` does not configure it. |
| `nvcc` not found | Install the Toolkit, reopen the terminal, and check its `bin` directory in PATH. |
| `nvidia-smi` not found | Check GPU hardware, driver installation, and command location; in Colab select a GPU runtime. |
| `nvcc` works but CUDA execution fails | A compiler installation alone does not prove GPU or driver availability. |
| `unsupported Microsoft Visual Studio version` | Match MSVC to the installed Toolkit's compatibility table. |
| `no kernel image is available` | Check GPU architecture support and rebuild for the target GPU. |
| PowerShell rejects `cd /d` | Use `Set-Location "path"`; `/d` belongs to Command Prompt. |
| `where cl` shows nothing in PowerShell | Use `Get-Command cl` or `where.exe cl`. |
| PowerShell blocks `.venv` activation | Invoke `.\.venv\Scripts\python.exe` directly; activation is optional. |
| Colab cannot allocate a GPU | Availability is limited; try again later or use another permitted GPU environment. |

## Official references

- [CUDA Toolkit download](https://developer.nvidia.com/cuda-downloads)
- [CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [CUDA nvcc compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
