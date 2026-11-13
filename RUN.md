# Run the CPU and CUDA comparison

Create an isolated environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

Compile, run both processors, and generate the analysis:

```powershell
python compare.py
```

CUDA requires a compatible NVIDIA GPU, driver, CUDA Toolkit (`nvcc`), and supported host C++ compiler. On Windows, use the **Visual Studio x64 Native Tools terminal** with C++ support. On Linux, use a compatible GCC compiler with OpenMP. The default command requires actual GPU execution and reports an error if CUDA cannot run.

See [CUDA_SETUP.md](CUDA_SETUP.md) for Windows compiler setup and explicit CUDA/C++ linking commands. Activation is optional: `.\.venv\Scripts\python.exe compare.py` always selects the virtual environment directly.

Without CUDA, benchmark CPU only:

```powershell
python compare.py --cpu-only
```

Reuse an existing executable:

```powershell
python compare.py --cpu-only --binary build/benchmark.exe
```

The program trains a small neural network (MLP 784 → 128 → 64 → 10) on synthetic data if no trained model exists. It compares **inference** with identical FP32 weights in C++ and CUDA. NumPy training is excluded from the timing comparison.

Results are saved in `results/comparison/`:

- `analysis.md`: timing tables and conclusions for each batch size.
- `comparison_real.png`: latency and speedup plots.
- `benchmark.csv`: timings, percentiles, numerical error, and accuracy.
- `metadata.txt` and `host.json`: execution context.

Defaults are five warmups and 21 repetitions. The benchmark validates against NumPy and compares naive/optimized CPU with naive/tiled CUDA. GPU measurements cover resident data and input/output transfers separately. Allocation and initial weight loading are excluded on both processors.

Speedup is `CPU time / GPU time`: above 1 favors GPU. Check the spread between the 10th and 90th percentiles; small differences need more runs. Conclusions apply to this network, these batches, and this machine. Without executing CUDA, there is no CPU/GPU winner. These educational implementations do not measure peak cuBLAS or framework performance.

Customize the workload:

```powershell
python compare.py --batches 1 32 256 1024 4096 --repetitions 31 --warmup 10
```

See [README.md](README.md) for the concepts and [CODE.md](CODE.md) for the implementation walkthrough.
