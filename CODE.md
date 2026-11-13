# Implementation walkthrough

Follow the data from training to the performance report. See [README.md](README.md) for concepts and [RUN.md](RUN.md) for commands.

## Entry point: `compare.py`

The launcher parses batches, repetitions, warmups, output directory, and CPU-only mode, then validates arguments. It compiles C++/CUDA by default; `--binary` reuses an executable. Matplotlib uses the noninteractive Agg backend so charts do not require a desktop GUI.

The workflow calls `build()`, `run_benchmark()`, and `analyze()`, then prints the report. Normal execution requires CUDA; `--cpu-only` explicitly selects a CPU experiment. Failures print an English diagnostic and return a nonzero exit status.

## Dataset and training: `python/ai.py`

`synthetic_dataset()` uses a seeded random generator. Class centers affect the first 32 features at most; remaining features are noise. Training, validation, and test samples are independently generated with shared class centers.

`MLP` stores FP32 weights as `[input_features, output_features]` matrices and biases as vectors. He initialization scales random weights by `sqrt(2 / input_features)`. `forward()` returns intermediate activations for backpropagation as well as output probabilities.

Each layer calculates `activation @ weights + bias`. Hidden layers apply ReLU; the output applies softmax. Softmax subtracts row maxima before exponentiating and normalizing.

`train_model()` shuffles minibatches, evaluates cross-entropy, and backpropagates. The output gradient is `probabilities - one_hot_target`, divided by batch size. Hidden gradients multiply by the next weight transpose and the ReLU derivative. Adam tracks gradient moments with bias correction and learning rate 0.001.

Validation accuracy selects checkpoints. The test split is evaluated only after loading the selected checkpoint. `MLP.save()` and `load()` preserve dimensions and parameters. Model artifacts are stored as `ai_model.npz`, `ai_test.npz`, and `ai_metrics.json`.

## Binary export

`export_native()` writes identical FP32 tensors for both processors:

| Bytes | Type | Meaning |
|---|---|---|
| 0–3 | Little-endian uint32 | Rows |
| 4–7 | Little-endian uint32 | Columns |
| 8 onward | Little-endian float32 | Row-major values |

Biases are one-row matrices. `model.txt` stores the layer count. `inputs.bin`, `labels.bin`, and `probabilities.bin` provide inputs, labels, and NumPy reference outputs.

## Common interface: `native/native.hpp`

`Matrix` stores contiguous row-major FP32 data. `Model` groups parameters, samples, labels, and references. Shared interfaces allow CPU-only compilation. Numerical comparison checks dimensions and finite values before calculating maximum absolute error.

## CPU: `native/cpu.cpp`

Loading validates matrix dimensions, file size, finite values, layer compatibility, bias widths, samples, and labels. The naive product uses i-j-k loops, accumulating one complete dot product per output.

The optimized product uses i-k-j traversal with reduction blocking. Its inner loop reads B and writes C contiguously, improving locality and enabling compiler SIMD. Optional OpenMP distributes rows. Parallel startup remains part of the timing.

Inference applies matrix multiplication, bias, and ReLU at hidden layers, followed by stable output softmax. Rectangular-product tests exercise dimensions and boundaries.

## CUDA: `native/gpu.cu`

The naive kernel assigns one output to each thread in a 16 × 16 block. Each valid thread accumulates across the reduction dimension.

The tiled kernel loads A and B tiles into shared memory. Threads synchronize, accumulate, then synchronize again before overwriting tiles. Edge threads load zeros and still participate in barriers. Skipping barriers at matrix boundaries would invalidate synchronization.

Separate kernels apply bias and activations. Softmax uses one thread per sample for the small ten-class output. RAII releases resources even when an operation throws.

Weights and buffers are allocated outside timing. Resident measurements retain input on GPU. Transfer-inclusive measurements copy input H2D, execute inference, copy output D2H, and synchronize. Weights remain resident in both cases.

## Measurement: `native/benchmark.cpp`

The CLI accepts folders, batches, repetitions, warmups, CPU-only mode, and required-GPU mode. It rejects invalid ranges and supports independent numerical self-tests.

Every implementation warms up, repeats the same network over the same data, validates against NumPy, and writes CSV measurements. Columns include batch, algorithm, metric, repetitions, median milliseconds, percentiles 10/90, maximum error, and accuracy. Percentiles use sorted samples and interpolation.

`metadata.txt` records build and device information. `--require-gpu` prevents a missing CUDA device from being interpreted as a CPU/GPU comparison.

## Analysis: `python/analysis.py`

`run_benchmark()` trains if needed, exports tensors, runs the executable, and records host metadata. It defaults `OMP_NUM_THREADS` to four unless already configured.

`load_measurements()` rejects non-finite metrics, duplicate rows, nonpositive times, too few repetitions, inconsistent percentiles, and errors above `1e-4`.

`analyze()` plots latency and speedup with percentile bands. It compares optimized CPU wall time with GPU transfer-inclusive wall time and also shows resident measurements. Without CUDA rows, it produces a CPU-only report. The first tested winning batch is not an exact crossover.

## Builds, notebooks, figures, and checks

`python/build_native.py` uses nvcc for CUDA and checks Windows MSVC requirements. CPU builds use GCC, Clang, or installed Zig. `CMakeLists.txt` provides optional CUDA and OpenMP.

`python/prepare_demo.py` generates English local/Colab notebooks and a portable source/data ZIP. Extraction checks keep archive members inside the target directory. An explicit package list excludes caches and unrelated files.

`python/figures.py` redraws training and CPU charts from saved measurements. SVGs illustrate architecture and matrix multiplication.

Run `python -m unittest discover -s tests -v`, then native `--self-test --cpu-only`, or `--self-test --require-gpu` on a CUDA machine. Performance conclusions remain specific to the tested hardware, compiler, network, batches, and runtime conditions.
