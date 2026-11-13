# CPU × CUDA — analysis of actual measurements

Timings use warmups and repeated runs. Bands show the 10th and 90th percentiles, not confidence intervals. Allocation and initial weight loading are excluded on both architectures.

**GPU was not run. This file contains no CPU × GPU result.** The comparison below is between two C++ CPU implementations.

| Batch | Sequential CPU (ms) | Optimized CPU (ms) | Sequential/optimized speedup |
|---:|---:|---:|---:|
| 1 | 0.097600 | 0.005000 | 19.520× |
| 8 | 0.661900 | 0.055800 | 11.862× |
| 64 | 5.577000 | 0.427700 | 13.040× |
| 128 | 10.828400 | 0.592800 | 18.267× |
| 256 | 21.560100 | 1.208000 | 17.848× |
| 1024 | 88.161500 | 5.165000 | 17.069× |
## Interpretation

Optimized CPU reuses contiguous data and uses OpenMP when enabled at compilation. Tiled CUDA reuses operands in shared memory; padding and barriers can hurt small matrices. The program does not predetermine a winner.

Events measure the interval between stream markers, including all kernels and possible submission gaps. They do not measure a single instruction or total host-visible latency.

Differences small relative to variability require more runs. The benchmark measures warmed-up execution, excluding allocation and weight loading. It does not measure context creation or a cold first call.

Run CPU and GPU on the same machine and preserve metadata.txt and host.json. These educational implementations do not represent peak BLAS, cuBLAS, Tensor Core, or framework performance.
