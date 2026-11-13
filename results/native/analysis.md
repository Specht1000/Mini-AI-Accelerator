# CPU × CUDA — analysis of actual measurements

Timings use warmups and repeated runs. Bands show the 10th and 90th percentiles, not confidence intervals. Allocation and initial weight loading are excluded on both architectures.

**GPU was not run. This file contains no CPU × GPU result.** The comparison below is between two C++ CPU implementations.

| Batch | Sequential CPU (ms) | Optimized CPU (ms) | Sequential/optimized speedup |
|---:|---:|---:|---:|
| 1 | 0.226600 | 0.013000 | 17.431× |
| 8 | 0.822100 | 0.056900 | 14.448× |
| 64 | 7.287900 | 0.595800 | 12.232× |
| 128 | 12.894800 | 0.687500 | 18.756× |
| 256 | 26.402200 | 1.615400 | 16.344× |
| 1024 | 104.038800 | 5.923500 | 17.564× |
## Interpretation

Optimized CPU reuses contiguous data and uses OpenMP when enabled at compilation. Tiled CUDA reuses operands in shared memory; padding and barriers can hurt small matrices. The program does not predetermine a winner.

Events measure the interval between stream markers, including all kernels and possible submission gaps. They do not measure a single instruction or total host-visible latency.

Differences small relative to variability require more runs. The benchmark measures warmed-up execution, excluding allocation and weight loading. It does not measure context creation or a cold first call.

Run CPU and GPU on the same machine and preserve metadata.txt and host.json. These educational implementations do not represent peak BLAS, cuBLAS, Tensor Core, or framework performance.
