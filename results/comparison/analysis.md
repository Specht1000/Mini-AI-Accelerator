# CPU × CUDA — analysis of actual measurements

Timings use warmups and repeated runs. Bands show the 10th and 90th percentiles, not confidence intervals. Allocation and initial weight loading are excluded on both architectures.

**GPU was not run. This file contains no CPU × GPU result.** The comparison below is between two C++ CPU implementations.

| Batch | Sequential CPU (ms) | Optimized CPU (ms) | Sequential/optimized speedup |
|---:|---:|---:|---:|
| 1 | 0.100800 | 0.005500 | 18.327× |
| 8 | 0.718100 | 0.053800 | 13.348× |
| 64 | 5.728300 | 0.446600 | 12.826× |
| 128 | 11.529000 | 0.814200 | 14.160× |
| 256 | 22.963800 | 1.304900 | 17.598× |
| 1024 | 112.419700 | 12.630600 | 8.901× |
## Interpretation

Optimized CPU reuses contiguous data and uses OpenMP when enabled at compilation. Tiled CUDA reuses operands in shared memory; padding and barriers can hurt small matrices. The program does not predetermine a winner.

Events measure the interval between stream markers, including all kernels and possible submission gaps. They do not measure a single instruction or total host-visible latency.

Differences small relative to variability require more runs. The benchmark measures warmed-up execution, excluding allocation and weight loading. It does not measure context creation or a cold first call.

Run CPU and GPU on the same machine and preserve metadata.txt and host.json. These educational implementations do not represent peak BLAS, cuBLAS, Tensor Core, or framework performance.
