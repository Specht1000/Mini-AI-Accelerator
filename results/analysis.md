# CPU × GPU — historical hypothetical simulation

These are not physical GPU measurements or calibrated predictions for this PC. This historical comparison covers dense matrix multiplications only, excluding bias, activations, and training. Current hardware results are in the results subdirectories.

The MLP was trained on synthetic data and achieved 99.83% accuracy on 600 test samples. This is not MNIST accuracy. Functional inference on eight samples was validated for both simulated architectures.

| Batch | CPU (µs) | Resident GPU (µs) | Cold GPU (µs) | CPU/resident GPU | Resident winner |
|---:|---:|---:|---:|---:|---|
| 1 | 8.829 | 54.958 | 101.353 | 0.161× | CPU |
| 2 | 8.923 | 55.223 | 101.617 | 0.162× | CPU |
| 4 | 9.112 | 55.752 | 102.147 | 0.163× | CPU |
| 8 | 9.489 | 56.811 | 103.205 | 0.167× | CPU |
| 16 | 18.261 | 60.276 | 106.671 | 0.303× | CPU |
| 32 | 36.523 | 70.382 | 116.777 | 0.519× | CPU |
| 64 | 73.045 | 90.617 | 137.012 | 0.806× | CPU |
| 128 | 146.091 | 131.234 | 177.629 | 1.113× | GPU |
| 256 | 292.181 | 212.468 | 258.863 | 1.375× | GPU |
| 512 | 584.363 | 374.936 | 421.330 | 1.559× | GPU |
| 1024 | 1168.725 | 699.871 | 746.266 | 1.670× | GPU |

The first tested batch where resident-weight GPU wins is **128 samples**. With transferred weights it is **256 samples**. Neither locates an exact crossover.

## Why results change

Small batches underfill GPU blocks while paying launch and transfer costs. Larger batches expose more parallelism and amortize overhead. The simulated CPU also uses cores and SIMD.

Outcomes depend on clock rates, execution units, memory bandwidth, tile size, and link overhead. These educational parameters do not correspond to specific commercial models.

## Functional validation

Maximum CPU error: 3.553e-15. Maximum GPU error: 3.553e-15. Both were compared with NumPy on the same rectangular product.

Cycles come from a virtual scheduler. Memory is modeled afterward with an aggregate bandwidth bound; caches, stalls, and individual instructions are not simulated cycle by cycle.
