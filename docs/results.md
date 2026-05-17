## Results

```bash
matrix multiplication optimizations (GEMM benchmark)

naive                   6771.89 ms  ± 133.84 ms   min  6624.90 ms       0.32 GFLOPS
reordered                194.53 ms  ±  14.77 ms   min   168.89 ms      11.04 GFLOPS
tiled-4                 1910.34 ms  ±  39.54 ms   min  1854.14 ms       1.12 GFLOPS
tiled-8                 1267.73 ms  ±  30.51 ms   min  1234.78 ms       1.69 GFLOPS
tiled-16                 692.57 ms  ±  12.16 ms   min   677.52 ms       3.10 GFLOPS
tiled-32                 404.90 ms  ±   5.37 ms   min   395.98 ms       5.30 GFLOPS
tiled-64                 260.19 ms  ±   4.17 ms   min   253.03 ms       8.25 GFLOPS
tiled-128                211.77 ms  ±   4.68 ms   min   202.77 ms      10.14 GFLOPS
parallel                 117.27 ms  ±   4.36 ms   min   109.54 ms      18.31 GFLOPS
tiled-parallel-4         219.03 ms  ±   9.48 ms   min   207.82 ms       9.80 GFLOPS
tiled-parallel-8         192.57 ms  ±   6.47 ms   min   179.84 ms      11.15 GFLOPS
tiled-parallel-16        198.85 ms  ±   7.96 ms   min   178.92 ms      10.80 GFLOPS
tiled-parallel-32        187.89 ms  ±  13.18 ms   min   165.12 ms      11.43 GFLOPS
tiled-parallel-64        179.60 ms  ±  10.31 ms   min   163.44 ms      11.96 GFLOPS
tiled-parallel-128       219.69 ms  ±  26.04 ms   min   177.65 ms       9.78 GFLOPS

n: 1024
threads: 16
```
