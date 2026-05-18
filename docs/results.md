## Results

```bash
matrix multiplication optimizations (GEMM benchmark)

naive                   6597.02 ms  ±  17.24 ms   min  6574.68 ms       0.33 GFLOPS
reordered                167.59 ms  ±   5.03 ms   min   161.74 ms      12.81 GFLOPS
tiled-4                 1755.71 ms  ±   2.08 ms   min  1753.20 ms       1.22 GFLOPS
tiled-8                 1096.29 ms  ±   6.08 ms   min  1088.78 ms       1.96 GFLOPS
tiled-16                 643.04 ms  ±   0.71 ms   min   641.32 ms       3.34 GFLOPS
tiled-32                 366.29 ms  ±   0.52 ms   min   365.15 ms       5.86 GFLOPS
tiled-64                 240.06 ms  ±   0.44 ms   min   239.37 ms       8.95 GFLOPS
tiled-128                203.61 ms  ±   0.47 ms   min   203.03 ms      10.55 GFLOPS
parallel                 107.67 ms  ±   1.43 ms   min   105.36 ms      19.94 GFLOPS
tiled-parallel-4         197.83 ms  ±   3.76 ms   min   191.93 ms      10.86 GFLOPS
tiled-parallel-8         178.11 ms  ±   4.27 ms   min   170.54 ms      12.06 GFLOPS
tiled-parallel-16        162.21 ms  ±   8.29 ms   min   153.13 ms      13.24 GFLOPS
tiled-parallel-32        156.95 ms  ±  10.11 ms   min   143.04 ms      13.68 GFLOPS
tiled-parallel-64        152.47 ms  ±   8.42 ms   min   139.62 ms      14.08 GFLOPS
tiled-parallel-128       189.03 ms  ±  20.26 ms   min   154.84 ms      11.36 GFLOPS
tiled-packed             108.28 ms  ±   2.14 ms   min   105.81 ms      19.83 GFLOPS
tiled-simd                78.74 ms  ±   1.59 ms   min    77.15 ms      27.27 GFLOPS

n: 1024
threads: 16
```
