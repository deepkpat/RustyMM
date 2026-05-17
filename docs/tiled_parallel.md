## Tiled Parallel

tiled parallel combines the benefits of tiling and parallelization.

like tiling, it breaks the matrices into small blocks to improve cache reuse, reducing memory bandwidth pressure.

like parallelization, it splits the work across multiple threads, letting each core compute different blocks concurrently. 

by operating on cache-friendly blocks in parallel, each thread can make full use of the cpu cache while the cpu cores work 
simultaneously on independent regions of the result matrix. 

in theory, this should give the best performance of all the optimizations. in practice, however, the sample runs show that 
tiled parallel is only slightly faster than the other approaches and sometimes even slower, likely due to overhead from 
synchronizing threads and managing many small blocks.

execution time (in ms) is similar to tiling or parallel alone, with a mean of ~152ms on `n = 1024`, `block_size = 32`, `threads = 16`.

---

## Code

we assume that `c` is a zero vector.

```rust
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

pub fn matmul_tiled_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize, block_size: usize) {
    c.par_chunks_mut(block_size * n)
        .enumerate()
        .for_each(|(block_idx, c_chunk)| {
            let ii = block_idx * block_size;

            let row_count = (n - ii).min(block_size);

            for kk in (0..n).step_by(block_size) {
                for jj in (0..n).step_by(block_size) {
                    for i_local in 0..row_count {
                        let i = ii + i_local;

                        for k in kk..(kk + block_size).min(n) {
                            let a_i_k = a[i * n + k];

                            for j in jj..(jj + block_size).min(n) {
                                c_chunk[i_local * n + j] += a_i_k * b[k * n + j];
                            }
                        }
                    }
                }
            }
        });
}
```
