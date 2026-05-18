## Tiled

tiling improves cache efficiency even further by dividing the matrices into smaller 
submatrices that fit into the cpu cache.

while the reordered version accesses rows of *b* sequentially, it still works on the full matrix at once, 
causing previously loaded cache lines to be evicted before they can be reused. 

blocking reduces this problem by operating on small chunks of *a*, *b*, and *c* repeatedly before moving 
on to the next region of memory.

this increases temporal locality: once a block is loaded into cache, many multiplications are performed using that same data before it is discarded. as a result, memory bandwidth pressure drops significantly and the cpu spends more time doing arithmetic instead of waiting on memory.

---

## Code

we assume that `c` is a zero vector.

```rust
pub fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize, block_size: usize) {
    for ii in (0..n).step_by(block_size) {
        for kk in (0..n).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                // process block
                for i in ii..(ii + block_size).min(n) {
                    for k in kk..(kk + block_size).min(n) {
                        let a_i_k = a[i * n + k];

                        for j in jj..(jj + block_size).min(n) {
                            c[i * n + j] += a_i_k * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}
```
