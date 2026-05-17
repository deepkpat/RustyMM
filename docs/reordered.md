## Reordered

by reordering the loops from `i, j, k` to `i, k, j`, we imporve *cache locality*.

in the naive version, the inner `k` loop accesses `b[k * n + j]` with a stride of `n`, which evicts cache lines on every 
iteration.

wrapping `j` and `k` makes `j` the innermost loop, so `b[k * n + j]` is accessed sequentially in memory, fully utilizing
the cpu's cache line.

this single change accounts for ~40x speed up (6880ms to 170ms on `n = 1024`).

---

## Code

we assume that `c` is a zero vector.

```rust
pub fn matmul_reordered(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for k in 0..n {
            let a_i_k = a[i * n + k];

            for j in 0..n {
                c[i * n + j] += a_i_k * b[k * n + j];
            }
        }
    }
}
```
