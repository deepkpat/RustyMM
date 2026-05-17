## Naive

a naive implementation of matrix multiplication.

---

## Code

```rust
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;

            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }

            c[i * n + j] = sum;
        }
    }
}
```

--- 

## Sample Runs

execution time (in ms)

`n = 1024`

1. 6982.044
2. 6932.011
3. 6712.814
4. 6797.968
5. 6866.102
6. 6976.003
7. 6884.703
8. 6894.566

mean: 6880.776
