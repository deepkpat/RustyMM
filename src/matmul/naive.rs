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
