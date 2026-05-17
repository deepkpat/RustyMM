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

const BS: usize = 16;

pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for ii in (0..n).step_by(BS) {
        for kk in (0..n).step_by(BS) {
            for jj in (0..n).step_by(BS) {
                // process block
                for i in ii..(ii + BS).min(n) {
                    for k in kk..(kk + BS).min(n) {
                        let a_i_k = a[i * n + k];

                        for j in jj..(jj + BS).min(n) {
                            c[i * n + j] += a_i_k * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

pub fn transpose(src: &[f32], dst: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

pub fn matmul_transposed(a: &[f32], bt: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;

            for k in 0..n {
                sum += a[i * n + k] * bt[j * n + k];
            }
            c[i * n + j] = sum;
        }
    }
}

pub fn matmul_blocked_transposed(a: &[f32], bt: &[f32], c: &mut [f32], n: usize) {
    for ii in (0..n).step_by(BS) {
        for jj in (0..n).step_by(BS) {
            for kk in (0..n).step_by(BS) {
                // process block
                for i in ii..(ii + BS).min(n) {
                    for j in jj..(jj + BS).min(n) {
                        let mut sum = c[i * n + j];

                        for k in kk..(kk + BS).min(n) {
                            sum += a[i * n + k] * bt[j * n + k];
                        }

                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}
