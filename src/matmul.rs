use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use std::arch::x86_64::*;

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

const BS: usize = 32;

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

pub fn matmul_blocked_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    c.par_chunks_mut(BS * n)
        .enumerate()
        .for_each(|(block_idx, c_chunk)| {
            let ii = block_idx * BS;

            let row_count = (n - ii).min(BS);

            for kk in (0..n).step_by(BS) {
                for jj in (0..n).step_by(BS) {
                    for i_local in 0..row_count {
                        let i = ii + i_local;

                        for k in kk..(kk + BS).min(n) {
                            let a_i_k = a[i * n + k];

                            for j in jj..(jj + BS).min(n) {
                                c_chunk[i_local * n + j] += a_i_k * b[k * n + j];
                            }
                        }
                    }
                }
            }
        });
}

#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();

    let mut sum = _mm256_setzero_ps();

    let mut k = 0;

    while k + 8 <= len {
        // load 8 floats from a
        let va = _mm256_loadu_ps(a.as_ptr().add(k));

        // load 8 bit from b;
        let vb = _mm256_loadu_ps(b.as_ptr().add(k));

        // multiply
        let prod = _mm256_mul_ps(va, vb);

        // accumulate
        sum = _mm256_add_ps(sum, prod);

        k += 8;
    }

    // horizontal reduction
    let mut temp = [0.0f32; 8];

    _mm256_storeu_ps(temp.as_mut_ptr(), sum);

    let mut result = temp.iter().sum::<f32>();

    // remainder loop
    while k < len {
        result += a[k] * b[k];
        k += 1;
    }

    result
}

pub fn matmul_transposed_simd(a: &[f32], bt: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let row_a = &a[i * n..(i + 1) * n];
            let row_bt = &bt[j * n..(j + 1) * n];

            c[i * n + j] = unsafe { dot_product_avx2(row_a, row_bt) };
        }
    }
}
