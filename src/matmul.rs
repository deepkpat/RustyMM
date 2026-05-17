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

pub fn get_batch_size() -> usize {
    BS
}

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

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_chunked(a: &[f32], b: &[f32], start: usize, end: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();

    let mut k = start;

    while k + 8 <= end {
        let va = _mm256_loadu_ps(a.as_ptr().add(k));

        let vb = _mm256_loadu_ps(b.as_ptr().add(k));

        sum = _mm256_fmadd_ps(va, vb, sum);

        k += 8;
    }

    let mut temp = [0.0f32; 8];

    _mm256_storeu_ps(temp.as_mut_ptr(), sum);

    let mut result = temp.iter().sum::<f32>();

    while k < end {
        result += a[k] * b[k];
        k += 1;
    }

    result
}

pub fn matmul_blocked_transposed_simd(a: &[f32], bt: &[f32], c: &mut [f32], n: usize) {
    for ii in (0..n).step_by(BS) {
        for jj in (0..n).step_by(BS) {
            for kk in (0..n).step_by(BS) {
                let k_end = (kk + BS).min(n);

                for i in ii..(ii + BS).min(n) {
                    let row_a = &a[i * n..(i + 1) * n];

                    for j in jj..(jj + BS).min(n) {
                        let row_bt = &bt[j * n..(j + 1) * n];

                        let partial = unsafe { dot_product_avx2_chunked(row_a, row_bt, kk, k_end) };

                        c[i * n + j] += partial;
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_1x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize, i: usize, j: usize) {
    // accumulator for 8 output values
    let mut acc = _mm256_setzero_ps();

    for k in 0..n {
        // broadcast A[i,k] into all 8 lanes
        let a_scalar = _mm256_set1_ps(a[i * n + k]);

        // load 8 contiguous B values
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(k * n + j));

        // fused multiply-add
        acc = _mm256_fmadd_ps(a_scalar, b_vec, acc);
    }

    // store results
    _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
}

pub fn matmul_microkernel_1x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        // step by 8 columns
        for j in (0..n).step_by(8) {
            unsafe {
                microkernel_1x8(a, b, c, n, i, j);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize, i: usize, j: usize) {
    // 4 accumulators
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for k in 0..n {
        // load B vector once
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(k * n + j));

        // broadcast A scalars
        let a0 = _mm256_set1_ps(a[(i + 0) * n + k]);
        let a1 = _mm256_set1_ps(a[(i + 1) * n + k]);
        let a2 = _mm256_set1_ps(a[(i + 2) * n + k]);
        let a3 = _mm256_set1_ps(a[(i + 3) * n + k]);

        // fused multiply-add
        acc0 = _mm256_fmadd_ps(a0, b_vec, acc0);
        acc1 = _mm256_fmadd_ps(a1, b_vec, acc1);
        acc2 = _mm256_fmadd_ps(a2, b_vec, acc2);
        acc3 = _mm256_fmadd_ps(a3, b_vec, acc3);
    }

    // store results
    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + j), acc0);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + j), acc1);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + j), acc2);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + j), acc3);
}

pub fn matmul_microkernel_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in (0..n).step_by(4) {
        for j in (0..n).step_by(8) {
            unsafe {
                microkernel_4x8(a, b, c, n, i, j);
            }
        }
    }
}
