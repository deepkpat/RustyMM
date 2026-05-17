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

const BS: usize = 64;

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

#[target_feature(enable = "avx2,fma")]
unsafe fn blocked_microkernel_4x8_helper(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    i: usize,
    j: usize,
    kk: usize,
    k_end: usize,
) {
    // load existing C values
    let mut acc0 = _mm256_loadu_ps(c.as_ptr().add((i + 0) * n + j));

    let mut acc1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + j));

    let mut acc2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + j));

    let mut acc3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + j));

    for k in kk..k_end {
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(k * n + j));

        let a0 = _mm256_set1_ps(a[(i + 0) * n + k]);
        let a1 = _mm256_set1_ps(a[(i + 1) * n + k]);
        let a2 = _mm256_set1_ps(a[(i + 2) * n + k]);
        let a3 = _mm256_set1_ps(a[(i + 3) * n + k]);

        acc0 = _mm256_fmadd_ps(a0, b_vec, acc0);
        acc1 = _mm256_fmadd_ps(a1, b_vec, acc1);
        acc2 = _mm256_fmadd_ps(a2, b_vec, acc2);
        acc3 = _mm256_fmadd_ps(a3, b_vec, acc3);
    }

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + j), acc0);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + j), acc1);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + j), acc2);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + j), acc3);
}

pub fn matmul_blocked_microkernel_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for ii in (0..n).step_by(BS) {
        for jj in (0..n).step_by(BS) {
            for kk in (0..n).step_by(BS) {
                let i_end = (ii + BS).min(n);
                let j_end = (jj + BS).min(n);
                let k_end = (kk + BS).min(n);

                for i in (ii..i_end).step_by(4) {
                    for j in (jj..j_end).step_by(8) {
                        unsafe {
                            blocked_microkernel_4x8_helper(a, b, c, n, i, j, kk, k_end);
                        }
                    }
                }
            }
        }
    }
}

pub fn pack_b_panel(
    b: &[f32],
    packed_b: &mut [f32],
    n: usize,
    kk: usize,
    k_end: usize,
    jj: usize,
    j_end: usize,
) {
    let block_width = j_end - jj;

    for k in kk..k_end {
        let packed_row = (k - kk) * block_width;

        let b_row = k * n;

        for j in jj..j_end {
            packed_b[packed_row + (j - jj)] = b[b_row + j];
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8_packed(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    n: usize,
    block_width: usize,
    i: usize,
    j_local: usize,
    kk: usize,
    k_end: usize,
    jj: usize,
) {
    let mut acc0 = _mm256_loadu_ps(c.as_ptr().add((i + 0) * n + jj + j_local));

    let mut acc1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + jj + j_local));

    let mut acc2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + jj + j_local));

    let mut acc3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + jj + j_local));

    for k in kk..k_end {
        let packed_row = (k - kk) * block_width;

        let b_vec = _mm256_loadu_ps(packed_b.as_ptr().add(packed_row + j_local));

        let a0 = _mm256_set1_ps(a[(i + 0) * n + k]);

        let a1 = _mm256_set1_ps(a[(i + 1) * n + k]);

        let a2 = _mm256_set1_ps(a[(i + 2) * n + k]);

        let a3 = _mm256_set1_ps(a[(i + 3) * n + k]);

        acc0 = _mm256_fmadd_ps(a0, b_vec, acc0);
        acc1 = _mm256_fmadd_ps(a1, b_vec, acc1);
        acc2 = _mm256_fmadd_ps(a2, b_vec, acc2);
        acc3 = _mm256_fmadd_ps(a3, b_vec, acc3);
    }

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + jj + j_local), acc0);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + jj + j_local), acc1);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + jj + j_local), acc2);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + jj + j_local), acc3);
}

pub fn matmul_packed_b_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut packed_b = vec![0.0f32; BS * BS];

    for ii in (0..n).step_by(BS) {
        for jj in (0..n).step_by(BS) {
            for kk in (0..n).step_by(BS) {
                let i_end = (ii + BS).min(n);
                let j_end = (jj + BS).min(n);
                let k_end = (kk + BS).min(n);

                let block_width = j_end - jj;

                // pack B panel
                pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                for i in (ii..i_end).step_by(4) {
                    for j_local in (0..block_width).step_by(8) {
                        unsafe {
                            microkernel_4x8_packed(
                                a,
                                &packed_b,
                                c,
                                n,
                                block_width,
                                i,
                                j_local,
                                kk,
                                k_end,
                                jj,
                            );
                        }
                    }
                }
            }
        }
    }
}

pub fn matmul_packed_b_4x8_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    c.par_chunks_mut(BS * n)
        .enumerate()
        .for_each(|(block_idx, c_chunk)| {
            let ii = block_idx * BS;

            let i_end = (ii + BS).min(n);

            // thread-local packed buffer
            let mut packed_b = vec![0.0f32; BS * BS];

            for jj in (0..n).step_by(BS) {
                for kk in (0..n).step_by(BS) {
                    let j_end = (jj + BS).min(n);

                    let k_end = (kk + BS).min(n);

                    let block_width = j_end - jj;

                    pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                    for i in (ii..i_end).step_by(4) {
                        for j_local in (0..block_width).step_by(8) {
                            unsafe {
                                microkernel_4x8_packed(
                                    a,
                                    &packed_b,
                                    c_chunk,
                                    n,
                                    block_width,
                                    i - ii,
                                    j_local,
                                    kk,
                                    k_end,
                                    jj,
                                );
                            }
                        }
                    }
                }
            }
        });
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_8x8_packed(
    a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    n: usize,
    block_width: usize,
    i: usize,
    j_local: usize,
    kk: usize,
    k_end: usize,
    jj: usize,
) {
    // 8 accumulators
    let mut acc0 = _mm256_loadu_ps(c.as_ptr().add((i + 0) * n + jj + j_local));

    let mut acc1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + jj + j_local));

    let mut acc2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + jj + j_local));

    let mut acc3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + jj + j_local));

    let mut acc4 = _mm256_loadu_ps(c.as_ptr().add((i + 4) * n + jj + j_local));

    let mut acc5 = _mm256_loadu_ps(c.as_ptr().add((i + 5) * n + jj + j_local));

    let mut acc6 = _mm256_loadu_ps(c.as_ptr().add((i + 6) * n + jj + j_local));

    let mut acc7 = _mm256_loadu_ps(c.as_ptr().add((i + 7) * n + jj + j_local));

    for k in kk..k_end {
        let packed_row = (k - kk) * block_width;

        let b_vec = _mm256_loadu_ps(packed_b.as_ptr().add(packed_row + j_local));

        let a0 = _mm256_set1_ps(a[(i + 0) * n + k]);
        let a1 = _mm256_set1_ps(a[(i + 1) * n + k]);
        let a2 = _mm256_set1_ps(a[(i + 2) * n + k]);
        let a3 = _mm256_set1_ps(a[(i + 3) * n + k]);
        let a4 = _mm256_set1_ps(a[(i + 4) * n + k]);
        let a5 = _mm256_set1_ps(a[(i + 5) * n + k]);
        let a6 = _mm256_set1_ps(a[(i + 6) * n + k]);
        let a7 = _mm256_set1_ps(a[(i + 7) * n + k]);

        acc0 = _mm256_fmadd_ps(a0, b_vec, acc0);
        acc1 = _mm256_fmadd_ps(a1, b_vec, acc1);
        acc2 = _mm256_fmadd_ps(a2, b_vec, acc2);
        acc3 = _mm256_fmadd_ps(a3, b_vec, acc3);
        acc4 = _mm256_fmadd_ps(a4, b_vec, acc4);
        acc5 = _mm256_fmadd_ps(a5, b_vec, acc5);
        acc6 = _mm256_fmadd_ps(a6, b_vec, acc6);
        acc7 = _mm256_fmadd_ps(a7, b_vec, acc7);
    }

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + jj + j_local), acc0);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + jj + j_local), acc1);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + jj + j_local), acc2);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + jj + j_local), acc3);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 4) * n + jj + j_local), acc4);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 5) * n + jj + j_local), acc5);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 6) * n + jj + j_local), acc6);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 7) * n + jj + j_local), acc7);
}

pub fn matmul_packed_b_8x8_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    c.par_chunks_mut(BS * n)
        .enumerate()
        .for_each(|(block_idx, c_chunk)| {
            let ii = block_idx * BS;

            let i_end = (ii + BS).min(n);

            let mut packed_b = vec![0.0f32; BS * BS];

            for jj in (0..n).step_by(BS) {
                for kk in (0..n).step_by(BS) {
                    let j_end = (jj + BS).min(n);

                    let k_end = (kk + BS).min(n);

                    let block_width = j_end - jj;

                    pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                    for i in (ii..i_end).step_by(8) {
                        for j_local in (0..block_width).step_by(8) {
                            unsafe {
                                microkernel_8x8_packed(
                                    a,
                                    &packed_b,
                                    c_chunk,
                                    n,
                                    block_width,
                                    i - ii,
                                    j_local,
                                    kk,
                                    k_end,
                                    jj,
                                );
                            }
                        }
                    }
                }
            }
        });
}

pub fn pack_a_panel_4(
    a: &[f32],
    packed_a: &mut [f32],
    n: usize,
    ii: usize,
    i_end: usize,
    kk: usize,
    k_end: usize,
) {
    let mut idx = 0;

    for k in kk..k_end {
        for i in ii..i_end {
            packed_a[idx] = a[i * n + k];
            idx += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8_packed_ab(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    n: usize,
    block_width: usize,
    i: usize,
    j_local: usize,
    kk: usize,
    k_end: usize,
    jj: usize,
) {
    // load accumulators
    let mut acc0 = _mm256_loadu_ps(c.as_ptr().add((i + 0) * n + jj + j_local));

    let mut acc1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + jj + j_local));

    let mut acc2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + jj + j_local));

    let mut acc3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + jj + j_local));

    for k in kk..k_end {
        // packed B row
        let packed_b_row = (k - kk) * block_width;

        let b_vec = _mm256_loadu_ps(packed_b.as_ptr().add(packed_b_row + j_local));

        // packed A row
        let packed_a_row = (k - kk) * 4;

        let a0 = _mm256_set1_ps(packed_a[packed_a_row + 0]);

        let a1 = _mm256_set1_ps(packed_a[packed_a_row + 1]);

        let a2 = _mm256_set1_ps(packed_a[packed_a_row + 2]);

        let a3 = _mm256_set1_ps(packed_a[packed_a_row + 3]);

        acc0 = _mm256_fmadd_ps(a0, b_vec, acc0);

        acc1 = _mm256_fmadd_ps(a1, b_vec, acc1);

        acc2 = _mm256_fmadd_ps(a2, b_vec, acc2);

        acc3 = _mm256_fmadd_ps(a3, b_vec, acc3);
    }

    // store accumulators
    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + jj + j_local), acc0);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + jj + j_local), acc1);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + jj + j_local), acc2);

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + jj + j_local), acc3);
}

pub fn matmul_packed_ab_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut packed_b = vec![0.0f32; BS * BS];

    let mut packed_a = vec![0.0f32; BS * 4];

    for ii in (0..n).step_by(BS) {
        for jj in (0..n).step_by(BS) {
            for kk in (0..n).step_by(BS) {
                let i_end = (ii + BS).min(n);

                let j_end = (jj + BS).min(n);

                let k_end = (kk + BS).min(n);

                let block_width = j_end - jj;

                // pack B panel
                pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                for i in (ii..i_end).step_by(4) {
                    // pack current A micro-panel
                    pack_a_panel_4(a, &mut packed_a, n, i, (i + 4).min(i_end), kk, k_end);

                    for j_local in (0..block_width).step_by(8) {
                        unsafe {
                            microkernel_4x8_packed_ab(
                                &packed_a,
                                &packed_b,
                                c,
                                n,
                                block_width,
                                i,
                                j_local,
                                kk,
                                k_end,
                                jj,
                            );
                        }
                    }
                }
            }
        }
    }
}

pub fn pack_a_block_4xk(
    a: &[f32],
    packed_a: &mut [f32],
    n: usize,
    ii: usize,
    i_end: usize,
    kk: usize,
    k_end: usize,
) {
    let mut idx = 0;

    for i in (ii..i_end).step_by(4) {
        for k in kk..k_end {
            packed_a[idx + 0] = a[(i + 0) * n + k];
            packed_a[idx + 1] = a[(i + 1) * n + k];
            packed_a[idx + 2] = a[(i + 2) * n + k];
            packed_a[idx + 3] = a[(i + 3) * n + k];

            idx += 4;
        }
    }
}

pub fn matmul_packed_ab_4x8_precompute(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // packed B block
    let mut packed_b = vec![0.0f32; BS * BS];

    // packed A block
    let mut packed_a = vec![0.0f32; BS * BS];

    for ii in (0..n).step_by(BS) {
        for kk in (0..n).step_by(BS) {
            let i_end = (ii + BS).min(n);

            let k_end = (kk + BS).min(n);

            // pack ENTIRE A block once
            pack_a_block_4xk(a, &mut packed_a, n, ii, i_end, kk, k_end);

            // reuse packed A across ALL jj blocks
            for jj in (0..n).step_by(BS) {
                let j_end = (jj + BS).min(n);

                let block_width = j_end - jj;

                // pack B panel
                pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                // iterate micro-panels
                for i in (ii..i_end).step_by(4) {
                    // which packed A micro-panel?
                    let panel_idx = (i - ii) / 4;

                    let panel_stride = (k_end - kk) * 4;

                    let packed_a_offset = panel_idx * panel_stride;

                    for j_local in (0..block_width).step_by(8) {
                        unsafe {
                            microkernel_4x8_packed_ab(
                                &packed_a[packed_a_offset..],
                                &packed_b,
                                c,
                                n,
                                block_width,
                                i,
                                j_local,
                                kk,
                                k_end,
                                jj,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8_blis(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    n: usize,
    block_width: usize,
    i: usize,
    j_local: usize,
    k_size: usize,
    jj: usize,
) {
    let mut c0 = _mm256_loadu_ps(c.as_ptr().add((i + 0) * n + jj + j_local));
    let mut c1 = _mm256_loadu_ps(c.as_ptr().add((i + 1) * n + jj + j_local));
    let mut c2 = _mm256_loadu_ps(c.as_ptr().add((i + 2) * n + jj + j_local));
    let mut c3 = _mm256_loadu_ps(c.as_ptr().add((i + 3) * n + jj + j_local));

    for k in 0..k_size {
        let b = _mm256_loadu_ps(packed_b.as_ptr().add(k * block_width + j_local));

        let a_base = k * 4;

        let a0 = _mm256_broadcast_ss(&packed_a[a_base + 0]);
        let a1 = _mm256_broadcast_ss(&packed_a[a_base + 1]);
        let a2 = _mm256_broadcast_ss(&packed_a[a_base + 2]);
        let a3 = _mm256_broadcast_ss(&packed_a[a_base + 3]);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }

    _mm256_storeu_ps(c.as_mut_ptr().add((i + 0) * n + jj + j_local), c0);
    _mm256_storeu_ps(c.as_mut_ptr().add((i + 1) * n + jj + j_local), c1);
    _mm256_storeu_ps(c.as_mut_ptr().add((i + 2) * n + jj + j_local), c2);
    _mm256_storeu_ps(c.as_mut_ptr().add((i + 3) * n + jj + j_local), c3);
}

const MC: usize = 128;
const NC: usize = 128;
const KC: usize = 64;

pub fn matmul_blis_4x8(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut packed_a = vec![0.0f32; MC * KC];
    let mut packed_b = vec![0.0f32; KC * NC];

    for ii in (0..n).step_by(MC) {
        let i_end = (ii + MC).min(n);

        for kk in (0..n).step_by(KC) {
            let k_end = (kk + KC).min(n);
            let k_size = k_end - kk;

            // PACK A ONCE PER (ii, kk)
            pack_a_block_4xk(a, &mut packed_a, n, ii, i_end, kk, k_end);

            for jj in (0..n).step_by(NC) {
                let j_end = (jj + NC).min(n);
                let block_width = j_end - jj;

                // PACK B ONCE PER (kk, jj)
                pack_b_panel(b, &mut packed_b, n, kk, k_end, jj, j_end);

                // MICROKERNELS
                for i in (ii..i_end).step_by(4) {
                    for j_local in (0..block_width).step_by(8) {
                        unsafe {
                            microkernel_4x8_blis(
                                &packed_a,
                                &packed_b,
                                c,
                                n,
                                block_width,
                                i,
                                j_local,
                                k_size,
                                jj,
                            );
                        }
                    }
                }
            }
        }
    }
}
