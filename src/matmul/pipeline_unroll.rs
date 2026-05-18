use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn macro_kernel_32_pipelined(
    packed_a_block_row: *const f32,
    packed_b: *const f32,
    c_chunk_ptr: *mut f32,
    jj_block: usize,
    blocks_per_dim: usize,
    n: usize,
) {
    let jj = jj_block * 32;

    for i in (0..32).step_by(8) {
        for j in (0..32).step_by(8) {
            let mut c0 = _mm256_loadu_ps(c_chunk_ptr.add(i * n + (jj + j)));
            let mut c1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 1) * n + (jj + j)));
            let mut c2 = _mm256_loadu_ps(c_chunk_ptr.add((i + 2) * n + (jj + j)));
            let mut c3 = _mm256_loadu_ps(c_chunk_ptr.add((i + 3) * n + (jj + j)));
            let mut c4 = _mm256_loadu_ps(c_chunk_ptr.add((i + 4) * n + (jj + j)));
            let mut c5 = _mm256_loadu_ps(c_chunk_ptr.add((i + 5) * n + (jj + j)));
            let mut c6 = _mm256_loadu_ps(c_chunk_ptr.add((i + 6) * n + (jj + j)));
            let mut c7 = _mm256_loadu_ps(c_chunk_ptr.add((i + 7) * n + (jj + j)));

            for kk in 0..blocks_per_dim {
                let a_block = packed_a_block_row.add(kk * 1024);
                let b_block = packed_b.add((kk * blocks_per_dim + jj_block) * 1024);

                // UNROLLED BY 2: We jump by 2 instead of 1 to feed instruction pipelines
                for k in (0..32).step_by(2) {
                    // === STREAM STEP 0 (k) ===
                    let b_val_0 = _mm256_loadu_ps(b_block.add(k * 32 + j));
                    let a0_0 = _mm256_set1_ps(*a_block.add(i * 32 + k));
                    let a1_0 = _mm256_set1_ps(*a_block.add((i + 1) * 32 + k));
                    let a2_0 = _mm256_set1_ps(*a_block.add((i + 2) * 32 + k));
                    let a3_0 = _mm256_set1_ps(*a_block.add((i + 3) * 32 + k));
                    let a4_0 = _mm256_set1_ps(*a_block.add((i + 4) * 32 + k));
                    let a5_0 = _mm256_set1_ps(*a_block.add((i + 5) * 32 + k));
                    let a6_0 = _mm256_set1_ps(*a_block.add((i + 6) * 32 + k));
                    let a7_0 = _mm256_set1_ps(*a_block.add((i + 7) * 32 + k));

                    // === STREAM STEP 1 (k + 1) ===
                    // Loaded immediately so the CPU overlaps this with step 0's math!
                    let b_val_1 = _mm256_loadu_ps(b_block.add((k + 1) * 32 + j));
                    let a0_1 = _mm256_set1_ps(*a_block.add(i * 32 + k + 1));
                    let a1_1 = _mm256_set1_ps(*a_block.add((i + 1) * 32 + k + 1));
                    let a2_1 = _mm256_set1_ps(*a_block.add((i + 2) * 32 + k + 1));
                    let a3_1 = _mm256_set1_ps(*a_block.add((i + 3) * 32 + k + 1));
                    let a4_1 = _mm256_set1_ps(*a_block.add((i + 4) * 32 + k + 1));
                    let a5_1 = _mm256_set1_ps(*a_block.add((i + 5) * 32 + k + 1));
                    let a6_1 = _mm256_set1_ps(*a_block.add((i + 6) * 32 + k + 1));
                    let a7_1 = _mm256_set1_ps(*a_block.add((i + 7) * 32 + k + 1));

                    // === ACCUMULATE MATH STEP 0 ===
                    c0 = _mm256_fmadd_ps(a0_0, b_val_0, c0);
                    c1 = _mm256_fmadd_ps(a1_0, b_val_0, c1);
                    c2 = _mm256_fmadd_ps(a2_0, b_val_0, c2);
                    c3 = _mm256_fmadd_ps(a3_0, b_val_0, c3);
                    c4 = _mm256_fmadd_ps(a4_0, b_val_0, c4);
                    c5 = _mm256_fmadd_ps(a5_0, b_val_0, c5);
                    c6 = _mm256_fmadd_ps(a6_0, b_val_0, c6);
                    c7 = _mm256_fmadd_ps(a7_0, b_val_0, c7);

                    // === ACCUMULATE MATH STEP 1 ===
                    c0 = _mm256_fmadd_ps(a0_1, b_val_1, c0);
                    c1 = _mm256_fmadd_ps(a1_1, b_val_1, c1);
                    c2 = _mm256_fmadd_ps(a2_1, b_val_1, c2);
                    c3 = _mm256_fmadd_ps(a3_1, b_val_1, c3);
                    c4 = _mm256_fmadd_ps(a4_1, b_val_1, c4);
                    c5 = _mm256_fmadd_ps(a5_1, b_val_1, c5);
                    c6 = _mm256_fmadd_ps(a6_1, b_val_1, c6);
                    c7 = _mm256_fmadd_ps(a7_1, b_val_1, c7);
                }
            }

            _mm256_storeu_ps(c_chunk_ptr.add(i * n + (jj + j)), c0);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 1) * n + (jj + j)), c1);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 2) * n + (jj + j)), c2);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 3) * n + (jj + j)), c3);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 4) * n + (jj + j)), c4);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 5) * n + (jj + j)), c5);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 6) * n + (jj + j)), c6);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 7) * n + (jj + j)), c7);
        }
    }
}

pub fn matmul_pipeline_unroll(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 32;
    let mut packed_a = vec![0.0f32; n * n];
    let mut packed_b = vec![0.0f32; n * n];

    for ii in 0..blocks_per_dim {
        for kk in 0..blocks_per_dim {
            let block_offset = (ii * blocks_per_dim + kk) * 1024;
            for i in 0..32 {
                for j in 0..32 {
                    packed_a[block_offset + i * 32 + j] = a[(ii * 32 + i) * n + (kk * 32 + j)];
                }
            }
        }
    }
    for kk in 0..blocks_per_dim {
        for jj in 0..blocks_per_dim {
            let block_offset = (kk * blocks_per_dim + jj) * 1024;
            for k in 0..32 {
                for j in 0..32 {
                    packed_b[block_offset + k * 32 + j] = b[(kk * 32 + k) * n + (jj * 32 + j)];
                }
            }
        }
    }

    c.par_chunks_mut(32 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            let a_ptr = packed_a.as_ptr();
            let b_ptr = packed_b.as_ptr();
            let c_chunk_ptr = c_chunk.as_mut_ptr();

            unsafe {
                let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 1024);
                for jj_block in 0..blocks_per_dim {
                    macro_kernel_32_pipelined(
                        packed_a_block_row,
                        b_ptr,
                        c_chunk_ptr,
                        jj_block,
                        blocks_per_dim,
                        n,
                    );
                }
            }
        });
}
