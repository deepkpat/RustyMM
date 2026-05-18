use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn macro_kernel_64_scaled(
    packed_a_block_row: *const f32,
    packed_b: *const f32,
    c_chunk_ptr: *mut f32,
    jj_block: usize,
    blocks_per_dim: usize,
    n: usize,
) {
    let jj = jj_block * 64;

    // Process the expanded 64x64 macro-block using our elite 8x8 register tiles
    for i in (0..64).step_by(8) {
        for j in (0..64).step_by(8) {
            let mut c0 = _mm256_loadu_ps(c_chunk_ptr.add(i * n + (jj + j)));
            let mut c1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 1) * n + (jj + j)));
            let mut c2 = _mm256_loadu_ps(c_chunk_ptr.add((i + 2) * n + (jj + j)));
            let mut c3 = _mm256_loadu_ps(c_chunk_ptr.add((i + 3) * n + (jj + j)));
            let mut c4 = _mm256_loadu_ps(c_chunk_ptr.add((i + 4) * n + (jj + j)));
            let mut c5 = _mm256_loadu_ps(c_chunk_ptr.add((i + 5) * n + (jj + j)));
            let mut c6 = _mm256_loadu_ps(c_chunk_ptr.add((i + 6) * n + (jj + j)));
            let mut c7 = _mm256_loadu_ps(c_chunk_ptr.add((i + 7) * n + (jj + j)));

            for kk in 0..blocks_per_dim {
                // Each packed sub-block now contains 64x64 = 4096 elements
                let a_block = packed_a_block_row.add(kk * 4096);
                let b_block = packed_b.add((kk * blocks_per_dim + jj_block) * 4096);

                // Deep execution loop scaled up to 64 iterations for maximized data reuse
                for k in 0..64 {
                    let b_val = _mm256_loadu_ps(b_block.add(k * 64 + j));

                    let a0 = _mm256_set1_ps(*a_block.add(i * 64 + k));
                    let a1 = _mm256_set1_ps(*a_block.add((i + 1) * 64 + k));
                    let a2 = _mm256_set1_ps(*a_block.add((i + 2) * 64 + k));
                    let a3 = _mm256_set1_ps(*a_block.add((i + 3) * 64 + k));
                    let a4 = _mm256_set1_ps(*a_block.add((i + 4) * 64 + k));
                    let a5 = _mm256_set1_ps(*a_block.add((i + 5) * 64 + k));
                    let a6 = _mm256_set1_ps(*a_block.add((i + 6) * 64 + k));
                    let a7 = _mm256_set1_ps(*a_block.add((i + 7) * 64 + k));

                    c0 = _mm256_fmadd_ps(a0, b_val, c0);
                    c1 = _mm256_fmadd_ps(a1, b_val, c1);
                    c2 = _mm256_fmadd_ps(a2, b_val, c2);
                    c3 = _mm256_fmadd_ps(a3, b_val, c3);
                    c4 = _mm256_fmadd_ps(a4, b_val, c4);
                    c5 = _mm256_fmadd_ps(a5, b_val, c5);
                    c6 = _mm256_fmadd_ps(a6, b_val, c6);
                    c7 = _mm256_fmadd_ps(a7, b_val, c7);
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

pub fn matmul_cache_tuned(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 64;

    let mut packed_a = vec![0.0f32; n * n];
    let mut packed_b = vec![0.0f32; n * n];

    // Parallel packing over 64x64 blocks (4096 floats per block chunk)
    packed_a
        .par_chunks_exact_mut(4096)
        .enumerate()
        .for_each(|(block_idx, block)| {
            let ii = block_idx / blocks_per_dim;
            let kk = block_idx % blocks_per_dim;
            for i in 0..64 {
                for j in 0..64 {
                    block[i * 64 + j] = a[(ii * 64 + i) * n + (kk * 64 + j)];
                }
            }
        });

    packed_b
        .par_chunks_exact_mut(4096)
        .enumerate()
        .for_each(|(block_idx, block)| {
            let kk = block_idx / blocks_per_dim;
            let jj = block_idx % blocks_per_dim;
            for k in 0..64 {
                for j in 0..64 {
                    block[k * 64 + j] = b[(kk * 64 + k) * n + (jj * 64 + j)];
                }
            }
        });

    // Parallel calculation over 64-row slices of Matrix C
    c.par_chunks_mut(64 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            let a_ptr = packed_a.as_ptr();
            let b_ptr = packed_b.as_ptr();
            let c_chunk_ptr = c_chunk.as_mut_ptr();

            unsafe {
                let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 4096);
                for jj_block in 0..blocks_per_dim {
                    macro_kernel_64_scaled(
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
