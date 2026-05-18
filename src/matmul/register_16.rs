use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_4x16(
    packed_a_block_row: *const f32,
    packed_b: *const f32,
    c_chunk_ptr: *mut f32,
    jj_block: usize,
    blocks_per_dim: usize,
    n: usize,
) {
    let jj = jj_block * 64;

    // Loop rows by 4, columns by 16 (2 YMM registers wide)
    for i in (0..64).step_by(4) {
        for j in (0..64).step_by(16) {
            // Allocate 8 registers for our 4x16 Accumulation Tile
            let mut c0_0 = _mm256_loadu_ps(c_chunk_ptr.add(i * n + j));
            let mut c0_1 = _mm256_loadu_ps(c_chunk_ptr.add(i * n + j + 8));

            let mut c1_0 = _mm256_loadu_ps(c_chunk_ptr.add((i + 1) * n + j));
            let mut c1_1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 1) * n + j + 8));

            let mut c2_0 = _mm256_loadu_ps(c_chunk_ptr.add((i + 2) * n + j));
            let mut c2_1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 2) * n + j + 8));

            let mut c3_0 = _mm256_loadu_ps(c_chunk_ptr.add((i + 3) * n + j));
            let mut c3_1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 3) * n + j + 8));

            for kk in 0..blocks_per_dim {
                let a_block = packed_a_block_row.add(kk * 4096);
                let b_block = packed_b.add((kk * blocks_per_dim + jj_block) * 4096);

                for k in 0..64 {
                    // Load 2 sequential vector registers from B (16 elements total)
                    let b_val_0 = _mm256_loadu_ps(b_block.add(k * 64 + j));
                    let b_val_1 = _mm256_loadu_ps(b_block.add(k * 64 + j + 8));

                    // Broadcast 4 elements from A
                    let a0 = _mm256_set1_ps(*a_block.add(i * 64 + k));
                    let a1 = _mm256_set1_ps(*a_block.add((i + 1) * 64 + k));
                    let a2 = _mm256_set1_ps(*a_block.add((i + 2) * 64 + k));
                    let a3 = _mm256_set1_ps(*a_block.add((i + 3) * 64 + k));

                    // Execute 8 FMAs with extreme register reuse!
                    c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                    c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);

                    c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                    c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);

                    c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                    c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);

                    c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                    c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);
                }
            }

            // Write back the finished 4x16 block to main memory
            _mm256_storeu_ps(c_chunk_ptr.add(i * n + j), c0_0);
            _mm256_storeu_ps(c_chunk_ptr.add(i * n + j + 8), c0_1);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 1) * n + j), c1_0);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 1) * n + j + 8), c1_1);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 2) * n + j), c2_0);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 2) * n + j + 8), c2_1);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 3) * n + j), c3_0);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 3) * n + j + 8), c3_1);
        }
    }
}

pub fn matmul_register_16(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 64;
    let mut packed_a = vec![0.0f32; n * n];
    let mut packed_b = vec![0.0f32; n * n];

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
                    block[k * 64 + j] = b[(kk * 32 + k) * n + (jj * 64 + j)];
                }
            }
        });

    c.par_chunks_mut(64 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            let a_ptr = packed_a.as_ptr();
            let b_ptr = packed_b.as_ptr();
            let c_chunk_ptr = c_chunk.as_mut_ptr();

            unsafe {
                let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 4096);
                for jj_block in 0..blocks_per_dim {
                    micro_kernel_4x16(
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
