use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn macro_kernel_32_parallel(
    packed_a_block_row: *const f32, // Pointer to the block-row of A for this thread
    packed_b: *const f32,           // Shared read-only global packed B
    c_chunk_ptr: *mut f32,          // Thread-local destination pointer for a C chunk
    jj_block: usize,
    blocks_per_dim: usize,
    n: usize,
) {
    let jj = jj_block * 32;

    // Process the 32x32 macro-block in micro-tiles of 4x8
    for i in (0..32).step_by(4) {
        for j in (0..32).step_by(8) {
            // Load C registers from our thread's safe, isolated slice pointer
            let mut c0 = _mm256_loadu_ps(c_chunk_ptr.add(i * n + (jj + j)));
            let mut c1 = _mm256_loadu_ps(c_chunk_ptr.add((i + 1) * n + (jj + j)));
            let mut c2 = _mm256_loadu_ps(c_chunk_ptr.add((i + 2) * n + (jj + j)));
            let mut c3 = _mm256_loadu_ps(c_chunk_ptr.add((i + 3) * n + (jj + j)));

            for kk in 0..blocks_per_dim {
                let a_block = packed_a_block_row.add(kk * 1024);
                let b_block = packed_b.add((kk * blocks_per_dim + jj_block) * 1024);

                for k in 0..32 {
                    let b_val = _mm256_loadu_ps(b_block.add(k * 32 + j));

                    let a0 = _mm256_set1_ps(*a_block.add(i * 32 + k));
                    let a1 = _mm256_set1_ps(*a_block.add((i + 1) * 32 + k));
                    let a2 = _mm256_set1_ps(*a_block.add((i + 2) * 32 + k));
                    let a3 = _mm256_set1_ps(*a_block.add((i + 3) * 32 + k));

                    c0 = _mm256_fmadd_ps(a0, b_val, c0);
                    c1 = _mm256_fmadd_ps(a1, b_val, c1);
                    c2 = _mm256_fmadd_ps(a2, b_val, c2);
                    c3 = _mm256_fmadd_ps(a3, b_val, c3);
                }
            }

            // Write back to our isolated thread partition
            _mm256_storeu_ps(c_chunk_ptr.add(i * n + (jj + j)), c0);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 1) * n + (jj + j)), c1);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 2) * n + (jj + j)), c2);
            _mm256_storeu_ps(c_chunk_ptr.add((i + 3) * n + (jj + j)), c3);
        }
    }
}

pub fn matmul_parallel_packed(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 32;

    let mut packed_a = vec![0.0f32; n * n];
    let mut packed_b = vec![0.0f32; n * n];

    // Upfront sequential packing (negligible cost relative to computation)
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

    // Parallelize over 32-row blocks of Matrix C
    c.par_chunks_mut(32 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            // Immutable references to packed vectors are safe to access across Rayon threads
            let a_ptr = packed_a.as_ptr();
            let b_ptr = packed_b.as_ptr();
            let c_chunk_ptr = c_chunk.as_mut_ptr();

            unsafe {
                // Pinpoint the exact base pointer for this specific thread's row-block of A
                let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 1024);

                for jj_block in 0..blocks_per_dim {
                    macro_kernel_32_parallel(
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
