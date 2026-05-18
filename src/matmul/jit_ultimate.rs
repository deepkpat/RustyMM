use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_4x16_local(
    local_a: &[f32; 64 * 64],
    local_b: &[f32; 64 * 64],
    local_c: &mut [f32; 64 * 64],
) {
    // Standard 4x16 execution loop inside the L1 Cache
    for i in (0..64).step_by(4) {
        for j in (0..64).step_by(16) {
            let mut c0_0 = _mm256_loadu_ps(local_c.as_ptr().add(i * 64 + j));
            let mut c0_1 = _mm256_loadu_ps(local_c.as_ptr().add(i * 64 + j + 8));
            let mut c1_0 = _mm256_loadu_ps(local_c.as_ptr().add((i + 1) * 64 + j));
            let mut c1_1 = _mm256_loadu_ps(local_c.as_ptr().add((i + 1) * 64 + j + 8));
            let mut c2_0 = _mm256_loadu_ps(local_c.as_ptr().add((i + 2) * 64 + j));
            let mut c2_1 = _mm256_loadu_ps(local_c.as_ptr().add((i + 2) * 64 + j + 8));
            let mut c3_0 = _mm256_loadu_ps(local_c.as_ptr().add((i + 3) * 64 + j));
            let mut c3_1 = _mm256_loadu_ps(local_c.as_ptr().add((i + 3) * 64 + j + 8));

            for k in 0..64 {
                let b_val_0 = _mm256_loadu_ps(local_b.as_ptr().add(k * 64 + j));
                let b_val_1 = _mm256_loadu_ps(local_b.as_ptr().add(k * 64 + j + 8));

                let a0 = _mm256_set1_ps(local_a[i * 64 + k]);
                let a1 = _mm256_set1_ps(local_a[(i + 1) * 64 + k]);
                let a2 = _mm256_set1_ps(local_a[(i + 2) * 64 + k]);
                let a3 = _mm256_set1_ps(local_a[(i + 3) * 64 + k]);

                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);
            }

            _mm256_storeu_ps(local_c.as_mut_ptr().add(i * 64 + j), c0_0);
            _mm256_storeu_ps(local_c.as_mut_ptr().add(i * 64 + j + 8), c0_1);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 1) * 64 + j), c1_0);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 1) * 64 + j + 8), c1_1);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 2) * 64 + j), c2_0);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 2) * 64 + j + 8), c2_1);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 3) * 64 + j), c3_0);
            _mm256_storeu_ps(local_c.as_mut_ptr().add((i + 3) * 64 + j + 8), c3_1);
        }
    }
}

pub fn matmul_jit_ultimate(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 64;

    // Parallelize over 64-row blocks of Matrix C
    c.par_chunks_mut(64 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            // Loop over column blocks of C
            for jj_block in 0..blocks_per_dim {
                // Allocate 16 KB destination buffer directly on this thread's stack
                let mut local_c = [0.0f32; 64 * 64];

                // Stream global C into our local L1 cache buffer
                for i in 0..64 {
                    for j in 0..64 {
                        local_c[i * 64 + j] = c_chunk[i * n + (jj_block * 64 + j)];
                    }
                }

                // Walk along the K-dimension updating our local stack tile
                for kk_block in 0..blocks_per_dim {
                    let mut local_a = [0.0f32; 64 * 64];
                    let mut local_b = [0.0f32; 64 * 64];

                    // JIT Pack Block A (Perfect stride-1 sequential reads)
                    for i in 0..64 {
                        for k in 0..64 {
                            local_a[i * 64 + k] = a[(ii_block * 64 + i) * n + (kk_block * 64 + k)];
                        }
                    }

                    // JIT Pack Block B (Perfect stride-1 sequential reads - Strides Corrected!)
                    for k in 0..64 {
                        for j in 0..64 {
                            local_b[k * 64 + j] = b[(kk_block * 64 + k) * n + (jj_block * 64 + j)];
                        }
                    }

                    // Execute math inside the isolated core caches
                    unsafe {
                        micro_kernel_4x16_local(&local_a, &local_b, &mut local_c);
                    }
                }

                // Write the finished 16 KB block back to global memory exactly once
                for i in 0..64 {
                    for j in 0..64 {
                        c_chunk[i * n + (jj_block * 64 + j)] = local_c[i * 64 + j];
                    }
                }
            }
        });
}
