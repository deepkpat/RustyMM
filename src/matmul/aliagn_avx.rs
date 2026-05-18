use rayon::prelude::*;
use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_store_ps,
    _mm256_storeu_ps,
};

// Force exact 32-byte alignment for AVX2 YMM registers
#[repr(C, align(32))]
struct AlignedBlock {
    data: [f32; 64 * 64],
}

impl AlignedBlock {
    #[inline(always)]
    fn new() -> Self {
        AlignedBlock {
            data: [0.0f32; 64 * 64],
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_4x16_aligned(
    local_a: &AlignedBlock,
    local_b: &AlignedBlock,
    local_c: &mut AlignedBlock,
) {
    let a_ptr = local_a.data.as_ptr();
    let b_ptr = local_b.data.as_ptr();
    let c_ptr = local_c.data.as_mut_ptr();

    for i in (0..64).step_by(4) {
        for j in (0..64).step_by(16) {
            // ALIGNED Loads for the destination registers
            let mut c0_0 = _mm256_load_ps(c_ptr.add(i * 64 + j));
            let mut c0_1 = _mm256_load_ps(c_ptr.add(i * 64 + j + 8));
            let mut c1_0 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j));
            let mut c1_1 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j + 8));
            let mut c2_0 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j));
            let mut c2_1 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j + 8));
            let mut c3_0 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j));
            let mut c3_1 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j + 8));

            // Unroll K by 4 to crush branch prediction overhead
            for k in (0..64).step_by(4) {
                // --- UNROLL 0 ---
                let mut b_val_0 = _mm256_load_ps(b_ptr.add(k * 64 + j));
                let mut b_val_1 = _mm256_load_ps(b_ptr.add(k * 64 + j + 8));
                let mut a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k));
                let mut a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k));
                let mut a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k));
                let mut a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k));

                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);

                // --- UNROLL 1 ---
                b_val_0 = _mm256_load_ps(b_ptr.add((k + 1) * 64 + j));
                b_val_1 = _mm256_load_ps(b_ptr.add((k + 1) * 64 + j + 8));
                a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k + 1));
                a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k + 1));
                a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k + 1));
                a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k + 1));

                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);

                // --- UNROLL 2 ---
                b_val_0 = _mm256_load_ps(b_ptr.add((k + 2) * 64 + j));
                b_val_1 = _mm256_load_ps(b_ptr.add((k + 2) * 64 + j + 8));
                a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k + 2));
                a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k + 2));
                a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k + 2));
                a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k + 2));

                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);

                // --- UNROLL 3 ---
                b_val_0 = _mm256_load_ps(b_ptr.add((k + 3) * 64 + j));
                b_val_1 = _mm256_load_ps(b_ptr.add((k + 3) * 64 + j + 8));
                a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k + 3));
                a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k + 3));
                a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k + 3));
                a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k + 3));

                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);
            }

            // ALIGNED Stores back to the local buffer
            _mm256_store_ps(c_ptr.add(i * 64 + j), c0_0);
            _mm256_store_ps(c_ptr.add(i * 64 + j + 8), c0_1);
            _mm256_store_ps(c_ptr.add((i + 1) * 64 + j), c1_0);
            _mm256_store_ps(c_ptr.add((i + 1) * 64 + j + 8), c1_1);
            _mm256_store_ps(c_ptr.add((i + 2) * 64 + j), c2_0);
            _mm256_store_ps(c_ptr.add((i + 2) * 64 + j + 8), c2_1);
            _mm256_store_ps(c_ptr.add((i + 3) * 64 + j), c3_0);
            _mm256_store_ps(c_ptr.add((i + 3) * 64 + j + 8), c3_1);
        }
    }
}

pub fn matmul_aligned_avx(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 64;

    c.par_chunks_mut(64 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            for jj_block in 0..blocks_per_dim {
                let mut local_c = AlignedBlock::new();

                // Initial load from global to aligned local
                for i in 0..64 {
                    for j in 0..64 {
                        local_c.data[i * 64 + j] = c_chunk[i * n + (jj_block * 64 + j)];
                    }
                }

                for kk_block in 0..blocks_per_dim {
                    let mut local_a = AlignedBlock::new();
                    let mut local_b = AlignedBlock::new();

                    for i in 0..64 {
                        for k in 0..64 {
                            local_a.data[i * 64 + k] =
                                a[(ii_block * 64 + i) * n + (kk_block * 64 + k)];
                        }
                    }

                    for k in 0..64 {
                        for j in 0..64 {
                            local_b.data[k * 64 + j] =
                                b[(kk_block * 64 + k) * n + (jj_block * 64 + j)];
                        }
                    }

                    unsafe {
                        micro_kernel_4x16_aligned(&local_a, &local_b, &mut local_c);
                    }
                }

                // Final store from aligned local back to global
                for i in 0..64 {
                    for j in 0..64 {
                        c_chunk[i * n + (jj_block * 64 + j)] = local_c.data[i * 64 + j];
                    }
                }
            }
        });
}
