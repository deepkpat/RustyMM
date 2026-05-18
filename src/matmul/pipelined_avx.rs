use rayon::prelude::*;
use std::arch::x86_64::{__m256, _mm256_fmadd_ps, _mm256_load_ps, _mm256_set1_ps, _mm256_store_ps};

// Guarantee exact 32-byte alignment on the stack for seamless AVX2 execution
#[repr(C, align(32))]
pub struct AlignedBlock {
    pub data: [f32; 64 * 64],
}

impl AlignedBlock {
    #[inline(always)]
    pub fn new() -> Self {
        AlignedBlock {
            data: [0.0f32; 64 * 64],
        }
    }
}

/// Software-Pipelined 4x16 Register Micro-Kernel
/// Manually interleaves memory loads for iteration K+1 during the math execution of iteration K
#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_4x16_pipelined(
    local_a: &AlignedBlock,
    local_b: &AlignedBlock,
    local_c: &mut AlignedBlock,
) {
    let a_ptr = local_a.data.as_ptr();
    let b_ptr = local_b.data.as_ptr();
    let c_ptr = local_c.data.as_mut_ptr();

    for i in (0..64).step_by(4) {
        for j in (0..64).step_by(16) {
            // Load destination accumulation registers (Guaranteed 32-Byte Aligned)
            let mut c0_0 = _mm256_load_ps(c_ptr.add(i * 64 + j));
            let mut c0_1 = _mm256_load_ps(c_ptr.add(i * 64 + j + 8));
            let mut c1_0 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j));
            let mut c1_1 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j + 8));
            let mut c2_0 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j));
            let mut c2_1 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j + 8));
            let mut c3_0 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j));
            let mut c3_1 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j + 8));

            // PROLOGUE: Pre-fetch data for the first step (k = 0) before entering the loop
            let mut b_val_0 = _mm256_load_ps(b_ptr.add(0 * 64 + j));
            let mut b_val_1 = _mm256_load_ps(b_ptr.add(0 * 64 + j + 8));
            let mut a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + 0));
            let mut a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + 0));
            let mut a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + 0));
            let mut a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + 0));

            // Loop unrolled by 2 to manage staggered pipeline loading cleanly
            for k in (0..64).step_by(2) {
                // --- STEP 1: EXECUTE CURRENT K ---
                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);

                // INTERLEAVE: Load partial data for next step (K + 1) during execution latency
                a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k + 1));
                a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k + 1));

                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);

                // INTERLEAVE: Fetch remaining data for step (K + 1)
                b_val_0 = _mm256_load_ps(b_ptr.add((k + 1) * 64 + j));
                b_val_1 = _mm256_load_ps(b_ptr.add((k + 1) * 64 + j + 8));
                a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k + 1));
                a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k + 1));

                // --- STEP 2: EXECUTE STEP K + 1 ---
                c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
                c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
                c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
                c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);

                // INTERLEAVE: Look-ahead load for the upcoming loop iteration (K + 2)
                if k + 2 < 64 {
                    a0 = _mm256_set1_ps(*a_ptr.add(i * 64 + k + 2));
                    a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * 64 + k + 2));
                }

                c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
                c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
                c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
                c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);

                if k + 2 < 64 {
                    b_val_0 = _mm256_load_ps(b_ptr.add((k + 2) * 64 + j));
                    b_val_1 = _mm256_load_ps(b_ptr.add((k + 2) * 64 + j + 8));
                    a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * 64 + k + 2));
                    a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * 64 + k + 2));
                }
            }

            // Write back results directly to the aligned local stack block
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

/// Main entry point for the Pipelined AVX2 Optimization
pub fn matmul_pipelined_avx(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 64;

    // Isolate work to 64-row horizontal chunks of Matrix C across threads
    c.par_chunks_mut(64 * n)
        .enumerate()
        .for_each(|(ii_block, c_chunk)| {
            for jj_block in 0..blocks_per_dim {
                // Zero-allocation thread stack workspace (16 KB)
                let mut local_c = AlignedBlock::new();

                // Initial layout migration into local L1 cache storage
                for i in 0..64 {
                    for j in 0..64 {
                        local_c.data[i * 64 + j] = c_chunk[i * n + (jj_block * 64 + j)];
                    }
                }

                // Traverse depth parameters packing and calculating dynamically
                for kk_block in 0..blocks_per_dim {
                    let mut local_a = AlignedBlock::new();
                    let mut local_b = AlignedBlock::new();

                    // JIT Stack Pack Block A (Perfect stride-1 sequential reading)
                    for i in 0..64 {
                        for k in 0..64 {
                            local_a.data[i * 64 + k] =
                                a[(ii_block * 64 + i) * n + (kk_block * 64 + k)];
                        }
                    }

                    // JIT Stack Pack Block B (Perfect stride-1 sequential reading)
                    for k in 0..64 {
                        for j in 0..64 {
                            local_b.data[k * 64 + j] =
                                b[(kk_block * 64 + k) * n + (jj_block * 64 + j)];
                        }
                    }

                    // Call the pipelined mathematical calculation core
                    unsafe {
                        micro_kernel_4x16_pipelined(&local_a, &local_b, &mut local_c);
                    }
                }

                // Push calculated local state back to global memory space exactly once
                for i in 0..64 {
                    for j in 0..64 {
                        c_chunk[i * n + (jj_block * 64 + j)] = local_c.data[i * 64 + j];
                    }
                }
            }
        });
}
