use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_32_simd(
    a_block: &[f32; 32 * 32],
    b_block: &[f32; 32 * 32],
    c_block: &mut [f32; 32 * 32],
) {
    for i in 0..32 {
        for k in 0..32 {
            // broadcast the single element a[i, k] into all 8 lanes of a register
            let a_idx = i * 32 + k;
            let a_vec: __m256 = _mm256_set1_ps(a_block[a_idx]);

            // step by 8 because we process 8 floats at a time
            for j in (0..32).step_by(8) {
                let b_idx = k * 32 + j;
                let c_idx = i * 32 + j;

                // load 8 sequential elements of matrix B and C
                // we use raw pointers to feed the simd load instructions
                let b_vec: __m256 = unsafe { _mm256_loadu_ps(b_block.as_ptr().add(b_idx)) };
                let c_vec: __m256 = unsafe { _mm256_loadu_ps(c_block.as_ptr().add(c_idx)) };

                // perform: C = (A * B) + C in a single clock cycle
                let result_vec: __m256 = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                // store the 8 computed values back into Matrix C
                unsafe { _mm256_storeu_ps(c_block.as_mut_ptr().add(c_idx), result_vec) };
            }
        }
    }
}

pub fn matmul_tiled_simd(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut a_block = [0.0f32; 32 * 32];
    let mut b_block = [0.0f32; 32 * 32];
    let mut c_block = [0.0f32; 32 * 32];

    for ii in (0..n).step_by(32) {
        for jj in (0..n).step_by(32) {
            c_block.fill(0.0);

            for kk in (0..n).step_by(32) {
                // pack global memory into L1 stack blocks
                for i in 0..32 {
                    for j in 0..32 {
                        a_block[i * 32 + j] = a[(ii + i) * n + (kk + j)];
                        b_block[i * 32 + j] = b[(kk + i) * n + (jj + j)];
                    }
                }

                // call our vectorized hardware engine safely
                unsafe {
                    micro_kernel_32_simd(&a_block, &b_block, &mut c_block);
                }
            }

            // write back local L1 results to global memory C
            for i in 0..32 {
                for j in 0..32 {
                    c[(ii + i) * n + (jj + j)] += c_block[i * 32 + j];
                }
            }
        }
    }
}
