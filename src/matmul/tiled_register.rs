use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn micro_kernel_32_register(
    a_block: &[f32; 32 * 32],
    b_block: &[f32; 32 * 32],
    c_block: &mut [f32; 32 * 32],
) {
    // step through rows by 4
    for i in (0..32).step_by(4) {
        // step through columns by 8
        // since one register holds 8 floats
        for j in (0..32).step_by(8) {
            // allocate 4 registers to hod our 4x8 tile of matrix C
            // we load them from memory exactly ONCE before the k-loop
            let mut c0 = unsafe { _mm256_loadu_ps(c_block.as_ptr().add(i * 32 + j)) };
            let mut c1 = unsafe { _mm256_loadu_ps(c_block.as_ptr().add((i + 1) * 32 + j)) };
            let mut c2 = unsafe { _mm256_loadu_ps(c_block.as_ptr().add((i + 2) * 32 + j)) };
            let mut c3 = unsafe { _mm256_loadu_ps(c_block.as_ptr().add((i + 3) * 32 + j)) };

            // the deep math loop: unrolled along k-dimension
            for k in 0..32 {
                // load 1 vector from B (8 elements)
                // CRITICAL: this mmoroy load is reused 4 times below!
                let b_val = unsafe { _mm256_loadu_ps(b_block.as_ptr().add(k * 32 + j)) };

                // broadcast 4 separate scalar scalar values from A
                let a0 = _mm256_set1_ps(a_block[i * 32 + k]);
                let a1 = _mm256_set1_ps(a_block[(i + 1) * 32 + k]);
                let a2 = _mm256_set1_ps(a_block[(i + 2) * 32 + k]);
                let a3 = _mm256_set1_ps(a_block[(i + 3) * 32 + k]);

                // perform 4 FMAs simultaneously
                // zero reads or writes to matrix C
                c0 = _mm256_fmadd_ps(a0, b_val, c0);
                c1 = _mm256_fmadd_ps(a1, b_val, c1);
                c2 = _mm256_fmadd_ps(a2, b_val, c2);
                c3 = _mm256_fmadd_ps(a3, b_val, c3);
            }

            // write the finished accumulated registers back to memory exactly ONCE
            unsafe { _mm256_storeu_ps(c_block.as_mut_ptr().add(i * 32 + j), c0) };
            unsafe { _mm256_storeu_ps(c_block.as_mut_ptr().add((i + 1) * 32 + j), c1) };
            unsafe { _mm256_storeu_ps(c_block.as_mut_ptr().add((i + 2) * 32 + j), c2) };
            unsafe { _mm256_storeu_ps(c_block.as_mut_ptr().add((i + 3) * 32 + j), c3) };
        }
    }
}

pub fn matmul_tiled_register(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut a_block = [0.0f32; 32 * 32];
    let mut b_block = [0.0f32; 32 * 32];
    let mut c_block = [0.0f32; 32 * 32];

    for ii in (0..n).step_by(32) {
        for jj in (0..n).step_by(32) {
            c_block.fill(0.0);

            for kk in (0..n).step_by(32) {
                // pack global values into memory blocks
                for i in 0..32 {
                    for j in 0..32 {
                        a_block[i * 32 + j] = a[(ii + i) * n + (kk + j)];
                        b_block[i * 32 + j] = b[(kk + i) * n + (jj + j)];
                    }
                }

                unsafe {
                    micro_kernel_32_register(&a_block, &b_block, &mut c_block);
                }
            }

            // write back to global C memory
            for i in 0..32 {
                for j in 0..32 {
                    c[(ii + i) * n + (jj + j)] += c_block[i * 32 + j];
                }
            }
        }
    }
}
