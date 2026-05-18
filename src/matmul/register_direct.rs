use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_4x8_direct(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ii: usize,
    kk: usize,
    jj: usize,
    n: usize,
) {
    let c_ptr = c.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // load the 4x8 destination tile of global Matrix C straight into registers
    let mut c0 = unsafe { _mm256_loadu_ps(c_ptr.add(ii * n + jj)) };
    let mut c1 = unsafe { _mm256_loadu_ps(c_ptr.add((ii + 1) * n + jj)) };
    let mut c2 = unsafe { _mm256_loadu_ps(c_ptr.add((ii + 2) * n + jj)) };
    let mut c3 = unsafe { _mm256_loadu_ps(c_ptr.add((ii + 3) * n + jj)) };

    // loop over the macro-block k-range (32 elements)
    for k in kk..(kk + 32) {
        // load 8 sequential elements from global Matrix B
        let b_val = unsafe { _mm256_loadu_ps(b_ptr.add(k * n + jj)) };

        // broadcast values directly out of global Matrix A
        let a0 = unsafe { _mm256_set1_ps(*a_ptr.add(ii * n + k)) };
        let a1 = unsafe { _mm256_set1_ps(*a_ptr.add((ii + 1) * n + k)) };
        let a2 = unsafe { _mm256_set1_ps(*a_ptr.add((ii + 2) * n + k)) };
        let a3 = unsafe { _mm256_set1_ps(*a_ptr.add((ii + 3) * n + k)) };

        // mathematical execution
        c0 = _mm256_fmadd_ps(a0, b_val, c0);
        c1 = _mm256_fmadd_ps(a1, b_val, c1);
        c2 = _mm256_fmadd_ps(a2, b_val, c2);
        c3 = _mm256_fmadd_ps(a3, b_val, c3);
    }

    // update global Matrix C exactly once
    unsafe { _mm256_storeu_ps(c_ptr.add(ii * n + jj), c0) };
    unsafe { _mm256_storeu_ps(c_ptr.add((ii + 1) * n + jj), c1) };
    unsafe { _mm256_storeu_ps(c_ptr.add((ii + 2) * n + jj), c2) };
    unsafe { _mm256_storeu_ps(c_ptr.add((ii + 3) * n + jj), c3) };
}

pub fn matmul_register_direct(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // macro-blocking loops for cache locality (remains 32x32)
    for ii in (0..n).step_by(32) {
        for jj in (0..n).step_by(32) {
            for kk in (0..n).step_by(32) {
                // zero-allocation micro-loops feeding our hardware registers
                unsafe {
                    for sub_i in (ii..ii + 32).step_by(4) {
                        for sub_j in (jj..jj + 32).step_by(8) {
                            micro_kernel_4x8_direct(a, b, c, sub_i, kk, sub_j, n);
                        }
                    }
                }
            }
        }
    }
}
