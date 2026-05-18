use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[target_feature(enable = "avx2,fma")]
pub unsafe fn macro_kernel_32_accumulate(
    packed_a: *const f32,
    packed_b: *const f32,
    c: *mut f32,
    ii_block: usize,
    jj_block: usize,
    blocks_per_dim: usize,
    n: usize,
) {
    let ii = ii_block * 32;
    let jj = jj_block * 32;

    // Process our 32x32 macro-block in micro-tiles of 4x8
    for i in (0..32).step_by(4) {
        for j in (0..32).step_by(8) {
            // 1. Load the target destination C tile into registers exactly ONCE
            let mut c0 = _mm256_loadu_ps(c.add((ii + i) * n + (jj + j)));
            let mut c1 = _mm256_loadu_ps(c.add((ii + i + 1) * n + (jj + j)));
            let mut c2 = _mm256_loadu_ps(c.add((ii + i + 2) * n + (jj + j)));
            let mut c3 = _mm256_loadu_ps(c.add((ii + i + 3) * n + (jj + j)));

            // 2. Loop through all K macro-blocks across the matrix width
            for kk in 0..blocks_per_dim {
                // Instantly resolve the memory address of our pre-packed sub-blocks
                let a_block = packed_a.add((ii_block * blocks_per_dim + kk) * 1024);
                let b_block = packed_b.add((kk * blocks_per_dim + jj_block) * 1024);

                // 3. Blazing fast inner loop with 100% sequential layout (stride-1)
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

            // 4. Store the completely calculated accumulation back to global C memory ONCE
            _mm256_storeu_ps(c.add((ii + i) * n + (jj + j)), c0);
            _mm256_storeu_ps(c.add((ii + i + 1) * n + (jj + j)), c1);
            _mm256_storeu_ps(c.add((ii + i + 2) * n + (jj + j)), c2);
            _mm256_storeu_ps(c.add((ii + i + 3) * n + (jj + j)), c3);
        }
    }
}

pub fn matmul_amortized_packed(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let blocks_per_dim = n / 32;

    // Allocate continuous linear memory for packed configurations
    let mut packed_a = vec![0.0f32; n * n];
    let mut packed_b = vec![0.0f32; n * n];

    // Rearrange Matrix A into sequential 32x32 blocks UPFRONT (Cost: 1x)
    for ii in 0..blocks_per_dim {
        for kk in 0..blocks_per_dim {
            let block_offset = (ii * blocks_per_dim + kk) * 1024;
            for i in 0..32 {
                for k in 0..32 {
                    packed_a[block_offset + i * 32 + k] = a[(ii * 32 + i) * n + (kk * 32 + k)];
                }
            }
        }
    }

    // Rearrange Matrix B into sequential 32x32 blocks UPFRONT (Cost: 1x)
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

    // Pure execution loops (Only 2 levels deep! Matrix C is kept clean)
    for ii_block in 0..blocks_per_dim {
        for jj_block in 0..blocks_per_dim {
            unsafe {
                macro_kernel_32_accumulate(
                    packed_a.as_ptr(),
                    packed_b.as_ptr(),
                    c.as_mut_ptr(),
                    ii_block,
                    jj_block,
                    blocks_per_dim,
                    n,
                );
            }
        }
    }
}
