#[inline(always)]
fn micro_kernel_32(
    a_block: &[f32; 32 * 32],
    b_block: &[f32; 32 * 32],
    c_block: &mut [f32; 32 * 32],
) {
    // the compiler can completely unroll these loops because
    // the bounds (32) and array sizes are completely static!
    for i in 0..32 {
        for k in 0..32 {
            let a_i_k = a_block[i * 32 + k];
            for j in 0..32 {
                c_block[i * 32 + j] += a_i_k * b_block[k * 32 + j];
            }
        }
    }
}

pub fn matmul_tiled_packed(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // statically allocated local cache blocks (fits perfectly in 32 KiB L1)
    let mut a_block = [0.0f32; 32 * 32];
    let mut b_block = [0.0f32; 32 * 32];
    let mut c_block = [0.0f32; 32 * 32];

    for ii in (0..n).step_by(32) {
        for jj in (0..n).step_by(32) {
            // clear our local c block accumulator
            c_block.fill(0.0);

            for kk in (0..n).step_by(32) {
                // pack global memory into continuous L1 stack blocks
                for i in 0..32 {
                    for j in 0..32 {
                        a_block[i * 32 + j] = a[(ii + i) * n + (kk + j)];
                        b_block[i * 32 + j] = b[(kk + i) * n + (jj + j)];
                    }
                }

                // run the pure, highly-optimized math kernel
                micro_kernel_32(&a_block, &b_block, &mut c_block);
            }

            // write back the finished results to global memory 'c'
            for i in 0..32 {
                for j in 0..32 {
                    c[(ii + i) * n + (jj + j)] += c_block[i * 32 + j];
                }
            }
        }
    }
}
