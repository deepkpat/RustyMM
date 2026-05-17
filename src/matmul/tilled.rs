pub fn matmul_tilled(a: &[f32], b: &[f32], c: &mut [f32], n: usize, block_size: usize) {
    for ii in (0..n).step_by(block_size) {
        for kk in (0..n).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                // process block
                for i in ii..(ii + block_size).min(n) {
                    for k in kk..(kk + block_size).min(n) {
                        let a_i_k = a[i * n + k];

                        for j in jj..(jj + block_size).min(n) {
                            c[i * n + j] += a_i_k * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}
