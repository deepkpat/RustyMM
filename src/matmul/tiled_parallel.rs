use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

pub fn matmul_tiled_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize, block_size: usize) {
    c.par_chunks_mut(block_size * n)
        .enumerate()
        .for_each(|(block_idx, c_chunk)| {
            let ii = block_idx * block_size;

            let row_count = (n - ii).min(block_size);

            for kk in (0..n).step_by(block_size) {
                for jj in (0..n).step_by(block_size) {
                    for i_local in 0..row_count {
                        let i = ii + i_local;

                        for k in kk..(kk + block_size).min(n) {
                            let a_i_k = a[i * n + k];

                            for j in jj..(jj + block_size).min(n) {
                                c_chunk[i_local * n + j] += a_i_k * b[k * n + j];
                            }
                        }
                    }
                }
            }
        });
}
