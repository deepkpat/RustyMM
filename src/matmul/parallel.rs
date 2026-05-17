use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

pub fn matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // parallelize over rows
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for k in 0..n {
            let a_i_k = a[i * n + k];
            for j in 0..n {
                c_row[j] += a_i_k * b[k * n + j];
            }
        }
    });
}
