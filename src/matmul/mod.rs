mod naive;
mod parallel;
mod reordered;
mod tiled;

pub use naive::matmul_naive;
pub use parallel::matmul_parallel;
pub use reordered::matmul_reordered;
pub use tiled::matmul_tiled;
