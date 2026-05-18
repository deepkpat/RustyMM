mod naive;
mod parallel;
mod reordered;
mod tiled;
mod tiled_packed;
mod tiled_parallel;

pub use naive::matmul_naive;
pub use parallel::matmul_parallel;
pub use reordered::matmul_reordered;
pub use tiled::matmul_tiled;
pub use tiled_packed::matmul_tiled_packed;
pub use tiled_parallel::matmul_tiled_parallel;
