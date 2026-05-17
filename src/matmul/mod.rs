mod naive;
mod reordered;
mod tilled;

pub use naive::matmul_naive;
pub use reordered::matmul_reordered;
pub use tilled::matmul_tilled;
