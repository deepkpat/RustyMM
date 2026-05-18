// // use rayon::prelude::*;
// // use std::arch::x86_64::{__m256, _mm256_fmadd_ps, _mm256_load_ps, _mm256_set1_ps, _mm256_store_ps};

// // #[repr(C, align(32))]
// // struct BlasAlignedBlock {
// //     data: [f32; 64 * 64],
// // }

// // /// Corrected Aligned Micro-Kernel
// // /// Takes the base pointer of B and correctly steps through memory without leaking out of bounds
// // #[target_feature(enable = "avx2,fma")]
// // pub unsafe fn micro_kernel_blas(
// //     packed_a_row_ptr: *const f32,
// //     packed_b_base_ptr: *const f32,
// //     c_ptr: *mut f32,
// //     jj_block: usize,
// //     blocks_per_dim: usize,
// // ) {
// //     for i in (0..64).step_by(4) {
// //         for j in (0..64).step_by(16) {
// //             let mut c0_0 = _mm256_load_ps(c_ptr.add(i * 64 + j));
// //             let mut c0_1 = _mm256_load_ps(c_ptr.add(i * 64 + j + 8));
// //             let mut c1_0 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j));
// //             let mut c1_1 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j + 8));
// //             let mut c2_0 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j));
// //             let mut c2_1 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j + 8));
// //             let mut c3_0 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j));
// //             let mut c3_1 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j + 8));

// //             for kk in 0..blocks_per_dim {
// //                 // A blocks are stacked linearly for this row segment: each block is 4096 elements
// //                 let a_block = packed_a_row_ptr.add(kk * 4096);

// //                 // B blocks layout formula: (kk * blocks_per_dim + jj_block) * 4096
// //                 let b_block = packed_b_base_ptr.add((kk * blocks_per_dim + jj_block) * 4096);

// //                 for k in 0..64 {
// //                     let b_val_0 = _mm256_load_ps(b_block.add(k * 64 + j));
// //                     let b_val_1 = _mm256_load_ps(b_block.add(k * 64 + j + 8));

// //                     let a0 = _mm256_set1_ps(*a_block.add(i * 64 + k));
// //                     let a1 = _mm256_set1_ps(*a_block.add((i + 1) * 64 + k));
// //                     let a2 = _mm256_set1_ps(*a_block.add((i + 2) * 64 + k));
// //                     let a3 = _mm256_set1_ps(*a_block.add((i + 3) * 64 + k));

// //                     c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
// //                     c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
// //                     c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
// //                     c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
// //                     c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
// //                     c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
// //                     c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
// //                     c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);
// //                 }
// //             }

// //             _mm256_store_ps(c_ptr.add(i * 64 + j), c0_0);
// //             _mm256_store_ps(c_ptr.add(i * 64 + j + 8), c0_1);
// //             _mm256_store_ps(c_ptr.add((i + 1) * 64 + j), c1_0);
// //             _mm256_store_ps(c_ptr.add((i + 1) * 64 + j + 8), c1_1);
// //             _mm256_store_ps(c_ptr.add((i + 2) * 64 + j), c2_0);
// //             _mm256_store_ps(c_ptr.add((i + 2) * 64 + j + 8), c2_1);
// //             _mm256_store_ps(c_ptr.add((i + 3) * 64 + j), c3_0);
// //             _mm256_store_ps(c_ptr.add((i + 3) * 64 + j + 8), c3_1);
// //         }
// //     }
// // }

// // pub fn matmul_blas_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
// //     let blocks_per_dim = n / 64;

// //     let mut packed_a = vec![0.0f32; n * n];
// //     let mut packed_b = vec![0.0f32; n * n];

// //     // Parallel block packing for Matrix A
// //     packed_a
// //         .par_chunks_exact_mut(4096)
// //         .enumerate()
// //         .for_each(|(block_idx, block)| {
// //             let ii = block_idx / blocks_per_dim;
// //             let kk = block_idx % blocks_per_dim;
// //             for i in 0..64 {
// //                 for j in 0..64 {
// //                     block[i * 64 + j] = a[(ii * 64 + i) * n + (kk * 64 + j)];
// //                 }
// //             }
// //         });

// //     // Parallel block packing for Matrix B
// //     packed_b
// //         .par_chunks_exact_mut(4096)
// //         .enumerate()
// //         .for_each(|(block_idx, block)| {
// //             let kk = block_idx / blocks_per_dim;
// //             let jj = block_idx % blocks_per_dim;
// //             for k in 0..64 {
// //                 for j in 0..64 {
// //                     block[k * 64 + j] = b[(kk * 64 + k) * n + (jj * 64 + j)];
// //                 }
// //             }
// //         });

// //     // Pure math thread pool distribution step
// //     c.par_chunks_mut(64 * n)
// //         .enumerate()
// //         .for_each(|(ii_block, c_chunk)| {
// //             let a_ptr = packed_a.as_ptr();
// //             let b_ptr = packed_b.as_ptr();

// //             for jj_block in 0..blocks_per_dim {
// //                 let mut local_c = BlasAlignedBlock {
// //                     data: [0.0f32; 64 * 64],
// //                 };

// //                 // Read standard memory layout slice inside our aligned sandbox
// //                 for i in 0..64 {
// //                     for j in 0..64 {
// //                         local_c.data[i * 64 + j] = c_chunk[i * n + (jj_block * 64 + j)];
// //                     }
// //                 }

// //                 unsafe {
// //                     // Extract the base offset for row chunk ii_block
// //                     let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 4096);

// //                     micro_kernel_blas(
// //                         packed_a_block_row,
// //                         b_ptr, // Pass base pointer safely
// //                         local_c.data.as_mut_ptr(),
// //                         jj_block,
// //                         blocks_per_dim,
// //                     );
// //                 }

// //                 // Unload computed values back to main program memory matrix space
// //                 for i in 0..64 {
// //                     for j in 0..64 {
// //                         c_chunk[i * n + (jj_block * 64 + j)] = local_c.data[i * 64 + j];
// //                     }
// //                 }
// //             }
// //         });
// // }

// use rayon::prelude::*;
// use std::arch::x86_64::{__m256, _mm256_fmadd_ps, _mm256_load_ps, _mm256_set1_ps, _mm256_store_ps};

// /// Safely returns a 32-byte aligned raw pointer from a standard heap-allocated Vector
// #[inline(always)]
// fn get_aligned_ptr(vec: &mut [f32]) -> *mut f32 {
//     let ptr = vec.as_mut_ptr() as usize;
//     // Calculate how many bytes we need to add to reach a 32-byte boundary
//     let remainder = ptr % 32;
//     if remainder == 0 {
//         ptr as *mut f32
//     } else {
//         ((ptr + 32 - remainder) as *mut f32)
//     }
// }

// #[target_feature(enable = "avx2,fma")]
// pub unsafe fn micro_kernel_blas(
//     packed_a_row_ptr: *const f32,
//     packed_b_base_ptr: *const f32,
//     c_ptr: *mut f32,
//     jj_block: usize,
//     blocks_per_dim: usize,
// ) {
//     for i in (0..64).step_by(4) {
//         for j in (0..64).step_by(16) {
//             let mut c0_0 = _mm256_load_ps(c_ptr.add(i * 64 + j));
//             let mut c0_1 = _mm256_load_ps(c_ptr.add(i * 64 + j + 8));
//             let mut c1_0 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j));
//             let mut c1_1 = _mm256_load_ps(c_ptr.add((i + 1) * 64 + j + 8));
//             let mut c2_0 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j));
//             let mut c2_1 = _mm256_load_ps(c_ptr.add((i + 2) * 64 + j + 8));
//             let mut c3_0 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j));
//             let mut c3_1 = _mm256_load_ps(c_ptr.add((i + 3) * 64 + j + 8));

//             for kk in 0..blocks_per_dim {
//                 let a_block = packed_a_row_ptr.add(kk * 4096);
//                 let b_block = packed_b_base_ptr.add((kk * blocks_per_dim + jj_block) * 4096);

//                 for k in 0..64 {
//                     let b_val_0 = _mm256_load_ps(b_block.add(k * 64 + j));
//                     let b_val_1 = _mm256_load_ps(b_block.add(k * 64 + j + 8));

//                     let a0 = _mm256_set1_ps(*a_block.add(i * 64 + k));
//                     let a1 = _mm256_set1_ps(*a_block.add((i + 1) * 64 + k));
//                     let a2 = _mm256_set1_ps(*a_block.add((i + 2) * 64 + k));
//                     let a3 = _mm256_set1_ps(*a_block.add((i + 3) * 64 + k));

//                     c0_0 = _mm256_fmadd_ps(a0, b_val_0, c0_0);
//                     c0_1 = _mm256_fmadd_ps(a0, b_val_1, c0_1);
//                     c1_0 = _mm256_fmadd_ps(a1, b_val_0, c1_0);
//                     c1_1 = _mm256_fmadd_ps(a1, b_val_1, c1_1);
//                     c2_0 = _mm256_fmadd_ps(a2, b_val_0, c2_0);
//                     c2_1 = _mm256_fmadd_ps(a2, b_val_1, c2_1);
//                     c3_0 = _mm256_fmadd_ps(a3, b_val_0, c3_0);
//                     c3_1 = _mm256_fmadd_ps(a3, b_val_1, c3_1);
//                 }
//             }

//             _mm256_store_ps(c_ptr.add(i * 64 + j), c0_0);
//             _mm256_store_ps(c_ptr.add(i * 64 + j + 8), c0_1);
//             _mm256_store_ps(c_ptr.add((i + 1) * 64 + j), c1_0);
//             _mm256_store_ps(c_ptr.add((i + 1) * 64 + j + 8), c1_1);
//             _mm256_store_ps(c_ptr.add((i + 2) * 64 + j), c2_0);
//             _mm256_store_ps(c_ptr.add((i + 2) * 64 + j + 8), c2_1);
//             _mm256_store_ps(c_ptr.add((i + 3) * 64 + j), c3_0);
//             _mm256_store_ps(c_ptr.add((i + 3) * 64 + j + 8), c3_1);
//         }
//     }
// }

// pub fn matmul_blas_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
//     let blocks_per_dim = n / 64;

//     let mut packed_a = vec![0.0f32; n * n];
//     let mut packed_b = vec![0.0f32; n * n];

//     // Parallel block packing for Matrix A
//     packed_a
//         .par_chunks_exact_mut(4096)
//         .enumerate()
//         .for_each(|(block_idx, block)| {
//             let ii = block_idx / blocks_per_dim;
//             let kk = block_idx % blocks_per_dim;
//             for i in 0..64 {
//                 for j in 0..64 {
//                     block[i * 64 + j] = a[(ii * 64 + i) * n + (kk * 64 + j)];
//                 }
//             }
//         });

//     // Parallel block packing for Matrix B
//     packed_b
//         .par_chunks_exact_mut(4096)
//         .enumerate()
//         .for_each(|(block_idx, block)| {
//             let kk = block_idx / blocks_per_dim;
//             let jj = block_idx % blocks_per_dim;
//             for k in 0..64 {
//                 for j in 0..64 {
//                     block[k * 64 + j] = b[(kk * 64 + k) * n + (jj * 64 + j)];
//                 }
//             }
//         });

//     // Pure math thread pool distribution step
//     c.par_chunks_mut(64 * n)
//         .enumerate()
//         .for_each(|(ii_block, c_chunk)| {
//             let a_ptr = packed_a.as_ptr();
//             let b_ptr = packed_b.as_ptr();

//             for jj_block in 0..blocks_per_dim {
//                 // Allocate 4096 elements + 8 elements padding on the HEAP to protect the stack frame
//                 let mut heap_c_buffer = vec![0.0f32; 4096 + 8];
//                 let aligned_c_ptr = get_aligned_ptr(&mut heap_c_buffer);

//                 // Read standard memory layout slice inside our aligned heap destination
//                 unsafe {
//                     for i in 0..64 {
//                         for j in 0..64 {
//                             *aligned_c_ptr.add(i * 64 + j) = c_chunk[i * n + (jj_block * 64 + j)];
//                         }
//                     }

//                     // Extract the base offset for row chunk ii_block
//                     let packed_a_block_row = a_ptr.add(ii_block * blocks_per_dim * 4096);

//                     micro_kernel_blas(
//                         packed_a_block_row,
//                         b_ptr,
//                         aligned_c_ptr,
//                         jj_block,
//                         blocks_per_dim,
//                     );

//                     // Unload computed values back to main program memory matrix space
//                     for i in 0..64 {
//                         for j in 0..64 {
//                             c_chunk[i * n + (jj_block * 64 + j)] = *aligned_c_ptr.add(i * 64 + j);
//                         }
//                     }
//                 }
//             }
//         });
// }

// blas_blocked.rs
//
// High-performance blocked SGEMM implementation for:
//
//     C = A * B
//
// Assumptions:
// - row major matrices
// - n is divisible by 64
// - AVX2 + FMA CPU
//
// Design goals:
// - easy to understand
// - stable (no segfaults)
// - reasonably fast
// - proper cache blocking
// - reusable packed matrices
//
// This is MUCH closer to real BLAS structure than the previous version.
//

use rayon::prelude::*;
use std::arch::x86_64::{
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

const BLOCK: usize = 64;
const TILE_M: usize = 4;
const TILE_N: usize = 16;

//
// ============================================================
// PACKING
// ============================================================
//
// We pack matrices into contiguous 64x64 tiles.
//
// This improves:
//
// - cache locality
// - TLB behavior
// - SIMD efficiency
// - prefetching effectiveness
//
// Layout:
//
// packed_a:
//     [A00][A01][A02]...
//
// packed_b:
//     [B00][B01][B02]...
//
// each tile is:
//     64 * 64 = 4096 floats
//

#[inline(always)]
fn pack_a(a: &[f32], n: usize) -> Vec<f32> {
    let blocks = n / BLOCK;

    let mut packed = vec![0.0f32; n * n];

    packed
        .par_chunks_exact_mut(BLOCK * BLOCK)
        .enumerate()
        .for_each(|(block_idx, block)| {
            let ii = block_idx / blocks;
            let kk = block_idx % blocks;

            for i in 0..BLOCK {
                let src_row = (ii * BLOCK + i) * n + kk * BLOCK;
                let dst_row = i * BLOCK;

                block[dst_row..dst_row + BLOCK].copy_from_slice(&a[src_row..src_row + BLOCK]);
            }
        });

    packed
}

#[inline(always)]
fn pack_b(b: &[f32], n: usize) -> Vec<f32> {
    let blocks = n / BLOCK;

    let mut packed = vec![0.0f32; n * n];

    packed
        .par_chunks_exact_mut(BLOCK * BLOCK)
        .enumerate()
        .for_each(|(block_idx, block)| {
            let kk = block_idx / blocks;
            let jj = block_idx % blocks;

            for k in 0..BLOCK {
                let src_row = (kk * BLOCK + k) * n + jj * BLOCK;
                let dst_row = k * BLOCK;

                block[dst_row..dst_row + BLOCK].copy_from_slice(&b[src_row..src_row + BLOCK]);
            }
        });

    packed
}

//
// ============================================================
// MICROKERNEL
// ============================================================
//
// Computes:
//
//     4x16 tile of C
//
// using:
//
//     A = 4x64
//     B = 64x16
//
// AVX2 register strategy:
//
// Each row uses:
//
//     2 registers for 16 columns
//
// Total:
//
//     8 accumulator registers
//
// This is a classic AVX2 microkernel shape.
//

#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x16(a_block: *const f32, b_block: *const f32, c_tile: *mut f32) {
    for i in (0..BLOCK).step_by(TILE_M) {
        for j in (0..BLOCK).step_by(TILE_N) {
            //
            // accumulator registers
            //

            let mut c00 = _mm256_setzero_ps();
            let mut c01 = _mm256_setzero_ps();

            let mut c10 = _mm256_setzero_ps();
            let mut c11 = _mm256_setzero_ps();

            let mut c20 = _mm256_setzero_ps();
            let mut c21 = _mm256_setzero_ps();

            let mut c30 = _mm256_setzero_ps();
            let mut c31 = _mm256_setzero_ps();

            //
            // reduction dimension
            //

            for k in 0..BLOCK {
                //
                // load 16 B values
                //

                let b0 = _mm256_loadu_ps(b_block.add(k * BLOCK + j));
                let b1 = _mm256_loadu_ps(b_block.add(k * BLOCK + j + 8));

                //
                // broadcast A scalars
                //

                let a0 = _mm256_set1_ps(*a_block.add((i + 0) * BLOCK + k));
                let a1 = _mm256_set1_ps(*a_block.add((i + 1) * BLOCK + k));
                let a2 = _mm256_set1_ps(*a_block.add((i + 2) * BLOCK + k));
                let a3 = _mm256_set1_ps(*a_block.add((i + 3) * BLOCK + k));

                //
                // fused multiply-add
                //

                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);

                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);

                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);

                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
            }

            //
            // store results
            //

            _mm256_storeu_ps(c_tile.add((i + 0) * BLOCK + j), c00);
            _mm256_storeu_ps(c_tile.add((i + 0) * BLOCK + j + 8), c01);

            _mm256_storeu_ps(c_tile.add((i + 1) * BLOCK + j), c10);
            _mm256_storeu_ps(c_tile.add((i + 1) * BLOCK + j + 8), c11);

            _mm256_storeu_ps(c_tile.add((i + 2) * BLOCK + j), c20);
            _mm256_storeu_ps(c_tile.add((i + 2) * BLOCK + j + 8), c21);

            _mm256_storeu_ps(c_tile.add((i + 3) * BLOCK + j), c30);
            _mm256_storeu_ps(c_tile.add((i + 3) * BLOCK + j + 8), c31);
        }
    }
}

//
// ============================================================
// TOP LEVEL GEMM
// ============================================================
//
// Threading strategy:
//
// Parallelize over block rows of C.
//
// Each worker:
//
// - owns one 64-row slab
// - reuses local scratch tile
// - avoids heap allocation in inner loops
//

pub fn matmul_blas_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    assert!(n % BLOCK == 0);

    let blocks = n / BLOCK;

    //
    // pack once
    //

    let packed_a = pack_a(a, n);
    let packed_b = pack_b(b, n);

    //
    // parallelize over C block rows
    //

    c.par_chunks_mut(BLOCK * n).enumerate().for_each_init(
        || vec![0.0f32; BLOCK * BLOCK],
        |local_tile, (ii, c_chunk)| {
            for jj in 0..blocks {
                //
                // clear local tile
                //

                local_tile.fill(0.0);

                //
                // accumulate over K dimension
                //

                for kk in 0..blocks {
                    let a_block_idx = (ii * blocks + kk) * BLOCK * BLOCK;
                    let b_block_idx = (kk * blocks + jj) * BLOCK * BLOCK;

                    let a_block = packed_a[a_block_idx..].as_ptr();

                    let b_block = packed_b[b_block_idx..].as_ptr();

                    unsafe {
                        microkernel_4x16(a_block, b_block, local_tile.as_mut_ptr());
                    }
                }

                //
                // write tile back to C
                //

                for i in 0..BLOCK {
                    let c_row = i * n + jj * BLOCK;

                    let tile_row = i * BLOCK;

                    c_chunk[c_row..c_row + BLOCK]
                        .copy_from_slice(&local_tile[tile_row..tile_row + BLOCK]);
                }
            }
        },
    );
}
