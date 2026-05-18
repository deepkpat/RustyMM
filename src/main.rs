use std::time::Instant;

use crate::matmul::matmul_naive;

mod matmul;

const WARMUP_ITERS: usize = 4;
const MEASURE_ITERS: usize = 16;

fn flops(n: usize) -> f64 {
    2.0 * (n as f64).powi(3)
}

fn benchmark<F>(name: &str, n: usize, f: &mut F) -> (f64, f64, f64)
where
    F: FnMut(&mut [f32]),
{
    let mut c = vec![0.0f32; n * n];

    // warm up phase
    for _ in 0..WARMUP_ITERS {
        c.fill(0.0);
        f(&mut c);
    }

    // measurement phase
    let mut times = Vec::with_capacity(MEASURE_ITERS);

    for _ in 0..MEASURE_ITERS {
        c.fill(0.0);

        let start = Instant::now();
        f(&mut c);
        let elapsed = start.elapsed();

        times.push(elapsed.as_secs_f64() * 1000.0);
    }

    // statistics
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    let variance = times
        .iter()
        .map(|t| {
            let d = t - mean;
            d * d
        })
        .sum::<f64>()
        / times.len() as f64;

    let std_dev = variance.sqrt();

    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);

    let gflops = flops(n) / (mean / 1000.0) / 1e9;

    println!(
        "{:<22} {:>8.2} ms  ± {:>6.2} ms   min {:>8.2} ms   {:>8.2} GFLOPS",
        name, mean, std_dev, min, gflops
    );

    (mean, std_dev, gflops)
}

fn main() {
    println!("matrix multiplication optimizations (GEMM benchmark)\n");

    let n = 1024;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    // IMPORTANT: single shared output buffer conceptually,
    // but we still pass mutable slices per benchmark for fairness

    let mut kernels: Vec<(
        &str,
        Box<dyn FnMut(&mut [f32], usize, &Vec<f32>, &Vec<f32>)>,
    )> = vec![
        // naive
        ("naive", Box::new(|c, n, a, b| matmul_naive(a, b, c, n))),
        // reordered
        (
            "reordered",
            Box::new(|c, n, a, b| matmul::matmul_reordered(a, b, c, n)),
        ),
        // tiled
        (
            "tiled-4",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 4)),
        ),
        (
            "tiled-8",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 8)),
        ),
        (
            "tiled-16",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 16)),
        ),
        (
            "tiled-32",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 32)),
        ),
        (
            "tiled-64",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 64)),
        ),
        (
            "tiled-128",
            Box::new(|c, n, a, b| matmul::matmul_tiled(a, b, c, n, 128)),
        ),
        // parallel
        (
            "parallel",
            Box::new(|c, n, a, b| matmul::matmul_parallel(a, b, c, n)),
        ),
        // tiled-parallel
        (
            "tiled-parallel-4",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 4)),
        ),
        (
            "tiled-parallel-8",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 8)),
        ),
        (
            "tiled-parallel-16",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 16)),
        ),
        (
            "tiled-parallel-32",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 32)),
        ),
        (
            "tiled-parallel-64",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 64)),
        ),
        (
            "tiled-parallel-128",
            Box::new(|c, n, a, b| matmul::matmul_tiled_parallel(a, b, c, n, 128)),
        ),
        // tiled-packed
        (
            "tiled-packed",
            Box::new(|c, n, a, b| matmul::matmul_tiled_packed(a, b, c, n)),
        ),
        // tiled-simd
        (
            "tiled-simd",
            Box::new(|c, n, a, b| matmul::matmul_tiled_simd(a, b, c, n)),
        ),
        // tiled-register
        (
            "tiled-register",
            Box::new(|c, n, a, b| matmul::matmul_tiled_register(a, b, c, n)),
        ),
    ];

    let order: Vec<usize> = (0..kernels.len()).collect();
    for &i in &order {
        let (name, kernel) = &mut kernels[i];

        let mut f = |c: &mut [f32]| {
            kernel(c, n, &a, &b);
        };

        benchmark(name, n, &mut f);
    }

    println!("\nn: {}", n);
    println!("threads: {}", rayon::current_num_threads());
}
