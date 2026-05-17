use std::time::Instant;

mod matmul;

fn benchmark<F>(name: &str, n: usize, mut f: F)
where
    F: FnMut(),
{
    let start = Instant::now();

    f();

    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0;

    let flops = 2.0 * (n as f64).powi(3);

    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!("{:<24} {:>10.3} ms   {:>8.2} GFLOPS", name, ms, gflops);
}

fn main() {
    println!("matrix multiplication optimizations!");

    let n = 1024 * 4;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    let mut bt = vec![0.0f32; n * n];
    matmul::transpose(&b, &mut bt, n);

    let mut c = vec![0.0f32; n * n];

    // c.fill(0.0f32);
    // benchmark("naive", n, || {
    //     matmul::matmul_naive(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("reordered", n, || {
    //     matmul::matmul_reordered(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("blocked", n, || {
    //     matmul::matmul_blocked(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("transposed", n, || {
    //     matmul::matmul_transposed(&a, &bt, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("blocked_transposed", n, || {
    //     matmul::matmul_blocked_transposed(&a, &bt, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("parallel", n, || {
    //     matmul::matmul_blocked_parallel(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("transposed_simd", n, || {
    //     matmul::matmul_transposed_simd(&a, &bt, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("blocked_trans_simd", n, || {
    //     matmul::matmul_blocked_transposed_simd(&a, &bt, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("microkernel_1x8", n, || {
    //     matmul::matmul_microkernel_1x8(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("microkernel_4x8", n, || {
    //     matmul::matmul_microkernel_4x8(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("blocked_micro_4x8", n, || {
    //     matmul::matmul_blocked_microkernel_4x8(&a, &b, &mut c, n);
    // });

    // c.fill(0.0f32);
    // benchmark("packed_b_4x8", n, || {
    //     matmul::matmul_packed_b_4x8(&a, &b, &mut c, n);
    // });

    c.fill(0.0f32);
    benchmark("packed_b_4x8_par", n, || {
        matmul::matmul_packed_b_4x8_parallel(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("packed_b_8x8_par", n, || {
        matmul::matmul_packed_b_8x8_parallel(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("packed_ab_4x8", n, || {
        matmul::matmul_packed_ab_4x8(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("packed_ab_4x8_precompute", n, || {
        matmul::matmul_packed_ab_4x8_precompute(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("blis_4x8", n, || {
        matmul::matmul_blis_4x8(&a, &b, &mut c, n);
    });

    println!("\nn: {n}");
    println!("batch_size: {}", matmul::get_batch_size());
    println!("threads: {}", rayon::current_num_threads());
    println!("first element: {}", c[0]);
}
