use std::time::Instant;

mod matmul;

fn benchmark<F>(name: &str, mut f: F)
where
    F: FnMut(),
{
    let start = Instant::now();

    f();

    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0;

    println!("{:<20} {:>16.5?} ms", name, ms);
}

fn main() {
    println!("matrix multiplication optimizations!");

    let n = 512;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    let mut bt = vec![0.0f32; n * n];
    matmul::transpose(&b, &mut bt, n);

    let mut c = vec![0.0f32; n * n];

    c.fill(0.0f32);
    benchmark("naive", || {
        matmul::matmul_naive(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("reordered", || {
        matmul::matmul_reordered(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("blocked", || {
        matmul::matmul_blocked(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("transposed", || {
        matmul::matmul_transposed(&a, &bt, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("blocked_transposed", || {
        matmul::matmul_blocked_transposed(&a, &bt, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("parallel", || {
        matmul::matmul_blocked_parallel(&a, &b, &mut c, n);
    });

    c.fill(0.0f32);
    benchmark("transposed_simd", || {
        matmul::matmul_transposed_simd(&a, &b, &mut c, n);
    });

    println!("\nn: {n}");
    println!("first element: {}", c[0]);
}
