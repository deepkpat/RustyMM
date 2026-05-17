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

    let n = 1024;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    let mut c = vec![0.0f32; n * n];

    c.fill(0.0f32);
    benchmark("naive", n, || {
        matmul::matmul_naive(&a, &b, &mut c, n);
    });

    println!("first element: {}", c[0]);
}
