use std::time::Instant;

mod matmul;

fn benchmark<F>(name: &str, f: F)
where
    F: FnOnce(),
{
    let start = Instant::now();

    f();

    let elapsed = start.elapsed();

    let ms = elapsed.as_secs_f64() * 1000.0;

    println!("{:<15} {:>15.5?} ms", name, ms);
}

fn main() {
    println!("matrix multiplication optimizations!");

    let n = 1024;

    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    let mut bt = vec![0.0f32; n * n];
    matmul::transpose(&b, &mut bt, n);

    let mut c1 = vec![0.0f32; n * n];
    let mut c2 = vec![0.0f32; n * n];
    let mut c3 = vec![0.0f32; n * n];
    let mut c4 = vec![0.0f32; n * n];
    let mut c5 = vec![0.0f32; n * n];

    benchmark("naive", || {
        matmul::matmul_naive(&a, &b, &mut c1, n);
    });

    benchmark("reordered", || {
        matmul::matmul_reordered(&a, &b, &mut c2, n);
    });

    benchmark("blocked", || {
        matmul::matmul_blocked(&a, &b, &mut c3, n);
    });

    benchmark("transposed", || {
        matmul::matmul_transposed(&a, &bt, &mut c4, n);
    });

    benchmark("blk+trans", || {
        matmul::matmul_blocked_transposed(&a, &bt, &mut c5, n);
    });

    println!("\nfirst elements");
    dbg!(c1[0]);
    dbg!(c2[0]);
    dbg!(c3[0]);
    dbg!(c4[0]);
    dbg!(c5[0]);
}
