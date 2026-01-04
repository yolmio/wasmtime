#![cfg(target_arch = "x86_64")]

//! Performance benchmarks for AVX-512 columnar HTAP workloads.
//!
//! These benchmarks measure the performance of vectorized database operations:
//! - Aggregate: SUM, MIN/MAX with horizontal reduction
//! - Vector arithmetic throughput
//!
//! Run with: cargo test -p cranelift-jit --test avx512_perf --release -- --nocapture

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::*;
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;
use std::mem;
use std::time::Instant;

// =============================================================================
// Test Configuration
// =============================================================================

/// Number of rows for benchmarks
const BENCH_ROWS: usize = 1_000_000;
/// Number of warmup iterations
const WARMUP_ITERS: usize = 3;
/// Number of benchmark iterations
const BENCH_ITERS: usize = 10;

// =============================================================================
// Infrastructure
// =============================================================================

fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return true;
        }
    }
    false
}

fn isa_with_avx512() -> Option<OwnedTargetIsa> {
    if !has_avx512() {
        return None;
    }

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder.set("opt_level", "speed").unwrap();

    let isa_builder = cranelift_native::builder().ok()?;
    isa_builder.finish(settings::Flags::new(flag_builder)).ok()
}

fn jit_module_with_avx512() -> Option<JITModule> {
    let isa = isa_with_avx512()?;
    Some(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

struct PerfCompiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl PerfCompiler {
    fn new() -> Option<Self> {
        let module = jit_module_with_avx512()?;
        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();
        Some(Self {
            module,
            ctx,
            func_ctx,
        })
    }

    fn ptr_type(&self) -> Type {
        self.module.target_config().pointer_type()
    }
}

fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn run_benchmark<F>(name: &str, rows: usize, bytes_per_row: usize, mut f: F)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        f();
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min_ns = *times.iter().min().unwrap();
    let max_ns = *times.iter().max().unwrap();
    let avg_ns = times.iter().sum::<u64>() / times.len() as u64;

    let rows_per_sec = rows as f64 / (avg_ns as f64 / 1_000_000_000.0);
    let bytes_total = rows * bytes_per_row;
    let gb_per_sec =
        (bytes_total as f64 / (1024.0 * 1024.0 * 1024.0)) / (avg_ns as f64 / 1_000_000_000.0);

    println!("\n=== {} ===", name);
    println!("  Rows:         {:>12}", format_num(rows));
    println!("  Min time:     {:>12.2} ms", min_ns as f64 / 1_000_000.0);
    println!("  Max time:     {:>12.2} ms", max_ns as f64 / 1_000_000.0);
    println!("  Avg time:     {:>12.2} ms", avg_ns as f64 / 1_000_000.0);
    println!(
        "  Throughput:   {:>12.2} M rows/sec",
        rows_per_sec / 1_000_000.0
    );
    println!("  Bandwidth:    {:>12.2} GB/sec", gb_per_sec);
}

// =============================================================================
// Compiled Function Builders
// =============================================================================

impl PerfCompiler {
    /// Compile a function that processes 8 vectors (64 i64s) and returns their sum.
    /// This tests vectorized addition throughput.
    fn compile_vector_add_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // src ptr
        sig.params.push(AbiParam::new(ptr)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src_ptr = params[0];
            let dst_ptr = params[1];

            // Load 8 vectors (unrolled for throughput testing)
            let v0 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);
            let v1 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 64);
            let v2 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 128);
            let v3 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 192);
            let v4 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 256);
            let v5 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 320);
            let v6 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 384);
            let v7 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 448);

            // Sum pairs (4 adds in parallel)
            let s01 = builder.ins().iadd(v0, v1);
            let s23 = builder.ins().iadd(v2, v3);
            let s45 = builder.ins().iadd(v4, v5);
            let s67 = builder.ins().iadd(v6, v7);

            // Sum quads (2 adds)
            let s0123 = builder.ins().iadd(s01, s23);
            let s4567 = builder.ins().iadd(s45, s67);

            // Final sum
            let result = builder.ins().iadd(s0123, s4567);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that does horizontal reduction (sum of I64X8 to scalar)
    fn compile_horizontal_sum_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // src ptr
        sig.returns.push(AbiParam::new(I64)); // sum
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src_ptr = params[0];

            // Load vector
            let vec = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);

            // Extract all lanes and sum
            let l0 = builder.ins().extractlane(vec, 0);
            let l1 = builder.ins().extractlane(vec, 1);
            let l2 = builder.ins().extractlane(vec, 2);
            let l3 = builder.ins().extractlane(vec, 3);
            let l4 = builder.ins().extractlane(vec, 4);
            let l5 = builder.ins().extractlane(vec, 5);
            let l6 = builder.ins().extractlane(vec, 6);
            let l7 = builder.ins().extractlane(vec, 7);

            // Reduce
            let s01 = builder.ins().iadd(l0, l1);
            let s23 = builder.ins().iadd(l2, l3);
            let s45 = builder.ins().iadd(l4, l5);
            let s67 = builder.ins().iadd(l6, l7);
            let s0123 = builder.ins().iadd(s01, s23);
            let s4567 = builder.ins().iadd(s45, s67);
            let total = builder.ins().iadd(s0123, s4567);

            builder.ins().return_(&[total]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile min/max reduction
    fn compile_minmax_i64x8(&mut self, name: &str, is_min: bool) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // src ptr (two vectors)
        sig.params.push(AbiParam::new(ptr)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src_ptr = params[0];
            let dst_ptr = params[1];

            // Load two vectors
            let v0 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);
            let v1 = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 64);

            // Compute min or max
            let result = if is_min {
                builder.ins().smin(v0, v1)
            } else {
                builder.ins().smax(v0, v1)
            };

            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile FMA (fused multiply-add) operation
    fn compile_fma_f64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // a ptr
        sig.params.push(AbiParam::new(ptr)); // b ptr
        sig.params.push(AbiParam::new(ptr)); // c ptr
        sig.params.push(AbiParam::new(ptr)); // dst ptr (a*b + c)
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let a_ptr = params[0];
            let b_ptr = params[1];
            let c_ptr = params[2];
            let dst_ptr = params[3];

            let a = builder.ins().load(F64X8, MemFlags::trusted(), a_ptr, 0);
            let b = builder.ins().load(F64X8, MemFlags::trusted(), b_ptr, 0);
            let c = builder.ins().load(F64X8, MemFlags::trusted(), c_ptr, 0);

            // FMA: a*b + c
            let result = builder.ins().fma(a, b, c);

            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile vector comparison (returns mask-like result)
    fn compile_compare_i32x16(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // values ptr
        sig.params.push(AbiParam::new(I32)); // threshold
        sig.params.push(AbiParam::new(ptr)); // mask output ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let values_ptr = params[0];
            let threshold = params[1];
            let mask_ptr = params[2];

            // Load values
            let values = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), values_ptr, 0);

            // Splat threshold
            let threshold_vec = builder.ins().splat(I32X16, threshold);

            // Compare: values > threshold
            let mask = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, values, threshold_vec);

            // Store mask
            builder.ins().store(MemFlags::trusted(), mask, mask_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }
}

// =============================================================================
// Benchmark Tests
// =============================================================================

#[test]
fn bench_vector_add_throughput() {
    let mut compiler = match PerfCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let func_ptr = compiler.compile_vector_add_i64x8("vec_add").unwrap();
    let func: fn(*const i64, *mut i64) = unsafe { mem::transmute(func_ptr) };

    // 512 bytes input (8 vectors of 8 i64s)
    let src: Vec<i64> = (0..64).collect();
    let mut dst = [0i64; 8];

    // Verify correctness
    func(src.as_ptr(), dst.as_mut_ptr());
    let expected: i64 = src.iter().sum();
    let actual: i64 = dst.iter().sum();
    assert_eq!(actual, expected, "Vector add correctness check");

    // Benchmark
    let iterations = BENCH_ROWS / 64; // Each call processes 64 i64s
    run_benchmark("Vector Add (I64X8 x 8)", BENCH_ROWS, 8, || {
        for _ in 0..iterations {
            func(src.as_ptr(), dst.as_mut_ptr());
        }
    });
}

#[test]
fn bench_horizontal_reduction() {
    let mut compiler = match PerfCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let func_ptr = compiler.compile_horizontal_sum_i64x8("hsum").unwrap();
    let func: fn(*const i64) -> i64 = unsafe { mem::transmute(func_ptr) };

    let src: Vec<i64> = (1..=8).collect();

    // Verify
    let expected: i64 = src.iter().sum();
    let actual = func(src.as_ptr());
    assert_eq!(actual, expected, "Horizontal sum correctness");

    // Benchmark - how fast can we reduce vectors?
    let iterations = BENCH_ROWS / 8;
    run_benchmark("Horizontal Sum (I64X8 → scalar)", BENCH_ROWS, 8, || {
        for _ in 0..iterations {
            let _ = func(src.as_ptr());
        }
    });
}

#[test]
fn bench_minmax_throughput() {
    let mut compiler = match PerfCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let min_ptr = compiler.compile_minmax_i64x8("vmin", true).unwrap();
    let max_ptr = compiler.compile_minmax_i64x8("vmax", false).unwrap();
    let min_fn: fn(*const i64, *mut i64) = unsafe { mem::transmute(min_ptr) };
    let max_fn: fn(*const i64, *mut i64) = unsafe { mem::transmute(max_ptr) };

    let src: Vec<i64> = (0..16).map(|i| (i * 7 + 13) % 100).collect();
    let mut dst = [0i64; 8];

    // Verify min
    min_fn(src.as_ptr(), dst.as_mut_ptr());
    for i in 0..8 {
        assert_eq!(dst[i], src[i].min(src[i + 8]), "Min lane {} incorrect", i);
    }

    // Verify max
    max_fn(src.as_ptr(), dst.as_mut_ptr());
    for i in 0..8 {
        assert_eq!(dst[i], src[i].max(src[i + 8]), "Max lane {} incorrect", i);
    }

    let iterations = BENCH_ROWS / 16;
    run_benchmark("Vector Min (I64X8)", BENCH_ROWS, 8, || {
        for _ in 0..iterations {
            min_fn(src.as_ptr(), dst.as_mut_ptr());
        }
    });

    run_benchmark("Vector Max (I64X8)", BENCH_ROWS, 8, || {
        for _ in 0..iterations {
            max_fn(src.as_ptr(), dst.as_mut_ptr());
        }
    });
}

#[test]
fn bench_fma_throughput() {
    let mut compiler = match PerfCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let func_ptr = compiler.compile_fma_f64x8("fma").unwrap();
    let func: fn(*const f64, *const f64, *const f64, *mut f64) =
        unsafe { mem::transmute(func_ptr) };

    let a: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect();
    let c: Vec<f64> = (0..8).map(|i| (i + 2) as f64).collect();
    let mut dst = [0f64; 8];

    // Verify: a*b + c
    func(a.as_ptr(), b.as_ptr(), c.as_ptr(), dst.as_mut_ptr());
    for i in 0..8 {
        let expected = a[i] * b[i] + c[i];
        assert!(
            (dst[i] - expected).abs() < 1e-10,
            "FMA lane {} incorrect",
            i
        );
    }

    let iterations = BENCH_ROWS / 8;
    run_benchmark("FMA (F64X8)", BENCH_ROWS, 8 * 3, || {
        for _ in 0..iterations {
            func(a.as_ptr(), b.as_ptr(), c.as_ptr(), dst.as_mut_ptr());
        }
    });
}

#[test]
fn bench_compare_throughput() {
    let mut compiler = match PerfCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let func_ptr = compiler.compile_compare_i32x16("vcmp").unwrap();
    let func: fn(*const i32, i32, *mut i32) = unsafe { mem::transmute(func_ptr) };

    let values: Vec<i32> = (0..16).collect();
    let mut mask = [0i32; 16];

    // Test with threshold 7: values 8-15 should pass
    func(values.as_ptr(), 7, mask.as_mut_ptr());
    for i in 0..16 {
        let expected = if values[i] > 7 { -1i32 } else { 0 };
        assert_eq!(mask[i], expected, "Compare lane {} incorrect", i);
    }

    let iterations = BENCH_ROWS / 16;
    run_benchmark("Compare I32X16 > threshold", BENCH_ROWS, 4, || {
        for _ in 0..iterations {
            func(values.as_ptr(), 7, mask.as_mut_ptr());
        }
    });
}

#[test]
fn bench_memory_bandwidth_scan() {
    println!("\n========================================");
    println!("Memory Bandwidth (Sequential Scan)");
    println!("========================================");

    // Pure memory bandwidth test using std (baseline)
    for &size_mb in &[1usize, 10, 100, 500] {
        let size = size_mb * 1024 * 1024 / 8; // i64 elements
        let data: Vec<i64> = (0..size as i64).collect();

        let mut sum = 0i64;
        let start = Instant::now();
        for _ in 0..5 {
            for &v in &data {
                sum = sum.wrapping_add(v);
            }
        }
        let elapsed = start.elapsed();
        let _ = sum; // prevent optimization

        let bytes_read = size * 8 * 5;
        let gb_per_sec = (bytes_read as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();

        println!(
            "  {} MB:  {:>8.2} GB/sec (scalar baseline)",
            size_mb, gb_per_sec
        );
    }
}

#[test]
fn bench_selectivity_impact() {
    println!("\n========================================");
    println!("Filter Selectivity Impact (Scalar Baseline)");
    println!("========================================");

    let rows = BENCH_ROWS;
    let data: Vec<i32> = (0..rows as i32).map(|i| i % 100).collect();

    for &threshold in &[0, 25, 50, 75, 99] {
        let start = Instant::now();
        let mut count = 0u64;
        for _ in 0..10 {
            for &v in &data {
                if v > threshold {
                    count += 1;
                }
            }
        }
        let elapsed = start.elapsed();

        let selectivity = (100 - threshold) as f64;
        let rows_per_sec = (rows * 10) as f64 / elapsed.as_secs_f64();

        println!(
            "  Selectivity {:>3}%: {:>8.2} M rows/sec, {} matches",
            selectivity as i32,
            rows_per_sec / 1_000_000.0,
            count / 10
        );
    }
}

#[test]
fn print_benchmark_summary() {
    println!("\n========================================");
    println!("AVX-512 HTAP Performance Benchmark Suite");
    println!("========================================");
    println!("\nRun individual benchmarks:");
    println!(
        "  cargo test -p cranelift-jit --test avx512_perf bench_vector_add_throughput --release -- --nocapture"
    );
    println!(
        "  cargo test -p cranelift-jit --test avx512_perf bench_horizontal_reduction --release -- --nocapture"
    );
    println!(
        "  cargo test -p cranelift-jit --test avx512_perf bench_minmax_throughput --release -- --nocapture"
    );
    println!(
        "  cargo test -p cranelift-jit --test avx512_perf bench_fma_throughput --release -- --nocapture"
    );
    println!(
        "  cargo test -p cranelift-jit --test avx512_perf bench_compare_throughput --release -- --nocapture"
    );
    println!("\nRun all benchmarks:");
    println!("  cargo test -p cranelift-jit --test avx512_perf --release -- --nocapture");
}
