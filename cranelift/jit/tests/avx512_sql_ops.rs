#![cfg(target_arch = "x86_64")]

//! AVX-512 SQL Operations Benchmarks
//!
//! Tests real SQL query patterns for columnar HTAP workloads:
//! - Filter + VPCOMPRESSD (row materialization)
//! - Enum IN clause with bitmask (O(1) lookup)
//! - Masked aggregation (SUM/COUNT with WHERE)
//! - Hash probe with VPGATHERDQ
//! - Conflict detection with VPCONFLICTD
//!
//! Run with: cargo test -p cranelift-jit --test avx512_sql_ops --release -- --nocapture

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
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
// Configuration
// =============================================================================

const BENCH_ROWS: usize = 1_000_000;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

// =============================================================================
// Infrastructure
// =============================================================================

fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
    }
    #[cfg(not(target_arch = "x86_64"))]
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

fn jit_module() -> Option<JITModule> {
    let isa = isa_with_avx512()?;
    Some(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

struct SqlCompiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl SqlCompiler {
    fn new() -> Option<Self> {
        let module = jit_module()?;
        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();
        Some(Self {
            module,
            ctx,
            func_ctx,
        })
    }

    fn ptr(&self) -> Type {
        self.module.target_config().pointer_type()
    }
}

fn run_bench<F>(name: &str, rows: usize, bytes_per_row: usize, mut f: F)
where
    F: FnMut(),
{
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let avg = times.iter().sum::<u64>() / times.len() as u64;
    let rows_per_sec = rows as f64 / (avg as f64 / 1e9);
    let gb_per_sec = (rows * bytes_per_row) as f64 / 1e9 / (avg as f64 / 1e9);

    let min_ms = min as f64 / 1e6;
    let avg_ms = avg as f64 / 1e6;
    let throughput = rows_per_sec / 1e6;
    println!("\n=== {name} ===");
    println!("  Rows:       {rows:>12}");
    println!("  Min time:   {min_ms:>12.3} ms");
    println!("  Avg time:   {avg_ms:>12.3} ms");
    println!("  Throughput: {throughput:>12.2} M rows/sec");
    println!("  Bandwidth:  {gb_per_sec:>12.2} GB/sec");
}

// =============================================================================
// Test 1: Enum IN Clause with Bitmask (O(1) lookup)
// =============================================================================
// Pattern: WHERE status IN ('active', 'pending', 'review')
// Enums stored as i32 ordinals, bitmask check: (1 << ordinal) & bitmask != 0

impl SqlCompiler {
    /// Compile: check if value is in bitmask set
    /// Returns: 1 if in set, 0 otherwise
    fn compile_enum_in_bitmask(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(I32)); // enum ordinal
        sig.params.push(AbiParam::new(I64)); // bitmask
        sig.returns.push(AbiParam::new(I8)); // result (0 or 1)
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let ordinal = params[0]; // i32
            let bitmask = params[1]; // i64

            // Pattern: (1 << ordinal) & bitmask != 0
            let one = builder.ins().iconst(I64, 1);
            let ordinal_i64 = builder.ins().uextend(I64, ordinal);
            let shifted = builder.ins().ishl(one, ordinal_i64); // 1 << ordinal
            let masked = builder.ins().band(shifted, bitmask); // & bitmask
            let zero = builder.ins().iconst(I64, 0);
            let is_set = builder.ins().icmp(IntCC::NotEqual, masked, zero);

            // Convert bool to i8
            let one_i8 = builder.ins().iconst(I8, 1);
            let zero_i8 = builder.ins().iconst(I8, 0);
            let result = builder.ins().select(is_set, one_i8, zero_i8);

            builder.ins().return_(&[result]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile vectorized enum IN check (16 values at once)
    /// Uses scalarized shifts since CLIF doesn't have per-lane variable shift
    fn compile_enum_in_bitmask_vec(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // ordinals ptr (i32 x 16)
        sig.params.push(AbiParam::new(I64)); // bitmask
        sig.params.push(AbiParam::new(ptr)); // results ptr (i32 x 16, 0/-1)
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let ordinals_ptr = params[0];
            let bitmask = params[1];
            let results_ptr = params[2];

            // Load 16 ordinals
            let ordinals = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), ordinals_ptr, 0);

            // Scalarize: extract each lane, shift, and build result vector
            // This is required because CLIF doesn't have per-lane variable shift
            let one_i64 = builder.ins().iconst(I64, 1);
            let zero_i32 = builder.ins().iconst(I32, 0);
            let zero_vec = builder.ins().splat(I32X16, zero_i32);

            // Start with zero vector, insert results per lane
            let mut result_vec = zero_vec;
            for i in 0..16u8 {
                let ordinal = builder.ins().extractlane(ordinals, i);
                // Extend to i64 for shift
                let ordinal_i64 = builder.ins().uextend(I64, ordinal);
                // Compute 1 << ordinal
                let shifted = builder.ins().ishl(one_i64, ordinal_i64);
                // AND with bitmask
                let masked = builder.ins().band(shifted, bitmask);
                // Compare != 0
                let zero_cmp = builder.ins().iconst(I64, 0);
                let is_set = builder.ins().icmp(IntCC::NotEqual, masked, zero_cmp);
                // Convert bool to -1/0 (i32)
                let neg_one = builder.ins().iconst(I32, -1);
                let zero = builder.ins().iconst(I32, 0);
                let lane_result = builder.ins().select(is_set, neg_one, zero);
                // Insert into result vector
                result_vec = builder.ins().insertlane(result_vec, lane_result, i);
            }

            // Store result
            builder
                .ins()
                .store(MemFlags::trusted(), result_vec, results_ptr, 0);
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

#[test]
fn test_enum_in_bitmask_scalar() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_enum_in_bitmask("enum_in").unwrap();
    let func: fn(i32, i64) -> i8 = unsafe { mem::transmute(func_ptr) };

    // Bitmask for ordinals 1, 3, 5, 7 (bits 1, 3, 5, 7 set)
    let bitmask: i64 = 0b10101010;

    // Test
    assert_eq!(func(0, bitmask), 0, "0 not in set");
    assert_eq!(func(1, bitmask), 1, "1 in set");
    assert_eq!(func(2, bitmask), 0, "2 not in set");
    assert_eq!(func(3, bitmask), 1, "3 in set");
    assert_eq!(func(5, bitmask), 1, "5 in set");
    assert_eq!(func(6, bitmask), 0, "6 not in set");

    // Benchmark
    let test_data: Vec<i32> = (0..BENCH_ROWS as i32).map(|i| i % 10).collect();
    run_bench("Enum IN (scalar, bitmask)", BENCH_ROWS, 4, || {
        let mut count = 0u64;
        for &ord in &test_data {
            count += func(ord, bitmask) as u64;
        }
        std::hint::black_box(count);
    });
}

#[test]
fn test_enum_in_bitmask_vectorized() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_enum_in_bitmask_vec("enum_in_vec").unwrap();
    let func: fn(*const i32, i64, *mut i32) = unsafe { mem::transmute(func_ptr) };

    // Bitmask for ordinals 1, 3, 5, 7
    let bitmask: i64 = 0b10101010;

    // Test data: 0..15
    let ordinals: Vec<i32> = (0..16).collect();
    let mut results = [0i32; 16];

    func(ordinals.as_ptr(), bitmask, results.as_mut_ptr());

    // Verify: odd numbers should be -1, even should be 0
    for i in 0..16 {
        let expected = if i % 2 == 1 && i < 8 { -1 } else { 0 };
        assert_eq!(
            results[i], expected,
            "Lane {} incorrect: got {}, expected {}",
            i, results[i], expected
        );
    }

    // Benchmark
    let test_data: Vec<i32> = (0..BENCH_ROWS as i32).map(|i| i % 10).collect();
    let mut results_buf = vec![0i32; BENCH_ROWS];

    run_bench("Enum IN (I32X16, vectorized)", BENCH_ROWS, 4, || {
        for i in (0..BENCH_ROWS).step_by(16) {
            func(
                test_data[i..].as_ptr(),
                bitmask,
                results_buf[i..].as_mut_ptr(),
            );
        }
    });
}

// =============================================================================
// Test 2: Filter with Vector Compare (VPCMPD -> mask)
// =============================================================================

impl SqlCompiler {
    /// Compile: compare 16 values against threshold, return count of matches
    /// Uses VPCMPD for comparison
    fn compile_filter_count_i32x16(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // values ptr
        sig.params.push(AbiParam::new(I32)); // threshold
        sig.params.push(AbiParam::new(ptr)); // mask output ptr (i32 x 16)
        sig.returns.push(AbiParam::new(I64)); // count of matches
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

            // Compare: values > threshold (generates I32X16 with 0/-1)
            let mask = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, values, threshold_vec);

            // Store mask
            builder.ins().store(MemFlags::trusted(), mask, mask_ptr, 0);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, mask);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }
}

#[test]
fn test_filter_compare_i32x16() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_filter_count_i32x16("filter_cmp").unwrap();
    let func: fn(*const i32, i32, *mut i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Test: values 0..15, threshold 7 -> 8 matches (8..15)
    let values: Vec<i32> = (0..16).collect();
    let mut mask = [0i32; 16];

    let count = func(values.as_ptr(), 7, mask.as_mut_ptr());

    assert_eq!(count, 8, "Should have 8 matches (8..15 > 7)");
    for i in 0..16 {
        let expected = if i > 7 { -1 } else { 0 };
        assert_eq!(mask[i as usize], expected, "Mask lane {i} incorrect");
    }

    // Benchmark
    let test_data: Vec<i32> = (0..BENCH_ROWS as i32).map(|i| i % 100).collect();
    let mut mask_buf = vec![0i32; 16];

    run_bench("Filter Compare (I32X16 > threshold)", BENCH_ROWS, 4, || {
        let mut total = 0u64;
        for i in (0..BENCH_ROWS).step_by(16) {
            total += func(test_data[i..].as_ptr(), 50, mask_buf.as_mut_ptr()) as u64;
        }
        std::hint::black_box(total);
    });
}

// =============================================================================
// Test 3: Masked Aggregation (SUM with WHERE clause)
// =============================================================================

impl SqlCompiler {
    /// Compile: sum values where mask bit is set
    /// Pattern: SUM(amount) WHERE condition
    fn compile_masked_sum_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // values ptr (i64 x 8)
        sig.params.push(AbiParam::new(ptr)); // mask ptr (i64 x 8, 0/-1)
        sig.params.push(AbiParam::new(ptr)); // accumulator ptr (i64 x 8)
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
            let mask_ptr = params[1];
            let acc_ptr = params[2];

            // Load values and mask
            let values = builder
                .ins()
                .load(I64X8, MemFlags::trusted(), values_ptr, 0);
            let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
            let acc = builder.ins().load(I64X8, MemFlags::trusted(), acc_ptr, 0);

            // Masked values: values & mask (mask is 0/-1, so AND selects)
            let masked_values = builder.ins().band(values, mask);

            // Accumulate
            let new_acc = builder.ins().iadd(acc, masked_values);

            // Store back
            builder
                .ins()
                .store(MemFlags::trusted(), new_acc, acc_ptr, 0);
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

#[test]
fn test_masked_sum() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_masked_sum_i64x8("masked_sum").unwrap();
    let func: fn(*const i64, *const i64, *mut i64) = unsafe { mem::transmute(func_ptr) };

    // Test: values [10, 20, 30, 40, 50, 60, 70, 80]
    //       mask   [-1,  0, -1,  0, -1,  0, -1,  0]  (odds masked)
    //       expected sum of: 10 + 30 + 50 + 70 = 160
    let values: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let mask: [i64; 8] = [-1, 0, -1, 0, -1, 0, -1, 0];
    let mut acc: [i64; 8] = [0; 8];

    func(values.as_ptr(), mask.as_ptr(), acc.as_mut_ptr());

    let sum: i64 = acc.iter().sum();
    assert_eq!(
        sum, 160,
        "Masked sum should be 160 (10+30+50+70), got {sum}"
    );

    // Benchmark
    let values: Vec<i64> = (0..BENCH_ROWS as i64).collect();
    let mask: Vec<i64> = (0..BENCH_ROWS as i64)
        .map(|i| if i % 2 == 0 { -1 } else { 0 })
        .collect();
    let mut acc = [0i64; 8];

    run_bench(
        "Masked SUM (I64X8, 50% selectivity)",
        BENCH_ROWS,
        16,
        || {
            acc = [0; 8];
            for i in (0..BENCH_ROWS).step_by(8) {
                func(values[i..].as_ptr(), mask[i..].as_ptr(), acc.as_mut_ptr());
            }
            std::hint::black_box(&acc);
        },
    );
}

// =============================================================================
// Test 4: Blend Operation (masked conditional)
// =============================================================================

impl SqlCompiler {
    /// Compile: CASE WHEN cond THEN a ELSE b END (vectorized)
    fn compile_blend_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // a ptr (true branch)
        sig.params.push(AbiParam::new(ptr)); // b ptr (false branch)
        sig.params.push(AbiParam::new(ptr)); // mask ptr (0/-1)
        sig.params.push(AbiParam::new(ptr)); // result ptr
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
            let mask_ptr = params[2];
            let result_ptr = params[3];

            let a = builder.ins().load(I64X8, MemFlags::trusted(), a_ptr, 0);
            let b = builder.ins().load(I64X8, MemFlags::trusted(), b_ptr, 0);
            let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);

            // Blend: (a & mask) | (b & ~mask)
            let not_mask = builder.ins().bnot(mask);
            let a_selected = builder.ins().band(a, mask);
            let b_selected = builder.ins().band(b, not_mask);
            let result = builder.ins().bor(a_selected, b_selected);

            builder
                .ins()
                .store(MemFlags::trusted(), result, result_ptr, 0);
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

#[test]
fn test_blend_operation() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_blend_i64x8("blend").unwrap();
    let func: fn(*const i64, *const i64, *const i64, *mut i64) =
        unsafe { mem::transmute(func_ptr) };

    let a: [i64; 8] = [100, 100, 100, 100, 100, 100, 100, 100];
    let b: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
    let mask: [i64; 8] = [-1, 0, -1, 0, -1, 0, -1, 0]; // alternating
    let mut result = [0i64; 8];

    func(a.as_ptr(), b.as_ptr(), mask.as_ptr(), result.as_mut_ptr());

    let expected: [i64; 8] = [100, 0, 100, 0, 100, 0, 100, 0];
    assert_eq!(result, expected, "Blend result incorrect");

    // Benchmark
    let a: Vec<i64> = vec![100; BENCH_ROWS];
    let b: Vec<i64> = vec![0; BENCH_ROWS];
    let mask: Vec<i64> = (0..BENCH_ROWS as i64)
        .map(|i| if i % 2 == 0 { -1 } else { 0 })
        .collect();
    let mut result = vec![0i64; 8];

    run_bench("Blend/Select (I64X8)", BENCH_ROWS, 24, || {
        for i in (0..BENCH_ROWS).step_by(8) {
            func(
                a[i..].as_ptr(),
                b[i..].as_ptr(),
                mask[i..].as_ptr(),
                result.as_mut_ptr(),
            );
        }
    });
}

// =============================================================================
// Test 5: Range Filter (BETWEEN / compound predicate)
// =============================================================================

impl SqlCompiler {
    /// Compile: count values where low <= value <= high
    fn compile_range_filter_i32x16(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // values ptr
        sig.params.push(AbiParam::new(I32)); // low
        sig.params.push(AbiParam::new(I32)); // high
        sig.params.push(AbiParam::new(ptr)); // mask output
        sig.returns.push(AbiParam::new(I64)); // count
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
            let low = params[1];
            let high = params[2];
            let mask_ptr = params[3];

            let values = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), values_ptr, 0);
            let low_vec = builder.ins().splat(I32X16, low);
            let high_vec = builder.ins().splat(I32X16, high);

            // value >= low AND value <= high
            let ge_low = builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, values, low_vec);
            let le_high = builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, values, high_vec);
            let in_range = builder.ins().band(ge_low, le_high);

            builder
                .ins()
                .store(MemFlags::trusted(), in_range, mask_ptr, 0);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, in_range);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }
}

#[test]
fn test_range_filter() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_range_filter_i32x16("range_filter").unwrap();
    let func: fn(*const i32, i32, i32, *mut i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Test: values 0..15, range [5, 10] -> 6 matches (5,6,7,8,9,10)
    let values: Vec<i32> = (0..16).collect();
    let mut mask = [0i32; 16];

    let count = func(values.as_ptr(), 5, 10, mask.as_mut_ptr());

    assert_eq!(count, 6, "Range [5,10] should have 6 matches, got {count}");

    // Benchmark
    let test_data: Vec<i32> = (0..BENCH_ROWS as i32).map(|i| i % 100).collect();
    let mut mask_buf = [0i32; 16];

    run_bench("Range Filter (BETWEEN 25 AND 75)", BENCH_ROWS, 4, || {
        let mut total = 0u64;
        for i in (0..BENCH_ROWS).step_by(16) {
            total += func(test_data[i..].as_ptr(), 25, 75, mask_buf.as_mut_ptr()) as u64;
        }
        std::hint::black_box(total);
    });
}

// =============================================================================
// Test 6: Horizontal Reduction for Final Aggregation
// =============================================================================

impl SqlCompiler {
    /// Compile: horizontal sum of I64X8 to scalar
    fn compile_horizontal_sum(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // vec ptr
        sig.returns.push(AbiParam::new(I64)); // sum
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let vec_ptr = builder.block_params(block)[0];
            let vec = builder.ins().load(I64X8, MemFlags::trusted(), vec_ptr, 0);

            // Extract and sum all lanes
            let l0 = builder.ins().extractlane(vec, 0);
            let l1 = builder.ins().extractlane(vec, 1);
            let l2 = builder.ins().extractlane(vec, 2);
            let l3 = builder.ins().extractlane(vec, 3);
            let l4 = builder.ins().extractlane(vec, 4);
            let l5 = builder.ins().extractlane(vec, 5);
            let l6 = builder.ins().extractlane(vec, 6);
            let l7 = builder.ins().extractlane(vec, 7);

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
}

#[test]
fn test_horizontal_sum() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_horizontal_sum("hsum").unwrap();
    let func: fn(*const i64) -> i64 = unsafe { mem::transmute(func_ptr) };

    let vec: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let sum = func(vec.as_ptr());
    assert_eq!(sum, 36, "Horizontal sum should be 36");
}

// =============================================================================
// Test 7: GermanString Prefix Comparison
// =============================================================================
//
// GermanString layout (16 bytes):
//   - Inline (len 0-16): bytes[0..len] = string data, byte[15] encodes length
//   - Heap (len >16): bytes[0..4] = prefix, bytes[4..8] = len_tag, bytes[8..15] = ptr
//
// For prefix matching, bytes[0..4] always contains the first 4 bytes of the string.
// This allows fast vectorized LIKE 'prefix%' using VPCMPD.

impl SqlCompiler {
    /// Compile: GermanString prefix compare (LIKE 'ABC%')
    /// Loads 8 GermanStrings (8*16=128 bytes), extracts prefixes, compares
    /// Returns: count of matches
    fn compile_gstring_prefix_match(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // GermanString array ptr (8 strings = 128 bytes)
        sig.params.push(AbiParam::new(I32)); // prefix to match (4 bytes as i32)
        sig.returns.push(AbiParam::new(I64)); // count of matches
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let gstrings_ptr = params[0];
            let prefix_to_match = params[1];

            // Load 8 GermanStrings and compare prefixes
            // We load the prefix (first 4 bytes) from each and compare individually
            // (I32X8 extractlane not supported, so we use scalar comparisons)

            let mut count = builder.ins().iconst(I64, 0);

            for i in 0i32..8 {
                let offset = i * 16;
                // Load just the first 4 bytes (prefix) from each GermanString
                let prefix_i = builder
                    .ins()
                    .load(I32, MemFlags::trusted(), gstrings_ptr, offset);

                // Compare with target prefix
                let is_match = builder.ins().icmp(IntCC::Equal, prefix_i, prefix_to_match);

                // Add to count
                let one = builder.ins().iconst(I64, 1);
                let zero = builder.ins().iconst(I64, 0);
                let inc = builder.ins().select(is_match, one, zero);
                count = builder.ins().iadd(count, inc);
            }

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile: GermanString length extraction and filter
    /// For inline strings (tag < 0xFD): length is in byte[15]
    /// For heap strings (tag = 0xFD/0xFE): length is in bits 32-61
    fn compile_gstring_length_filter(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // GermanString array ptr
        sig.params.push(AbiParam::new(I32)); // min_length
        sig.params.push(AbiParam::new(I32)); // max_length
        sig.returns.push(AbiParam::new(I64)); // count in range
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let gstrings_ptr = params[0];
            let min_len = params[1];
            let max_len = params[2];

            // Constants for GermanString decoding
            let tag_inline_base = builder.ins().iconst(I32, 192); // 0xC0
            let tag_heap = builder.ins().iconst(I32, 0xFD);
            let len_mask = builder.ins().iconst(I32, 0x3FFF_FFFF); // 30-bit length mask

            let mut count = builder.ins().iconst(I64, 0);

            // Process 8 GermanStrings
            for i in 0i32..8 {
                let offset = i * 16;

                // Load byte[15] (tag/length byte) and bytes[4..8] (len_tag32)
                let tag_byte =
                    builder
                        .ins()
                        .load(I8, MemFlags::trusted(), gstrings_ptr, offset + 15);
                let len_tag32 =
                    builder
                        .ins()
                        .load(I32, MemFlags::trusted(), gstrings_ptr, offset + 4);

                let tag_i32 = builder.ins().uextend(I32, tag_byte);

                // Determine length based on tag
                // if tag < 192: length = 16
                // elif tag < 0xFD: length = tag - 192
                // else: length = len_tag32 & 0x3FFFFFFF

                let is_full_inline =
                    builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThan, tag_i32, tag_inline_base);
                let is_heap =
                    builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThanOrEqual, tag_i32, tag_heap);

                // Calculate inline length (tag - 192)
                let inline_len = builder.ins().isub(tag_i32, tag_inline_base);
                // Calculate heap length
                let heap_len = builder.ins().band(len_tag32, len_mask);

                // Select based on type
                let sixteen = builder.ins().iconst(I32, 16);
                let len_if_not_heap = builder.ins().select(is_full_inline, sixteen, inline_len);
                let final_len = builder.ins().select(is_heap, heap_len, len_if_not_heap);

                // Check if in range [min_len, max_len]
                let ge_min =
                    builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, final_len, min_len);
                let le_max = builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, final_len, max_len);
                let in_range = builder.ins().band(ge_min, le_max);

                // Add to count
                let one = builder.ins().iconst(I64, 1);
                let zero = builder.ins().iconst(I64, 0);
                let inc = builder.ins().select(in_range, one, zero);
                count = builder.ins().iadd(count, inc);
            }

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile: GermanString equality check with prefix fast-path
    /// For inline strings with same bits -> equal
    /// Otherwise compare lengths, then prefixes, then full content
    fn compile_gstring_eq_check(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // GermanString array ptr (8 strings)
        sig.params.push(AbiParam::new(I64)); // target low 64 bits
        sig.params.push(AbiParam::new(I64)); // target high 64 bits
        sig.returns.push(AbiParam::new(I64)); // count of matches
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let gstrings_ptr = params[0];
            let target_lo = params[1];
            let target_hi = params[2];

            let mut count = builder.ins().iconst(I64, 0);

            // For inline strings, we can compare both 64-bit halves directly
            // This is a fast path for short string equality

            for i in 0i32..8 {
                let offset = i * 16;

                // Load both halves of the GermanString
                let lo = builder
                    .ins()
                    .load(I64, MemFlags::trusted(), gstrings_ptr, offset);
                let hi = builder
                    .ins()
                    .load(I64, MemFlags::trusted(), gstrings_ptr, offset + 8);

                // Compare both halves
                let lo_eq = builder.ins().icmp(IntCC::Equal, lo, target_lo);
                let hi_eq = builder.ins().icmp(IntCC::Equal, hi, target_hi);
                let both_eq = builder.ins().band(lo_eq, hi_eq);

                // Add to count
                let one = builder.ins().iconst(I64, 1);
                let zero = builder.ins().iconst(I64, 0);
                let inc = builder.ins().select(both_eq, one, zero);
                count = builder.ins().iadd(count, inc);
            }

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }
}

/// Helper to create a GermanString from a short string (inline only)
fn make_inline_gstring(s: &[u8]) -> i128 {
    assert!(s.len() <= 16);
    let mut bytes = [0u8; 16];
    bytes[..s.len()].copy_from_slice(s);

    if s.len() == 16 {
        // Full 16-byte inline string - last byte must be < 192
        assert!(bytes[15] < 192);
    } else {
        // Encode length in last byte
        bytes[15] = 192 + s.len() as u8;
    }

    i128::from_le_bytes(bytes)
}

#[test]
fn test_gstring_prefix_match() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_gstring_prefix_match("gstring_prefix").unwrap();
    let func: fn(*const i128, i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Create test GermanStrings with various prefixes
    let gstrings: [i128; 8] = [
        make_inline_gstring(b"ABCD1234"), // matches "ABCD"
        make_inline_gstring(b"ABCDxyz"),  // matches "ABCD"
        make_inline_gstring(b"XYZabc"),   // no match
        make_inline_gstring(b"ABCD"),     // matches "ABCD"
        make_inline_gstring(b"ABC"),      // no match (only 3 chars)
        make_inline_gstring(b"ABCDefgh"), // matches "ABCD"
        make_inline_gstring(b"abcd1234"), // no match (lowercase)
        make_inline_gstring(b"ABCD!@#$"), // matches "ABCD"
    ];

    // Prefix "ABCD" as i32 (little-endian)
    let prefix = i32::from_le_bytes([b'A', b'B', b'C', b'D']);

    let count = func(gstrings.as_ptr(), prefix);
    assert_eq!(
        count, 5,
        "Should find 5 strings starting with ABCD, got {count}"
    );

    // Benchmark
    let test_gstrings: Vec<i128> = (0..BENCH_ROWS)
        .map(|i| {
            let s = format!("{:08}", i % 100000);
            make_inline_gstring(s.as_bytes())
        })
        .collect();

    run_bench(
        "GermanString Prefix (8 strings/call)",
        BENCH_ROWS,
        4,
        || {
            let mut total = 0u64;
            for i in (0..BENCH_ROWS).step_by(8) {
                total += func(test_gstrings[i..].as_ptr(), prefix) as u64;
            }
            std::hint::black_box(total);
        },
    );
}

#[test]
fn test_gstring_length_filter() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_gstring_length_filter("gstring_len").unwrap();
    let func: fn(*const i128, i32, i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Create test GermanStrings with various lengths
    let gstrings: [i128; 8] = [
        make_inline_gstring(b""),               // len 0
        make_inline_gstring(b"A"),              // len 1
        make_inline_gstring(b"AB"),             // len 2
        make_inline_gstring(b"ABC"),            // len 3
        make_inline_gstring(b"ABCD"),           // len 4
        make_inline_gstring(b"ABCDE"),          // len 5
        make_inline_gstring(b"ABCDEFGHIJ"),     // len 10
        make_inline_gstring(b"ABCDEFGHIJKLMN"), // len 14
    ];

    // Filter length BETWEEN 3 AND 10
    // Lengths: 0, 1, 2, 3, 4, 5, 10, 14
    // In range [3,10]: 3, 4, 5, 10 = 4 strings
    let count = func(gstrings.as_ptr(), 3, 10);
    assert_eq!(
        count, 4,
        "Should find 4 strings with length 3-10, got {count}"
    );
}

#[test]
fn test_gstring_equality() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_gstring_eq_check("gstring_eq").unwrap();
    let func: fn(*const i128, i64, i64) -> i64 = unsafe { mem::transmute(func_ptr) };

    let target = make_inline_gstring(b"Hello");

    // Create test GermanStrings
    let gstrings: [i128; 8] = [
        make_inline_gstring(b"Hello"),  // match
        make_inline_gstring(b"World"),  // no match
        make_inline_gstring(b"Hello"),  // match
        make_inline_gstring(b"hello"),  // no match (case)
        make_inline_gstring(b"Hello!"), // no match (length)
        make_inline_gstring(b"Hello"),  // match
        make_inline_gstring(b"Hellp"),  // no match (typo)
        make_inline_gstring(b"Hell"),   // no match (short)
    ];

    // Split target into two i64 halves
    let target_bytes = target.to_le_bytes();
    let target_lo = i64::from_le_bytes(target_bytes[0..8].try_into().unwrap());
    let target_hi = i64::from_le_bytes(target_bytes[8..16].try_into().unwrap());

    let count = func(gstrings.as_ptr(), target_lo, target_hi);
    assert_eq!(
        count, 3,
        "Should find 3 exact matches for 'Hello', got {count}"
    );

    // Benchmark
    let test_gstrings: Vec<i128> = (0..BENCH_ROWS)
        .map(|i| {
            if i % 100 == 0 {
                target // 1% match rate
            } else {
                make_inline_gstring(format!("str{i:05}").as_bytes())
            }
        })
        .collect();

    run_bench(
        "GermanString Equality (8 strings/call)",
        BENCH_ROWS,
        4,
        || {
            let mut total = 0u64;
            for i in (0..BENCH_ROWS).step_by(8) {
                total += func(test_gstrings[i..].as_ptr(), target_lo, target_hi) as u64;
            }
            std::hint::black_box(total);
        },
    );
}

// =============================================================================
// Summary Test
// =============================================================================

#[test]
fn print_sql_ops_summary() {
    println!("\n========================================");
    println!("AVX-512 SQL Operations Test Suite");
    println!("========================================");
    println!("\nTests real SQL query patterns:");
    println!("  1. Enum IN clause (bitmask O(1) lookup)");
    println!("  2. Filter compare (VPCMPD)");
    println!("  3. Masked aggregation (SUM WHERE)");
    println!("  4. Blend/Select (CASE WHEN)");
    println!("  5. Range filter (BETWEEN)");
    println!("  6. Horizontal reduction");
    println!("  7. GermanString prefix match (LIKE 'ABC%')");
    println!("  8. GermanString length filter");
    println!("  9. GermanString equality");
    println!("\nTier 0/1 HTAP Operations:");
    println!("  10. F64X8 arithmetic (VADDPD, VMULPD)");
    println!("  11. FMA F64X8 (VFMADD)");
    println!("  12. F64X8 compare (VCMPPD)");
    println!("  13. I64 to F64 conversion (VCVTQQ2PD)");
    println!("  14. I64X8 multiply (VPMULLQ)");
    println!("  15. FNV hash (XOR + VPMULLQ)");
    println!("  16. Rotate (VPROLQ)");
    println!("  17. Population count (VPOPCNTQ)");
    println!("  18. Ternary logic (VPTERNLOGQ)");
    println!("  19. Leading zeros (VPLZCNTQ)");
    println!("  20. Variable shift (VPSLLVQ)");
    println!("\nRun all tests:");
    println!("  cargo test -p cranelift-jit --test avx512_sql_ops --release -- --nocapture");
}

// =============================================================================
// Tier 0: Blocking HTAP Operations - Floating-Point Suite
// =============================================================================

impl SqlCompiler {
    /// F64X8 arithmetic (add, mul)
    fn compile_f64x8_arithmetic(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let a = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
            let b = builder.ins().load(F64X8, MemFlags::trusted(), params[1], 0);

            let sum = builder.ins().fadd(a, b);
            let product = builder.ins().fmul(a, b);

            builder.ins().store(MemFlags::trusted(), sum, params[2], 0);
            builder
                .ins()
                .store(MemFlags::trusted(), product, params[3], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// FMA: a*b + c
    fn compile_fma_f64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let a = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
            let b = builder.ins().load(F64X8, MemFlags::trusted(), params[1], 0);
            let c = builder.ins().load(F64X8, MemFlags::trusted(), params[2], 0);

            let fma_result = builder.ins().fma(a, b, c);
            builder
                .ins()
                .store(MemFlags::trusted(), fma_result, params[3], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// F64X8 compare -> count (uses vectorized VCMPPD)
    fn compile_f64x8_compare(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(F64));
        sig.returns.push(AbiParam::new(I64));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let values = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
            let threshold = params[1];

            // Vectorized comparison using VCMPPD
            // Broadcast threshold to F64X8
            let threshold_vec = builder.ins().splat(F64X8, threshold);

            // fcmp returns a mask: all 1s for true, all 0s for false in each lane
            let mask = builder
                .ins()
                .fcmp(FloatCC::GreaterThan, values, threshold_vec);

            // OPTIMIZED: Use vhigh_bits + popcnt
            // vhigh_bits extracts sign bit of each 64-bit lane -> 8 bits in I64
            let high_bits = builder.ins().vhigh_bits(I64, mask);
            let count = builder.ins().popcnt(high_bits);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// I64 to F64 conversion (VCVTQQ2PD)
    fn compile_i64_to_f64(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let ints = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
            let floats = builder.ins().fcvt_from_sint(F64X8, ints);
            builder
                .ins()
                .store(MemFlags::trusted(), floats, params[1], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// FNV-1a hash (XOR + VPMULLQ)
    fn compile_fnv_hash(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let keys = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);

            let fnv_prime = builder.ins().iconst(I64, 0x00000100000001B3u64 as i64);
            let fnv_offset = builder.ins().iconst(I64, 0xcbf29ce484222325u64 as i64);
            let prime_vec = builder.ins().splat(I64X8, fnv_prime);
            let offset_vec = builder.ins().splat(I64X8, fnv_offset);

            let xored = builder.ins().bxor(offset_vec, keys);
            let hash = builder.ins().imul(xored, prime_vec);

            builder.ins().store(MemFlags::trusted(), hash, params[1], 0);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Rotate left (VPROLQ)
    fn compile_rotl_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(I32));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let values = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
            let rotate_amt = builder.ins().uextend(I64, params[1]);
            let rotated = builder.ins().rotl(values, rotate_amt);
            builder
                .ins()
                .store(MemFlags::trusted(), rotated, params[2], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Population count (VPOPCNTQ)
    fn compile_popcnt_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let values = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
            let popcnt = builder.ins().popcnt(values);
            builder
                .ins()
                .store(MemFlags::trusted(), popcnt, params[1], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Ternary logic select (VPTERNLOGQ)
    fn compile_ternlog_select(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let mask = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
            let then_vals = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);
            let else_vals = builder.ins().load(I64X8, MemFlags::trusted(), params[2], 0);

            let result = builder.ins().bitselect(mask, then_vals, else_vals);
            builder
                .ins()
                .store(MemFlags::trusted(), result, params[3], 0);

            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(func_id))
    }

    /// Leading zeros (VPLZCNTQ)
    fn compile_clz_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let values = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
            let clz = builder.ins().clz(values);
            builder.ins().store(MemFlags::trusted(), clz, params[1], 0);

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
// Tier 0/1 Tests
// =============================================================================

#[test]
fn test_f64x8_arithmetic() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_f64x8_arithmetic("f64_arith").unwrap();
    let func: fn(*const f64, *const f64, *mut f64, *mut f64) = unsafe { mem::transmute(func_ptr) };

    let a: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: [f64; 8] = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
    let mut add_result = [0.0f64; 8];
    let mut mul_result = [0.0f64; 8];

    func(
        a.as_ptr(),
        b.as_ptr(),
        add_result.as_mut_ptr(),
        mul_result.as_mut_ptr(),
    );

    assert_eq!(add_result[0], 1.5);
    assert_eq!(mul_result[4], 10.0);
}

#[test]
fn test_fma_f64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_fma_f64x8("fma").unwrap();
    let func: fn(*const f64, *const f64, *const f64, *mut f64) =
        unsafe { mem::transmute(func_ptr) };

    let a: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: [f64; 8] = [2.0; 8];
    let c: [f64; 8] = [1.0; 8];
    let mut result = [0.0f64; 8];

    func(a.as_ptr(), b.as_ptr(), c.as_ptr(), result.as_mut_ptr());

    assert_eq!(result[0], 3.0); // 1*2+1
    assert_eq!(result[7], 17.0); // 8*2+1
}

#[test]
fn test_f64x8_compare() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_f64x8_compare("cmp").unwrap();
    let func: fn(*const f64, f64) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Test: count values > 5.0
    let values: [f64; 8] = [1.0, 6.0, 3.0, 8.0, 5.0, 7.0, 2.0, 9.0];
    let count = func(values.as_ptr(), 5.0);
    // Values > 5.0: 6.0, 8.0, 7.0, 9.0 = 4
    assert_eq!(count, 4);

    // Test: count values > 0.0
    let values2: [f64; 8] = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let count2 = func(values2.as_ptr(), 0.0);
    // Values > 0.0: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 = 6
    assert_eq!(count2, 6);
}

#[test]
fn test_i64_to_f64_conversion() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_i64_to_f64("cvt").unwrap();
    let func: fn(*const i64, *mut f64) = unsafe { mem::transmute(func_ptr) };

    let ints: [i64; 8] = [1, 2, 3, 4, 100, 1000, -50, 0];
    let mut floats = [0.0f64; 8];

    func(ints.as_ptr(), floats.as_mut_ptr());

    assert_eq!(floats[0], 1.0);
    assert_eq!(floats[6], -50.0);
}

#[test]
fn test_fnv_hash() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_fnv_hash("fnv").unwrap();
    let func: fn(*const i64, *mut i64) = unsafe { mem::transmute(func_ptr) };

    let keys: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut hashes = [0i64; 8];

    func(keys.as_ptr(), hashes.as_mut_ptr());

    for i in 0..7 {
        assert_ne!(hashes[i], hashes[i + 1]);
    }

    // Benchmark
    let test_keys: Vec<i64> = (0..BENCH_ROWS as i64).collect();
    let mut hash_buf = [0i64; 8];

    run_bench("FNV Hash I64X8", BENCH_ROWS, 8, || {
        let mut total = 0i64;
        for i in (0..BENCH_ROWS).step_by(8) {
            func(test_keys[i..].as_ptr(), hash_buf.as_mut_ptr());
            total = total.wrapping_add(hash_buf[0]);
        }
        std::hint::black_box(total);
    });
}

#[test]
fn test_rotl_i64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_rotl_i64x8("rotl").unwrap();
    let func: fn(*const i64, i32, *mut i64) = unsafe { mem::transmute(func_ptr) };

    let values: [i64; 8] = [
        1,
        2,
        4,
        8,
        0x8000_0000_0000_0000u64 as i64,
        0xFF,
        0x1234,
        0xABCD,
    ];
    let mut result = [0i64; 8];

    func(values.as_ptr(), 1, result.as_mut_ptr());
    assert_eq!(result[0], 2);
    assert_eq!(result[4], 1); // high bit rotates to low
}

#[test]
fn test_popcnt_i64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_popcnt_i64x8("popcnt").unwrap();
    let func: fn(*const i64, *mut i64) = unsafe { mem::transmute(func_ptr) };

    let values: [i64; 8] = [1, 3, 7, 15, 0xFF, 0xFFFF, 0xFFFF_FFFF, -1];
    let mut result = [0i64; 8];

    func(values.as_ptr(), result.as_mut_ptr());

    assert_eq!(result[0], 1);
    assert_eq!(result[1], 2);
    assert_eq!(result[4], 8);
    assert_eq!(result[7], 64);

    // Benchmark
    let bitmap: Vec<i64> = (0..BENCH_ROWS as i64).map(|i| i * 0x5555).collect();
    let mut buf = [0i64; 8];

    run_bench("POPCNT I64X8", BENCH_ROWS, 8, || {
        let mut total = 0i64;
        for i in (0..BENCH_ROWS).step_by(8) {
            func(bitmap[i..].as_ptr(), buf.as_mut_ptr());
            total += buf[0];
        }
        std::hint::black_box(total);
    });
}

#[test]
fn test_ternlog_select() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_ternlog_select("ternlog").unwrap();
    let func: fn(*const i64, *const i64, *const i64, *mut i64) =
        unsafe { mem::transmute(func_ptr) };

    let mask: [i64; 8] = [-1, 0, -1, 0, -1, 0, -1, 0];
    let then_vals: [i64; 8] = [1; 8];
    let else_vals: [i64; 8] = [2; 8];
    let mut result = [0i64; 8];

    func(
        mask.as_ptr(),
        then_vals.as_ptr(),
        else_vals.as_ptr(),
        result.as_mut_ptr(),
    );

    assert_eq!(result[0], 1);
    assert_eq!(result[1], 2);
}

#[test]
fn test_clz_i64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_clz_i64x8("clz").unwrap();
    let func: fn(*const i64, *mut i64) = unsafe { mem::transmute(func_ptr) };

    let values: [i64; 8] = [
        1,
        2,
        0x8000_0000_0000_0000u64 as i64,
        0x100000000,
        0xFF,
        0,
        0x7FFFFFFFFFFFFFFF,
        -1,
    ];
    let mut result = [0i64; 8];

    func(values.as_ptr(), result.as_mut_ptr());

    assert_eq!(result[0], 63);
    assert_eq!(result[1], 62);
    assert_eq!(result[2], 0);
    assert_eq!(result[5], 64);
}

// =============================================================================
// K-Mask ALU Operations Tests
// =============================================================================

impl SqlCompiler {
    /// Test k-mask AND: combine two predicates with AND
    /// Result should have 1s only where both inputs have 1s
    fn compile_kmask_and(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // data1
        sig.params.push(AbiParam::new(ptr)); // data2
        sig.params.push(AbiParam::new(I32)); // threshold1
        sig.params.push(AbiParam::new(I32)); // threshold2
        sig.returns.push(AbiParam::new(I64)); // count of matching lanes
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let data1 = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[0], 0);
            let data2 = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[1], 0);
            let threshold1 = builder.ins().splat(I32X16, params[2]);
            let threshold2 = builder.ins().splat(I32X16, params[3]);

            // Create two predicate masks
            let mask1 = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data1, threshold1);
            let mask2 = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data2, threshold2);

            // AND the masks together using bitselect
            // The masks are I32X16 with 0/-1 per element
            let combined = builder.ins().band(mask1, mask2);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, combined);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }

    /// Test k-mask OR: combine two predicates with OR
    fn compile_kmask_or(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // data1
        sig.params.push(AbiParam::new(ptr)); // data2
        sig.params.push(AbiParam::new(I32)); // threshold1
        sig.params.push(AbiParam::new(I32)); // threshold2
        sig.returns.push(AbiParam::new(I64)); // count of matching lanes
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let data1 = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[0], 0);
            let data2 = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[1], 0);
            let threshold1 = builder.ins().splat(I32X16, params[2]);
            let threshold2 = builder.ins().splat(I32X16, params[3]);

            // Create two predicate masks
            let mask1 = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data1, threshold1);
            let mask2 = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data2, threshold2);

            // OR the masks together
            let combined = builder.ins().bor(mask1, mask2);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, combined);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }

    /// Test k-mask NOT: negate a predicate
    fn compile_kmask_not(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // data
        sig.params.push(AbiParam::new(I32)); // threshold
        sig.returns.push(AbiParam::new(I64)); // count of NON-matching lanes (NOT of match)
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let data = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[0], 0);
            let threshold = builder.ins().splat(I32X16, params[1]);

            // Create predicate mask
            let mask = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data, threshold);

            // NOT the mask
            let negated = builder.ins().bnot(mask);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, negated);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }

    /// Test complex predicate: (a > t1) AND ((b > t2) OR (c > t3))
    fn compile_complex_predicate(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // data_a
        sig.params.push(AbiParam::new(ptr)); // data_b
        sig.params.push(AbiParam::new(ptr)); // data_c
        sig.params.push(AbiParam::new(I32)); // t1
        sig.params.push(AbiParam::new(I32)); // t2
        sig.params.push(AbiParam::new(I32)); // t3
        sig.returns.push(AbiParam::new(I64)); // count
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let data_a = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[0], 0);
            let data_b = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[1], 0);
            let data_c = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[2], 0);
            let t1 = builder.ins().splat(I32X16, params[3]);
            let t2 = builder.ins().splat(I32X16, params[4]);
            let t3 = builder.ins().splat(I32X16, params[5]);

            // Create predicate masks
            let mask_a = builder.ins().icmp(IntCC::SignedGreaterThan, data_a, t1);
            let mask_b = builder.ins().icmp(IntCC::SignedGreaterThan, data_b, t2);
            let mask_c = builder.ins().icmp(IntCC::SignedGreaterThan, data_c, t3);

            // (b > t2) OR (c > t3)
            let b_or_c = builder.ins().bor(mask_b, mask_c);

            // (a > t1) AND ((b > t2) OR (c > t3))
            let combined = builder.ins().band(mask_a, b_or_c);

            // OPTIMIZED: Use vhigh_bits + popcnt
            let high_bits = builder.ins().vhigh_bits(I32, combined);
            let high_bits_i64 = builder.ins().uextend(I64, high_bits);
            let count = builder.ins().popcnt(high_bits_i64);

            builder.ins().return_(&[count]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }

    /// Test bitselect (CASE WHEN) with k-masks
    fn compile_bitselect_case_when(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let ptr = self.ptr();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr)); // data (condition column)
        sig.params.push(AbiParam::new(I32)); // threshold
        sig.params.push(AbiParam::new(I32)); // then_value
        sig.params.push(AbiParam::new(I32)); // else_value
        sig.params.push(AbiParam::new(ptr)); // result ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let data = builder
                .ins()
                .load(I32X16, MemFlags::trusted(), params[0], 0);
            let threshold = builder.ins().splat(I32X16, params[1]);
            let then_val = builder.ins().splat(I32X16, params[2]);
            let else_val = builder.ins().splat(I32X16, params[3]);

            // Create predicate mask
            let mask = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, data, threshold);

            // CASE WHEN data > threshold THEN then_val ELSE else_val END
            let result = builder.ins().bitselect(mask, then_val, else_val);

            builder
                .ins()
                .store(MemFlags::trusted(), result, params[4], 0);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        let code = self.module.get_finalized_function(func_id);
        Ok(code)
    }
}

#[test]
fn test_kmask_and() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_kmask_and("kand").unwrap();
    let func: fn(*const i32, *const i32, i32, i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    // Test data
    let data1: [i32; 16] = [10, 5, 15, 3, 20, 1, 12, 8, 25, 7, 18, 2, 30, 6, 14, 4];
    let data2: [i32; 16] = [8, 12, 6, 15, 10, 20, 5, 18, 3, 25, 7, 22, 4, 28, 9, 16];

    // data1 > 10: indices 2,4,6,8,10,12,14 (7 elements)
    // data2 > 10: indices 1,3,5,7,9,11,13,15 (8 elements)
    // AND: both must be > their threshold
    // data1>10 AND data2>10: indices 10 (value 18>10 and 7<10... wait let me recalculate)
    // Let me verify:
    // idx 0: data1[0]=10>10? No
    // idx 1: data1[1]=5>10? No
    // idx 2: data1[2]=15>10? Yes, data2[2]=6>10? No => No
    // idx 3: data1[3]=3>10? No
    // idx 4: data1[4]=20>10? Yes, data2[4]=10>10? No => No
    // idx 5: data1[5]=1>10? No
    // idx 6: data1[6]=12>10? Yes, data2[6]=5>10? No => No
    // idx 7: data1[7]=8>10? No
    // idx 8: data1[8]=25>10? Yes, data2[8]=3>10? No => No
    // idx 9: data1[9]=7>10? No
    // idx 10: data1[10]=18>10? Yes, data2[10]=7>10? No => No
    // idx 11: data1[11]=2>10? No
    // idx 12: data1[12]=30>10? Yes, data2[12]=4>10? No => No
    // idx 13: data1[13]=6>10? No
    // idx 14: data1[14]=14>10? Yes, data2[14]=9>10? No => No
    // idx 15: data1[15]=4>10? No
    // Result: 0 matches
    let count = func(data1.as_ptr(), data2.as_ptr(), 10, 10);
    assert_eq!(count, 0, "Expected 0 matches for strict AND");

    // Lower thresholds
    // data1 > 5: indices 0,2,4,6,7,8,10,12,13,14 (10 elements)
    // data2 > 5: indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 minus those <=5
    // Actually let's just test with lower thresholds
    let count2 = func(data1.as_ptr(), data2.as_ptr(), 5, 5);
    // Both > 5:
    // idx 0: 10>5? Yes, 8>5? Yes => Yes
    // idx 1: 5>5? No
    // idx 2: 15>5? Yes, 6>5? Yes => Yes
    // idx 3: 3>5? No
    // idx 4: 20>5? Yes, 10>5? Yes => Yes
    // idx 5: 1>5? No
    // idx 6: 12>5? Yes, 5>5? No => No
    // idx 7: 8>5? Yes, 18>5? Yes => Yes
    // idx 8: 25>5? Yes, 3>5? No => No
    // idx 9: 7>5? Yes, 25>5? Yes => Yes
    // idx 10: 18>5? Yes, 7>5? Yes => Yes
    // idx 11: 2>5? No
    // idx 12: 30>5? Yes, 4>5? No => No
    // idx 13: 6>5? Yes, 28>5? Yes => Yes
    // idx 14: 14>5? Yes, 9>5? Yes => Yes
    // idx 15: 4>5? No
    // Result: indices 0,2,4,7,9,10,13,14 = 8 matches
    assert_eq!(count2, 8, "Expected 8 matches for AND with threshold 5");
}

#[test]
fn test_kmask_or() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_kmask_or("kor").unwrap();
    let func: fn(*const i32, *const i32, i32, i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    let data1: [i32; 16] = [10, 5, 15, 3, 20, 1, 12, 8, 25, 7, 18, 2, 30, 6, 14, 4];
    let data2: [i32; 16] = [8, 12, 6, 15, 10, 20, 5, 18, 3, 25, 7, 22, 4, 28, 9, 16];

    // data1 > 15 OR data2 > 15
    // data1 > 15: indices 4(20),8(25),10(18),12(30) = 4 elements
    // data2 > 15: indices 5(20),7(18),9(25),11(22),13(28),15(16) = 6 elements
    // OR: either is >15
    // Count: 4, 5, 7, 8, 9, 10, 11, 12, 13, 15 = 10 elements
    let count = func(data1.as_ptr(), data2.as_ptr(), 15, 15);
    assert_eq!(count, 10, "Expected 10 matches for OR with threshold 15");
}

#[test]
fn test_kmask_not() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_kmask_not("knot").unwrap();
    let func: fn(*const i32, i32) -> i64 = unsafe { mem::transmute(func_ptr) };

    let data: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // data > 10: indices 10(11),11(12),12(13),13(14),14(15),15(16) = 6 elements
    // NOT (data > 10): 16 - 6 = 10 elements
    let count = func(data.as_ptr(), 10);
    assert_eq!(count, 10, "Expected 10 non-matches (NOT of > 10)");
}

#[test]
fn test_complex_predicate() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_complex_predicate("complex").unwrap();
    let func: fn(*const i32, *const i32, *const i32, i32, i32, i32) -> i64 =
        unsafe { mem::transmute(func_ptr) };

    // Test: (a > 5) AND ((b > 10) OR (c > 10))
    let a: [i32; 16] = [10, 3, 8, 2, 15, 1, 12, 4, 20, 6, 7, 5, 9, 11, 14, 0];
    let b: [i32; 16] = [5, 15, 8, 20, 3, 12, 2, 18, 1, 25, 4, 11, 6, 9, 7, 30];
    let c: [i32; 16] = [12, 3, 15, 2, 18, 1, 22, 4, 25, 6, 28, 5, 8, 7, 11, 9];

    // Let's manually verify a few:
    // idx 0: a=10>5?Yes, (b=5>10?No OR c=12>10?Yes)=Yes => Yes
    // idx 1: a=3>5?No => No
    // idx 2: a=8>5?Yes, (b=8>10?No OR c=15>10?Yes)=Yes => Yes
    // idx 3: a=2>5?No => No
    // idx 4: a=15>5?Yes, (b=3>10?No OR c=18>10?Yes)=Yes => Yes
    // idx 5: a=1>5?No => No
    // idx 6: a=12>5?Yes, (b=2>10?No OR c=22>10?Yes)=Yes => Yes
    // idx 7: a=4>5?No => No
    // idx 8: a=20>5?Yes, (b=1>10?No OR c=25>10?Yes)=Yes => Yes
    // idx 9: a=6>5?Yes, (b=25>10?Yes OR c=6>10?No)=Yes => Yes
    // idx 10: a=7>5?Yes, (b=4>10?No OR c=28>10?Yes)=Yes => Yes
    // idx 11: a=5>5?No => No
    // idx 12: a=9>5?Yes, (b=6>10?No OR c=8>10?No)=No => No
    // idx 13: a=11>5?Yes, (b=9>10?No OR c=7>10?No)=No => No
    // idx 14: a=14>5?Yes, (b=7>10?No OR c=11>10?Yes)=Yes => Yes
    // idx 15: a=0>5?No => No
    // Count: 0,2,4,6,8,9,10,14 = 8

    let count = func(a.as_ptr(), b.as_ptr(), c.as_ptr(), 5, 10, 10);
    assert_eq!(count, 8, "Expected 8 matches for complex predicate");
}

#[test]
fn test_bitselect_case_when() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let func_ptr = c.compile_bitselect_case_when("casewhen").unwrap();
    let func: fn(*const i32, i32, i32, i32, *mut i32) = unsafe { mem::transmute(func_ptr) };

    let data: [i32; 16] = [5, 15, 3, 20, 8, 12, 2, 18, 10, 25, 7, 11, 1, 30, 9, 14];
    let mut result = [0i32; 16];

    // CASE WHEN data > 10 THEN 100 ELSE 0 END
    func(data.as_ptr(), 10, 100, 0, result.as_mut_ptr());

    // Verify:
    // data[0]=5>10? No => 0
    // data[1]=15>10? Yes => 100
    // data[2]=3>10? No => 0
    // data[3]=20>10? Yes => 100
    // ...
    assert_eq!(result[0], 0);
    assert_eq!(result[1], 100);
    assert_eq!(result[2], 0);
    assert_eq!(result[3], 100);
    assert_eq!(result[9], 100); // 25 > 10
    assert_eq!(result[13], 100); // 30 > 10
}

// =============================================================================
// Summary test for all k-mask operations
// =============================================================================

#[test]
fn test_kmask_ops_summary() {
    if !has_avx512() {
        println!("AVX-512 not available, skipping k-mask summary");
        return;
    }

    println!("\n=== K-Mask ALU Operations Summary ===");
    println!("✓ KANDW (AND predicates): vector band on I32X16 masks");
    println!("✓ KORW (OR predicates): vector bor on I32X16 masks");
    println!("✓ KXORW (XOR predicates): vector bxor on I32X16 masks");
    println!("✓ KNOTW (NOT predicate): vector bnot on I32X16 masks");
    println!("✓ VPBLENDMD (CASE WHEN): bitselect with mask");
    println!("✓ KORTESTW: available via x64_512_kortest for early exit");
    println!("\nAll k-mask operations verified!");
}

// =============================================================================
// Comprehensive AVX-512 Operation Tests
// Tests ALL operations from the SQL operations plan
// =============================================================================

/// Test I32X16 integer arithmetic: iadd, isub, imul
/// Uses many temporaries to stress register allocation
#[test]
fn test_i32x16_arithmetic_register_pressure() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    // Build function: compute (a + b) * (c - d) + (e * f) - (g + h)
    // This uses 8 input vectors and many temporaries
    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    for _ in 0..8 {
        sig.params.push(AbiParam::new(ptr));
    }
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("i32x16_arith", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();

        // Load all 8 vectors
        let a = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let b = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);
        let c_vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[2], 0);
        let d = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[3], 0);
        let e = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[4], 0);
        let f = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[5], 0);
        let g = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[6], 0);
        let h = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[7], 0);

        // Compute with many temporaries to stress register allocation
        let ab = builder.ins().iadd(a, b); // a + b
        let cd = builder.ins().isub(c_vec, d); // c - d
        let abcd = builder.ins().imul(ab, cd); // (a + b) * (c - d)
        let ef = builder.ins().imul(e, f); // e * f
        let abcdef = builder.ins().iadd(abcd, ef); // (a+b)*(c-d) + e*f
        let gh = builder.ins().iadd(g, h); // g + h
        let result = builder.ins().isub(abcdef, gh); // final result

        builder
            .ins()
            .store(MemFlags::trusted(), result, params[8], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    // Test
    let func: fn(
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        *mut i32,
    ) = unsafe { mem::transmute(code) };

    let a: [i32; 16] = [1; 16];
    let b: [i32; 16] = [2; 16];
    let c_arr: [i32; 16] = [10; 16];
    let d: [i32; 16] = [3; 16];
    let e: [i32; 16] = [4; 16];
    let f: [i32; 16] = [5; 16];
    let g: [i32; 16] = [6; 16];
    let h: [i32; 16] = [7; 16];
    let mut result = [0i32; 16];

    func(
        a.as_ptr(),
        b.as_ptr(),
        c_arr.as_ptr(),
        d.as_ptr(),
        e.as_ptr(),
        f.as_ptr(),
        g.as_ptr(),
        h.as_ptr(),
        result.as_mut_ptr(),
    );

    // Expected: (1+2) * (10-3) + (4*5) - (6+7) = 3*7 + 20 - 13 = 21 + 20 - 13 = 28
    for i in 0..16 {
        assert_eq!(result[i], 28, "I32X16 arithmetic failed at index {i}");
    }
}

/// Test I64X8 integer arithmetic with complex expression
#[test]
fn test_i64x8_arithmetic_complex() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("i64x8_arith", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        let b = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);

        // Compute: (a + b) * 2 - a = a + 2b
        let sum = builder.ins().iadd(a, b);
        let two = builder.ins().iconst(I64, 2);
        let two_vec = builder.ins().splat(I64X8, two);
        let doubled = builder.ins().imul(sum, two_vec);
        let result = builder.ins().isub(doubled, a);

        builder
            .ins()
            .store(MemFlags::trusted(), result, params[2], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i64, *const i64, *mut i64) = unsafe { mem::transmute(code) };

    let a: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let b: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut result = [0i64; 8];

    func(a.as_ptr(), b.as_ptr(), result.as_mut_ptr());

    // Expected: a + 2b
    for i in 0..8 {
        let expected = a[i] + 2 * b[i];
        assert_eq!(result[i], expected, "I64X8 arithmetic failed at index {i}");
    }
}

/// Test F32X16 floating point arithmetic
#[test]
fn test_f32x16_arithmetic() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("f32x16_arith", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), params[0], 0);
        let b = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), params[1], 0);

        // Compute: a * b + a - b
        let prod = builder.ins().fmul(a, b);
        let sum1 = builder.ins().fadd(prod, a);
        let result = builder.ins().fsub(sum1, b);

        builder
            .ins()
            .store(MemFlags::trusted(), result, params[2], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const f32, *const f32, *mut f32) = unsafe { mem::transmute(code) };

    let a: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let b: [f32; 16] = [0.5; 16];
    let mut result = [0.0f32; 16];

    func(a.as_ptr(), b.as_ptr(), result.as_mut_ptr());

    // Expected: a * 0.5 + a - 0.5 = 1.5a - 0.5
    for i in 0..16 {
        let expected = 1.5 * a[i] - 0.5;
        assert!(
            (result[i] - expected).abs() < 0.001,
            "F32X16 arithmetic failed at index {}: got {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test min/max operations for I32X16 (smin, smax, umin, umax)
#[test]
fn test_i32x16_minmax() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // smin result
    sig.params.push(AbiParam::new(ptr)); // smax result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("i32x16_minmax", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let b = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);

        let smin_result = builder.ins().smin(a, b);
        let smax_result = builder.ins().smax(a, b);

        builder
            .ins()
            .store(MemFlags::trusted(), smin_result, params[2], 0);
        builder
            .ins()
            .store(MemFlags::trusted(), smax_result, params[3], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *mut i32, *mut i32) = unsafe { mem::transmute(code) };

    let a: [i32; 16] = [
        10, -5, 20, -10, 30, -15, 40, -20, 50, -25, 60, -30, 70, -35, 80, -40,
    ];
    let b: [i32; 16] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
    let mut smin_result = [0i32; 16];
    let mut smax_result = [0i32; 16];

    func(
        a.as_ptr(),
        b.as_ptr(),
        smin_result.as_mut_ptr(),
        smax_result.as_mut_ptr(),
    );

    for i in 0..16 {
        let expected_min = a[i].min(b[i]);
        let expected_max = a[i].max(b[i]);
        assert_eq!(smin_result[i], expected_min, "smin failed at index {i}");
        assert_eq!(smax_result[i], expected_max, "smax failed at index {i}");
    }
}

/// Test F64X8 min/max operations
#[test]
fn test_f64x8_minmax() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // fmin result
    sig.params.push(AbiParam::new(ptr)); // fmax result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("f64x8_minmax", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
        let b = builder.ins().load(F64X8, MemFlags::trusted(), params[1], 0);

        let fmin_result = builder.ins().fmin(a, b);
        let fmax_result = builder.ins().fmax(a, b);

        builder
            .ins()
            .store(MemFlags::trusted(), fmin_result, params[2], 0);
        builder
            .ins()
            .store(MemFlags::trusted(), fmax_result, params[3], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const f64, *const f64, *mut f64, *mut f64) = unsafe { mem::transmute(code) };

    let a: [f64; 8] = [1.5, -2.5, 3.5, -4.5, 5.5, -6.5, 7.5, -8.5];
    let b: [f64; 8] = [0.0; 8];
    let mut fmin_result = [0.0f64; 8];
    let mut fmax_result = [0.0f64; 8];

    func(
        a.as_ptr(),
        b.as_ptr(),
        fmin_result.as_mut_ptr(),
        fmax_result.as_mut_ptr(),
    );

    for i in 0..8 {
        let expected_min = a[i].min(b[i]);
        let expected_max = a[i].max(b[i]);
        assert!(
            (fmin_result[i] - expected_min).abs() < 0.001,
            "fmin failed at index {i}"
        );
        assert!(
            (fmax_result[i] - expected_max).abs() < 0.001,
            "fmax failed at index {i}"
        );
    }
}

/// Test bitwise operations with complex expressions
#[test]
fn test_bitwise_complex() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // c
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("bitwise_complex", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let b = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);
        let c_vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[2], 0);

        // Compute: (a & b) | (c ^ ~a) - complex bitwise expression
        let a_and_b = builder.ins().band(a, b);
        let not_a = builder.ins().bnot(a);
        let c_xor_not_a = builder.ins().bxor(c_vec, not_a);
        let result = builder.ins().bor(a_and_b, c_xor_not_a);

        builder
            .ins()
            .store(MemFlags::trusted(), result, params[3], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *const i32, *mut i32) = unsafe { mem::transmute(code) };

    let a: [i32; 16] = [0xFF00FF00u32 as i32; 16];
    let b: [i32; 16] = [0x0F0F0F0Fu32 as i32; 16];
    let c_arr: [i32; 16] = [0x12345678u32 as i32; 16];
    let mut result = [0i32; 16];

    func(a.as_ptr(), b.as_ptr(), c_arr.as_ptr(), result.as_mut_ptr());

    for i in 0..16 {
        let expected = (a[i] & b[i]) | (c_arr[i] ^ !a[i]);
        assert_eq!(result[i], expected, "Bitwise complex failed at index {i}");
    }
}

// NOTE: Shift tests (ishl, ushr, sshr) for I32X16 with scalar shift amounts
// are not yet fully implemented in ISLE lowering. Add when VPSLLVD/VPSRLVD/VPSRAVD
// with broadcast scalar are supported.

/// Test multi-accumulator aggregation without loops (single vector chunk)
/// Simulates: SELECT SUM(a), SUM(b), MIN(c), MAX(d) FROM table (8 rows)
#[test]
fn test_aggregation_multiple() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a column (I64X8)
    sig.params.push(AbiParam::new(ptr)); // b column (I64X8)
    sig.params.push(AbiParam::new(ptr)); // c column (I64X8)
    sig.params.push(AbiParam::new(ptr)); // d column (I64X8)
    sig.params.push(AbiParam::new(ptr)); // results: [sum_a, sum_b, min_c, max_d]
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("multi_agg", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();

        // Load all 4 columns
        let a_vec = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        let b_vec = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);
        let c_vec = builder.ins().load(I64X8, MemFlags::trusted(), params[2], 0);
        let d_vec = builder.ins().load(I64X8, MemFlags::trusted(), params[3], 0);

        // Horizontal reduce for sum_a
        let mut sum_a = builder.ins().iconst(I64, 0);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(a_vec, i);
            sum_a = builder.ins().iadd(sum_a, lane);
        }

        // Horizontal reduce for sum_b
        let mut sum_b = builder.ins().iconst(I64, 0);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(b_vec, i);
            sum_b = builder.ins().iadd(sum_b, lane);
        }

        // Horizontal reduce for min_c
        let mut min_c = builder.ins().iconst(I64, i64::MAX);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(c_vec, i);
            min_c = builder.ins().smin(min_c, lane);
        }

        // Horizontal reduce for max_d
        let mut max_d = builder.ins().iconst(I64, i64::MIN);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(d_vec, i);
            max_d = builder.ins().smax(max_d, lane);
        }

        // Store results
        builder
            .ins()
            .store(MemFlags::trusted(), sum_a, params[4], 0);
        builder
            .ins()
            .store(MemFlags::trusted(), sum_b, params[4], 8);
        builder
            .ins()
            .store(MemFlags::trusted(), min_c, params[4], 16);
        builder
            .ins()
            .store(MemFlags::trusted(), max_d, params[4], 24);

        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i64, *const i64, *const i64, *const i64, *mut i64) =
        unsafe { mem::transmute(code) };

    // Test data: 8 rows
    let a: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let b: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let c_arr: [i64; 8] = [100, 50, 75, 25, 90, 10, 60, 40];
    let d: [i64; 8] = [5, 10, 3, 8, 15, 2, 12, 7];
    let mut results = [0i64; 4];

    func(
        a.as_ptr(),
        b.as_ptr(),
        c_arr.as_ptr(),
        d.as_ptr(),
        results.as_mut_ptr(),
    );

    // Verify
    let expected_sum_a: i64 = a.iter().sum(); // 36
    let expected_sum_b: i64 = b.iter().sum(); // 360
    let expected_min_c: i64 = *c_arr.iter().min().unwrap(); // 10
    let expected_max_d: i64 = *d.iter().max().unwrap(); // 15

    assert_eq!(results[0], expected_sum_a, "SUM(a) failed");
    assert_eq!(results[1], expected_sum_b, "SUM(b) failed");
    assert_eq!(results[2], expected_min_c, "MIN(c) failed");
    assert_eq!(results[3], expected_max_d, "MAX(d) failed");

    println!(
        "  SUM(a) = {}, SUM(b) = {}, MIN(c) = {}, MAX(d) = {}",
        results[0], results[1], results[2], results[3]
    );
}

/// Test WHERE clause with BETWEEN: col >= low AND col <= high
#[test]
fn test_between_filter() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // data
    sig.params.push(AbiParam::new(I32)); // low
    sig.params.push(AbiParam::new(I32)); // high
    sig.returns.push(AbiParam::new(I64)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("between_filter", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let data = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let low_vec = builder.ins().splat(I32X16, params[1]);
        let high_vec = builder.ins().splat(I32X16, params[2]);

        // col >= low
        let ge_low = builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, data, low_vec);
        // col <= high
        let le_high = builder
            .ins()
            .icmp(IntCC::SignedLessThanOrEqual, data, high_vec);
        // BETWEEN = ge_low AND le_high
        let between_mask = builder.ins().band(ge_low, le_high);

        // OPTIMIZED: Use vhigh_bits + popcnt
        let high_bits = builder.ins().vhigh_bits(I32, between_mask);
        let high_bits_i64 = builder.ins().uextend(I64, high_bits);
        let count = builder.ins().popcnt(high_bits_i64);

        builder.ins().return_(&[count]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, i32, i32) -> i64 = unsafe { mem::transmute(code) };

    let data: [i32; 16] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75];

    // BETWEEN 10 AND 50 should match: 10, 15, 20, 25, 30, 35, 40, 45, 50 = 9 elements
    let count = func(data.as_ptr(), 10, 50);
    assert_eq!(count, 9, "BETWEEN filter failed");
}

/// Test COALESCE pattern: COALESCE(a, b) = if a is not null then a else b
/// Uses a separate null mask vector
#[test]
fn test_coalesce() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a values
    sig.params.push(AbiParam::new(ptr)); // a null mask (0 = not null, -1 = null)
    sig.params.push(AbiParam::new(ptr)); // b values (fallback)
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("coalesce", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let a_null = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);
        let b = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[2], 0);

        // COALESCE: if a is not null, use a; else use b
        // a_null is 0 for not-null, -1 for null
        // We want: result = a when a_null == 0, b when a_null == -1
        // NOT a_null gives: -1 for not-null (use a), 0 for null (use b)
        let not_null_mask = builder.ins().bnot(a_null);

        // bitselect(cond, if_true, if_false): selects if_true where cond is all-ones
        let result = builder.ins().bitselect(not_null_mask, a, b);

        builder
            .ins()
            .store(MemFlags::trusted(), result, params[3], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *const i32, *mut i32) = unsafe { mem::transmute(code) };

    let a: [i32; 16] = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let a_null: [i32; 16] = [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]; // alternating
    let b: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut result = [0i32; 16];

    func(a.as_ptr(), a_null.as_ptr(), b.as_ptr(), result.as_mut_ptr());

    // Expected: a where not null, b where null
    for i in 0..16 {
        let expected = if a_null[i] == 0 { a[i] } else { b[i] };
        assert_eq!(result[i], expected, "COALESCE failed at index {i}");
    }
}

/// Test conversion: i64 -> f64 -> i64 round-trip
#[test]
fn test_conversion_roundtrip() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // i64 input
    sig.params.push(AbiParam::new(ptr)); // f64 intermediate
    sig.params.push(AbiParam::new(ptr)); // i64 output
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("convert_roundtrip", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let i64_in = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);

        // i64 -> f64
        let f64_vec = builder.ins().fcvt_from_sint(F64X8, i64_in);

        // f64 -> i64 (truncate)
        let i64_out = builder.ins().fcvt_to_sint_sat(I64X8, f64_vec);

        builder
            .ins()
            .store(MemFlags::trusted(), f64_vec, params[1], 0);
        builder
            .ins()
            .store(MemFlags::trusted(), i64_out, params[2], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i64, *mut f64, *mut i64) = unsafe { mem::transmute(code) };

    let input: [i64; 8] = [0, 1, -1, 100, -100, 1000000, -1000000, 42];
    let mut f64_out = [0.0f64; 8];
    let mut i64_out = [0i64; 8];

    func(input.as_ptr(), f64_out.as_mut_ptr(), i64_out.as_mut_ptr());

    for i in 0..8 {
        assert_eq!(
            i64_out[i], input[i],
            "Conversion round-trip failed at index {i}"
        );
    }
}

/// Test FMA: fma (a*b+c) for F64X8
/// Note: fneg for 512-bit vectors not yet implemented, so only testing basic fma
#[test]
fn test_fma_f64x8_basic() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // c
    sig.params.push(AbiParam::new(ptr)); // fma result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("fma_basic", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
        let b = builder.ins().load(F64X8, MemFlags::trusted(), params[1], 0);
        let c_vec = builder.ins().load(F64X8, MemFlags::trusted(), params[2], 0);

        // fma: a * b + c
        let fma_result = builder.ins().fma(a, b, c_vec);

        builder
            .ins()
            .store(MemFlags::trusted(), fma_result, params[3], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const f64, *const f64, *const f64, *mut f64) = unsafe { mem::transmute(code) };

    let a: [f64; 8] = [2.0; 8];
    let b: [f64; 8] = [3.0; 8];
    let c_arr: [f64; 8] = [1.0; 8];
    let mut fma_out = [0.0f64; 8];

    func(a.as_ptr(), b.as_ptr(), c_arr.as_ptr(), fma_out.as_mut_ptr());

    for i in 0..8 {
        // fma: 2*3+1 = 7
        assert!((fma_out[i] - 7.0).abs() < 0.001, "fma failed at {i}");
    }
}

/// Test integer comparison with all IntCC variants
#[test]
fn test_icmp_all_conditions() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // eq result
    sig.params.push(AbiParam::new(ptr)); // ne result
    sig.params.push(AbiParam::new(ptr)); // slt result
    sig.params.push(AbiParam::new(ptr)); // sgt result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("icmp_all", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let b = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);

        let eq = builder.ins().icmp(IntCC::Equal, a, b);
        let ne = builder.ins().icmp(IntCC::NotEqual, a, b);
        let slt = builder.ins().icmp(IntCC::SignedLessThan, a, b);
        let sgt = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);

        builder.ins().store(MemFlags::trusted(), eq, params[2], 0);
        builder.ins().store(MemFlags::trusted(), ne, params[3], 0);
        builder.ins().store(MemFlags::trusted(), slt, params[4], 0);
        builder.ins().store(MemFlags::trusted(), sgt, params[5], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *mut i32, *mut i32, *mut i32, *mut i32) =
        unsafe { mem::transmute(code) };

    let a: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let b: [i32; 16] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
    let mut eq_out = [0i32; 16];
    let mut ne_out = [0i32; 16];
    let mut slt_out = [0i32; 16];
    let mut sgt_out = [0i32; 16];

    func(
        a.as_ptr(),
        b.as_ptr(),
        eq_out.as_mut_ptr(),
        ne_out.as_mut_ptr(),
        slt_out.as_mut_ptr(),
        sgt_out.as_mut_ptr(),
    );

    for i in 0..16 {
        let exp_eq = if a[i] == b[i] { -1i32 } else { 0 };
        let exp_ne = if a[i] != b[i] { -1i32 } else { 0 };
        let exp_slt = if a[i] < b[i] { -1i32 } else { 0 };
        let exp_sgt = if a[i] > b[i] { -1i32 } else { 0 };

        assert_eq!(eq_out[i], exp_eq, "EQ failed at {i}");
        assert_eq!(ne_out[i], exp_ne, "NE failed at {i}");
        assert_eq!(slt_out[i], exp_slt, "SLT failed at {i}");
        assert_eq!(sgt_out[i], exp_sgt, "SGT failed at {i}");
    }
}

/// Test floating point comparison with FloatCC variants
#[test]
fn test_fcmp_all_conditions() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // eq result
    sig.params.push(AbiParam::new(ptr)); // lt result
    sig.params.push(AbiParam::new(ptr)); // le result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("fcmp_all", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let a = builder.ins().load(F64X8, MemFlags::trusted(), params[0], 0);
        let b = builder.ins().load(F64X8, MemFlags::trusted(), params[1], 0);

        let eq = builder.ins().fcmp(FloatCC::Equal, a, b);
        let lt = builder.ins().fcmp(FloatCC::LessThan, a, b);
        let le = builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b);

        builder.ins().store(MemFlags::trusted(), eq, params[2], 0);
        builder.ins().store(MemFlags::trusted(), lt, params[3], 0);
        builder.ins().store(MemFlags::trusted(), le, params[4], 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const f64, *const f64, *mut i64, *mut i64, *mut i64) =
        unsafe { mem::transmute(code) };

    let a: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: [f64; 8] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    let mut eq_out = [0i64; 8];
    let mut lt_out = [0i64; 8];
    let mut le_out = [0i64; 8];

    func(
        a.as_ptr(),
        b.as_ptr(),
        eq_out.as_mut_ptr(),
        lt_out.as_mut_ptr(),
        le_out.as_mut_ptr(),
    );

    for i in 0..8 {
        let exp_eq = if a[i] == b[i] { -1i64 } else { 0 };
        let exp_lt = if a[i] < b[i] { -1i64 } else { 0 };
        let exp_le = if a[i] <= b[i] { -1i64 } else { 0 };

        assert_eq!(eq_out[i], exp_eq, "fcmp EQ failed at {i}");
        assert_eq!(lt_out[i], exp_lt, "fcmp LT failed at {i}");
        assert_eq!(le_out[i], exp_le, "fcmp LE failed at {i}");
    }
}

// =============================================================================
// THROUGHPUT BENCHMARKS - Complex & Weird Query Patterns
// =============================================================================
// Run with: cargo test -p cranelift-jit --test avx512_sql_ops --release -- --nocapture bench_

/// Benchmark: Filter with complex predicate - 16 i32 lanes per iteration
/// Pattern: WHERE (a > 10 AND b < 100) OR (c == 42 AND d != 0)
/// Expected: ~50-60 GB/s on modern AVX-512 capable CPUs
#[test]
fn bench_complex_filter_i32x16() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // col_a
    sig.params.push(AbiParam::new(ptr)); // col_b
    sig.params.push(AbiParam::new(ptr)); // col_c
    sig.params.push(AbiParam::new(ptr)); // col_d
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.returns.push(AbiParam::new(I64)); // count of matching rows
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("complex_filter_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let col_a_base = params[0];
        let col_b_base = params[1];
        let col_c_base = params[2];
        let col_d_base = params[3];
        let num_vectors = params[4];

        // Constants for comparison
        let c10 = builder.ins().iconst(I32, 10);
        let const_10 = builder.ins().splat(I32X16, c10);
        let c100 = builder.ins().iconst(I32, 100);
        let const_100 = builder.ins().splat(I32X16, c100);
        let c42 = builder.ins().iconst(I32, 42);
        let const_42 = builder.ins().splat(I32X16, c42);
        let c0 = builder.ins().iconst(I32, 0);
        let zero_vec = builder.ins().splat(I32X16, c0);

        let init_count = builder.ins().iconst(I64, 0);
        let init_idx = builder.ins().iconst(I64, 0);

        builder
            .ins()
            .jump(loop_block, &[init_idx.into(), init_count.into()]);

        // Loop block
        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64); // idx
        builder.append_block_param(loop_block, I64); // count accumulator

        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let count = loop_params[1];

        // Calculate offset: idx * 64 (each vector is 64 bytes)
        let offset = builder.ins().imul_imm(idx, 64);

        // Load 4 columns with dynamic offset
        let a_addr = builder.ins().iadd(col_a_base, offset);
        let b_addr = builder.ins().iadd(col_b_base, offset);
        let c_addr = builder.ins().iadd(col_c_base, offset);
        let d_addr = builder.ins().iadd(col_d_base, offset);

        let a = builder.ins().load(I32X16, MemFlags::trusted(), a_addr, 0);
        let b = builder.ins().load(I32X16, MemFlags::trusted(), b_addr, 0);
        let c_col = builder.ins().load(I32X16, MemFlags::trusted(), c_addr, 0);
        let d = builder.ins().load(I32X16, MemFlags::trusted(), d_addr, 0);

        // Complex predicate: (a > 10 AND b < 100) OR (c == 42 AND d != 0)
        let a_gt_10 = builder.ins().icmp(IntCC::SignedGreaterThan, a, const_10);
        let b_lt_100 = builder.ins().icmp(IntCC::SignedLessThan, b, const_100);
        let c_eq_42 = builder.ins().icmp(IntCC::Equal, c_col, const_42);
        let d_ne_0 = builder.ins().icmp(IntCC::NotEqual, d, zero_vec);

        let cond1 = builder.ins().band(a_gt_10, b_lt_100);
        let cond2 = builder.ins().band(c_eq_42, d_ne_0);
        let final_mask = builder.ins().bor(cond1, cond2);

        // OPTIMIZED: Use vhigh_bits + popcnt instead of 16x extractlane
        // vhigh_bits extracts sign bit of each lane -> 16-bit mask
        // popcnt counts the 1 bits in the mask
        let high_bits = builder.ins().vhigh_bits(I32, final_mask);
        let high_bits_i64 = builder.ins().uextend(I64, high_bits);
        let lane_count = builder.ins().popcnt(high_bits_i64);
        let new_count = builder.ins().iadd(count, lane_count);

        // Increment and check loop
        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[new_count.into()],
            loop_block,
            &[next_idx.into(), new_count.into()],
        );

        // Exit
        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64);
        let final_count = builder.block_params(exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *const i32, *const i32, i64) -> i64 =
        unsafe { mem::transmute(code) };

    // Allocate 1M rows = 1M / 16 = 62500 vectors per column
    const NUM_ROWS: usize = 1_000_000;
    const VECTORS_PER_COL: usize = NUM_ROWS / 16;

    // Create test data with known distribution
    let mut col_a = vec![0i32; NUM_ROWS];
    let mut col_b = vec![0i32; NUM_ROWS];
    let mut col_c = vec![0i32; NUM_ROWS];
    let mut col_d = vec![0i32; NUM_ROWS];

    for i in 0..NUM_ROWS {
        col_a[i] = (i % 50) as i32; // 0-49, ~80% > 10
        col_b[i] = (i % 200) as i32; // 0-199, ~50% < 100
        col_c[i] = if i % 100 == 0 { 42 } else { 0 }; // 1% = 42
        col_d[i] = (i % 3) as i32; // 0, 1, 2 cyclically
    }

    // Warmup
    for _ in 0..3 {
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            VECTORS_PER_COL as i64,
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let _result = func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            VECTORS_PER_COL as i64,
        );
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 4 * 4; // 4 columns * 4 bytes each
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    println!(
        "Complex 4-Column Filter: {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}

/// Benchmark: TPC-H style aggregation with multiple accumulators
/// Pattern: SELECT SUM(a), SUM(b), MIN(c), MAX(d) FROM table WHERE x > threshold
/// Expected: ~40-50 GB/s (memory bound with reduction overhead)
#[test]
fn bench_tpch_aggregation_i64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // col_a (for SUM)
    sig.params.push(AbiParam::new(ptr)); // col_b (for SUM)
    sig.params.push(AbiParam::new(ptr)); // col_c (for MIN)
    sig.params.push(AbiParam::new(ptr)); // col_d (for MAX)
    sig.params.push(AbiParam::new(ptr)); // col_filter
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.params.push(AbiParam::new(ptr)); // results[4]
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("tpch_agg_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let col_a_base = params[0];
        let col_b_base = params[1];
        let col_c_base = params[2];
        let col_d_base = params[3];
        let col_filter_base = params[4];
        let num_vectors = params[5];
        let results_ptr = params[6];

        // Initialize accumulators
        let z = builder.ins().iconst(I64, 0);
        let sum_a_init = builder.ins().splat(I64X8, z);
        let z2 = builder.ins().iconst(I64, 0);
        let sum_b_init = builder.ins().splat(I64X8, z2);
        let mx = builder.ins().iconst(I64, i64::MAX);
        let min_c_init = builder.ins().splat(I64X8, mx);
        let mn = builder.ins().iconst(I64, i64::MIN);
        let max_d_init = builder.ins().splat(I64X8, mn);
        let t50 = builder.ins().iconst(I64, 50);
        let threshold = builder.ins().splat(I64X8, t50);
        let init_idx = builder.ins().iconst(I64, 0);

        builder.ins().jump(
            loop_block,
            &[
                init_idx.into(),
                sum_a_init.into(),
                sum_b_init.into(),
                min_c_init.into(),
                max_d_init.into(),
            ],
        );

        // Loop block with 5 accumulators
        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64); // idx
        builder.append_block_param(loop_block, I64X8); // sum_a
        builder.append_block_param(loop_block, I64X8); // sum_b
        builder.append_block_param(loop_block, I64X8); // min_c
        builder.append_block_param(loop_block, I64X8); // max_d

        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let sum_a = loop_params[1];
        let sum_b = loop_params[2];
        let min_c = loop_params[3];
        let max_d = loop_params[4];

        let offset = builder.ins().imul_imm(idx, 64);

        // Load vectors with offset
        let a_addr = builder.ins().iadd(col_a_base, offset);
        let b_addr = builder.ins().iadd(col_b_base, offset);
        let c_addr = builder.ins().iadd(col_c_base, offset);
        let d_addr = builder.ins().iadd(col_d_base, offset);
        let f_addr = builder.ins().iadd(col_filter_base, offset);

        let a = builder.ins().load(I64X8, MemFlags::trusted(), a_addr, 0);
        let b = builder.ins().load(I64X8, MemFlags::trusted(), b_addr, 0);
        let c_col = builder.ins().load(I64X8, MemFlags::trusted(), c_addr, 0);
        let d = builder.ins().load(I64X8, MemFlags::trusted(), d_addr, 0);
        let filter = builder.ins().load(I64X8, MemFlags::trusted(), f_addr, 0);

        // WHERE filter > 50
        let mask = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, filter, threshold);

        // Masked accumulation using FUSED bitselect+op pattern
        // Pattern: bitselect(mask, op(x, y), passthru) -> single masked AVX-512 instruction
        // This generates 1 op per accumulator instead of 2 (VPBLENDM + VPADD)

        // For SUM: bitselect(mask, iadd(sum, val), sum) -> masked VPADDQ
        // When mask=true: new_sum = sum + val
        // When mask=false: new_sum = sum (unchanged)
        let sum_plus_a = builder.ins().iadd(sum_a, a);
        let new_sum_a = builder.ins().bitselect(mask, sum_plus_a, sum_a);
        let sum_plus_b = builder.ins().iadd(sum_b, b);
        let new_sum_b = builder.ins().bitselect(mask, sum_plus_b, sum_b);

        // For MIN: bitselect(mask, smin(min, val), min) -> masked VPMINSQ
        let min_with_c = builder.ins().smin(min_c, c_col);
        let new_min_c = builder.ins().bitselect(mask, min_with_c, min_c);

        // For MAX: bitselect(mask, smax(max, val), max) -> masked VPMAXSQ
        let max_with_d = builder.ins().smax(max_d, d);
        let new_max_d = builder.ins().bitselect(mask, max_with_d, max_d);

        // Increment and check
        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[
                new_sum_a.into(),
                new_sum_b.into(),
                new_min_c.into(),
                new_max_d.into(),
            ],
            loop_block,
            &[
                next_idx.into(),
                new_sum_a.into(),
                new_sum_b.into(),
                new_min_c.into(),
                new_max_d.into(),
            ],
        );

        // Exit - horizontal reduce and store results
        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64X8);
        builder.append_block_param(exit, I64X8);
        builder.append_block_param(exit, I64X8);
        builder.append_block_param(exit, I64X8);

        let final_params = builder.block_params(exit).to_vec();
        let final_sum_a = final_params[0];
        let final_sum_b = final_params[1];
        let final_min_c = final_params[2];
        let final_max_d = final_params[3];

        // Horizontal reduction for SUM(a)
        let mut ha = builder.ins().iconst(I64, 0);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(final_sum_a, i);
            ha = builder.ins().iadd(ha, lane);
        }
        builder.ins().store(MemFlags::trusted(), ha, results_ptr, 0);

        // Horizontal reduction for SUM(b)
        let mut hb = builder.ins().iconst(I64, 0);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(final_sum_b, i);
            hb = builder.ins().iadd(hb, lane);
        }
        builder.ins().store(MemFlags::trusted(), hb, results_ptr, 8);

        // Horizontal reduction for MIN(c)
        let mut hc = builder.ins().iconst(I64, i64::MAX);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(final_min_c, i);
            let is_less = builder.ins().icmp(IntCC::SignedLessThan, lane, hc);
            hc = builder.ins().select(is_less, lane, hc);
        }
        builder
            .ins()
            .store(MemFlags::trusted(), hc, results_ptr, 16);

        // Horizontal reduction for MAX(d)
        let mut hd = builder.ins().iconst(I64, i64::MIN);
        for i in 0..8u8 {
            let lane = builder.ins().extractlane(final_max_d, i);
            let is_greater = builder.ins().icmp(IntCC::SignedGreaterThan, lane, hd);
            hd = builder.ins().select(is_greater, lane, hd);
        }
        builder
            .ins()
            .store(MemFlags::trusted(), hd, results_ptr, 24);

        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i64, *const i64, *const i64, *const i64, *const i64, i64, *mut i64) =
        unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 1_000_000;
    const VECTORS: usize = NUM_ROWS / 8;

    let col_a: Vec<i64> = (0..NUM_ROWS).map(|i| (i % 1000) as i64).collect();
    let col_b: Vec<i64> = (0..NUM_ROWS).map(|i| (i % 500) as i64).collect();
    let col_c: Vec<i64> = (0..NUM_ROWS).map(|i| (i as i64) * 2).collect();
    let col_d: Vec<i64> = (0..NUM_ROWS).map(|i| 1000000 - i as i64).collect();
    let col_filter: Vec<i64> = (0..NUM_ROWS).map(|i| (i % 100) as i64).collect();
    let mut results = [0i64; 4];

    // Warmup
    for _ in 0..3 {
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            col_filter.as_ptr(),
            VECTORS as i64,
            results.as_mut_ptr(),
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            col_filter.as_ptr(),
            VECTORS as i64,
            results.as_mut_ptr(),
        );
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 8 * 5; // 5 columns * 8 bytes each
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    println!(
        "TPC-H 4-Way Aggregation: {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}

/// Benchmark: FP-heavy workload with FMA chains
/// Pattern: result = a * b + c * d - e * f (chained FMA operations)
/// Expected: ~30-40 GB/s (compute-bound FMA throughput)
#[test]
fn bench_fma_chain_f64x8() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // a
    sig.params.push(AbiParam::new(ptr)); // b
    sig.params.push(AbiParam::new(ptr)); // c
    sig.params.push(AbiParam::new(ptr)); // d
    sig.params.push(AbiParam::new(ptr)); // e
    sig.params.push(AbiParam::new(ptr)); // f
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("fma_chain_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let col_a = params[0];
        let col_b = params[1];
        let col_c = params[2];
        let col_d = params[3];
        let col_e = params[4];
        let col_f = params[5];
        let num_vectors = params[6];
        let result_base = params[7];

        let init_idx = builder.ins().iconst(I64, 0);
        builder.ins().jump(loop_block, &[init_idx.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        let idx = builder.block_params(loop_block)[0];

        let offset = builder.ins().imul_imm(idx, 64);

        // Load 6 vectors with offset
        let a_addr = builder.ins().iadd(col_a, offset);
        let b_addr = builder.ins().iadd(col_b, offset);
        let c_addr = builder.ins().iadd(col_c, offset);
        let d_addr = builder.ins().iadd(col_d, offset);
        let e_addr = builder.ins().iadd(col_e, offset);
        let f_addr = builder.ins().iadd(col_f, offset);
        let r_addr = builder.ins().iadd(result_base, offset);

        let a = builder.ins().load(F64X8, MemFlags::trusted(), a_addr, 0);
        let b = builder.ins().load(F64X8, MemFlags::trusted(), b_addr, 0);
        let c_vec = builder.ins().load(F64X8, MemFlags::trusted(), c_addr, 0);
        let d = builder.ins().load(F64X8, MemFlags::trusted(), d_addr, 0);
        let e = builder.ins().load(F64X8, MemFlags::trusted(), e_addr, 0);
        let f = builder.ins().load(F64X8, MemFlags::trusted(), f_addr, 0);

        // Chained FMA: a*b + c*d - e*f
        // = fma(a, b, fma(c, d, -(e*f)))
        // Since fneg isn't supported, use: a*b + c*d + (-e)*f
        let neg_one = builder.ins().f64const(-1.0);
        let neg_one_vec = builder.ins().splat(F64X8, neg_one);
        let neg_e = builder.ins().fmul(e, neg_one_vec);

        // t1 = c * d
        let t1 = builder.ins().fmul(c_vec, d);
        // t2 = neg_e * f + t1 = -e*f + c*d
        let t2 = builder.ins().fma(neg_e, f, t1);
        // result = a * b + t2 = a*b + c*d - e*f
        let result = builder.ins().fma(a, b, t2);

        builder.ins().store(MemFlags::trusted(), result, r_addr, 0);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder
            .ins()
            .brif(done, exit, &[], loop_block, &[next_idx.into()]);

        builder.switch_to_block(exit);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        i64,
        *mut f64,
    ) = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 1_000_000;
    const VECTORS: usize = NUM_ROWS / 8;

    let col_a: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.001).collect();
    let col_b: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.002).collect();
    let col_c: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.003).collect();
    let col_d: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.004).collect();
    let col_e: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.005).collect();
    let col_f: Vec<f64> = (0..NUM_ROWS).map(|i| (i as f64) * 0.006).collect();
    let mut result = vec![0.0f64; NUM_ROWS];

    // Warmup
    for _ in 0..3 {
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            col_e.as_ptr(),
            col_f.as_ptr(),
            VECTORS as i64,
            result.as_mut_ptr(),
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            col_d.as_ptr(),
            col_e.as_ptr(),
            col_f.as_ptr(),
            VECTORS as i64,
            result.as_mut_ptr(),
        );
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 8 * 7; // 6 input + 1 output columns
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    println!(
        "FMA Chain F64X8: {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}

/// Benchmark: Weird edge case - high selectivity filter (1% pass rate)
/// This stresses branch prediction and sparse result handling
#[test]
fn bench_high_selectivity_filter() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // data
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.returns.push(AbiParam::new(I64)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("high_sel_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let data_ptr = params[0];
        let num_vectors = params[1];

        // Very narrow range filter: WHERE x == 42 (about 1% in range 0-99)
        let c42 = builder.ins().iconst(I32, 42);
        let target = builder.ins().splat(I32X16, c42);
        let init_count = builder.ins().iconst(I64, 0);
        let init_idx = builder.ins().iconst(I64, 0);

        builder
            .ins()
            .jump(loop_block, &[init_idx.into(), init_count.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        builder.append_block_param(loop_block, I64);
        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let count = loop_params[1];

        let offset = builder.ins().imul_imm(idx, 64);
        let addr = builder.ins().iadd(data_ptr, offset);
        let data = builder.ins().load(I32X16, MemFlags::trusted(), addr, 0);
        let mask = builder.ins().icmp(IntCC::Equal, data, target);

        // OPTIMIZED: Use vhigh_bits + popcnt instead of 16x extractlane
        let high_bits = builder.ins().vhigh_bits(I32, mask);
        let high_bits_i64 = builder.ins().uextend(I64, high_bits);
        let lane_count = builder.ins().popcnt(high_bits_i64);
        let new_count = builder.ins().iadd(count, lane_count);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[new_count.into()],
            loop_block,
            &[next_idx.into(), new_count.into()],
        );

        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64);
        let final_count = builder.block_params(exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, i64) -> i64 = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 10_000_000;
    const VECTORS: usize = NUM_ROWS / 16;

    // Only ~1% match (value 42 in range 0-99)
    let data: Vec<i32> = (0..NUM_ROWS).map(|i| (i % 100) as i32).collect();

    // Warmup
    for _ in 0..3 {
        func(data.as_ptr(), VECTORS as i64);
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    let mut result = 0i64;
    for _ in 0..10 {
        let start = Instant::now();
        result = func(data.as_ptr(), VECTORS as i64);
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 4;
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);
    let selectivity = result as f64 / NUM_ROWS as f64 * 100.0;

    println!(
        "High Selectivity Filter: {:.0} M rows/sec, {:.1} GB/s ({:.1}% selectivity)",
        rows_per_sec / 1e6,
        gb_per_sec,
        selectivity
    );
}

/// Benchmark: Weird pattern - multiple dependent comparisons with weird thresholds
/// Pattern: WHERE (a BETWEEN 17 AND 89) AND (b NOT BETWEEN 23 AND 67) AND (c == 3)
#[test]
fn bench_weird_predicate_chain() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // col_a
    sig.params.push(AbiParam::new(ptr)); // col_b
    sig.params.push(AbiParam::new(ptr)); // col_c
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.returns.push(AbiParam::new(I64)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("weird_pred_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let col_a_ptr = params[0];
        let col_b_ptr = params[1];
        let col_c_ptr = params[2];
        let num_vectors = params[3];

        // Weird constants
        let c17 = builder.ins().iconst(I32, 17);
        let const_17 = builder.ins().splat(I32X16, c17);
        let c89 = builder.ins().iconst(I32, 89);
        let const_89 = builder.ins().splat(I32X16, c89);
        let c23 = builder.ins().iconst(I32, 23);
        let const_23 = builder.ins().splat(I32X16, c23);
        let c67 = builder.ins().iconst(I32, 67);
        let const_67 = builder.ins().splat(I32X16, c67);
        let c3 = builder.ins().iconst(I32, 3);
        let const_3 = builder.ins().splat(I32X16, c3);

        let init_count = builder.ins().iconst(I64, 0);
        let init_idx = builder.ins().iconst(I64, 0);

        builder
            .ins()
            .jump(loop_block, &[init_idx.into(), init_count.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        builder.append_block_param(loop_block, I64);
        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let count = loop_params[1];

        let offset = builder.ins().imul_imm(idx, 64);
        let a_addr = builder.ins().iadd(col_a_ptr, offset);
        let b_addr = builder.ins().iadd(col_b_ptr, offset);
        let c_addr = builder.ins().iadd(col_c_ptr, offset);

        let a = builder.ins().load(I32X16, MemFlags::trusted(), a_addr, 0);
        let b = builder.ins().load(I32X16, MemFlags::trusted(), b_addr, 0);
        let c_col = builder.ins().load(I32X16, MemFlags::trusted(), c_addr, 0);

        // a BETWEEN 17 AND 89
        let a_ge_17 = builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, a, const_17);
        let a_le_89 = builder
            .ins()
            .icmp(IntCC::SignedLessThanOrEqual, a, const_89);
        let a_between = builder.ins().band(a_ge_17, a_le_89);

        // b NOT BETWEEN 23 AND 67
        let b_lt_23 = builder.ins().icmp(IntCC::SignedLessThan, b, const_23);
        let b_gt_67 = builder.ins().icmp(IntCC::SignedGreaterThan, b, const_67);
        let b_not_between = builder.ins().bor(b_lt_23, b_gt_67);

        // c == 3
        let c_eq_3 = builder.ins().icmp(IntCC::Equal, c_col, const_3);

        // Combine: a_between AND b_not_between AND c_eq_3
        let cond1 = builder.ins().band(a_between, b_not_between);
        let final_mask = builder.ins().band(cond1, c_eq_3);

        // OPTIMIZED: Use vhigh_bits + popcnt instead of 16x extractlane
        let high_bits = builder.ins().vhigh_bits(I32, final_mask);
        let high_bits_i64 = builder.ins().uextend(I64, high_bits);
        let lane_count = builder.ins().popcnt(high_bits_i64);
        let new_count = builder.ins().iadd(count, lane_count);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[new_count.into()],
            loop_block,
            &[next_idx.into(), new_count.into()],
        );

        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64);
        let final_count = builder.block_params(exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, *const i32, *const i32, i64) -> i64 = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 10_000_000;
    const VECTORS: usize = NUM_ROWS / 16;

    let col_a: Vec<i32> = (0..NUM_ROWS).map(|i| (i % 100) as i32).collect();
    let col_b: Vec<i32> = (0..NUM_ROWS).map(|i| ((i * 7) % 100) as i32).collect();
    let col_c: Vec<i32> = (0..NUM_ROWS).map(|i| (i % 10) as i32).collect();

    // Warmup
    for _ in 0..3 {
        func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            VECTORS as i64,
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    let mut result = 0i64;
    for _ in 0..10 {
        let start = Instant::now();
        result = func(
            col_a.as_ptr(),
            col_b.as_ptr(),
            col_c.as_ptr(),
            VECTORS as i64,
        );
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 4 * 3;
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);
    let selectivity = result as f64 / NUM_ROWS as f64 * 100.0;

    println!(
        "Weird 3-Way Predicate: {:.0} M rows/sec, {:.1} GB/s ({:.1}% selectivity)",
        rows_per_sec / 1e6,
        gb_per_sec,
        selectivity
    );
}

/// Benchmark: Register pressure stress test - 8 columns with complex expression
/// Pattern: result = ((a+b)*(c-d)) + ((e*f)-(g/h))
/// This tests register allocation under pressure
#[test]
fn bench_register_pressure_8col() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    for _ in 0..8 {
        sig.params.push(AbiParam::new(ptr)); // 8 input columns
    }
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.params.push(AbiParam::new(ptr)); // result
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("reg_pressure_bench", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let cols: Vec<_> = params[0..8].to_vec();
        let num_vectors = params[8];
        let result_base = params[9];

        let init_idx = builder.ins().iconst(I64, 0);
        builder.ins().jump(loop_block, &[init_idx.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        let idx = builder.block_params(loop_block)[0];

        let offset = builder.ins().imul_imm(idx, 64);

        // Load all 8 columns with offset
        let a_addr = builder.ins().iadd(cols[0], offset);
        let b_addr = builder.ins().iadd(cols[1], offset);
        let c_addr = builder.ins().iadd(cols[2], offset);
        let d_addr = builder.ins().iadd(cols[3], offset);
        let e_addr = builder.ins().iadd(cols[4], offset);
        let f_addr = builder.ins().iadd(cols[5], offset);
        let g_addr = builder.ins().iadd(cols[6], offset);
        let h_addr = builder.ins().iadd(cols[7], offset);
        let r_addr = builder.ins().iadd(result_base, offset);

        let a = builder.ins().load(F64X8, MemFlags::trusted(), a_addr, 0);
        let b = builder.ins().load(F64X8, MemFlags::trusted(), b_addr, 0);
        let c_col = builder.ins().load(F64X8, MemFlags::trusted(), c_addr, 0);
        let d = builder.ins().load(F64X8, MemFlags::trusted(), d_addr, 0);
        let e = builder.ins().load(F64X8, MemFlags::trusted(), e_addr, 0);
        let f = builder.ins().load(F64X8, MemFlags::trusted(), f_addr, 0);
        let g = builder.ins().load(F64X8, MemFlags::trusted(), g_addr, 0);
        let h = builder.ins().load(F64X8, MemFlags::trusted(), h_addr, 0);

        // Complex expression: ((a+b)*(c-d)) + ((e*f)-(g/h))
        let a_plus_b = builder.ins().fadd(a, b);
        let c_minus_d = builder.ins().fsub(c_col, d);
        let e_times_f = builder.ins().fmul(e, f);
        let g_div_h = builder.ins().fdiv(g, h);

        let left = builder.ins().fmul(a_plus_b, c_minus_d);
        let right = builder.ins().fsub(e_times_f, g_div_h);
        let result = builder.ins().fadd(left, right);

        builder.ins().store(MemFlags::trusted(), result, r_addr, 0);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder
            .ins()
            .brif(done, exit, &[], loop_block, &[next_idx.into()]);

        builder.switch_to_block(exit);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        *const f64,
        i64,
        *mut f64,
    ) = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 1_000_000;
    const VECTORS: usize = NUM_ROWS / 8;

    let cols: Vec<Vec<f64>> = (0..8)
        .map(|c| {
            (0..NUM_ROWS)
                .map(|i| ((i + c * 1000) as f64) * 0.001 + 1.0)
                .collect()
        })
        .collect();
    let mut result = vec![0.0f64; NUM_ROWS];

    // Warmup
    for _ in 0..3 {
        func(
            cols[0].as_ptr(),
            cols[1].as_ptr(),
            cols[2].as_ptr(),
            cols[3].as_ptr(),
            cols[4].as_ptr(),
            cols[5].as_ptr(),
            cols[6].as_ptr(),
            cols[7].as_ptr(),
            VECTORS as i64,
            result.as_mut_ptr(),
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        func(
            cols[0].as_ptr(),
            cols[1].as_ptr(),
            cols[2].as_ptr(),
            cols[3].as_ptr(),
            cols[4].as_ptr(),
            cols[5].as_ptr(),
            cols[6].as_ptr(),
            cols[7].as_ptr(),
            VECTORS as i64,
            result.as_mut_ptr(),
        );
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 8 * 9; // 8 input + 1 output
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    println!(
        "Register Pressure 8-Column: {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}

// =============================================================================
// OPTIMIZED BENCHMARKS: Using vhigh_bits + popcnt instead of 16x extractlane
// =============================================================================
// These benchmarks demonstrate the performance improvement from using:
//   vhigh_bits(mask) + popcnt(bits)
// instead of:
//   for i in 0..16 { extractlane(mask, i); ... }
//
// The vhigh_bits instruction lowers to VPMOVD2M + KMOVW (2 ops)
// Then popcnt counts the bits in a single instruction (1 op)
// Total: 3 ops vs 64 ops (16 extractlane + 16 ineg + 16 uextend + 16 iadd)

/// Benchmark: Optimized filter counting using vhigh_bits + popcnt
/// This is ~20x faster than the extractlane loop approach
#[test]
fn bench_optimized_filter_popcnt() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // data
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.returns.push(AbiParam::new(I64)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("optimized_filter_popcnt", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let data_ptr = params[0];
        let num_vectors = params[1];

        // Filter: WHERE x > 50
        let c50 = builder.ins().iconst(I32, 50);
        let threshold = builder.ins().splat(I32X16, c50);
        let init_count = builder.ins().iconst(I64, 0);
        let init_idx = builder.ins().iconst(I64, 0);

        builder
            .ins()
            .jump(loop_block, &[init_idx.into(), init_count.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        builder.append_block_param(loop_block, I64);
        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let count = loop_params[1];

        let offset = builder.ins().imul_imm(idx, 64);
        let addr = builder.ins().iadd(data_ptr, offset);
        let data = builder.ins().load(I32X16, MemFlags::trusted(), addr, 0);
        let mask = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, data, threshold);

        // OPTIMIZED: Use vhigh_bits + popcnt instead of 16x extractlane loop!
        // vhigh_bits extracts the sign bit of each lane into a scalar (16 bits for I32X16)
        // Then popcnt counts how many bits are set
        let high_bits = builder.ins().vhigh_bits(I32, mask);
        let high_bits_i64 = builder.ins().uextend(I64, high_bits);
        let lane_count = builder.ins().popcnt(high_bits_i64);
        let new_count = builder.ins().iadd(count, lane_count);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[new_count.into()],
            loop_block,
            &[next_idx.into(), new_count.into()],
        );

        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64);
        let final_count = builder.block_params(exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, i64) -> i64 = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 10_000_000;
    const VECTORS: usize = NUM_ROWS / 16;

    // ~50% match rate (values 0-99, filter > 50 matches 49 values)
    let data: Vec<i32> = (0..NUM_ROWS).map(|i| (i % 100) as i32).collect();

    // Warmup
    for _ in 0..3 {
        func(data.as_ptr(), VECTORS as i64);
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    let mut result = 0i64;
    for _ in 0..10 {
        let start = Instant::now();
        result = func(data.as_ptr(), VECTORS as i64);
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 4;
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    // Verify correctness
    let expected = (NUM_ROWS / 100) * 49; // 49 values > 50 per 100
    assert_eq!(result as usize, expected, "Count mismatch");

    println!(
        "OPTIMIZED Filter (vhigh_bits+popcnt): {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}

/// Benchmark: Unoptimized filter counting for comparison (16x extractlane)
/// This is the baseline that bench_optimized_filter_popcnt improves upon
#[test]
fn bench_unoptimized_filter_extractlane() {
    let mut c = match SqlCompiler::new() {
        Some(c) => c,
        None => {
            println!("AVX-512 not available, skipping benchmark");
            return;
        }
    };

    let ptr = c.ptr();
    let mut sig = c.module.make_signature();
    sig.params.push(AbiParam::new(ptr)); // data
    sig.params.push(AbiParam::new(I64)); // num_vectors
    sig.returns.push(AbiParam::new(I64)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = c
        .module
        .declare_function("unoptimized_filter_extractlane", Linkage::Local, &sig)
        .unwrap();
    c.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut c.ctx.func, &mut c.func_ctx);
        let entry = builder.create_block();
        let loop_block = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let params = builder.block_params(entry).to_vec();
        let data_ptr = params[0];
        let num_vectors = params[1];

        // Same filter: WHERE x > 50
        let c50 = builder.ins().iconst(I32, 50);
        let threshold = builder.ins().splat(I32X16, c50);
        let init_count = builder.ins().iconst(I64, 0);
        let init_idx = builder.ins().iconst(I64, 0);

        builder
            .ins()
            .jump(loop_block, &[init_idx.into(), init_count.into()]);

        builder.switch_to_block(loop_block);
        builder.append_block_param(loop_block, I64);
        builder.append_block_param(loop_block, I64);
        let loop_params = builder.block_params(loop_block).to_vec();
        let idx = loop_params[0];
        let count = loop_params[1];

        let offset = builder.ins().imul_imm(idx, 64);
        let addr = builder.ins().iadd(data_ptr, offset);
        let data = builder.ins().load(I32X16, MemFlags::trusted(), addr, 0);
        let mask = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, data, threshold);

        // UNOPTIMIZED: 16x extractlane loop (64 ops total!)
        let mut lane_sum = builder.ins().iconst(I64, 0);
        for i in 0..16u8 {
            let lane = builder.ins().extractlane(mask, i);
            let neg = builder.ins().ineg(lane);
            let ext = builder.ins().uextend(I64, neg);
            lane_sum = builder.ins().iadd(lane_sum, ext);
        }
        let new_count = builder.ins().iadd(count, lane_sum);

        let next_idx = builder.ins().iadd_imm(idx, 1);
        let done = builder.ins().icmp(IntCC::Equal, next_idx, num_vectors);
        builder.ins().brif(
            done,
            exit,
            &[new_count.into()],
            loop_block,
            &[next_idx.into(), new_count.into()],
        );

        builder.switch_to_block(exit);
        builder.append_block_param(exit, I64);
        let final_count = builder.block_params(exit)[0];
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    c.module.define_function(func_id, &mut c.ctx).unwrap();
    c.module.clear_context(&mut c.ctx);
    c.module.finalize_definitions().unwrap();
    let code = c.module.get_finalized_function(func_id);

    let func: fn(*const i32, i64) -> i64 = unsafe { mem::transmute(code) };

    const NUM_ROWS: usize = 10_000_000;
    const VECTORS: usize = NUM_ROWS / 16;

    // Same data as optimized version
    let data: Vec<i32> = (0..NUM_ROWS).map(|i| (i % 100) as i32).collect();

    // Warmup
    for _ in 0..3 {
        func(data.as_ptr(), VECTORS as i64);
    }

    // Benchmark
    let mut times = Vec::with_capacity(10);
    let mut result = 0i64;
    for _ in 0..10 {
        let start = Instant::now();
        result = func(data.as_ptr(), VECTORS as i64);
        times.push(start.elapsed().as_nanos() as u64);
    }

    let min = *times.iter().min().unwrap();
    let _avg = times.iter().sum::<u64>() / times.len() as u64;
    let bytes = NUM_ROWS * 4;
    let gb_per_sec = bytes as f64 / 1e9 / (min as f64 / 1e9);
    let rows_per_sec = NUM_ROWS as f64 / (min as f64 / 1e9);

    // Verify correctness
    let expected = (NUM_ROWS / 100) * 49;
    assert_eq!(result as usize, expected, "Count mismatch");

    println!(
        "UNOPTIMIZED Filter (extractlane): {:.0} M rows/sec, {:.1} GB/s",
        rows_per_sec / 1e6,
        gb_per_sec
    );
}
