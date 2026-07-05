#![cfg(target_arch = "x86_64")]

//! Regression tests for the 512-bit `bitselect` lane-mask strictness fix.
//!
//! CLIF `bitselect` is BITWISE: `(c & x) | (!c & y)`. The AVX-512
//! VPBLENDMD/Q and masked-fusion lowerings sample only each lane's sign bit
//! (VPMOVD2M/Q), so they are only legal when the condition is a whole-LANE
//! mask at the bitselect type's granularity. The old gate
//! (`all_ones_or_all_zeros`) proved only BYTE granularity, so:
//!
//! - a `vconst` condition whose bytes are all 0x00/0xFF but whose lanes mix
//!   them (e.g. i32 lanes of 0x00FF00FF) was whole-lane blended, and
//! - `bitcast.i64x8(fcmp on f32x16)` (32-bit lane mask reinterpreted at
//!   64-bit lanes) was whole-lane blended,
//!
//! both producing whole-lane results instead of the required bitwise mix.
//! These tests execute those exact adversarial shapes and compare against a
//! Rust bitwise reference, plus positive cases proving that genuine lane
//! masks (icmp/fcmp results, lane-uniform vconsts, same-lane-width bitcasts
//! of comparisons) still compute correct results on the blend/fusion paths.

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

/// Check if AVX-512 is available on this machine (same guard as
/// avx512_e2e.rs).
fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

fn isa_with_avx512() -> Option<OwnedTargetIsa> {
    if !has_avx512() {
        return None;
    }
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().ok()?;
    isa_builder.finish(settings::Flags::new(flag_builder)).ok()
}

struct TestCompiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl TestCompiler {
    fn new() -> Option<Self> {
        let isa = isa_with_avx512()?;
        let module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();
        Some(Self {
            module,
            ctx,
            func_ctx,
        })
    }

    /// Compile a `fn(*ptr, *ptr, ...)` style function with `nparams` pointer
    /// arguments and no return value. The closure receives the builder and
    /// the pointer parameter values and must emit all loads/stores.
    fn compile_ptr_fn<F>(
        &mut self,
        name: &str,
        nparams: usize,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, &[Value]),
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        for _ in 0..nparams {
            sig.params.push(AbiParam::new(ptr_type));
        }
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);
            let params = builder.block_params(block).to_vec();
            build_fn(&mut builder, &params);
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

macro_rules! require_avx512 {
    () => {
        match TestCompiler::new() {
            Some(c) => c,
            None => {
                println!("Skipping test: AVX-512 not available");
                return;
            }
        }
    };
}

#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Debug)]
struct V512<T: Copy + PartialEq + std::fmt::Debug, const N: usize>([T; N]);

type I32x16 = V512<i32, 16>;
type I64x8 = V512<i64, 8>;
type F32x16 = V512<f32, 16>;

/// Emit `vconst.i32x16` with the given per-lane bit patterns.
fn vconst_i32x16(b: &mut FunctionBuilder, lanes: [u32; 16]) -> Value {
    let mut bytes = [0u8; 64];
    for (i, l) in lanes.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&l.to_le_bytes());
    }
    let handle = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
    b.ins().vconst(I32X16, handle)
}

/// Emit `vconst.i64x8` with the given per-lane bit patterns.
fn vconst_i64x8(b: &mut FunctionBuilder, lanes: [u64; 8]) -> Value {
    let mut bytes = [0u8; 64];
    for (i, l) in lanes.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&l.to_le_bytes());
    }
    let handle = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
    b.ins().vconst(I64X8, handle)
}

fn bitwise_ref_i32(c: u32, x: i32, y: i32) -> i32 {
    ((c & x as u32) | (!c & y as u32)) as i32
}

fn bitwise_ref_i64(c: u64, x: i64, y: i64) -> i64 {
    ((c & x as u64) | (!c & y as u64)) as i64
}

fn test_x_i32() -> I32x16 {
    V512(std::array::from_fn(|i| {
        0x1111_1111u32.wrapping_mul(i as u32 + 1) as i32
    }))
}

fn test_y_i32() -> I32x16 {
    V512(std::array::from_fn(|i| {
        0xF0E0_D0C0u32.wrapping_add(i as u32 * 0x0101_0101) as i32
    }))
}

// =============================================================================
// Adversarial: vconst conditions with 0x00/0xFF BYTES but mixed lanes
// =============================================================================

/// Compile `bitselect.i32x16` with a baked-in vconst condition and check the
/// result lane-by-lane against the bitwise reference.
fn run_i32x16_vconst_cond(name: &str, lanes: [u32; 16]) {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn(name, 3, |b, p| {
            let cond = vconst_i32x16(b, lanes);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[2], 0);
        })
        .unwrap_or_else(|e| panic!("compile {name}: {e}"));
    let func: extern "C" fn(*const I32x16, *const I32x16, *mut I32x16) =
        unsafe { mem::transmute(code) };

    let x = test_x_i32();
    let y = test_y_i32();
    let mut result = V512([0i32; 16]);
    func(&x, &y, &mut result);
    for i in 0..16 {
        let expect = bitwise_ref_i32(lanes[i], x.0[i], y.0[i]);
        assert_eq!(
            result.0[i], expect,
            "{name} lane {i} (cond {:#010x}): got {:#010x} want {expect:#010x}",
            lanes[i], result.0[i]
        );
    }
}

#[test]
fn test_bitselect_i32x16_vconst_bytemask_00ff00ff() {
    // Bytes all 0x00/0xFF, but each 32-bit lane mixes them: NOT a lane mask.
    // A sign-bit-sampling blend would select whole lane `y` (sign bit 0);
    // the bitwise reference mixes bytes of x and y.
    run_i32x16_vconst_cond("bitselect_i32x16_bytemask_00ff00ff", [0x00FF_00FF; 16]);
}

#[test]
fn test_bitselect_i32x16_vconst_bytemask_ff0000ff() {
    // Sign bit SET but low lane bytes mixed: a whole-lane blend would take
    // all of x, the bitwise reference takes only the top and bottom bytes.
    run_i32x16_vconst_cond("bitselect_i32x16_bytemask_ff0000ff", [0xFF00_00FF; 16]);
}

#[test]
fn test_bitselect_i32x16_vconst_bytemask_varying() {
    // Per-lane variety, including genuine lane masks mixed in.
    run_i32x16_vconst_cond(
        "bitselect_i32x16_bytemask_varying",
        [
            0x00FF_00FF,
            0xFF00_00FF,
            0xFFFF_FFFF,
            0x0000_0000,
            0xFF00_FF00,
            0x00FF_FF00,
            0xFFFF_0000,
            0x0000_FFFF,
            0xFF00_0000,
            0x0000_00FF,
            0xFFFF_FF00,
            0x00FF_FFFF,
            0xFF00_FFFF,
            0xFFFF_00FF,
            0x00FF_0000,
            0x0000_FF00,
        ],
    );
}

#[test]
fn test_bitselect_i64x8_vconst_bytemask() {
    // 64-bit lanes of 0x00000000FFFFFFFF: uniform at 32-bit granularity but
    // not at the bitselect type's 64-bit granularity. VPMOVQ2M samples bit
    // 63 (= 0) so a whole-lane blend would take all of y; bitwise takes the
    // low half of x.
    let lanes: [u64; 8] = [
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x0000_0000_FFFF_FFFF,
        0x00FF_00FF_00FF_00FF,
        0xFF00_0000_0000_00FF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0000,
        0xFFFF_0000_0000_FFFF,
    ];
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i64x8_bytemask", 3, |b, p| {
            let cond = vconst_i64x8(b, lanes);
            let x = b.ins().load(I64X8, MemFlags::trusted(), p[0], 0);
            let y = b.ins().load(I64X8, MemFlags::trusted(), p[1], 0);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[2], 0);
        })
        .expect("compile bitselect_i64x8_bytemask");
    let func: extern "C" fn(*const I64x8, *const I64x8, *mut I64x8) =
        unsafe { mem::transmute(code) };

    let x = V512(std::array::from_fn::<i64, 8, _>(|i| {
        0x0123_4567_89AB_CDEFu64.wrapping_mul(i as u64 + 1) as i64
    }));
    let y = V512(std::array::from_fn::<i64, 8, _>(|i| {
        0xFEDC_BA98_7654_3210u64.wrapping_add(i as u64 * 0x1111) as i64
    }));
    let mut result = V512([0i64; 8]);
    func(&x, &y, &mut result);
    for i in 0..8 {
        let expect = bitwise_ref_i64(lanes[i], x.0[i], y.0[i]);
        assert_eq!(
            result.0[i], expect,
            "lane {i} (cond {:#018x}): got {:#018x} want {expect:#018x}",
            lanes[i], result.0[i]
        );
    }
}

// =============================================================================
// Adversarial: fusion-shaped bitselect(vconst-mask, iadd(x, y), passthru)
// =============================================================================

#[test]
fn test_fusion_shape_iadd_i32x16_bytemask() {
    // The masked-fusion rules (bitselect(mask, op(x, y), passthru) ->
    // merge-masked instruction) copy whole lanes too; a byte-granular vconst
    // mask must fall back to plain op + bitwise select.
    let lanes: [u32; 16] = [0x00FF_00FF; 16];
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("fusion_iadd_i32x16_bytemask", 4, |b, p| {
            let mask = vconst_i32x16(b, lanes);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let passthru = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let sum = b.ins().iadd(x, y);
            let r = b.ins().bitselect(mask, sum, passthru);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })
        .expect("compile fusion_iadd_i32x16_bytemask");
    let func: extern "C" fn(*const I32x16, *const I32x16, *const I32x16, *mut I32x16) =
        unsafe { mem::transmute(code) };

    let x = test_x_i32();
    let y = test_y_i32();
    let passthru = V512([0x5A5A_5A5Au32 as i32; 16]);
    let mut result = V512([0i32; 16]);
    func(&x, &y, &passthru, &mut result);
    for i in 0..16 {
        let sum = x.0[i].wrapping_add(y.0[i]);
        let expect = bitwise_ref_i32(lanes[i], sum, passthru.0[i]);
        assert_eq!(
            result.0[i], expect,
            "lane {i}: got {:#010x} want {expect:#010x}",
            result.0[i]
        );
    }
}

// =============================================================================
// Adversarial: bitcast.i64x8 of an fcmp on f32x16 (lane-count-changing)
// =============================================================================

#[test]
fn test_bitselect_i64x8_bitcast_fcmp_cond() {
    // fcmp.f32x16 produces a 32-bit lane mask; reinterpreted as i64x8 it is
    // NOT a 64-bit lane mask whenever the two 32-bit halves of a 64-bit lane
    // compare differently. No constants involved: the condition is computed
    // at runtime, so this exercises the (bitcast (fcmp ...)) extractor arm
    // rejection specifically.
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i64x8_bitcast_fcmp", 5, |b, p| {
            let a = b.ins().load(F32X16, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(F32X16, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(I64X8, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(I64X8, MemFlags::trusted(), p[3], 0);
            let cmp = b.ins().fcmp(FloatCC::Equal, a, bb); // i32x16 mask
            // Lane-count-changing bitcasts require an explicit byte order.
            let mut little = MemFlags::new();
            little.set_endianness(Endianness::Little);
            let cond = b.ins().bitcast(I64X8, little, cmp);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[4], 0);
        })
        .expect("compile bitselect_i64x8_bitcast_fcmp");
    let func: extern "C" fn(
        *const F32x16,
        *const F32x16,
        *const I64x8,
        *const I64x8,
        *mut I64x8,
    ) = unsafe { mem::transmute(code) };

    // Lane pattern: even 32-bit lanes equal, odd ones differ (and vice
    // versa), so every 64-bit lane has exactly one half-mask set.
    let a = V512(std::array::from_fn::<f32, 16, _>(|i| i as f32));
    let b = V512(std::array::from_fn::<f32, 16, _>(|i| {
        if i % 2 == 0 { i as f32 } else { -1.0 }
    }));
    let x = V512(std::array::from_fn::<i64, 8, _>(|i| {
        0x1111_2222_3333_4444u64.wrapping_mul(i as u64 + 1) as i64
    }));
    let y = V512(std::array::from_fn::<i64, 8, _>(|i| {
        0xAAAA_BBBB_CCCC_DDDDu64.wrapping_add(i as u64) as i64
    }));
    let mut result = V512([0i64; 8]);
    func(&a, &b, &x, &y, &mut result);
    for i in 0..8 {
        // Recompute the i32x16 fcmp mask, then reinterpret as u64 (little
        // endian: lane 2*i is the LOW half).
        let lo = if a.0[2 * i] == b.0[2 * i] {
            0xFFFF_FFFFu64
        } else {
            0
        };
        let hi = if a.0[2 * i + 1] == b.0[2 * i + 1] {
            0xFFFF_FFFFu64
        } else {
            0
        };
        let cond = lo | (hi << 32);
        let expect = bitwise_ref_i64(cond, x.0[i], y.0[i]);
        assert_eq!(
            result.0[i], expect,
            "lane {i} (cond {cond:#018x}): got {:#018x} want {expect:#018x}",
            result.0[i]
        );
    }
}

// =============================================================================
// Positive cases: genuine lane masks still produce correct results
// (these take the VPBLENDM / masked-fusion paths; correctness is identical
// to the bitwise reference because the lanes are uniform)
// =============================================================================

#[test]
fn test_bitselect_i32x16_icmp_cond_still_correct() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i32x16_icmp_pos", 5, |b, p| {
            let a = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[3], 0);
            let cond = b.ins().icmp(IntCC::SignedGreaterThan, a, bb);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[4], 0);
        })
        .expect("compile bitselect_i32x16_icmp_pos");
    let func: extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    ) = unsafe { mem::transmute(code) };

    let a = V512(std::array::from_fn::<i32, 16, _>(|i| i as i32 - 8));
    let b = V512([0i32; 16]);
    let x = test_x_i32();
    let y = test_y_i32();
    let mut result = V512([0i32; 16]);
    func(&a, &b, &x, &y, &mut result);
    for i in 0..16 {
        let expect = if a.0[i] > 0 { x.0[i] } else { y.0[i] };
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}

#[test]
fn test_bitselect_i32x16_vconst_lanemask_still_correct() {
    // Lane-uniform vconst mask: still eligible for the blend path, and the
    // result must (trivially) equal the bitwise reference.
    let mut lanes = [0u32; 16];
    for i in (0..16).step_by(2) {
        lanes[i] = 0xFFFF_FFFF;
    }
    run_i32x16_vconst_cond("bitselect_i32x16_lanemask_pos", lanes);
}

#[test]
fn test_fusion_iadd_i32x16_icmp_mask_still_correct() {
    // bitselect(icmp-mask, iadd(x, y), passthru): the masked-VPADDD fusion
    // path must still produce merge-masked results.
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("fusion_iadd_i32x16_icmp_pos", 6, |b, p| {
            let a = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[3], 0);
            let passthru = b.ins().load(I32X16, MemFlags::trusted(), p[4], 0);
            let mask = b.ins().icmp(IntCC::Equal, a, bb);
            let sum = b.ins().iadd(x, y);
            let r = b.ins().bitselect(mask, sum, passthru);
            b.ins().store(MemFlags::trusted(), r, p[5], 0);
        })
        .expect("compile fusion_iadd_i32x16_icmp_pos");
    let func: extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    ) = unsafe { mem::transmute(code) };

    let a = V512(std::array::from_fn::<i32, 16, _>(|i| (i % 3) as i32));
    let b = V512([1i32; 16]);
    let x = test_x_i32();
    let y = test_y_i32();
    let passthru = V512([0x7BAD_F00Di32; 16]);
    let mut result = V512([0i32; 16]);
    func(&a, &b, &x, &y, &passthru, &mut result);
    for i in 0..16 {
        let expect = if a.0[i] == 1 {
            x.0[i].wrapping_add(y.0[i])
        } else {
            passthru.0[i]
        };
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}

#[test]
fn test_fusion_fadd_f32x16_bitcast_fcmp_mask_still_correct() {
    // The canonical float-mask shape: fcmp -> bitcast.f32x16 -> bitselect
    // over fadd. The bitcast keeps the 32-bit lane structure, so the
    // same-lane-count bitcast arm of the strict extractor accepts it and the
    // masked-VADDPS fusion stays available; either way the result must match
    // the per-lane select reference.
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("fusion_fadd_f32x16_bitcast_fcmp_pos", 6, |b, p| {
            let a = b.ins().load(F32X16, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(F32X16, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(F32X16, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(F32X16, MemFlags::trusted(), p[3], 0);
            let passthru = b.ins().load(F32X16, MemFlags::trusted(), p[4], 0);
            let cmp = b.ins().fcmp(FloatCC::GreaterThan, a, bb); // i32x16
            let mask = b.ins().bitcast(F32X16, MemFlags::new(), cmp);
            let sum = b.ins().fadd(x, y);
            let r = b.ins().bitselect(mask, sum, passthru);
            b.ins().store(MemFlags::trusted(), r, p[5], 0);
        })
        .expect("compile fusion_fadd_f32x16_bitcast_fcmp_pos");
    let func: extern "C" fn(
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    ) = unsafe { mem::transmute(code) };

    let a = V512(std::array::from_fn::<f32, 16, _>(|i| i as f32 - 7.5));
    let b = V512([0.0f32; 16]);
    let x = V512(std::array::from_fn::<f32, 16, _>(|i| i as f32 + 0.25));
    let y = V512(std::array::from_fn::<f32, 16, _>(|i| i as f32 * 2.0 + 0.5));
    let passthru = V512([-999.0f32; 16]);
    let mut result = V512([0.0f32; 16]);
    func(&a, &b, &x, &y, &passthru, &mut result);
    for i in 0..16 {
        let expect = if a.0[i] > 0.0 {
            x.0[i] + y.0[i]
        } else {
            passthru.0[i]
        };
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}
