#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for 512-bit bitwise/lane operations:
//!
//! - `bitselect` on I32X16/I64X8 with ARBITRARY bit-pattern conditions
//!   (partial-lane masks such as 0x0000FFFF, alternating bits). CLIF
//!   `bitselect` is bitwise, `(c & x) | (!c & y)`; these lock the
//!   VPTERNLOG(0xCA) lowering. Also with comparison-produced conditions,
//!   which take the VPBLENDMD/Q whole-lane blend path.
//! - `swizzle` on I32X16/I64X8: out-of-range indices must produce zero
//!   lanes (masked VPERMD/Q lowering), matching the semantics of the
//!   long-standing 128-bit i8x16 swizzle.
//! - `extractlane`/`insertlane` on F32X16/F64X8: every lane, with values
//!   distinguishable per lane, including NaN payloads which must be
//!   preserved bit-exactly (the float paths are pure XMM moves/shuffles).

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
type F64x8 = V512<f64, 8>;

// =============================================================================
// bitselect: arbitrary bit-pattern conditions (VPTERNLOG path)
// =============================================================================

#[test]
fn test_bitselect_i32x16_bitwise() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i32x16_bitwise", 4, |b, p| {
            let cond = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })
        .expect("compile bitselect_i32x16_bitwise");
    let func: extern "C" fn(*const I32x16, *const I32x16, *const I32x16, *mut I32x16) =
        unsafe { mem::transmute(code) };

    // Per-lane condition patterns, deliberately NOT all-ones/all-zeros:
    // partial-lane masks, alternating bits, single bits, etc.
    let conds: [u32; 7] = [
        0x0000_FFFF,
        0xFFFF_0000,
        0xFF00_FF00,
        0xAAAA_AAAA,
        0x5555_5555,
        0x8000_0001,
        0x1234_5678,
    ];
    for &cbits in &conds {
        let cond = V512([cbits as i32; 16]);
        let x = V512(std::array::from_fn::<i32, 16, _>(|i| {
            (0x1111_1111u32.wrapping_mul(i as u32 + 1)) as i32
        }));
        let y = V512(std::array::from_fn::<i32, 16, _>(|i| {
            (0xF0F0_F0F0u32.wrapping_add(i as u32 * 0x0101_0101)) as i32
        }));
        let mut result = V512([0i32; 16]);
        func(&cond, &x, &y, &mut result);
        for i in 0..16 {
            let expect = ((cbits & x.0[i] as u32) | (!cbits & y.0[i] as u32)) as i32;
            assert_eq!(
                result.0[i], expect,
                "cond={cbits:#010x} lane {i}: got {:#010x} want {expect:#010x}",
                result.0[i]
            );
        }
    }

    // A condition varying per lane too.
    let cond = V512(std::array::from_fn::<i32, 16, _>(|i| {
        (0x0000_FFFFu32.rotate_left(i as u32 * 2)) as i32
    }));
    let x = V512([0x7777_7777i32; 16]);
    let y = V512([-0x0F0F_0F10i32; 16]); // 0xF0F0F0F0 as i32
    let mut result = V512([0i32; 16]);
    func(&cond, &x, &y, &mut result);
    for i in 0..16 {
        let cb = cond.0[i] as u32;
        let expect = ((cb & x.0[i] as u32) | (!cb & y.0[i] as u32)) as i32;
        assert_eq!(result.0[i], expect, "varying cond lane {i}");
    }
}

#[test]
fn test_bitselect_i64x8_bitwise() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i64x8_bitwise", 4, |b, p| {
            let cond = b.ins().load(I64X8, MemFlags::trusted(), p[0], 0);
            let x = b.ins().load(I64X8, MemFlags::trusted(), p[1], 0);
            let y = b.ins().load(I64X8, MemFlags::trusted(), p[2], 0);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })
        .expect("compile bitselect_i64x8_bitwise");
    let func: extern "C" fn(*const I64x8, *const I64x8, *const I64x8, *mut I64x8) =
        unsafe { mem::transmute(code) };

    let conds: [u64; 6] = [
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x0000_FFFF_0000_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0x8000_0000_0000_0001,
    ];
    for &cbits in &conds {
        let cond = V512([cbits as i64; 8]);
        let x = V512(std::array::from_fn::<i64, 8, _>(|i| {
            0x0123_4567_89AB_CDEFu64.wrapping_mul(i as u64 + 1) as i64
        }));
        let y = V512(std::array::from_fn::<i64, 8, _>(|i| {
            0xFEDC_BA98_7654_3210u64.wrapping_add(i as u64) as i64
        }));
        let mut result = V512([0i64; 8]);
        func(&cond, &x, &y, &mut result);
        for i in 0..8 {
            let expect = ((cbits & x.0[i] as u64) | (!cbits & y.0[i] as u64)) as i64;
            assert_eq!(
                result.0[i], expect,
                "cond={cbits:#018x} lane {i}: got {:#018x} want {expect:#018x}",
                result.0[i]
            );
        }
    }
}

// =============================================================================
// bitselect: comparison-produced conditions (VPBLENDM path)
// =============================================================================

#[test]
fn test_bitselect_i32x16_cmp_condition() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i32x16_cmp", 5, |b, p| {
            let a = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[3], 0);
            let cond = b.ins().icmp(IntCC::Equal, a, bb);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[4], 0);
        })
        .expect("compile bitselect_i32x16_cmp");
    let func: extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    ) = unsafe { mem::transmute(code) };

    let a = V512(std::array::from_fn::<i32, 16, _>(|i| i as i32));
    let bv = V512(std::array::from_fn::<i32, 16, _>(|i| {
        if i % 3 == 0 { i as i32 } else { -1 }
    }));
    let x = V512([111i32; 16]);
    let y = V512([-222i32; 16]);
    let mut result = V512([0i32; 16]);
    func(&a, &bv, &x, &y, &mut result);
    for i in 0..16 {
        let expect = if a.0[i] == bv.0[i] { 111 } else { -222 };
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}

#[test]
fn test_bitselect_i64x8_cmp_condition() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("bitselect_i64x8_cmp", 5, |b, p| {
            let a = b.ins().load(I64X8, MemFlags::trusted(), p[0], 0);
            let bb = b.ins().load(I64X8, MemFlags::trusted(), p[1], 0);
            let x = b.ins().load(I64X8, MemFlags::trusted(), p[2], 0);
            let y = b.ins().load(I64X8, MemFlags::trusted(), p[3], 0);
            let cond = b.ins().icmp(IntCC::SignedGreaterThan, a, bb);
            let r = b.ins().bitselect(cond, x, y);
            b.ins().store(MemFlags::trusted(), r, p[4], 0);
        })
        .expect("compile bitselect_i64x8_cmp");
    let func: extern "C" fn(*const I64x8, *const I64x8, *const I64x8, *const I64x8, *mut I64x8) =
        unsafe { mem::transmute(code) };

    let a = V512([5i64, -5, 100, i64::MIN, i64::MAX, 0, 7, -7]);
    let bv = V512([4i64, -4, 100, i64::MAX, i64::MIN, 0, -8, 8]);
    let x = V512([1i64; 8]);
    let y = V512([0i64; 8]);
    let mut result = V512([-1i64; 8]);
    func(&a, &bv, &x, &y, &mut result);
    for i in 0..8 {
        let expect = if a.0[i] > bv.0[i] { 1 } else { 0 };
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}

// =============================================================================
// swizzle: out-of-range indices must produce zero lanes
// =============================================================================

fn swizzle_model_i32(data: &[i32; 16], idx: &[i32; 16]) -> [i32; 16] {
    std::array::from_fn(|i| {
        let j = idx[i] as u32;
        if j < 16 { data[j as usize] } else { 0 }
    })
}

#[test]
fn test_swizzle_i32x16() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("swizzle_i32x16", 3, |b, p| {
            let data = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let idx = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let r = b.ins().swizzle(data, idx);
            b.ins().store(MemFlags::trusted(), r, p[2], 0);
        })
        .expect("compile swizzle_i32x16");
    let func: extern "C" fn(*const I32x16, *const I32x16, *mut I32x16) =
        unsafe { mem::transmute(code) };

    let data = V512(std::array::from_fn::<i32, 16, _>(|i| 1000 + i as i32));
    let idx_cases: [[i32; 16]; 6] = [
        std::array::from_fn(|i| i as i32),          // identity [0..15]
        std::array::from_fn(|i| 15 - i as i32),     // reversed [15..0]
        [16; 16],                                   // first OOR value
        [17; 16],                                   // OOR
        [255; 16],                                  // far OOR
        [0, 15, 16, 17, 255, -1, 8, 3, 2, 1, 31, 5, 100, 14, 7, 12], // mixed, incl. -1 (unsigned huge)
    ];
    for idx in &idx_cases {
        let mut result = V512([i32::MIN; 16]);
        func(&data, &V512(*idx), &mut result);
        let expect = swizzle_model_i32(&data.0, idx);
        assert_eq!(result.0, expect, "idx={idx:?}");
    }
}

#[test]
fn test_swizzle_i64x8() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("swizzle_i64x8", 3, |b, p| {
            let data = b.ins().load(I64X8, MemFlags::trusted(), p[0], 0);
            let idx = b.ins().load(I64X8, MemFlags::trusted(), p[1], 0);
            let r = b.ins().swizzle(data, idx);
            b.ins().store(MemFlags::trusted(), r, p[2], 0);
        })
        .expect("compile swizzle_i64x8");
    let func: extern "C" fn(*const I64x8, *const I64x8, *mut I64x8) =
        unsafe { mem::transmute(code) };

    let data = V512(std::array::from_fn::<i64, 8, _>(|i| {
        0x1_0000_0000i64 + i as i64
    }));
    let idx_cases: [[i64; 8]; 6] = [
        std::array::from_fn(|i| i as i64),      // identity
        std::array::from_fn(|i| 7 - i as i64),  // reversed
        [8; 8],                                 // first OOR value
        [9; 8],                                 // OOR
        [255; 8],                               // far OOR
        [0, 7, 8, -1, 255, 3, 100, 5],          // mixed, incl. -1 (unsigned huge)
    ];
    for idx in &idx_cases {
        let mut result = V512([i64::MIN; 8]);
        func(&data, &V512(*idx), &mut result);
        let expect: [i64; 8] = std::array::from_fn(|i| {
            let j = idx[i] as u64;
            if j < 8 { data.0[j as usize] } else { 0 }
        });
        assert_eq!(result.0, expect, "idx={idx:?}");
    }
}

/// The 128-bit i8x16 swizzle analogue: same out-of-range semantics
/// (index >= 16 -> 0), long supported upstream. This anchors the 512-bit
/// expectations above to the established behavior.
#[test]
fn test_swizzle_i8x16_analogue() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("swizzle_i8x16", 3, |b, p| {
            let data = b.ins().load(I8X16, MemFlags::trusted(), p[0], 0);
            let idx = b.ins().load(I8X16, MemFlags::trusted(), p[1], 0);
            let r = b.ins().swizzle(data, idx);
            b.ins().store(MemFlags::trusted(), r, p[2], 0);
        })
        .expect("compile swizzle_i8x16");
    let func: extern "C" fn(*const [i8; 16], *const [i8; 16], *mut [i8; 16]) =
        unsafe { mem::transmute(code) };

    let data: [i8; 16] = std::array::from_fn(|i| 100 + i as i8);
    let idx_cases: [[i8; 16]; 4] = [
        std::array::from_fn(|i| i as i8),
        std::array::from_fn(|i| 15 - i as i8),
        [16; 16],
        [0, 15, 16, 17, -1, 3, 127, 5, 8, 2, 1, 31, 100, 14, 7, 12],
    ];
    for idx in &idx_cases {
        let mut result = [0i8; 16];
        func(&data, idx, &mut result);
        let expect: [i8; 16] = std::array::from_fn(|i| {
            let j = idx[i] as u8;
            if j < 16 { data[j as usize] } else { 0 }
        });
        assert_eq!(result, expect, "idx={idx:?}");
    }
}

// =============================================================================
// extractlane on F32X16 / F64X8: all lanes, NaN payloads preserved
// =============================================================================

#[test]
fn test_extractlane_f32x16_all_lanes() {
    let mut c = require_avx512!();
    // One function extracting every lane and storing them contiguously.
    let code = c
        .compile_ptr_fn("extract_f32x16_all", 2, |b, p| {
            let v = b.ins().load(F32X16, MemFlags::trusted(), p[0], 0);
            for lane in 0..16u8 {
                let s = b.ins().extractlane(v, lane);
                b.ins()
                    .store(MemFlags::trusted(), s, p[1], (lane as i32) * 4);
            }
        })
        .expect("compile extract_f32x16_all");
    let func: extern "C" fn(*const F32x16, *mut [f32; 16]) = unsafe { mem::transmute(code) };

    // Distinguishable per-lane values, including NaNs with payloads (must be
    // preserved bit-exactly: the lowering is pure moves/shuffles).
    let mut src = V512(std::array::from_fn::<f32, 16, _>(|i| {
        (i as f32) * 1.5 - 7.25
    }));
    src.0[3] = f32::from_bits(0x7FC1_2345); // quiet NaN with payload
    src.0[11] = f32::from_bits(0xFFA0_0001); // signaling-ish NaN, sign bit set
    src.0[15] = f32::from_bits(0x7F80_0001); // signaling NaN

    let mut out = [0.0f32; 16];
    func(&src, &mut out);
    for i in 0..16 {
        assert_eq!(
            out[i].to_bits(),
            src.0[i].to_bits(),
            "lane {i}: got {:#010x} want {:#010x}",
            out[i].to_bits(),
            src.0[i].to_bits()
        );
    }
}

#[test]
fn test_extractlane_f64x8_all_lanes() {
    let mut c = require_avx512!();
    let code = c
        .compile_ptr_fn("extract_f64x8_all", 2, |b, p| {
            let v = b.ins().load(F64X8, MemFlags::trusted(), p[0], 0);
            for lane in 0..8u8 {
                let s = b.ins().extractlane(v, lane);
                b.ins()
                    .store(MemFlags::trusted(), s, p[1], (lane as i32) * 8);
            }
        })
        .expect("compile extract_f64x8_all");
    let func: extern "C" fn(*const F64x8, *mut [f64; 8]) = unsafe { mem::transmute(code) };

    let mut src = V512(std::array::from_fn::<f64, 8, _>(|i| {
        (i as f64) * -3.25 + 0.5
    }));
    src.0[2] = f64::from_bits(0x7FF8_0000_DEAD_BEEF); // quiet NaN with payload
    src.0[5] = f64::from_bits(0xFFF0_0000_0000_0001); // signaling NaN, sign set
    src.0[7] = f64::from_bits(0x7FF4_1234_5678_9ABC);

    let mut out = [0.0f64; 8];
    func(&src, &mut out);
    for i in 0..8 {
        assert_eq!(
            out[i].to_bits(),
            src.0[i].to_bits(),
            "lane {i}: got {:#018x} want {:#018x}",
            out[i].to_bits(),
            src.0[i].to_bits()
        );
    }
}

// =============================================================================
// insertlane on F32X16 / F64X8: roundtrip every lane
// =============================================================================

#[test]
fn test_insertlane_f32x16_roundtrip() {
    let mut c = require_avx512!();
    // One function per lane (lane index is an immediate).
    let mut funcs = Vec::new();
    for lane in 0..16u8 {
        let code = c
            .compile_ptr_fn(&format!("insert_f32x16_{lane}"), 3, move |b, p| {
                let v = b.ins().load(F32X16, MemFlags::trusted(), p[0], 0);
                let s = b.ins().load(F32, MemFlags::trusted(), p[1], 0);
                let r = b.ins().insertlane(v, s, lane);
                b.ins().store(MemFlags::trusted(), r, p[2], 0);
            })
            .expect("compile insert_f32x16");
        let func: extern "C" fn(*const F32x16, *const f32, *mut F32x16) =
            unsafe { mem::transmute(code) };
        funcs.push(func);
    }

    let base = V512(std::array::from_fn::<f32, 16, _>(|i| i as f32 + 0.25));
    let vals = [42.5f32, f32::from_bits(0x7FC0_BEEF), -0.0f32];
    for (lane, func) in funcs.iter().enumerate() {
        for &val in &vals {
            let mut out = V512([0.0f32; 16]);
            func(&base, &val, &mut out);
            for i in 0..16 {
                let want = if i == lane { val } else { base.0[i] };
                assert_eq!(
                    out.0[i].to_bits(),
                    want.to_bits(),
                    "insert lane {lane}, check lane {i}"
                );
            }
        }
    }
}

#[test]
fn test_insertlane_f64x8_roundtrip() {
    let mut c = require_avx512!();
    let mut funcs = Vec::new();
    for lane in 0..8u8 {
        let code = c
            .compile_ptr_fn(&format!("insert_f64x8_{lane}"), 3, move |b, p| {
                let v = b.ins().load(F64X8, MemFlags::trusted(), p[0], 0);
                let s = b.ins().load(F64, MemFlags::trusted(), p[1], 0);
                let r = b.ins().insertlane(v, s, lane);
                b.ins().store(MemFlags::trusted(), r, p[2], 0);
            })
            .expect("compile insert_f64x8");
        let func: extern "C" fn(*const F64x8, *const f64, *mut F64x8) =
            unsafe { mem::transmute(code) };
        funcs.push(func);
    }

    let base = V512(std::array::from_fn::<f64, 8, _>(|i| i as f64 - 3.5));
    let vals = [
        1234.5678f64,
        f64::from_bits(0x7FF8_0000_CAFE_F00D),
        -0.0f64,
    ];
    for (lane, func) in funcs.iter().enumerate() {
        for &val in &vals {
            let mut out = V512([0.0f64; 8]);
            func(&base, &val, &mut out);
            for i in 0..8 {
                let want = if i == lane { val } else { base.0[i] };
                assert_eq!(
                    out.0[i].to_bits(),
                    want.to_bits(),
                    "insert lane {lane}, check lane {i}"
                );
            }
        }
    }
}

// =============================================================================
// Combined: bitselect result of comparison feeding arithmetic fusion still
// bitwise-correct when the condition is NOT a comparison (regression guard
// for the masked-fusion rules' all_ones_or_all_zeros gate).
// =============================================================================

#[test]
fn test_bitselect_i32x16_fusion_gate() {
    let mut c = require_avx512!();
    // bitselect(rawmask, iadd(x, y), passthru): previously fused into a
    // masked VPADDD keyed off each lane's SIGN BIT only; must now be
    // evaluated bitwise.
    let code = c
        .compile_ptr_fn("bitselect_i32x16_fusion_gate", 4, |b, p| {
            let cond = b.ins().load(I32X16, MemFlags::trusted(), p[0], 0);
            let x = b.ins().load(I32X16, MemFlags::trusted(), p[1], 0);
            let y = b.ins().load(I32X16, MemFlags::trusted(), p[2], 0);
            let sum = b.ins().iadd(x, y);
            let r = b.ins().bitselect(cond, sum, x);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })
        .expect("compile bitselect_i32x16_fusion_gate");
    let func: extern "C" fn(*const I32x16, *const I32x16, *const I32x16, *mut I32x16) =
        unsafe { mem::transmute(code) };

    // Positive-sign partial mask: old sign-bit lowering would have chosen
    // the passthru for every lane; bitwise semantics mix bits per lane.
    let cbits: u32 = 0x0000_FF00;
    let cond = V512([cbits as i32; 16]);
    let x = V512(std::array::from_fn::<i32, 16, _>(|i| i as i32 * 3 + 1));
    let y = V512(std::array::from_fn::<i32, 16, _>(|i| i as i32 * -7 + 5));
    let mut result = V512([0i32; 16]);
    func(&cond, &x, &y, &mut result);
    for i in 0..16 {
        let sum = x.0[i].wrapping_add(y.0[i]) as u32;
        let expect = ((cbits & sum) | (!cbits & x.0[i] as u32)) as i32;
        assert_eq!(result.0[i], expect, "lane {i}");
    }
}
