#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for the CLIF semantics of 512-bit floating-point
//! `fmin`/`fmax` and `fcvt_to_sint_sat`.
//!
//! These operations cannot be lowered to bare VMINPS/VMAXPS/VCVTTPS2DQ
//! because the machine instructions disagree with CLIF semantics for NaN,
//! signed zeros, and out-of-range conversions. The tests here:
//!
//! 1. Check CLIF semantics lane-by-lane against a Rust-computed reference
//!    (NaN lanes assert `is_nan`, signed-zero lanes assert the sign bit,
//!    everything else asserts exact bits).
//! 2. Cross-check every lane of the 512-bit lowering against the fork's own
//!    128-bit lowering of the same operation (f32x4 / f64x2, and the scalar
//!    f64 -> i64 conversion where no 128-bit vector lowering exists), which
//!    proves the 512-bit lowering agrees with the narrower ones.

use cranelift_codegen::Context;
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::*;
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;
use std::mem;

/// Check if AVX-512 is available on this machine.
fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// Create an ISA configured for AVX-512 testing.
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

fn jit_module_with_avx512() -> Option<JITModule> {
    let isa = isa_with_avx512()?;
    Some(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

/// All compiled functions use pointer-based signatures so that any vector
/// width can share the same Rust-side function type.
type BinaryFn = unsafe extern "C" fn(*const u8, *const u8, *mut u8);
type UnaryFn = unsafe extern "C" fn(*const u8, *mut u8);

struct TestCompiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl TestCompiler {
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

    /// Compile `fn(src1_ptr, src2_ptr, dst_ptr)`: loads two vectors of type
    /// `ty`, applies `build_fn`, stores the result vector at `dst_ptr`.
    fn compile_binary<F>(&mut self, name: &str, ty: Type, build_fn: F) -> *const u8
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .expect("declare_function");

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1 = builder.ins().load(ty, MemFlags::trusted(), params[0], 0);
            let src2 = builder.ins().load(ty, MemFlags::trusted(), params[1], 0);
            let result = build_fn(&mut builder, src1, src2);
            builder.ins().store(MemFlags::trusted(), result, params[2], 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions");
        self.module.get_finalized_function(func_id)
    }

    /// Compile `fn(src_ptr, dst_ptr)`: loads a value of type `in_ty`,
    /// applies `fcvt_to_sint_sat` to `out_ty`, stores the result.
    fn compile_fcvt_to_sint_sat(&mut self, name: &str, in_ty: Type, out_ty: Type) -> *const u8 {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .expect("declare_function");

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src = builder.ins().load(in_ty, MemFlags::trusted(), params[0], 0);
            let result = builder.ins().fcvt_to_sint_sat(out_ty, src);
            builder.ins().store(MemFlags::trusted(), result, params[1], 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function");
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .expect("finalize_definitions");
        self.module.get_finalized_function(func_id)
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Buf64([u8; 64]);

impl Buf64 {
    fn zero() -> Self {
        Buf64([0; 64])
    }
    fn from_f32(vals: &[f32; 16]) -> Self {
        let mut b = Self::zero();
        for (i, v) in vals.iter().enumerate() {
            b.0[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
    fn from_f64(vals: &[f64; 8]) -> Self {
        let mut b = Self::zero();
        for (i, v) in vals.iter().enumerate() {
            b.0[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
    fn as_f32(&self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = f32::from_le_bytes(self.0[i * 4..i * 4 + 4].try_into().unwrap());
        }
        out
    }
    fn as_f64(&self) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        for i in 0..8 {
            out[i] = f64::from_le_bytes(self.0[i * 8..i * 8 + 8].try_into().unwrap());
        }
        out
    }
    fn as_i32(&self) -> [i32; 16] {
        let mut out = [0i32; 16];
        for i in 0..16 {
            out[i] = i32::from_le_bytes(self.0[i * 4..i * 4 + 4].try_into().unwrap());
        }
        out
    }
    fn as_i64(&self) -> [i64; 8] {
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = i64::from_le_bytes(self.0[i * 8..i * 8 + 8].try_into().unwrap());
        }
        out
    }
}

unsafe fn run_binary(f: BinaryFn, a: &Buf64, b: &Buf64) -> Buf64 {
    let mut dst = Buf64::zero();
    unsafe { f(a.0.as_ptr(), b.0.as_ptr(), dst.0.as_mut_ptr()) };
    dst
}

unsafe fn run_unary(f: UnaryFn, a: &Buf64) -> Buf64 {
    let mut dst = Buf64::zero();
    unsafe { f(a.0.as_ptr(), dst.0.as_mut_ptr()) };
    dst
}

/// Lane pairs exercising every CLIF fmin/fmax edge case:
/// NaN either side, both NaN, +0/-0 both orders, Inf/-Inf, ordinary values,
/// subnormal vs zero, equal values, sign-mixed values.
const F32_PAIRS: [(f32, f32); 16] = [
    (f32::NAN, 1.0),
    (1.0, f32::NAN),
    (f32::NAN, f32::NAN),
    (0.0, -0.0),
    (-0.0, 0.0),
    (f32::INFINITY, f32::NEG_INFINITY),
    (f32::NEG_INFINITY, 3.0),
    (5.0, 2.0),
    (1e-45, 0.0), // smallest positive subnormal vs +0
    (0.0, 1e-45),
    (-1.5, -1.5),
    (2.0, 5.0),
    (3.0, -3.0),
    (-3.0, 3.0),
    (100.0, -100.0),
    (-100.0, 100.0),
];

const F64_PAIRS: [(f64, f64); 8] = [
    (f64::NAN, 1.0),
    (1.0, f64::NAN),
    (0.0, -0.0),
    (-0.0, 0.0),
    (f64::INFINITY, f64::NEG_INFINITY),
    (f64::NEG_INFINITY, 3.0),
    (5.0, 2.0),
    (5e-324, 0.0), // smallest positive subnormal vs +0
];

/// Extra f64 pairs run through the same compiled code to also cover
/// both-NaN and subnormal-second-operand lanes at 64 bits.
const F64_PAIRS_2: [(f64, f64); 8] = [
    (f64::NAN, f64::NAN),
    (0.0, 5e-324),
    (-1.5, -1.5),
    (2.0, 5.0),
    (3.0, -3.0),
    (-3.0, 3.0),
    (100.0, -100.0),
    (-100.0, 100.0),
];

/// Assert CLIF fmin/fmax semantics for one lane. `is_min` selects fmin.
fn check_minmax_lane(a: f64, b: f64, r_bits: u64, r_is_nan: bool, bits: u32, is_min: bool) {
    let ctx = format!(
        "{}{} lane a={a:?} b={b:?} r_bits={r_bits:#x}",
        if is_min { "fmin" } else { "fmax" },
        bits,
    );
    if a.is_nan() || b.is_nan() {
        assert!(r_is_nan, "expected NaN: {ctx}");
        return;
    }
    let sign_mask: u64 = if bits == 32 { 1 << 31 } else { 1 << 63 };
    if a == 0.0 && b == 0.0 {
        // Zero-valued result: check the sign bit, not just the value.
        // fmin(+0, -0) = -0 in either order; fmax(-0, +0) = +0.
        let a_neg = a.is_sign_negative();
        let b_neg = b.is_sign_negative();
        let want_neg = if is_min { a_neg || b_neg } else { a_neg && b_neg };
        assert_eq!(r_bits & !sign_mask, 0, "expected zero value: {ctx}");
        assert_eq!(
            r_bits & sign_mask != 0,
            want_neg,
            "wrong zero sign: {ctx} (want_neg={want_neg})"
        );
        return;
    }
    let want = if is_min == (a < b) { a } else { b };
    let want_bits = if bits == 32 {
        (want as f32).to_bits() as u64
    } else {
        want.to_bits()
    };
    assert_eq!(r_bits, want_bits, "wrong value: {ctx}");
}

/// A 512-bit lane and a narrower-lowering lane "agree" if they are both NaN
/// (payload may legitimately differ between lowerings) or bit-identical.
fn lanes_agree(bits512: u64, nan512: bool, bits_narrow: u64, nan_narrow: bool) -> bool {
    (nan512 && nan_narrow) || bits512 == bits_narrow
}

#[test]
fn test_f32x16_fmin_fmax_semantics_and_128bit_crosscheck() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let fmin512 = compiler.compile_binary("fmin_f32x16", F32X16, |b, x, y| b.ins().fmin(x, y));
    let fmax512 = compiler.compile_binary("fmax_f32x16", F32X16, |b, x, y| b.ins().fmax(x, y));
    let fmin128 = compiler.compile_binary("fmin_f32x4", F32X4, |b, x, y| b.ins().fmin(x, y));
    let fmax128 = compiler.compile_binary("fmax_f32x4", F32X4, |b, x, y| b.ins().fmax(x, y));
    let fmin512: BinaryFn = unsafe { mem::transmute(fmin512) };
    let fmax512: BinaryFn = unsafe { mem::transmute(fmax512) };
    let fmin128: BinaryFn = unsafe { mem::transmute(fmin128) };
    let fmax128: BinaryFn = unsafe { mem::transmute(fmax128) };

    let mut xs = [0f32; 16];
    let mut ys = [0f32; 16];
    for (i, (x, y)) in F32_PAIRS.iter().enumerate() {
        xs[i] = *x;
        ys[i] = *y;
    }
    let a = Buf64::from_f32(&xs);
    let b = Buf64::from_f32(&ys);

    for (func128, func512, is_min) in [(fmin128, fmin512, true), (fmax128, fmax512, false)] {
        let r512 = unsafe { run_binary(func512, &a, &b) };
        let r512_f = r512.as_f32();

        // 1. CLIF semantics, lane by lane.
        for i in 0..16 {
            check_minmax_lane(
                xs[i] as f64,
                ys[i] as f64,
                r512_f[i].to_bits() as u64,
                r512_f[i].is_nan(),
                32,
                is_min,
            );
        }

        // 2. Cross-check against the 128-bit (f32x4) lowering: run the same
        // lanes through the fork's own f32x4 fmin/fmax, 4 lanes at a time.
        for chunk in 0..4 {
            let mut xa = Buf64::zero();
            let mut yb = Buf64::zero();
            xa.0[..16].copy_from_slice(&a.0[chunk * 16..chunk * 16 + 16]);
            yb.0[..16].copy_from_slice(&b.0[chunk * 16..chunk * 16 + 16]);
            let r128 = unsafe { run_binary(func128, &xa, &yb) };
            let r128_f = r128.as_f32();
            for lane in 0..4 {
                let v512 = r512_f[chunk * 4 + lane];
                let v128 = r128_f[lane];
                assert!(
                    lanes_agree(
                        v512.to_bits() as u64,
                        v512.is_nan(),
                        v128.to_bits() as u64,
                        v128.is_nan()
                    ),
                    "{} f32 512-vs-128 mismatch lane {}: x={:?} y={:?} v512={:?}({:#x}) v128={:?}({:#x})",
                    if is_min { "fmin" } else { "fmax" },
                    chunk * 4 + lane,
                    xs[chunk * 4 + lane],
                    ys[chunk * 4 + lane],
                    v512,
                    v512.to_bits(),
                    v128,
                    v128.to_bits(),
                );
            }
        }
    }
}

#[test]
fn test_f64x8_fmin_fmax_semantics_and_128bit_crosscheck() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let fmin512 = compiler.compile_binary("fmin_f64x8", F64X8, |b, x, y| b.ins().fmin(x, y));
    let fmax512 = compiler.compile_binary("fmax_f64x8", F64X8, |b, x, y| b.ins().fmax(x, y));
    let fmin128 = compiler.compile_binary("fmin_f64x2", F64X2, |b, x, y| b.ins().fmin(x, y));
    let fmax128 = compiler.compile_binary("fmax_f64x2", F64X2, |b, x, y| b.ins().fmax(x, y));
    let fmin512: BinaryFn = unsafe { mem::transmute(fmin512) };
    let fmax512: BinaryFn = unsafe { mem::transmute(fmax512) };
    let fmin128: BinaryFn = unsafe { mem::transmute(fmin128) };
    let fmax128: BinaryFn = unsafe { mem::transmute(fmax128) };

    for pairs in [&F64_PAIRS, &F64_PAIRS_2] {
        let mut xs = [0f64; 8];
        let mut ys = [0f64; 8];
        for (i, (x, y)) in pairs.iter().enumerate() {
            xs[i] = *x;
            ys[i] = *y;
        }
        let a = Buf64::from_f64(&xs);
        let b = Buf64::from_f64(&ys);

        for (func128, func512, is_min) in [(fmin128, fmin512, true), (fmax128, fmax512, false)] {
            let r512 = unsafe { run_binary(func512, &a, &b) };
            let r512_f = r512.as_f64();

            // 1. CLIF semantics, lane by lane.
            for i in 0..8 {
                check_minmax_lane(xs[i], ys[i], r512_f[i].to_bits(), r512_f[i].is_nan(), 64, is_min);
            }

            // 2. Cross-check against the 128-bit (f64x2) lowering.
            for chunk in 0..4 {
                let mut xa = Buf64::zero();
                let mut yb = Buf64::zero();
                xa.0[..16].copy_from_slice(&a.0[chunk * 16..chunk * 16 + 16]);
                yb.0[..16].copy_from_slice(&b.0[chunk * 16..chunk * 16 + 16]);
                let r128 = unsafe { run_binary(func128, &xa, &yb) };
                let r128_f = r128.as_f64();
                for lane in 0..2 {
                    let v512 = r512_f[chunk * 2 + lane];
                    let v128 = r128_f[lane];
                    assert!(
                        lanes_agree(v512.to_bits(), v512.is_nan(), v128.to_bits(), v128.is_nan()),
                        "{} f64 512-vs-128 mismatch lane {}: x={:?} y={:?} v512={:?}({:#x}) v128={:?}({:#x})",
                        if is_min { "fmin" } else { "fmax" },
                        chunk * 2 + lane,
                        xs[chunk * 2 + lane],
                        ys[chunk * 2 + lane],
                        v512,
                        v512.to_bits(),
                        v128,
                        v128.to_bits(),
                    );
                }
            }
        }
    }
}

#[test]
fn test_fcvt_to_sint_sat_f32x16_semantics_and_128bit_crosscheck() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let cvt512 = compiler.compile_fcvt_to_sint_sat("cvt_f32x16", F32X16, I32X16);
    let cvt128 = compiler.compile_fcvt_to_sint_sat("cvt_f32x4", F32X4, I32X4);
    let cvt512: UnaryFn = unsafe { mem::transmute(cvt512) };
    let cvt128: UnaryFn = unsafe { mem::transmute(cvt128) };

    // Inputs and CLIF-required outputs. Note 2147483647 (i32::MAX) is NOT
    // representable as f32: the largest f32 below 2^31 is 2147483520.0 and
    // the literal 2147483648.0 is exactly 2^31 (out of range => i32::MAX).
    // -2147483648.0 is exactly -2^31 (in range); -2147483904.0 is the first
    // f32 below i32::MIN (out of range => i32::MIN).
    let cases: [(f32, i32); 16] = [
        (f32::NAN, 0),
        (f32::INFINITY, i32::MAX),
        (f32::NEG_INFINITY, i32::MIN),
        // (3.5e38 exceeds f32::MAX, so use 3.4e38: a large finite f32 that
        // still overflows the i32 range.)
        (3.4e38, i32::MAX),
        (-3.4e38, i32::MIN),
        (2147483648.0, i32::MAX),
        (-2147483904.0, i32::MIN),
        (2147483520.0, 2147483520),
        (-2147483648.0, i32::MIN),
        (0.5, 0),
        (-0.5, 0),
        (1.5, 1),
        (-1.5, -1),
        (0.0, 0),
        (100.25, 100),
        (-100.9, -100),
    ];

    let mut xs = [0f32; 16];
    for (i, (x, _)) in cases.iter().enumerate() {
        xs[i] = *x;
    }
    let a = Buf64::from_f32(&xs);
    let r512 = unsafe { run_unary(cvt512, &a) };
    let r512_i = r512.as_i32();

    // 1. CLIF semantics.
    for (i, (x, want)) in cases.iter().enumerate() {
        assert_eq!(
            r512_i[i], *want,
            "fcvt_to_sint_sat f32x16 lane {i}: input {x:?} => got {} want {want}",
            r512_i[i],
        );
    }

    // 2. Cross-check against the 128-bit (f32x4 -> i32x4) lowering.
    for chunk in 0..4 {
        let mut xa = Buf64::zero();
        xa.0[..16].copy_from_slice(&a.0[chunk * 16..chunk * 16 + 16]);
        let r128 = unsafe { run_unary(cvt128, &xa) };
        let r128_i = r128.as_i32();
        for lane in 0..4 {
            assert_eq!(
                r512_i[chunk * 4 + lane],
                r128_i[lane],
                "fcvt_to_sint_sat 512-vs-128 mismatch lane {}: input {:?}",
                chunk * 4 + lane,
                xs[chunk * 4 + lane],
            );
        }
    }
}

#[test]
fn test_fcvt_to_sint_sat_f64x8_semantics_and_scalar_crosscheck() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let cvt512 = compiler.compile_fcvt_to_sint_sat("cvt_f64x8", F64X8, I64X8);
    // There is no 128-bit vector f64x2 -> i64x2 lowering on x64, so
    // cross-check against the fork's scalar f64 -> i64 lowering instead.
    let cvt_scalar = compiler.compile_fcvt_to_sint_sat("cvt_f64_scalar", F64, I64);
    let cvt512: UnaryFn = unsafe { mem::transmute(cvt512) };
    let cvt_scalar: UnaryFn = unsafe { mem::transmute(cvt_scalar) };

    // Note i64::MAX is NOT representable as f64: the largest f64 below 2^63
    // is 9223372036854774784.0 (= 2^63 - 1024); 9223372036854775808.0 is
    // exactly 2^63 (out of range => i64::MAX). -9223372036854775808.0 is
    // exactly -2^63 (in range); -9223372036854777856.0 is the first f64
    // below i64::MIN (out of range => i64::MIN).
    let batches: [[(f64, i64); 8]; 2] = [
        [
            (f64::NAN, 0),
            (f64::INFINITY, i64::MAX),
            (f64::NEG_INFINITY, i64::MIN),
            (9223372036854775808.0, i64::MAX),
            (9223372036854774784.0, 9223372036854774784),
            (-9223372036854775808.0, i64::MIN),
            (-9223372036854777856.0, i64::MIN),
            (0.5, 0),
        ],
        [
            (3.5e38, i64::MAX),
            (-3.5e38, i64::MIN),
            (1.0e19, i64::MAX),
            (-1.0e19, i64::MIN),
            (12345.75, 12345),
            (-12345.75, -12345),
            (-0.5, 0),
            (0.0, 0),
        ],
    ];

    for cases in &batches {
        let mut xs = [0f64; 8];
        for (i, (x, _)) in cases.iter().enumerate() {
            xs[i] = *x;
        }
        let a = Buf64::from_f64(&xs);
        let r512 = unsafe { run_unary(cvt512, &a) };
        let r512_i = r512.as_i64();

        // 1. CLIF semantics.
        for (i, (x, want)) in cases.iter().enumerate() {
            assert_eq!(
                r512_i[i], *want,
                "fcvt_to_sint_sat f64x8 lane {i}: input {x:?} => got {} want {want}",
                r512_i[i],
            );
        }

        // 2. Cross-check every lane against the scalar lowering.
        for i in 0..8 {
            let mut xa = Buf64::zero();
            xa.0[..8].copy_from_slice(&xs[i].to_le_bytes());
            let r = unsafe { run_unary(cvt_scalar, &xa) };
            assert_eq!(
                r512_i[i],
                r.as_i64()[0],
                "fcvt_to_sint_sat 512-vs-scalar mismatch lane {i}: input {:?}",
                xs[i],
            );
        }
    }
}
