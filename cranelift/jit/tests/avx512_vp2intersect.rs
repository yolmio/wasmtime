#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for `x86_simd_vp2intersect_mask` (AVX-512
//! VP2INTERSECT).
//!
//! Semantics under test (must match both the CLIF documentation and actual
//! hardware): result lane i is all-ones iff a[i] compares equal to ANY lane
//! of b, all-zeros otherwise. Matching is by value equality; duplicates all
//! match on both sides.
//!
//! These tests are skipped unless the host supports the full AVX-512
//! F/DQ/BW/VL baseline AND the separate AVX512_VP2INTERSECT feature.

use cranelift_codegen::Context;
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::*;
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;
use std::mem;

/// Baseline AVX-512 guard (same as avx512_e2e.rs).
fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// VP2INTERSECT is its own CPUID feature (leaf 7, EDX bit 8) on top of the
/// AVX-512 baseline.
fn has_vp2intersect() -> bool {
    has_avx512() && std::arch::is_x86_feature_detected!("avx512vp2intersect")
}

fn isa_with_avx512() -> Option<OwnedTargetIsa> {
    if !has_vp2intersect() {
        return None;
    }
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().ok()?;
    isa_builder.finish(settings::Flags::new(flag_builder)).ok()
}

/// cranelift-native must report the new flag on hosts that have the CPUID
/// bit (this host does when the std detector returns true).
#[test]
fn test_native_reports_vp2intersect_flag() {
    if !std::arch::is_x86_feature_detected!("avx512vp2intersect") {
        println!("Skipping: avx512vp2intersect not detected on host");
        return;
    }
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();
    let value = isa
        .isa_flags()
        .iter()
        .find(|v| v.name == "has_avx512vp2intersect")
        .expect("has_avx512vp2intersect flag must exist on x64")
        .as_bool()
        .expect("has_avx512vp2intersect is a boolean flag");
    assert!(
        value,
        "cranelift-native must enable has_avx512vp2intersect on this host"
    );
}

struct Compiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl Compiler {
    fn new() -> Option<Self> {
        let isa = isa_with_avx512()?;
        let module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let ctx = module.make_context();
        Some(Self {
            module,
            ctx,
            func_ctx: FunctionBuilderContext::new(),
        })
    }

    /// Compile `fn(in0, ..., out)` over `n_inputs` pointers to 64-byte
    /// vectors of type `ty`. Returns the code pointer and VCode disassembly.
    fn compile<F>(&mut self, name: &str, ty: Type, n_inputs: usize, build: F) -> (*const u8, String)
    where
        F: FnOnce(&mut FunctionBuilder, &[Value]) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        for _ in 0..n_inputs {
            sig.params.push(AbiParam::new(ptr_type));
        }
        sig.params.push(AbiParam::new(ptr_type)); // out
        sig.call_conv = CallConv::SystemV;

        let func_id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .unwrap();
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let inputs: Vec<Value> = params[..n_inputs]
                .iter()
                .map(|&p| builder.ins().load(ty, MemFlags::trusted(), p, 0))
                .collect();
            let out_ptr = params[n_inputs];

            let result = build(&mut builder, &inputs);
            builder.ins().store(MemFlags::trusted(), result, out_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module
            .define_function(func_id, &mut self.ctx)
            .expect("define_function");
        let disasm = self
            .ctx
            .compiled_code()
            .expect("compiled code")
            .vcode
            .clone()
            .expect("vcode disasm");
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();
        (self.module.get_finalized_function(func_id), disasm)
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct I32x16([i32; 16]);

#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct I64x8([i64; 8]);

type Fn2 = unsafe extern "C" fn(*const u8, *const u8, *mut u8);

fn run2_i32(code: *const u8, a: &I32x16, b: &I32x16) -> I32x16 {
    let f: Fn2 = unsafe { mem::transmute(code) };
    let mut out = I32x16([0; 16]);
    unsafe {
        f(
            a.0.as_ptr().cast(),
            b.0.as_ptr().cast(),
            out.0.as_mut_ptr().cast(),
        )
    };
    out
}

fn run2_i64(code: *const u8, a: &I64x8, b: &I64x8) -> I64x8 {
    let f: Fn2 = unsafe { mem::transmute(code) };
    let mut out = I64x8([0; 8]);
    unsafe {
        f(
            a.0.as_ptr().cast(),
            b.0.as_ptr().cast(),
            out.0.as_mut_ptr().cast(),
        )
    };
    out
}

/// Documented semantics, computed in Rust: lane i of the result is all-ones
/// iff a[i] equals any lane of b (duplicates all match, on both sides).
fn ref_mask_i32(a: &I32x16, b: &I32x16) -> I32x16 {
    let mut out = I32x16([0; 16]);
    for i in 0..16 {
        out.0[i] = if b.0.contains(&a.0[i]) { -1 } else { 0 };
    }
    out
}

fn ref_mask_i64(a: &I64x8, b: &I64x8) -> I64x8 {
    let mut out = I64x8([0; 8]);
    for i in 0..8 {
        out.0[i] = if b.0.contains(&a.0[i]) { -1 } else { 0 };
    }
    out
}

fn assert_vp2intersect_disasm(disasm: &str, mnemonic: &str, what: &str) {
    assert!(
        disasm.contains(mnemonic),
        "{what}: expected `{mnemonic}` in disassembly:\n{disasm}"
    );
    // The destination pair is pinned to k6/k7 by the lowering.
    assert!(
        disasm.contains("%k6") && disasm.contains("%k7"),
        "{what}: expected the fixed %k6/%k7 pair in disassembly:\n{disasm}"
    );
}

#[test]
fn test_vp2intersect_i32x16_basic_sets() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    let (code, disasm) = compiler.compile("vp2i_d_basic", I32X16, 2, |f, v| {
        f.ins().x86_simd_vp2intersect_mask(v[0], v[1])
    });
    assert_vp2intersect_disasm(&disasm, "vp2intersectd", "i32x16 basic");

    // Disjoint sets: result must be all zeros.
    let a = I32x16([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let b = I32x16([
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    ]);
    assert_eq!(run2_i32(code, &a, &b), I32x16([0; 16]), "disjoint sets");
    assert_eq!(run2_i32(code, &a, &b), ref_mask_i32(&a, &b));

    // Identical sets: result must be all ones.
    assert_eq!(run2_i32(code, &a, &a), I32x16([-1; 16]), "identical sets");

    // Partial overlap with DUPLICATE values on both sides. Per the SDM,
    // intersection is by value equality: every duplicate occurrence matches.
    let a = I32x16([7, 7, 2, 3, 42, 42, 42, 8, 9, 7, 2, -1, 0, 13, 5, 5]);
    let b = I32x16([42, 42, 7, 7, 7, 5, 1000, 1000, -5, -5, 0, 0, 77, 88, 99, 111]);
    let got = run2_i32(code, &a, &b);
    let want = ref_mask_i32(&a, &b);
    assert_eq!(got, want, "partial overlap with duplicates");
    // Explicitly pin the duplicate semantics: all of a's 7s, 42s, 5s and the
    // 0 lane match; 2/3/8/9/-1/13 do not.
    assert_eq!(
        got,
        I32x16([-1, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0, -1, -1]),
        "duplicate-match semantics differ from documentation"
    );

    // Zero is a value like any other (no spurious matching of empty lanes).
    let a = I32x16([0; 16]);
    let b = I32x16([1; 16]);
    assert_eq!(run2_i32(code, &a, &b), I32x16([0; 16]), "zeros vs ones");
}

#[test]
fn test_vp2intersect_i64x8_basic_sets() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    let (code, disasm) = compiler.compile("vp2i_q_basic", I64X8, 2, |f, v| {
        f.ins().x86_simd_vp2intersect_mask(v[0], v[1])
    });
    assert_vp2intersect_disasm(&disasm, "vp2intersectq", "i64x8 basic");

    // Disjoint.
    let a = I64x8([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I64x8([10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(run2_i64(code, &a, &b), I64x8([0; 8]), "disjoint sets");

    // Identical.
    assert_eq!(run2_i64(code, &a, &a), I64x8([-1; 8]), "identical sets");

    // Partial overlap with duplicates on both sides; values exercise the
    // full 64-bit width.
    let a = I64x8([
        i64::MIN,
        i64::MIN,
        0x1234_5678_9ABC_DEF0,
        -1,
        42,
        42,
        i64::MAX,
        0,
    ]);
    let b = I64x8([
        42,
        i64::MIN,
        i64::MIN,
        0x1234_5678_9ABC_DEF0,
        7,
        7,
        -2,
        i64::MAX,
    ]);
    let got = run2_i64(code, &a, &b);
    assert_eq!(got, ref_mask_i64(&a, &b), "partial overlap with duplicates");
    assert_eq!(
        got,
        I64x8([-1, -1, -1, 0, -1, -1, -1, 0]),
        "duplicate-match semantics differ from documentation"
    );
}

/// Swapping the operands yields the b-side mask (documented way to get the
/// second VP2INTERSECT output).
#[test]
fn test_vp2intersect_swapped_operands() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    let (code, _) = compiler.compile("vp2i_d_swapped", I32X16, 2, |f, v| {
        f.ins().x86_simd_vp2intersect_mask(v[1], v[0])
    });

    let a = I32x16([7, 7, 2, 3, 42, 42, 42, 8, 9, 7, 2, -1, 0, 13, 5, 5]);
    let b = I32x16([42, 42, 7, 7, 7, 5, 1000, 1000, -5, -5, 0, 0, 77, 88, 99, 111]);
    // mask(b, a): membership of b's lanes in a.
    assert_eq!(run2_i32(code, &a, &b), ref_mask_i32(&b, &a), "swapped");
}

/// Seeded randomized cross-check of hardware semantics against the model,
/// with a small value range to force many collisions and duplicates.
#[test]
fn test_vp2intersect_randomized_seeded() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    let (code_d, _) = compiler.compile("vp2i_d_rand", I32X16, 2, |f, v| {
        f.ins().x86_simd_vp2intersect_mask(v[0], v[1])
    });
    let (code_q, _) = compiler.compile("vp2i_q_rand", I64X8, 2, |f, v| {
        f.ins().x86_simd_vp2intersect_mask(v[0], v[1])
    });

    let mut s: u64 = 0x1234_5678_DEAD_BEEF;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    for iter in 0..64 {
        // Small range [0, 8) forces heavy duplication and overlap.
        let mut a = I32x16([0; 16]);
        let mut b = I32x16([0; 16]);
        for i in 0..16 {
            a.0[i] = (next() % 8) as i32;
            b.0[i] = (next() % 8) as i32;
        }
        assert_eq!(
            run2_i32(code_d, &a, &b),
            ref_mask_i32(&a, &b),
            "i32x16 randomized iteration {iter}: a={a:?} b={b:?}"
        );

        let mut aq = I64x8([0; 8]);
        let mut bq = I64x8([0; 8]);
        for i in 0..8 {
            aq.0[i] = (next() % 6) as i64 - 3; // include negatives
            bq.0[i] = (next() % 6) as i64 - 3;
        }
        assert_eq!(
            run2_i64(code_q, &aq, &bq),
            ref_mask_i64(&aq, &bq),
            "i64x8 randomized iteration {iter}: a={aq:?} b={bq:?}"
        );
    }
}

/// Semi-join kernel: keep only the probe lanes whose key exists on the
/// build side (the intended SQL-engine usage of this instruction).
#[test]
fn test_vp2intersect_semi_join_filter() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    let (code, disasm) = compiler.compile("vp2i_semi_join", I32X16, 2, |f, v| {
        let mask = f.ins().x86_simd_vp2intersect_mask(v[0], v[1]);
        f.ins().band(v[0], mask)
    });
    assert_vp2intersect_disasm(&disasm, "vp2intersectd", "semi-join");

    let probe = I32x16([3, 17, 4, 99, 5, 23, 8, 42, 3, 3, 77, 100, 5, 6, 7, 2]);
    let build = I32x16([5, 42, 3, 7, 11, 13, 17, 19, 5, 42, 3, 7, 11, 13, 17, 19]);
    let got = run2_i32(code, &probe, &build);
    let mask = ref_mask_i32(&probe, &build);
    let mut want = I32x16([0; 16]);
    for i in 0..16 {
        want.0[i] = probe.0[i] & mask.0[i];
    }
    assert_eq!(got, want, "semi-join filtered probe lanes");
}

/// Register-pressure test: many zmm values stay live across two
/// vp2intersect operations (whose k6/k7 pair defs must not disturb any live
/// vector register), mirroring the gather alias regression style of
/// avx512_e2e.rs.
#[test]
fn test_vp2intersect_under_register_pressure() {
    let Some(mut compiler) = Compiler::new() else {
        println!("Skipping: AVX-512 VP2INTERSECT not available");
        return;
    };

    // fn(a, b, c, d, out):
    //   m1 = vp2i_mask(a, b); m2 = vp2i_mask(c, d)
    //   out = (a & m1) | (c & m2) | (b ^ d)
    // All four inputs stay live across both intersections.
    let (code, disasm) = compiler.compile("vp2i_pressure", I32X16, 4, |f, v| {
        let m1 = f.ins().x86_simd_vp2intersect_mask(v[0], v[1]);
        let m2 = f.ins().x86_simd_vp2intersect_mask(v[2], v[3]);
        let t1 = f.ins().band(v[0], m1);
        let t2 = f.ins().band(v[2], m2);
        let t3 = f.ins().bxor(v[1], v[3]);
        let t4 = f.ins().bor(t1, t2);
        f.ins().bor(t4, t3)
    });
    assert_eq!(
        disasm.matches("vp2intersectd").count(),
        2,
        "expected two vp2intersectd instructions:\n{disasm}"
    );

    type Fn4 = unsafe extern "C" fn(*const u8, *const u8, *const u8, *const u8, *mut u8);
    let f: Fn4 = unsafe { mem::transmute(code) };

    let a = I32x16([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let b = I32x16([2, 2, 4, 4, 6, 6, 8, 8, 100, 100, 12, 12, 200, 200, 16, 16]);
    let c = I32x16([-1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8]);
    let d = I32x16([-3, -3, -3, 0, 0, 0, -8, -8, 5, 5, 5, 5, 5, 5, 5, 5]);
    let mut out = I32x16([0; 16]);
    unsafe {
        f(
            a.0.as_ptr().cast(),
            b.0.as_ptr().cast(),
            c.0.as_ptr().cast(),
            d.0.as_ptr().cast(),
            out.0.as_mut_ptr().cast(),
        )
    };

    let m1 = ref_mask_i32(&a, &b);
    let m2 = ref_mask_i32(&c, &d);
    let mut want = I32x16([0; 16]);
    for i in 0..16 {
        want.0[i] = (a.0[i] & m1.0[i]) | (c.0[i] & m2.0[i]) | (b.0[i] ^ d.0[i]);
    }
    assert_eq!(out, want, "register-pressure combined result");
}
