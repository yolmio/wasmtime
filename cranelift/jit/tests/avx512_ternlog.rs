#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for the VPTERNLOG 3-input boolean fusion rules.
//!
//! Each fused shape (xor3 / and3 / or3 / (a&b)|c / (a|b)&c, in both
//! associations / operand orders) must:
//! 1. compile to exactly ONE vpternlog instruction with the documented imm8
//!    (asserted on the VCode disassembly), and
//! 2. produce bit-exact results against a Rust-computed reference over
//!    adversarial bit patterns (all-ones, all-zeros, alternating,
//!    per-lane-varying) and seeded pseudo-random inputs, on all four 512-bit
//!    integer types (I8X64, I16X32, I32X16, I64X8).
//!
//! Aliased-input expressions (the same value feeding two of the three
//! VPTERNLOG operands, including the tied/RMW first operand) are exercised
//! separately to catch tie/aliasing bugs.

use cranelift_codegen::Context;
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::*;
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::*;
use cranelift_jit::*;
use cranelift_module::*;
use std::mem;

/// Check if AVX-512 is available on this machine (same guard as
/// avx512_e2e.rs: the backend requires F+BW+DQ+VL together).
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

struct TernCompiler {
    module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
}

impl TernCompiler {
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

    /// Compile `fn(in0, in1, ..., out)` over `n_inputs` pointers to 64-byte
    /// vectors of type `ty`; the body is produced by `build`. Returns the
    /// code pointer and the VCode disassembly.
    fn compile<F>(&mut self, name: &str, ty: Type, n_inputs: usize, build: F) -> (*const u8, String)
    where
        F: FnOnce(&mut FunctionBuilder, &[Value]) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        for _ in 0..n_inputs {
            sig.params.push(AbiParam::new(ptr_type));
        }
        sig.params.push(AbiParam::new(ptr_type)); // out ptr
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
struct V512([u8; 64]);

type Fn3 = unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8);
type Fn2 = unsafe extern "C" fn(*const u8, *const u8, *mut u8);

fn run3(code: *const u8, a: &V512, b: &V512, c: &V512) -> V512 {
    let f: Fn3 = unsafe { mem::transmute(code) };
    let mut out = V512([0; 64]);
    unsafe { f(a.0.as_ptr(), b.0.as_ptr(), c.0.as_ptr(), out.0.as_mut_ptr()) };
    out
}

fn run2(code: *const u8, a: &V512, b: &V512) -> V512 {
    let f: Fn2 = unsafe { mem::transmute(code) };
    let mut out = V512([0; 64]);
    unsafe { f(a.0.as_ptr(), b.0.as_ptr(), out.0.as_mut_ptr()) };
    out
}

/// Bytewise reference: bitwise ops are lane-size agnostic, so the reference
/// works on bytes regardless of the CLIF vector type.
fn ref3(f: impl Fn(u8, u8, u8) -> u8, a: &V512, b: &V512, c: &V512) -> V512 {
    let mut out = V512([0; 64]);
    for i in 0..64 {
        out.0[i] = f(a.0[i], b.0[i], c.0[i]);
    }
    out
}

/// Deterministic xorshift64* generator for seeded pseudo-random patterns.
fn xorshift_fill(seed: u64) -> V512 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut out = V512([0; 64]);
    for chunk in out.0.chunks_exact_mut(8) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        chunk.copy_from_slice(&s.to_le_bytes());
    }
    out
}

/// Adversarial + seeded input patterns.
fn patterns() -> Vec<V512> {
    let mut v = vec![
        V512([0x00; 64]), // all zeros
        V512([0xFF; 64]), // all ones
        V512([0xAA; 64]), // alternating bits
        V512([0x55; 64]), // alternating bits, inverted phase
    ];
    // Per-lane-varying pattern (every byte different).
    let mut lanes = V512([0; 64]);
    for i in 0..64 {
        lanes.0[i] = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    v.push(lanes);
    // Alternating whole lanes of ones/zeros at 32-bit granularity.
    let mut alt32 = V512([0; 64]);
    for i in 0..64 {
        alt32.0[i] = if (i / 4) % 2 == 0 { 0xFF } else { 0x00 };
    }
    v.push(alt32);
    // Seeded pseudo-random patterns.
    v.push(xorshift_fill(1));
    v.push(xorshift_fill(2));
    v.push(xorshift_fill(0xDEADBEEF));
    v
}

/// The ten fused shapes: (name, imm8, IR builder, byte reference).
struct Shape {
    name: &'static str,
    imm8: u8,
    build: fn(&mut FunctionBuilder, &[Value]) -> Value,
    reference: fn(u8, u8, u8) -> u8,
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "xor3_left", // bxor(bxor(a,b),c): 0xF0^0xCC^0xAA = 0x96
            imm8: 0x96,
            build: |f, v| {
                let t = f.ins().bxor(v[0], v[1]);
                f.ins().bxor(t, v[2])
            },
            reference: |a, b, c| a ^ b ^ c,
        },
        Shape {
            name: "xor3_right", // bxor(a,bxor(b,c)): 0x96
            imm8: 0x96,
            build: |f, v| {
                let t = f.ins().bxor(v[1], v[2]);
                f.ins().bxor(v[0], t)
            },
            reference: |a, b, c| a ^ b ^ c,
        },
        Shape {
            name: "and3_left", // band(band(a,b),c): 0xF0&0xCC&0xAA = 0x80
            imm8: 0x80,
            build: |f, v| {
                let t = f.ins().band(v[0], v[1]);
                f.ins().band(t, v[2])
            },
            reference: |a, b, c| a & b & c,
        },
        Shape {
            name: "and3_right", // band(a,band(b,c)): 0x80
            imm8: 0x80,
            build: |f, v| {
                let t = f.ins().band(v[1], v[2]);
                f.ins().band(v[0], t)
            },
            reference: |a, b, c| a & b & c,
        },
        Shape {
            name: "or3_left", // bor(bor(a,b),c): 0xF0|0xCC|0xAA = 0xFE
            imm8: 0xFE,
            build: |f, v| {
                let t = f.ins().bor(v[0], v[1]);
                f.ins().bor(t, v[2])
            },
            reference: |a, b, c| a | b | c,
        },
        Shape {
            name: "or3_right", // bor(a,bor(b,c)): 0xFE
            imm8: 0xFE,
            build: |f, v| {
                let t = f.ins().bor(v[1], v[2]);
                f.ins().bor(v[0], t)
            },
            reference: |a, b, c| a | b | c,
        },
        Shape {
            name: "andor", // bor(band(a,b),c): (0xF0&0xCC)|0xAA = 0xEA
            imm8: 0xEA,
            build: |f, v| {
                let t = f.ins().band(v[0], v[1]);
                f.ins().bor(t, v[2])
            },
            reference: |a, b, c| (a & b) | c,
        },
        Shape {
            name: "andor_commuted", // bor(c,band(a,b)): 0xEA
            imm8: 0xEA,
            build: |f, v| {
                let t = f.ins().band(v[0], v[1]);
                f.ins().bor(v[2], t)
            },
            reference: |a, b, c| (a & b) | c,
        },
        Shape {
            name: "orand", // band(bor(a,b),c): (0xF0|0xCC)&0xAA = 0xA8
            imm8: 0xA8,
            build: |f, v| {
                let t = f.ins().bor(v[0], v[1]);
                f.ins().band(t, v[2])
            },
            reference: |a, b, c| (a | b) & c,
        },
        Shape {
            name: "orand_commuted", // band(c,bor(a,b)): 0xA8
            imm8: 0xA8,
            build: |f, v| {
                let t = f.ins().bor(v[0], v[1]);
                f.ins().band(v[2], t)
            },
            reference: |a, b, c| (a | b) & c,
        },
    ]
}

/// Assert the disassembly contains exactly one vpternlog, with the expected
/// D/Q mnemonic for the type and the expected imm8.
fn assert_single_vpternlog(disasm: &str, ty: Type, imm8: u8, what: &str) {
    let count = disasm.matches("vpternlog").count();
    assert_eq!(count, 1, "{what}: expected exactly 1 vpternlog, got {count}:\n{disasm}");
    let mnemonic = if ty == I64X8 { "vpternlogq" } else { "vpternlogd" };
    let expected = format!("{mnemonic} $0x{imm8:x}");
    assert!(
        disasm.contains(&expected),
        "{what}: expected `{expected}` in disassembly:\n{disasm}"
    );
}

fn all_types() -> [Type; 4] {
    [I8X64, I16X32, I32X16, I64X8]
}

#[test]
fn test_ternlog_fused_shapes_all_types() {
    let Some(mut compiler) = TernCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let pats = patterns();
    for ty in all_types() {
        for shape in shapes() {
            let name = format!("tern_{}_{ty}", shape.name);
            let build = shape.build;
            let (code, disasm) = compiler.compile(&name, ty, 3, |f, v| build(f, v));
            assert_single_vpternlog(&disasm, ty, shape.imm8, &name);

            for (ai, a) in pats.iter().enumerate() {
                for (bi, b) in pats.iter().enumerate() {
                    for (ci, c) in pats.iter().enumerate() {
                        let got = run3(code, a, b, c);
                        let want = ref3(shape.reference, a, b, c);
                        assert_eq!(
                            got, want,
                            "{name}: mismatch for pattern combo ({ai},{bi},{ci})"
                        );
                    }
                }
            }
        }
    }
}

/// Aliased inputs: the same SSA value feeds two of the three VPTERNLOG
/// operands (including the tied first operand). These are the shapes most
/// likely to expose tie/aliasing bugs in the RMW modeling.
#[test]
fn test_ternlog_aliased_inputs() {
    let Some(mut compiler) = TernCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    struct AliasCase {
        name: &'static str,
        imm8: u8,
        build: fn(&mut FunctionBuilder, &[Value]) -> Value,
        reference: fn(u8, u8) -> u8,
    }
    let cases = [
        AliasCase {
            // a ^ a ^ c: A and B are the same register (result == c).
            name: "xor_aac",
            imm8: 0x96,
            build: |f, v| {
                let t = f.ins().bxor(v[0], v[0]);
                f.ins().bxor(t, v[1])
            },
            reference: |_a, c| c,
        },
        AliasCase {
            // (a & a) | c: A and B are the same register (result == a|c).
            name: "and_aa_or_c",
            imm8: 0xEA,
            build: |f, v| {
                let t = f.ins().band(v[0], v[0]);
                f.ins().bor(t, v[1])
            },
            reference: |a, c| a | c,
        },
        AliasCase {
            // (a | b) & a: A (tied) is also C (result == a).
            name: "orand_aba",
            imm8: 0xA8,
            build: |f, v| {
                let t = f.ins().bor(v[0], v[1]);
                f.ins().band(t, v[0])
            },
            reference: |a, _b| a,
        },
        AliasCase {
            // (a & b) | b: C aliases B (result == b).
            name: "andor_abb",
            imm8: 0xEA,
            build: |f, v| {
                let t = f.ins().band(v[0], v[1]);
                f.ins().bor(t, v[1])
            },
            reference: |_a, b| b,
        },
    ];

    let pats = patterns();
    for ty in all_types() {
        for case in &cases {
            let name = format!("alias_{}_{ty}", case.name);
            let build = case.build;
            let (code, disasm) = compiler.compile(&name, ty, 2, |f, v| build(f, v));
            assert_single_vpternlog(&disasm, ty, case.imm8, &name);

            for (ai, a) in pats.iter().enumerate() {
                for (bi, b) in pats.iter().enumerate() {
                    let got = run2(code, a, b);
                    let mut want = V512([0; 64]);
                    for i in 0..64 {
                        want.0[i] = (case.reference)(a.0[i], b.0[i]);
                    }
                    assert_eq!(got, want, "{name}: mismatch for pattern combo ({ai},{bi})");
                }
            }
        }
    }
}

/// The tied operand staying live after the vpternlog must be preserved via a
/// regalloc copy; verify value correctness of both the fused result and a
/// later use of the tied input.
#[test]
fn test_ternlog_tied_operand_live_after() {
    let Some(mut compiler) = TernCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // out = (a ^ b ^ c) | a; the outer bor keeps `a` (the tied operand of
    // the fused inner xor3 tree) live after the vpternlog, forcing regalloc
    // to copy it. The outer bor cannot fuse (its inner op is a bxor, which
    // no bor-rooted fusion rule matches), so exactly one vpternlog remains.
    let (code, disasm) = compiler.compile("tied_live_after", I32X16, 3, |f, v| {
        let t = f.ins().bxor(v[0], v[1]);
        let x3 = f.ins().bxor(t, v[2]);
        f.ins().bor(x3, v[0])
    });
    assert_eq!(
        disasm.matches("vpternlog").count(),
        1,
        "expected exactly 1 vpternlog:\n{disasm}"
    );

    let pats = patterns();
    for a in &pats {
        for b in &pats {
            for c in &pats {
                let got = run3(code, a, b, c);
                let want = ref3(|a, b, c| (a ^ b ^ c) | a, a, b, c);
                assert_eq!(got, want, "tied-live-after mismatch");
            }
        }
    }
}
