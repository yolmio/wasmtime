#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for the mixed-width AVX-512 gather/scatter forms
//! VPGATHERDQ / VPSCATTERDQ: dword (i32x8) indices addressing qword (i64x8)
//! data, with non-zero displacements.
//!
//! The same-width DD/QQ forms have runtime coverage in
//! `avx512_scatter_offset.rs` and `avx512_e2e.rs` (which also has a
//! zero-offset DQ gather); this file locks down the DQ forms with offsets
//! +8, -8 and +1024. The EVEX compressed displacement (disp8*N, Intel SDM
//! Vol. 2A section 2.7.5) scales by the *element* size (8 for DQ, not the
//! 4-byte index size), so a bug conflating index and element width shows up
//! exactly here. Scatter tests assert both that every value lands at the
//! correct address and that nothing else in a sentinel-filled guard region
//! around the targets was written.

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

    // The native builder auto-detects AVX-512 features.
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

#[repr(C, align(32))]
struct I32x8([i32; 8]);

#[repr(C, align(64))]
struct I64x8([i64; 8]);

const SENTINEL64: i64 = 0x5EADBEEF5EADBEEFu64 as i64;

/// Build and finalize `fn(base, indices_ptr, out_ptr)` performing an
/// `x86_simd_gather.i64x8` with i32x8 indices (VPGATHERDQ) and the given
/// scale/offset; returns the code pointer. Asserts the VCode actually
/// selected the DQ form.
fn compile_gather_dq(module: &mut JITModule, name: &str, scale: u8, offset: i32) -> *const u8 {
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I32X8)
    sig.params.push(AbiParam::new(ptr_type)); // out ptr (I64X8)
    sig.call_conv = CallConv::SystemV;

    let func_id = module.declare_function(name, Linkage::Local, &sig).unwrap();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let out_ptr = params[2];

        let indices = builder
            .ins()
            .load(I32X8, MemFlags::trusted(), indices_ptr, 0);

        let gathered = builder.ins().x86_simd_gather(
            I64X8,
            MemFlags::trusted(),
            base_ptr,
            indices,
            scale,
            offset,
        );

        builder
            .ins()
            .store(MemFlags::trusted(), gathered, out_ptr, 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    ctx.set_disasm(true);
    module.define_function(func_id, &mut ctx).unwrap();
    if let Some(compiled) = ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            assert!(
                disasm.contains("vpgatherdq"),
                "expected VPGATHERDQ for i32x8 indices + i64x8 data:\n{disasm}"
            );
        }
    }
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    module.get_finalized_function(func_id)
}

/// Build and finalize `fn(base, indices_ptr, values_ptr, mask_ptr)`
/// performing an `x86_simd_scatter` of i64x8 data with i32x8 indices
/// (VPSCATTERDQ) and the given scale/offset; returns the code pointer.
/// The mask is an i64x8 vector (same type as the data, per the CLIF mask
/// convention). Asserts the VCode actually selected the DQ form.
fn compile_scatter_dq(module: &mut JITModule, name: &str, scale: u8, offset: i32) -> *const u8 {
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I32X8)
    sig.params.push(AbiParam::new(ptr_type)); // values ptr (I64X8)
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I64X8)
    sig.call_conv = CallConv::SystemV;

    let func_id = module.declare_function(name, Linkage::Local, &sig).unwrap();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let values_ptr = params[2];
        let mask_ptr = params[3];

        let indices = builder
            .ins()
            .load(I32X8, MemFlags::trusted(), indices_ptr, 0);
        let values = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);

        builder.ins().x86_simd_scatter(
            MemFlags::trusted(),
            mask,
            values,
            base_ptr,
            indices,
            scale,
            offset,
        );

        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    ctx.set_disasm(true);
    module.define_function(func_id, &mut ctx).unwrap();
    if let Some(compiled) = ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            assert!(
                disasm.contains("vpscatterdq"),
                "expected VPSCATTERDQ for i32x8 indices + i64x8 data:\n{disasm}"
            );
        }
    }
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    module.get_finalized_function(func_id)
}

type GatherFn = unsafe extern "C" fn(*const i64, *const I32x8, *mut I64x8);
type ScatterFn = unsafe extern "C" fn(*mut i64, *const I32x8, *const I64x8, *const I64x8);

/// VPGATHERDQ with offset +8 (compressed disp8 = 1 with 8-byte elements):
/// lane i reads the qword at base + i*16 + 8, i.e. element 2*i + 1.
#[test]
fn test_gather_dq_offset_plus8() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_gather_dq(&mut module, "gather_dq_off8", 1, 8);

    #[repr(C, align(64))]
    struct Data([i64; 32]);
    let data = Data(std::array::from_fn(|i| 100 + i as i64));

    // Byte offsets 0, 16, 32, ..., 112.
    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 16));
    let mut out = I64x8([0; 8]);

    let func: GatherFn = unsafe { mem::transmute(code) };
    unsafe { func(data.0.as_ptr(), &indices, &mut out) };

    for i in 0..8 {
        assert_eq!(
            out.0[i],
            100 + (2 * i as i64 + 1),
            "gathered lane {i} (expected element {})",
            2 * i + 1
        );
    }
}

/// VPGATHERDQ with offset -8 (negative compressed disp8): base points into
/// the middle of the data so lane i reads element 16 + 2*i - 1.
#[test]
fn test_gather_dq_offset_minus8() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_gather_dq(&mut module, "gather_dq_offm8", 1, -8);

    #[repr(C, align(64))]
    struct Data([i64; 48]);
    let data = Data(std::array::from_fn(|i| 200 + i as i64));
    let base_elem = 16;

    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 16));
    let mut out = I64x8([0; 8]);

    let func: GatherFn = unsafe { mem::transmute(code) };
    unsafe { func(data.0.as_ptr().add(base_elem), &indices, &mut out) };

    for i in 0..8 {
        let expected_elem = base_elem + 2 * i - 1;
        assert_eq!(
            out.0[i],
            200 + expected_elem as i64,
            "gathered lane {i} (expected element {expected_elem})"
        );
    }
}

/// VPGATHERDQ with scale 8 (element indices instead of byte offsets) and
/// offset -8: lane i reads element base_elem + 2*i - 1.
#[test]
fn test_gather_dq_scale8_offset_minus8() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_gather_dq(&mut module, "gather_dq_s8_offm8", 8, -8);

    #[repr(C, align(64))]
    struct Data([i64; 32]);
    let data = Data(std::array::from_fn(|i| 300 + i as i64));
    let base_elem = 4;

    // Element indices 0, 2, 4, ..., 14 (scaled by 8 in the address).
    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 2));
    let mut out = I64x8([0; 8]);

    let func: GatherFn = unsafe { mem::transmute(code) };
    unsafe { func(data.0.as_ptr().add(base_elem), &indices, &mut out) };

    for i in 0..8 {
        let expected_elem = base_elem + 2 * i - 1;
        assert_eq!(
            out.0[i],
            300 + expected_elem as i64,
            "gathered lane {i} (expected element {expected_elem})"
        );
    }
}

/// VPSCATTERDQ with offset +8: lane i writes the qword at base + i*16 + 8,
/// i.e. element base_elem + 2*i + 1; everything else in the guard region
/// must keep its sentinel.
#[test]
fn test_scatter_dq_offset_plus8_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter_dq(&mut module, "scatter_dq_off8", 1, 8);

    #[repr(C, align(64))]
    struct Buf([i64; 48]);
    let mut buf = Buf([SENTINEL64; 48]);
    let base_elem = 16;

    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 16));
    let values = I64x8(std::array::from_fn(|i| 1000 + i as i64));
    let mask = I64x8([-1; 8]); // all lanes active

    let func: ScatterFn = unsafe { mem::transmute(code) };
    unsafe {
        func(
            buf.0.as_mut_ptr().add(base_elem),
            &indices,
            &values,
            &mask,
        );
    }

    // Lane i writes at element base_elem + 2*i + 1.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem > base_elem
            && (elem - base_elem - 1) % 2 == 0
            && (elem - base_elem - 1) / 2 < 8
        {
            1000 + ((elem - base_elem - 1) / 2) as i64
        } else {
            SENTINEL64
        };
        assert_eq!(
            actual, expected,
            "buffer element {elem} (byte offset {} from base)",
            (elem as isize - base_elem as isize) * 8
        );
    }
}

/// VPSCATTERDQ with offset -8 and an alternating mask: only even lanes
/// write, at element base_elem + 2*i - 1; masked-off lanes must not write.
#[test]
fn test_scatter_dq_offset_minus8_masked_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter_dq(&mut module, "scatter_dq_offm8", 1, -8);

    #[repr(C, align(64))]
    struct Buf([i64; 48]);
    let mut buf = Buf([SENTINEL64; 48]);
    let base_elem = 16;

    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 16));
    let values = I64x8(std::array::from_fn(|i| 2000 + i as i64));
    // Lanes 0, 2, 4, 6 active; 1, 3, 5, 7 masked off.
    let mask = I64x8(std::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 }));

    let func: ScatterFn = unsafe { mem::transmute(code) };
    unsafe {
        func(
            buf.0.as_mut_ptr().add(base_elem),
            &indices,
            &values,
            &mask,
        );
    }

    // Active lane i (even) writes at element base_elem + 2*i - 1.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem + 1 >= base_elem
            && (elem + 1 - base_elem) % 2 == 0
            && (elem + 1 - base_elem) / 2 < 8
            && ((elem + 1 - base_elem) / 2) % 2 == 0
        {
            2000 + ((elem + 1 - base_elem) / 2) as i64
        } else {
            SENTINEL64
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}

/// VPSCATTERDQ with offset +1024: 1024/8 = 128 does not fit in a signed
/// disp8, exercising the disp32 encoding path (and the guard region proves
/// the displacement was not truncated or mis-scaled).
#[test]
fn test_scatter_dq_offset_1024_disp32_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter_dq(&mut module, "scatter_dq_off1024", 1, 1024);

    #[repr(C, align(64))]
    struct Buf([i64; 160]);
    let mut buf = Buf([SENTINEL64; 160]);
    let base_elem = 8; // +1024 bytes = +128 elements from base

    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 8));
    let values = I64x8(std::array::from_fn(|i| 3000 + i as i64));
    let mask = I64x8([-1; 8]);

    let func: ScatterFn = unsafe { mem::transmute(code) };
    unsafe {
        func(
            buf.0.as_mut_ptr().add(base_elem),
            &indices,
            &values,
            &mask,
        );
    }

    // Lane i writes at element base_elem + 128 + i.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem >= base_elem + 128 && elem < base_elem + 128 + 8 {
            3000 + (elem - base_elem - 128) as i64
        } else {
            SENTINEL64
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}

/// Scatter/gather DQ roundtrip with offset +8: scattering through dword
/// indices and gathering back with the same indices and offset must be the
/// identity.
#[test]
fn test_scatter_gather_dq_offset_roundtrip() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let scatter = compile_scatter_dq(&mut module, "rt_scatter_dq_off8", 1, 8);
    let gather = compile_gather_dq(&mut module, "rt_gather_dq_off8", 1, 8);

    #[repr(C, align(64))]
    struct Buf([i64; 48]);
    let mut buf = Buf([SENTINEL64; 48]);

    let indices = I32x8(std::array::from_fn(|i| (i as i32) * 24));
    let values = I64x8(std::array::from_fn(|i| 4000 + i as i64));
    let mask = I64x8([-1; 8]);
    let mut out = I64x8([0; 8]);

    let scatter: ScatterFn = unsafe { mem::transmute(scatter) };
    let gather: GatherFn = unsafe { mem::transmute(gather) };

    unsafe {
        scatter(buf.0.as_mut_ptr(), &indices, &values, &mask);
        gather(buf.0.as_ptr(), &indices, &mut out);
    }

    assert_eq!(out.0, values.0, "scatter+gather DQ with offset must roundtrip");

    // Lane i wrote at byte i*24 + 8 = 8 * (3*i + 1) -> element 3*i + 1.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem >= 1 && (elem - 1) % 3 == 0 && (elem - 1) / 3 < 8 {
            4000 + ((elem - 1) / 3) as i64
        } else {
            SENTINEL64
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}
