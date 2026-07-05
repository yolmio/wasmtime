#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for AVX-512 gather/scatter with a non-zero
//! displacement (the `offset` immediate of `x86_simd_scatter` /
//! `x86_simd_gather`).
//!
//! Regression tests for a miscompile in scatter displacement encoding:
//! EVEX uses a *compressed* 8-bit displacement (disp8*N, Intel SDM Vol. 2A
//! section 2.7.5) where the emitted byte is `disp / element_size` and the
//! hardware multiplies it back by the element size. The scatter emission
//! path used to emit the raw displacement as disp8, so a scatter with
//! offset +8 and 4-byte elements actually stored at +32.
//!
//! The scatter tests therefore assert both that the values land at the
//! correct addresses and that *nothing else* in a guard region around them
//! was written.

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

#[repr(C, align(64))]
struct I32x16([i32; 16]);

#[repr(C, align(64))]
struct I64x8([i64; 8]);

/// Build and finalize a function `fn(base, indices, values, mask)` that
/// performs `x86_simd_scatter` of `vec_ty` values with the given scatter
/// `offset`, and return the code pointer.
fn compile_scatter(module: &mut JITModule, vec_ty: Type, name: &str, offset: i32) -> *const u8 {
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr
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

        let indices = builder.ins().load(vec_ty, MemFlags::trusted(), indices_ptr, 0);
        let values = builder.ins().load(vec_ty, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(vec_ty, MemFlags::trusted(), mask_ptr, 0);

        builder.ins().x86_simd_scatter(
            MemFlags::trusted(),
            mask,
            values,
            base_ptr,
            indices,
            1u8, // scale: indices are byte offsets
            offset,
        );

        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    module.get_finalized_function(func_id)
}

/// Build and finalize a function `fn(base, indices, out)` that performs
/// `x86_simd_gather` of `vec_ty` values with the given gather `offset`, and
/// return the code pointer.
fn compile_gather(module: &mut JITModule, vec_ty: Type, name: &str, offset: i32) -> *const u8 {
    let mut ctx = module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    let mut sig = module.make_signature();
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr
    sig.params.push(AbiParam::new(ptr_type)); // out ptr
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

        let indices = builder.ins().load(vec_ty, MemFlags::trusted(), indices_ptr, 0);

        let gathered = builder.ins().x86_simd_gather(
            vec_ty,
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8, // scale: indices are byte offsets
            offset,
        );

        builder
            .ins()
            .store(MemFlags::trusted(), gathered, out_ptr, 0);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    module.get_finalized_function(func_id)
}

const SENTINEL: i32 = 0x5EADBEEFu32 as i32;
const SENTINEL64: i64 = 0x5EADBEEF5EADBEEFu64 as i64;

/// Scatter 16 i32 values with offset +8 (VPSCATTERDD, 4-byte elements).
///
/// With the compressed-displacement bug, offset +8 was emitted as a raw
/// disp8 of 8, which the hardware scales by the element size (4) to +32:
/// every value landed 24 bytes past its target and the guard check below
/// fails.
#[test]
fn test_scatter_dd_offset_plus8_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter(&mut module, I32X16, "scatter_dd_off8", 8);

    // Buffer with a generous guard region on both sides of the target area.
    // The scatter writes through `base` which points into the middle.
    #[repr(C, align(64))]
    struct Buf([i32; 96]);
    let mut buf = Buf([SENTINEL; 96]);
    let base_elem = 32; // base points at &buf.0[32]

    // Byte-offset indices for 16 lanes: element strides of 8 bytes, i.e.
    // every second i32 slot: 0, 8, 16, ..., 120.
    let mut indices = [0i32; 16];
    for (i, idx) in indices.iter_mut().enumerate() {
        *idx = (i as i32) * 8;
    }
    let indices = I32x16(indices);
    let values = I32x16(std::array::from_fn(|i| 1000 + i as i32));
    let mask = I32x16([-1; 16]); // all lanes active

    type ScatterFn = unsafe extern "C" fn(*mut i32, *const I32x16, *const I32x16, *const I32x16);
    let func: ScatterFn = unsafe { mem::transmute(code) };
    unsafe {
        func(
            buf.0.as_mut_ptr().add(base_elem),
            &indices,
            &values,
            &mask,
        );
    }

    // Lane i writes 4 bytes at base + i*8 + 8, i.e. element base_elem + 2*i + 2.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem >= base_elem + 2
            && (elem - base_elem - 2) % 2 == 0
            && (elem - base_elem - 2) / 2 < 16
        {
            1000 + ((elem - base_elem - 2) / 2) as i32
        } else {
            SENTINEL
        };
        assert_eq!(
            actual, expected,
            "buffer element {elem} (byte offset {} from base)",
            (elem as isize - base_elem as isize) * 4
        );
    }
}

/// Scatter 8 i64 values with offset +8 (VPSCATTERQQ, 8-byte elements,
/// compressed disp8 = 1).
#[test]
fn test_scatter_qq_offset_plus8_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter(&mut module, I64X8, "scatter_qq_off8", 8);

    #[repr(C, align(64))]
    struct Buf([i64; 48]);
    let mut buf = Buf([SENTINEL64; 48]);
    let base_elem = 16;

    // Byte-offset indices: 0, 16, 32, ..., 112 (every second i64 slot).
    let indices = I64x8(std::array::from_fn(|i| (i as i64) * 16));
    let values = I64x8(std::array::from_fn(|i| 2000 + i as i64));
    let mask = I64x8([-1; 8]);

    type ScatterFn = unsafe extern "C" fn(*mut i64, *const I64x8, *const I64x8, *const I64x8);
    let func: ScatterFn = unsafe { mem::transmute(code) };
    unsafe {
        func(
            buf.0.as_mut_ptr().add(base_elem),
            &indices,
            &values,
            &mask,
        );
    }

    // Lane i writes 8 bytes at base + i*16 + 8, i.e. element base_elem + 2*i + 1.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem > base_elem
            && (elem - base_elem - 1) % 2 == 0
            && (elem - base_elem - 1) / 2 < 8
        {
            2000 + ((elem - base_elem - 1) / 2) as i64
        } else {
            SENTINEL64
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}

/// Scatter with an offset too large for a compressed disp8 (+512 with
/// 4-byte elements scales to 128, which does not fit in an i8), exercising
/// the disp32 encoding path at runtime.
#[test]
fn test_scatter_dd_offset_512_disp32_exact_addresses() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_scatter(&mut module, I32X16, "scatter_dd_off512", 512);

    #[repr(C, align(64))]
    struct Buf([i32; 192]);
    let mut buf = Buf([SENTINEL; 192]);
    let base_elem = 16; // +512 bytes = +128 elements from base

    let indices = I32x16(std::array::from_fn(|i| (i as i32) * 4));
    let values = I32x16(std::array::from_fn(|i| 3000 + i as i32));
    let mask = I32x16([-1; 16]);

    type ScatterFn = unsafe extern "C" fn(*mut i32, *const I32x16, *const I32x16, *const I32x16);
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
        let expected = if elem >= base_elem + 128 && elem < base_elem + 128 + 16 {
            3000 + (elem - base_elem - 128) as i32
        } else {
            SENTINEL
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}

/// Gather with offset +8 (VPGATHERDD): roundtrip against a known data
/// pattern. Regression guard for the same compressed-displacement rules on
/// the gather side.
#[test]
fn test_gather_dd_offset_plus8_roundtrip() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let code = compile_gather(&mut module, I32X16, "gather_dd_off8", 8);

    #[repr(C, align(64))]
    struct Data([i32; 64]);
    let data = Data(std::array::from_fn(|i| 100 + i as i32));

    // Byte offsets 0, 8, 16, ..., 120 -> elements 0, 2, 4, ..., 30.
    let indices = I32x16(std::array::from_fn(|i| (i as i32) * 8));
    let mut out = I32x16([0; 16]);

    type GatherFn = unsafe extern "C" fn(*const i32, *const I32x16, *mut I32x16);
    let func: GatherFn = unsafe { mem::transmute(code) };
    unsafe {
        func(data.0.as_ptr(), &indices, &mut out);
    }

    // Lane i reads 4 bytes at base + i*8 + 8, i.e. element 2*i + 2.
    for i in 0..16 {
        assert_eq!(
            out.0[i],
            100 + (2 * i as i32 + 2),
            "gathered lane {i} (expected element {})",
            2 * i + 2
        );
    }
}

/// Gather/scatter roundtrip with offset +8: scatter values out with an
/// offset, then gather them back with the same offset; the roundtrip must
/// be the identity.
#[test]
fn test_scatter_gather_offset_roundtrip() {
    let Some(mut module) = jit_module_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };
    let scatter = compile_scatter(&mut module, I32X16, "rt_scatter_off8", 8);
    let gather = compile_gather(&mut module, I32X16, "rt_gather_off8", 8);

    #[repr(C, align(64))]
    struct Buf([i32; 64]);
    let mut buf = Buf([SENTINEL; 64]);

    let indices = I32x16(std::array::from_fn(|i| (i as i32) * 12));
    let values = I32x16(std::array::from_fn(|i| 4000 + i as i32));
    let mask = I32x16([-1; 16]);
    let mut out = I32x16([0; 16]);

    type ScatterFn = unsafe extern "C" fn(*mut i32, *const I32x16, *const I32x16, *const I32x16);
    type GatherFn = unsafe extern "C" fn(*const i32, *const I32x16, *mut I32x16);
    let scatter: ScatterFn = unsafe { mem::transmute(scatter) };
    let gather: GatherFn = unsafe { mem::transmute(gather) };

    unsafe {
        scatter(buf.0.as_mut_ptr(), &indices, &values, &mask);
        gather(buf.0.as_ptr(), &indices, &mut out);
    }

    assert_eq!(out.0, values.0, "scatter+gather with offset must roundtrip");

    // And the scatter side must have hit exactly the expected slots:
    // lane i writes at byte i*12 + 8; only offsets divisible by 4 are
    // i32-aligned slots. i*12 + 8 = 4 * (3*i + 2) -> element 3*i + 2.
    for (elem, &actual) in buf.0.iter().enumerate() {
        let expected = if elem >= 2 && (elem - 2) % 3 == 0 && (elem - 2) / 3 < 16 {
            4000 + ((elem - 2) / 3) as i32
        } else {
            SENTINEL
        };
        assert_eq!(actual, expected, "buffer element {elem}");
    }
}
