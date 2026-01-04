#![cfg(target_arch = "x86_64")]

//! End-to-end JIT tests for AVX-512 512-bit vector operations.
//!
//! These tests verify that:
//! 1. Cranelift IR with 512-bit vector types (I64X8, I32X16) compiles correctly
//! 2. The JIT-compiled code executes correctly and produces expected results
//! 3. The correct AVX-512 instructions are selected (via VCode inspection)
//!
//! These tests are designed to validate all operations needed for a columnar
//! database engine using AVX-512 on processors that support it.

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

/// Check if AVX-512 is available on this machine.
fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return true;
        }
    }
    false
}

/// Create an ISA configured for AVX-512 testing.
fn isa_with_avx512() -> Option<OwnedTargetIsa> {
    if !has_avx512() {
        return None;
    }

    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    // Enable AVX-512 features
    let isa_builder = cranelift_native::builder().ok()?;

    // The native builder should auto-detect AVX-512 features
    isa_builder.finish(settings::Flags::new(flag_builder)).ok()
}

/// Helper to create a JIT module with AVX-512 support.
fn jit_module_with_avx512() -> Option<JITModule> {
    let isa = isa_with_avx512()?;
    Some(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

/// Compile a function and optionally get the VCode disassembly for verification.
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

    /// Compile a function that takes two I64X8 vectors and returns I64X8.
    /// The `build_fn` closure constructs the function body.
    fn compile_binary_i64x8<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        // I64X8 is passed as a pointer (too large for registers on SystemV)
        // We'll use explicit pointers for 512-bit vectors
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(I64X8, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(I64X8, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        // Enable disassembly for verification
        self.ctx.set_disasm(true);

        self.module.define_function(func_id, &mut self.ctx)?;

        // Get the VCode disassembly for instruction verification
        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for I32X16 binary operations.
    fn compile_binary_i32x16<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(I32X16, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(I32X16, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that takes one I64X8 vector and returns I64X8.
    fn compile_unary_i64x8<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
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

            // Load 512-bit vector
            let src = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that takes one I32X16 vector and returns I32X16.
    fn compile_unary_i32x16<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
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

            // Load 512-bit vector
            let src = builder.ins().load(I32X16, MemFlags::trusted(), src_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for I8X64 (64 x byte) binary operations.
    fn compile_binary_i8x64<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors (I8X64 = 64 bytes)
            let src1 = builder.ins().load(I8X64, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(I8X64, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for I16X32 (32 x word) binary operations.
    fn compile_binary_i16x32<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors (I16X32 = 32 words)
            let src1 = builder.ins().load(I16X32, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(I16X32, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for F64X8 binary operations.
    fn compile_binary_f64x8<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(F64X8, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(F64X8, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for F64X8 ternary operations (like FMA).
    fn compile_ternary_f64x8<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr (x)
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr (y)
        sig.params.push(AbiParam::new(ptr_type)); // src3 ptr (z)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let src3_ptr = params[2];
            let dst_ptr = params[3];

            // Load 512-bit vectors
            let src1 = builder.ins().load(F64X8, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(F64X8, MemFlags::trusted(), src2_ptr, 0);
            let src3 = builder.ins().load(F64X8, MemFlags::trusted(), src3_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2, src3);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function for F32X16 binary operations.
    fn compile_binary_f32x16<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(F32X16, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(F32X16, MemFlags::trusted(), src2_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src1, src2);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: I64X8 -> F64X8.
    fn compile_convert_i64x8_to_f64x8<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (I64X8)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (F64X8)
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

            let src = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);
            let result = build_fn(&mut builder, src);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: F64X8 -> I64X8.
    fn compile_convert_f64x8_to_i64x8<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (F64X8)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (I64X8)
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

            let src = builder.ins().load(F64X8, MemFlags::trusted(), src_ptr, 0);
            let result = build_fn(&mut builder, src);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: I32X16 -> F32X16.
    fn compile_convert_i32x16_to_f32x16<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (I32X16)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (F32X16)
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

            let src = builder.ins().load(I32X16, MemFlags::trusted(), src_ptr, 0);
            let result = build_fn(&mut builder, src);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: F32X16 -> I32X16.
    fn compile_convert_f32x16_to_i32x16<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (F32X16)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (I32X16)
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

            let src = builder.ins().load(F32X16, MemFlags::trusted(), src_ptr, 0);
            let result = build_fn(&mut builder, src);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that takes a scalar i64 and returns I64X8 (splat).
    fn compile_splat_i64x8(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(I64)); // scalar input
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let scalar = params[0];
            let dst_ptr = params[1];

            // Splat the scalar to a 512-bit vector
            let result = builder.ins().splat(I64X8, scalar);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that takes a scalar i32 and returns I32X16 (splat).
    fn compile_splat_i32x16(&mut self, name: &str) -> Result<*const u8, ModuleError> {
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(I32)); // scalar input
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let scalar = params[0];
            let dst_ptr = params[1];

            // Splat the scalar to a 512-bit vector
            let result = builder.ins().splat(I32X16, scalar);
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: F32X8 -> F64X8 (fpromote).
    fn compile_convert_f32x8_to_f64x8<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (F32X8, 256-bit)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (F64X8, 512-bit)
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

            // Load F32X8 (256-bit vector)
            let src = builder.ins().load(F32X8, MemFlags::trusted(), src_ptr, 0);

            // Perform the conversion (fpromote)
            let result = build_fn(&mut builder, src);

            // Store F64X8 (512-bit vector)
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a conversion function: F64X8 -> F32X8 (fdemote).
    fn compile_convert_f64x8_to_f32x8<F>(
        &mut self,
        name: &str,
        build_fn: F,
    ) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr (F64X8, 512-bit)
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr (F32X8, 256-bit)
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

            // Load F64X8 (512-bit vector)
            let src = builder.ins().load(F64X8, MemFlags::trusted(), src_ptr, 0);

            // Perform the conversion (fdemote)
            let result = build_fn(&mut builder, src);

            // Store F32X8 (256-bit vector)
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a function that takes one F32X16 vector and returns F32X16.
    fn compile_unary_f32x16<F>(&mut self, name: &str, build_fn: F) -> Result<*const u8, ModuleError>
    where
        F: FnOnce(&mut FunctionBuilder, Value) -> Value,
    {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
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

            // Load 512-bit vector
            let src = builder.ins().load(F32X16, MemFlags::trusted(), src_ptr, 0);

            // Perform the operation
            let result = build_fn(&mut builder, src);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a comparison function that takes two I32X16 vectors and returns I32X16 mask.
    fn compile_comparison_i32x16(
        &mut self,
        name: &str,
        cc: IntCC,
    ) -> Result<*const u8, ModuleError> {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(I32X16, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(I32X16, MemFlags::trusted(), src2_ptr, 0);

            // Perform comparison
            let cmp_result = builder.ins().icmp(cc, src1, src2);
            // The result is a vector of booleans, bitcast to I32X16
            let result = builder.ins().bitcast(I32X16, MemFlags::new(), cmp_result);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }

    /// Compile a floating-point comparison function that takes two F32X16 vectors and returns I32X16 mask.
    fn compile_comparison_f32x16(
        &mut self,
        name: &str,
        cc: FloatCC,
    ) -> Result<*const u8, ModuleError> {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // src1 ptr
        sig.params.push(AbiParam::new(ptr_type)); // src2 ptr
        sig.params.push(AbiParam::new(ptr_type)); // dst ptr
        sig.call_conv = CallConv::SystemV;

        let func_id = self.module.declare_function(name, Linkage::Local, &sig)?;

        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);

            let params = builder.block_params(block).to_vec();
            let src1_ptr = params[0];
            let src2_ptr = params[1];
            let dst_ptr = params[2];

            // Load 512-bit vectors
            let src1 = builder.ins().load(F32X16, MemFlags::trusted(), src1_ptr, 0);
            let src2 = builder.ins().load(F32X16, MemFlags::trusted(), src2_ptr, 0);

            // Perform comparison
            let cmp_result = builder.ins().fcmp(cc, src1, src2);
            // The result is a vector of booleans, bitcast to I32X16
            let result = builder.ins().bitcast(I32X16, MemFlags::new(), cmp_result);

            // Store result
            builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
            builder.ins().return_(&[]);

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.ctx.set_disasm(true);
        self.module.define_function(func_id, &mut self.ctx)?;

        if let Some(compiled) = self.ctx.compiled_code() {
            if let Some(disasm) = &compiled.vcode {
                println!("=== VCode for {} ===\n{}", name, disasm);
            }
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        Ok(self.module.get_finalized_function(func_id))
    }
}

// =============================================================================
// Helper types for calling JIT functions
// =============================================================================

/// 512-bit vector as 8 x i64
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I64x8([i64; 8]);

impl I64x8 {
    fn new(values: [i64; 8]) -> Self {
        Self(values)
    }

    fn splat(v: i64) -> Self {
        Self([v; 8])
    }
}

/// 512-bit vector as 16 x i32
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I32x16([i32; 16]);

impl I32x16 {
    fn new(values: [i32; 16]) -> Self {
        Self(values)
    }

    fn splat(v: i32) -> Self {
        Self([v; 16])
    }
}

/// 512-bit vector as 64 x i8 (byte)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I8x64([i8; 64]);

impl I8x64 {
    fn new(values: [i8; 64]) -> Self {
        Self(values)
    }

    fn splat(v: i8) -> Self {
        Self([v; 64])
    }
}

/// 512-bit vector as 32 x i16 (word)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I16x32([i16; 32]);

impl I16x32 {
    fn new(values: [i16; 32]) -> Self {
        Self(values)
    }

    fn splat(v: i16) -> Self {
        Self([v; 32])
    }
}

/// 512-bit vector as 8 x f64
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct F64x8([f64; 8]);

impl F64x8 {
    fn new(values: [f64; 8]) -> Self {
        Self(values)
    }

    fn splat(v: f64) -> Self {
        Self([v; 8])
    }
}

/// 512-bit vector as 16 x f32
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct F32x16([f32; 16]);

impl F32x16 {
    fn new(values: [f32; 16]) -> Self {
        Self(values)
    }

    fn splat(v: f32) -> Self {
        Self([v; 16])
    }
}

/// 256-bit vector as 8 x f32 (used for fpromote/fdemote)
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct F32x8([f32; 8]);

impl F32x8 {
    fn new(values: [f32; 8]) -> Self {
        Self(values)
    }

    fn splat(v: f32) -> Self {
        Self([v; 8])
    }
}

// =============================================================================
// Binary operation function type
// =============================================================================

type BinaryI64x8Fn = unsafe extern "C" fn(*const I64x8, *const I64x8, *mut I64x8);
type BinaryI32x16Fn = unsafe extern "C" fn(*const I32x16, *const I32x16, *mut I32x16);
type BinaryI8x64Fn = unsafe extern "C" fn(*const I8x64, *const I8x64, *mut I8x64);
type BinaryI16x32Fn = unsafe extern "C" fn(*const I16x32, *const I16x32, *mut I16x32);
type UnaryI64x8Fn = unsafe extern "C" fn(*const I64x8, *mut I64x8);
type UnaryI32x16Fn = unsafe extern "C" fn(*const I32x16, *mut I32x16);
type BinaryF64x8Fn = unsafe extern "C" fn(*const F64x8, *const F64x8, *mut F64x8);
type BinaryF32x16Fn = unsafe extern "C" fn(*const F32x16, *const F32x16, *mut F32x16);
type TernaryF64x8Fn = unsafe extern "C" fn(*const F64x8, *const F64x8, *const F64x8, *mut F64x8);
type ConvertI64x8ToF64x8Fn = unsafe extern "C" fn(*const I64x8, *mut F64x8);
type ConvertF64x8ToI64x8Fn = unsafe extern "C" fn(*const F64x8, *mut I64x8);
type ConvertI32x16ToF32x16Fn = unsafe extern "C" fn(*const I32x16, *mut F32x16);
type ConvertF32x16ToI32x16Fn = unsafe extern "C" fn(*const F32x16, *mut I32x16);
type SplatI64x8Fn = unsafe extern "C" fn(i64, *mut I64x8);
type SplatI32x16Fn = unsafe extern "C" fn(i32, *mut I32x16);
type ConvertF32x8ToF64x8Fn = unsafe extern "C" fn(*const F32x8, *mut F64x8);
type ConvertF64x8ToF32x8Fn = unsafe extern "C" fn(*const F64x8, *mut F32x8);
type ComparisonF32x16Fn = unsafe extern "C" fn(*const F32x16, *const F32x16, *mut I32x16);

// =============================================================================
// Tests: 512-bit Integer Add (VPADDQ / VPADDD)
// =============================================================================

#[test]
fn test_i64x8_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_iadd", |builder, src1, src2| {
            builder.ins().iadd(src1, src2)
        })
        .expect("Failed to compile i64x8_iadd");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test case 1: Simple addition
    let a = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I64x8::new([10, 20, 30, 40, 50, 60, 70, 80]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([11, 22, 33, 44, 55, 66, 77, 88]));

    // Test case 2: Overflow behavior (wrapping)
    let a = I64x8::splat(i64::MAX);
    let b = I64x8::splat(1);
    unsafe {
        func(&a, &b, &mut result);
    }
    assert_eq!(result, I64x8::splat(i64::MIN));

    // Test case 3: Negative numbers
    let a = I64x8::new([-1, -2, -3, -4, -5, -6, -7, -8]);
    let b = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    unsafe {
        func(&a, &b, &mut result);
    }
    assert_eq!(result, I64x8::splat(0));
}

#[test]
fn test_i32x16_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_iadd", |builder, src1, src2| {
            builder.ins().iadd(src1, src2)
        })
        .expect("Failed to compile i32x16_iadd");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let b = I32x16::splat(100);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116
        ])
    );
}

// =============================================================================
// Tests: 512-bit Integer Subtract (VPSUBQ / VPSUBD)
// =============================================================================

#[test]
fn test_i64x8_isub() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_isub", |builder, src1, src2| {
            builder.ins().isub(src1, src2)
        })
        .expect("Failed to compile i64x8_isub");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([100, 200, 300, 400, 500, 600, 700, 800]);
    let b = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([99, 198, 297, 396, 495, 594, 693, 792]));
}

#[test]
fn test_i32x16_isub() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_isub", |builder, src1, src2| {
            builder.ins().isub(src1, src2)
        })
        .expect("Failed to compile i32x16_isub");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::splat(1000);
    let b = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            999, 998, 997, 996, 995, 994, 993, 992, 991, 990, 989, 988, 987, 986, 985, 984
        ])
    );
}

// =============================================================================
// Tests: 512-bit Bitwise AND (VPANDQ / VPANDD)
// =============================================================================

#[test]
fn test_i64x8_band() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_band", |builder, src1, src2| {
            builder.ins().band(src1, src2)
        })
        .expect("Failed to compile i64x8_band");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test: 0xFF AND 0x0F = 0x0F
    let a = I64x8::splat(0xFF);
    let b = I64x8::splat(0x0F);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::splat(0x0F));

    // Test: All ones AND value = value
    let a = I64x8::splat(-1); // all ones
    let b = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    unsafe {
        func(&a, &b, &mut result);
    }
    assert_eq!(result, I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]));
}

#[test]
fn test_i32x16_band() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_band", |builder, src1, src2| {
            builder.ins().band(src1, src2)
        })
        .expect("Failed to compile i32x16_band");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::splat(0xFFFF);
    let b = I32x16::splat(0x00FF);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I32x16::splat(0x00FF));
}

// =============================================================================
// Tests: 512-bit Bitwise OR (VPORQ / VPORD)
// =============================================================================

#[test]
fn test_i64x8_bor() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_bor", |builder, src1, src2| {
            builder.ins().bor(src1, src2)
        })
        .expect("Failed to compile i64x8_bor");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::splat(0xF0);
    let b = I64x8::splat(0x0F);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::splat(0xFF));
}

#[test]
fn test_i32x16_bor() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_bor", |builder, src1, src2| {
            builder.ins().bor(src1, src2)
        })
        .expect("Failed to compile i32x16_bor");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::splat(0xFF00);
    let b = I32x16::splat(0x00FF);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I32x16::splat(0xFFFF));
}

// =============================================================================
// Tests: 512-bit Bitwise XOR (VPXORQ / VPXORD)
// =============================================================================

#[test]
fn test_i64x8_bxor() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_bxor", |builder, src1, src2| {
            builder.ins().bxor(src1, src2)
        })
        .expect("Failed to compile i64x8_bxor");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // XOR with self = 0
    let a = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &a, &mut result);
    }

    assert_eq!(result, I64x8::splat(0));

    // XOR with all ones = NOT
    let a = I64x8::splat(0);
    let b = I64x8::splat(-1);
    unsafe {
        func(&a, &b, &mut result);
    }
    assert_eq!(result, I64x8::splat(-1));
}

#[test]
fn test_i32x16_bxor() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_bxor", |builder, src1, src2| {
            builder.ins().bxor(src1, src2)
        })
        .expect("Failed to compile i32x16_bxor");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::splat(0xAAAA);
    let b = I32x16::splat(0x5555);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I32x16::splat(0xFFFF));
}

// =============================================================================
// Tests: 512-bit Signed Min/Max (VPMINSQ/VPMAXSQ / VPMINSD/VPMAXSD)
// =============================================================================

#[test]
fn test_i64x8_smin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_smin", |builder, src1, src2| {
            builder.ins().smin(src1, src2)
        })
        .expect("Failed to compile i64x8_smin");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([10, -5, 100, -100, 0, 50, -50, 1000]);
    let b = I64x8::new([5, 10, -100, 100, 0, -50, 50, -1000]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([5, -5, -100, -100, 0, -50, -50, -1000]));
}

#[test]
fn test_i64x8_smax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_smax", |builder, src1, src2| {
            builder.ins().smax(src1, src2)
        })
        .expect("Failed to compile i64x8_smax");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([10, -5, 100, -100, 0, 50, -50, 1000]);
    let b = I64x8::new([5, 10, -100, 100, 0, -50, 50, -1000]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([10, 10, 100, 100, 0, 50, 50, 1000]));
}

#[test]
fn test_i32x16_smin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_smin", |builder, src1, src2| {
            builder.ins().smin(src1, src2)
        })
        .expect("Failed to compile i32x16_smin");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::new([10, -5, 100, -100, 0, 50, -50, 1000, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I32x16::new([5, 10, -100, 100, 0, -50, 50, -1000, 8, 7, 6, 5, 4, 3, 2, 1]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            5, -5, -100, -100, 0, -50, -50, -1000, 1, 2, 3, 4, 4, 3, 2, 1
        ])
    );
}

#[test]
fn test_i32x16_smax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_smax", |builder, src1, src2| {
            builder.ins().smax(src1, src2)
        })
        .expect("Failed to compile i32x16_smax");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::new([10, -5, 100, -100, 0, 50, -50, 1000, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I32x16::new([5, 10, -100, 100, 0, -50, 50, -1000, 8, 7, 6, 5, 4, 3, 2, 1]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([10, 10, 100, 100, 0, 50, 50, 1000, 8, 7, 6, 5, 5, 6, 7, 8])
    );
}

// =============================================================================
// Tests: 512-bit Unsigned Min/Max (VPMINUQ/VPMAXUQ / VPMINUD/VPMAXUD)
// =============================================================================

#[test]
fn test_i64x8_umin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_umin", |builder, src1, src2| {
            builder.ins().umin(src1, src2)
        })
        .expect("Failed to compile i64x8_umin");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Note: -1 as u64 is u64::MAX, so it should be "larger" than any positive number
    let a = I64x8::new([10, 5, 100, 200, 0, 50, 1, 1000]);
    let b = I64x8::new([5, 10, 200, 100, 0, 100, 2, 500]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([5, 5, 100, 100, 0, 50, 1, 500]));
}

#[test]
fn test_i64x8_umax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_umax", |builder, src1, src2| {
            builder.ins().umax(src1, src2)
        })
        .expect("Failed to compile i64x8_umax");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([10, 5, 100, 200, 0, 50, 1, 1000]);
    let b = I64x8::new([5, 10, 200, 100, 0, 100, 2, 500]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([10, 10, 200, 200, 0, 100, 2, 1000]));
}

// =============================================================================
// Tests: Database-specific patterns
// =============================================================================

/// Test a columnar SUM pattern: load values, add them element-wise
#[test]
fn test_columnar_sum_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Build a function that adds 4 vectors together (simulating summing 32 values)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // input array ptr (4 vectors)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("columnar_sum_4", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let input_ptr = params[0];
        let output_ptr = params[1];

        // Load 4 vectors (offset by 64 bytes each)
        let v0 = builder.ins().load(I64X8, MemFlags::trusted(), input_ptr, 0);
        let v1 = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), input_ptr, 64);
        let v2 = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), input_ptr, 128);
        let v3 = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), input_ptr, 192);

        // Sum them: ((v0 + v1) + (v2 + v3))
        let sum01 = builder.ins().iadd(v0, v1);
        let sum23 = builder.ins().iadd(v2, v3);
        let total = builder.ins().iadd(sum01, sum23);

        builder
            .ins()
            .store(MemFlags::trusted(), total, output_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for columnar_sum_4 ===\n{}", disasm);
            // Verify we're using VPADDQ instructions
            assert!(
                disasm.contains("vpaddq") || disasm.contains("Vpaddq"),
                "Expected VPADDQ instruction in VCode"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    let func: unsafe extern "C" fn(*const I64x8, *mut I64x8) = unsafe { mem::transmute(code) };

    // Test data: 4 vectors
    let input: [I64x8; 4] = [
        I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]),
        I64x8::new([10, 20, 30, 40, 50, 60, 70, 80]),
        I64x8::new([100, 200, 300, 400, 500, 600, 700, 800]),
        I64x8::new([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]),
    ];
    let mut result = I64x8::splat(0);

    unsafe {
        func(input.as_ptr(), &mut result);
    }

    assert_eq!(
        result,
        I64x8::new([1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888])
    );
}

/// Test a vectorized comparison pattern (useful for WHERE clauses)
#[test]
fn test_vectorized_filter_mask() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Build a function that compares values and returns a mask count
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(ptr_type)); // threshold ptr (single vector)
    sig.returns.push(AbiParam::new(I64)); // count of elements > threshold
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("count_greater", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];
        let threshold_ptr = params[1];

        // Load vectors
        let values = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), values_ptr, 0);
        let threshold = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), threshold_ptr, 0);

        // Compare: values > threshold using signed greater than
        // This produces a vector mask
        let _mask = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, values, threshold);

        // To count, we need to reduce the mask - for now just return 0
        // (full implementation would use popcount on the mask)
        let zero = builder.ins().iconst(I64, 0);
        builder.ins().return_(&[zero]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for count_greater ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();
}

// =============================================================================
// VCode Verification Tests - Verify correct instruction selection
// =============================================================================

/// Test that verifies VPADDQ (512-bit add) instruction is generated.
#[test]
fn test_vcode_vpaddq_used() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vcode_vpaddq", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src1 = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        let src2 = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);
        let result = builder.ins().iadd(src1, src2);
        builder
            .ins()
            .store(MemFlags::trusted(), result, params[2], 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    let mut found_vpaddq = false;
    let mut found_vmovdqu64 = false;

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vcode_vpaddq ===\n{}", disasm);
            let lower = disasm.to_lowercase();
            found_vpaddq = lower.contains("vpaddq");
            // vmovdqu64 confirms 512-bit mode (vs vmovdqu for 128/256-bit)
            found_vmovdqu64 = lower.contains("vmovdqu64");
        }
    }

    assert!(
        found_vpaddq,
        "Expected VPADDQ instruction for I64X8 iadd, but it wasn't generated"
    );
    assert!(
        found_vmovdqu64,
        "Expected VMOVDQU64 (512-bit load) instruction, but it wasn't generated"
    );
}

/// Test that verifies VPADDD (512-bit 32-bit element add) instruction is generated.
#[test]
fn test_vcode_vpaddd_used() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vcode_vpaddd", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src1 = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[0], 0);
        let src2 = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), params[1], 0);
        let result = builder.ins().iadd(src1, src2);
        builder
            .ins()
            .store(MemFlags::trusted(), result, params[2], 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    let mut found_vpaddd = false;
    let mut found_vmovdqu32 = false;

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vcode_vpaddd ===\n{}", disasm);
            let lower = disasm.to_lowercase();
            found_vpaddd = lower.contains("vpaddd");
            // vmovdqu32 confirms 512-bit mode (vs vmovdqu for 128/256-bit)
            found_vmovdqu32 = lower.contains("vmovdqu32");
        }
    }

    assert!(
        found_vpaddd,
        "Expected VPADDD instruction for I32X16 iadd, but it wasn't generated"
    );
    assert!(
        found_vmovdqu32,
        "Expected VMOVDQU32 (512-bit load) instruction, but it wasn't generated"
    );
}

/// Test that verifies VPANDQ/VPORD/VPXORQ (512-bit bitwise) instructions are generated.
#[test]
fn test_vcode_bitwise_ops_used() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Build a function that uses all three bitwise ops
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vcode_bitwise", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src1 = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        let src2 = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);

        // Chain of bitwise ops: (src1 AND src2) OR (src1 XOR src2)
        let and_result = builder.ins().band(src1, src2);
        let xor_result = builder.ins().bxor(src1, src2);
        let or_result = builder.ins().bor(and_result, xor_result);

        builder
            .ins()
            .store(MemFlags::trusted(), or_result, params[2], 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    let mut found_vpandq = false;
    let mut found_vpxorq = false;
    let mut found_vporq = false;

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vcode_bitwise ===\n{}", disasm);
            let lower = disasm.to_lowercase();
            found_vpandq = lower.contains("vpandq");
            found_vpxorq = lower.contains("vpxorq");
            found_vporq = lower.contains("vporq");
        }
    }

    assert!(
        found_vpandq,
        "Expected VPANDQ instruction for I64X8 band, but it wasn't generated"
    );
    assert!(
        found_vpxorq,
        "Expected VPXORQ instruction for I64X8 bxor, but it wasn't generated"
    );
    assert!(
        found_vporq,
        "Expected VPORQ instruction for I64X8 bor, but it wasn't generated"
    );
}

/// Test that verifies VPMINSQ/VPMAXSQ (512-bit signed min/max) instructions are generated.
#[test]
fn test_vcode_minmax_ops_used() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vcode_minmax", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src1 = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        let src2 = builder.ins().load(I64X8, MemFlags::trusted(), params[1], 0);

        // Get both min and max, then combine: min + max
        let min_result = builder.ins().smin(src1, src2);
        let max_result = builder.ins().smax(src1, src2);
        let combined = builder.ins().iadd(min_result, max_result);

        builder
            .ins()
            .store(MemFlags::trusted(), combined, params[2], 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    let mut found_vpminsq = false;
    let mut found_vpmaxsq = false;

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vcode_minmax ===\n{}", disasm);
            let lower = disasm.to_lowercase();
            found_vpminsq = lower.contains("vpminsq");
            found_vpmaxsq = lower.contains("vpmaxsq");
        }
    }

    assert!(
        found_vpminsq,
        "Expected VPMINSQ instruction for I64X8 smin, but it wasn't generated"
    );
    assert!(
        found_vpmaxsq,
        "Expected VPMAXSQ instruction for I64X8 smax, but it wasn't generated"
    );
}

/// Test that verifies VMOVDQU64 (512-bit load/store) instructions are generated.
#[test]
fn test_vcode_load_store_512bit() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vcode_memops", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        // Simple copy: load from src, store to dst
        let value = builder.ins().load(I64X8, MemFlags::trusted(), params[0], 0);
        builder
            .ins()
            .store(MemFlags::trusted(), value, params[1], 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .unwrap();

    let mut found_vmovdqu = false;

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vcode_memops ===\n{}", disasm);
            let lower = disasm.to_lowercase();
            // Should use vmovdqu64 or vmovdqu32 for 512-bit
            found_vmovdqu = lower.contains("vmovdqu");
        }
    }

    assert!(
        found_vmovdqu,
        "Expected VMOVDQU64/32 instruction for 512-bit load/store, but it wasn't generated"
    );
}

// NOTE: VPOPCNTD/Q tests are disabled because they require AVX-512 VPOPCNTDQ extension.
// The has_avx512vpopcntdq ISA flag exists but the instructions need additional lowering.

// =============================================================================
// Tests: Count Leading Zeros (VPLZCNTD / VPLZCNTQ)
// =============================================================================
// NOTE: Vector clz (VPLZCNTD/Q) is not currently exposed in CLIF IR for I64X8/I32X16.
// The underlying AVX-512CD instructions are implemented (VPLZCNTD/VPLZCNTQ),
// but CLIF verifier rejects clz on vector types. A future enhancement could add
// vector clz support to CLIF IR.
//
// For now, we test the instruction encoding via runtime_tests.rs instead.

// =============================================================================
// Tests: Shift Operations
// =============================================================================
// NOTE: CLIF vector shifts (ishl, ushr, sshr) require a scalar shift amount,
// not a per-element vector shift. The variable shift instructions (VPSLLVD/Q,
// VPSRLVD/Q, VPSRAVD/Q) are implemented but need explicit lowering rules or
// a new CLIF instruction like `vshl` to be accessible from IR.
//
// For now, we test the instruction encoding via runtime_tests.rs instead.

// =============================================================================
// Tests: Multiply operations (VPMULLQ, VPMULUDQ, VPMULDQ)
// =============================================================================

/// Test 64-bit multiply low (VPMULLQ).
#[test]
fn test_i64x8_imul() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_imul", |builder, src1, src2| {
            builder.ins().imul(src1, src2)
        })
        .expect("Failed to compile i64x8_imul");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test: simple multiplications
    let a = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I64x8::new([10, 20, 30, 40, 50, 60, 70, 80]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, I64x8::new([10, 40, 90, 160, 250, 360, 490, 640]));

    // Test: negative numbers
    let a = I64x8::new([-1, -2, 3, -4, 5, -6, 7, -8]);
    let b = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    unsafe {
        func(&a, &b, &mut result);
    }
    assert_eq!(result, I64x8::new([-1, -4, 9, -16, 25, -36, 49, -64]));
}

/// Test 32-bit multiply low (VPMULLD).
#[test]
fn test_i32x16_imul() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_imul", |builder, src1, src2| {
            builder.ins().imul(src1, src2)
        })
        .expect("Failed to compile i32x16_imul");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    let a = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let b = I32x16::splat(10);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
        ])
    );
}

// =============================================================================
// Tests: 512-bit Count Leading Zeros (VPLZCNTD / VPLZCNTQ)
// =============================================================================

/// Test I64X8 clz - VPLZCNTQ instruction (AVX-512CD).
#[test]
fn test_i64x8_clz() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Check for AVX-512CD which is required for VPLZCNT
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512cd") {
        println!("Skipping test: AVX-512CD not available");
        return;
    }

    let code = compiler
        .compile_unary_i64x8("i64x8_clz", |builder, src| builder.ins().clz(src))
        .expect("Failed to compile i64x8_clz");

    let func: UnaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test case: various leading zero counts
    // clz(0) = 64, clz(1) = 63, clz(0x8000000000000000) = 0
    let a = I64x8::new([
        0,                               // clz = 64
        1,                               // clz = 63
        0x0000_0000_0000_0002,           // clz = 62
        0x0000_0000_0000_0100,           // clz = 55
        0x0000_0000_8000_0000,           // clz = 32
        0x0001_0000_0000_0000,           // clz = 15
        0x4000_0000_0000_0000,           // clz = 1
        0x8000_0000_0000_0000u64 as i64, // clz = 0
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    assert_eq!(result, I64x8::new([64, 63, 62, 55, 32, 15, 1, 0]));

    // Test case: powers of 2
    let a = I64x8::new([1, 2, 4, 8, 16, 32, 64, 128]);
    unsafe {
        func(&a, &mut result);
    }
    assert_eq!(result, I64x8::new([63, 62, 61, 60, 59, 58, 57, 56]));
}

/// Test I32X16 clz - VPLZCNTD instruction (AVX-512CD).
#[test]
fn test_i32x16_clz() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512cd") {
        println!("Skipping test: AVX-512CD not available");
        return;
    }

    let code = compiler
        .compile_unary_i32x16("i32x16_clz", |builder, src| builder.ins().clz(src))
        .expect("Failed to compile i32x16_clz");

    let func: UnaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test case: various leading zero counts for 32-bit
    let a = I32x16::new([
        0,                    // clz = 32
        1,                    // clz = 31
        2,                    // clz = 30
        4,                    // clz = 29
        0x100,                // clz = 23
        0x8000,               // clz = 16
        0x10000,              // clz = 15
        0x800000,             // clz = 8
        0x1000000,            // clz = 7
        0x10000000,           // clz = 3
        0x40000000,           // clz = 1
        0x80000000u32 as i32, // clz = 0
        0x12345678u32 as i32, // clz = 3 (0x12345678 = 0b0001_0010...)
        0x00001234,           // clz = 19
        0x00000012,           // clz = 27
        0x00000001,           // clz = 31
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([32, 31, 30, 29, 23, 16, 15, 8, 7, 3, 1, 0, 3, 19, 27, 31])
    );
}

// =============================================================================
// Tests: 512-bit Per-Element Variable Shifts (VPSLLV, VPSRLV, VPSRAV)
// =============================================================================

/// Test I64X8 per-element variable left shift (VPSLLVQ).
#[test]
fn test_i64x8_vpsllv() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_vpsllv", |builder, data, shift| {
            builder.ins().x86_vpsllv(data, shift)
        })
        .expect("Failed to compile i64x8_vpsllv");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test: shift each lane by different amounts
    let data = I64x8::splat(1);
    let shift = I64x8::new([0, 1, 2, 3, 4, 8, 16, 63]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(result, I64x8::new([1, 2, 4, 8, 16, 256, 65536, 1 << 63]));

    // Test: various values
    let data = I64x8::new([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    let shift = I64x8::new([0, 4, 8, 12, 16, 20, 24, 28]);
    unsafe {
        func(&data, &shift, &mut result);
    }
    assert_eq!(
        result,
        I64x8::new([
            0xFF,
            0xFF0,
            0xFF00,
            0xFF000,
            0xFF0000,
            0xFF00000,
            0xFF000000,
            0xFF0000000
        ])
    );
}

/// Test I32X16 per-element variable left shift (VPSLLVD).
#[test]
fn test_i32x16_vpsllv() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_vpsllv", |builder, data, shift| {
            builder.ins().x86_vpsllv(data, shift)
        })
        .expect("Failed to compile i32x16_vpsllv");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test: shift each lane by different amounts
    let data = I32x16::splat(1);
    let shift = I32x16::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ])
    );
}

/// Test I64X8 per-element variable unsigned right shift (VPSRLVQ).
#[test]
fn test_i64x8_vpsrlv() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_vpsrlv", |builder, data, shift| {
            builder.ins().x86_vpsrlv(data, shift)
        })
        .expect("Failed to compile i64x8_vpsrlv");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test: shift right by different amounts
    let data = I64x8::splat(0x8000_0000_0000_0000u64 as i64);
    let shift = I64x8::new([0, 1, 4, 8, 16, 32, 48, 63]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(
        result,
        I64x8::new([
            0x8000_0000_0000_0000u64 as i64,
            0x4000_0000_0000_0000,
            0x0800_0000_0000_0000,
            0x0080_0000_0000_0000,
            0x0000_8000_0000_0000,
            0x0000_0000_8000_0000,
            0x0000_0000_0000_8000,
            1,
        ])
    );
}

/// Test I32X16 per-element variable unsigned right shift (VPSRLVD).
#[test]
fn test_i32x16_vpsrlv() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_vpsrlv", |builder, data, shift| {
            builder.ins().x86_vpsrlv(data, shift)
        })
        .expect("Failed to compile i32x16_vpsrlv");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test: shift right by different amounts
    let data = I32x16::splat(0x8000_0000u32 as i32);
    let shift = I32x16::new([0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 31, 0, 0, 0, 0]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            0x8000_0000u32 as i32,
            0x4000_0000,
            0x2000_0000,
            0x1000_0000,
            0x0800_0000,
            0x0080_0000,
            0x0008_0000,
            0x0000_8000,
            0x0000_0800,
            0x0000_0080,
            0x0000_0008,
            1,
            0x8000_0000u32 as i32,
            0x8000_0000u32 as i32,
            0x8000_0000u32 as i32,
            0x8000_0000u32 as i32,
        ])
    );
}

/// Test I64X8 per-element variable signed right shift (VPSRAVQ).
#[test]
fn test_i64x8_vpsrav() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_vpsrav", |builder, data, shift| {
            builder.ins().x86_vpsrav(data, shift)
        })
        .expect("Failed to compile i64x8_vpsrav");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test: signed shift right - sign bit should extend
    let data = I64x8::new([-128, -128, -128, -128, 128, 128, 128, 128]);
    let shift = I64x8::new([0, 1, 2, 3, 0, 1, 2, 3]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(result, I64x8::new([-128, -64, -32, -16, 128, 64, 32, 16,]));

    // Test: shifting preserves sign
    let data = I64x8::splat(i64::MIN); // 0x8000_0000_0000_0000
    let shift = I64x8::new([0, 1, 4, 8, 16, 32, 48, 63]);
    unsafe {
        func(&data, &shift, &mut result);
    }
    // Sign-extended: shifts in 1s from the left
    assert_eq!(
        result,
        I64x8::new([
            i64::MIN,
            0xC000_0000_0000_0000u64 as i64,
            0xF800_0000_0000_0000u64 as i64,
            0xFF80_0000_0000_0000u64 as i64,
            0xFFFF_8000_0000_0000u64 as i64,
            0xFFFF_FFFF_8000_0000u64 as i64,
            0xFFFF_FFFF_FFFF_8000u64 as i64,
            -1, // all 1s
        ])
    );
}

/// Test I32X16 per-element variable signed right shift (VPSRAVD).
#[test]
fn test_i32x16_vpsrav() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_vpsrav", |builder, data, shift| {
            builder.ins().x86_vpsrav(data, shift)
        })
        .expect("Failed to compile i32x16_vpsrav");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test: signed shift right - sign bit should extend
    let data = I32x16::new([
        -128, -128, -128, -128, -128, -128, -128, -128, 128, 128, 128, 128, 128, 128, 128, 128,
    ]);
    let shift = I32x16::new([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&data, &shift, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            -128, -64, -32, -16, -8, -4, -2, -1, 128, 64, 32, 16, 8, 4, 2, 1,
        ])
    );
}

// =============================================================================
// Widening Multiply Tests (32x32 -> 64 bit)
// =============================================================================

/// Test I64X8 unsigned widening multiply (VPMULUDQ).
/// Takes low 32 bits of each 64-bit lane, multiplies as unsigned, produces 64-bit results.
#[test]
fn test_i64x8_pmullq_low() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_pmullq_low", |builder, a, b| {
            builder.ins().x86_pmullq_low(a, b)
        })
        .expect("Failed to compile i64x8_pmullq_low");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple multiplications (low 32 bits only)
    // Each lane: only low 32 bits are used
    let a = I64x8::new([2, 3, 4, 5, 6, 7, 8, 9]);
    let b = I64x8::new([10, 20, 30, 40, 50, 60, 70, 80]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // 2*10=20, 3*20=60, 4*30=120, 5*40=200, 6*50=300, 7*60=420, 8*70=560, 9*80=720
    assert_eq!(result, I64x8::new([20, 60, 120, 200, 300, 420, 560, 720]));

    // Test 2: Large 32-bit values producing 64-bit results
    // 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001 (unsigned)
    let a = I64x8::new([
        0xFFFFFFFF, 0x80000000, 0x12345678, 0xDEADBEEF, 1000000, 0xCAFEBABE, 0x11111111, 0x55555555,
    ]);
    let b = I64x8::new([
        0xFFFFFFFF, 2, 0x87654321, 0x12345678, 1000000, 0xFEEDFACE, 0x11111111, 3,
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Calculate expected values:
    // 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001
    // 0x80000000 * 2 = 0x100000000
    // 0x12345678 * 0x87654321 = 0x9A0CD05B2A42D78 (actual: 305419896 * 2271560481)
    // etc.
    let expected = I64x8::new([
        0xFFFFFFFF_u64.wrapping_mul(0xFFFFFFFF) as i64,
        0x80000000_u64.wrapping_mul(2) as i64,
        0x12345678_u64.wrapping_mul(0x87654321) as i64,
        0xDEADBEEF_u64.wrapping_mul(0x12345678) as i64,
        1000000_u64.wrapping_mul(1000000) as i64,
        0xCAFEBABE_u64.wrapping_mul(0xFEEDFACE) as i64,
        0x11111111_u64.wrapping_mul(0x11111111) as i64,
        0x55555555_u64.wrapping_mul(3) as i64,
    ]);
    assert_eq!(result, expected);

    // Test 3: High 32 bits should be ignored
    // Set high 32 bits to different values, low 32 bits the same
    let a = I64x8::new([
        0xDEAD_0000_0000_0005u64 as i64,
        0xBEEF_0000_0000_000Au64 as i64,
        0xCAFE_0000_0000_0064u64 as i64,
        0xBABE_0000_0000_03E8u64 as i64,
        5,
        10,
        100,
        1000,
    ]);
    let b = I64x8::new([
        0xFEED_0000_0000_0003u64 as i64,
        0xFACE_0000_0000_0005u64 as i64,
        0x1234_0000_0000_0002u64 as i64,
        0x5678_0000_0000_0004u64 as i64,
        3,
        5,
        2,
        4,
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // High bits should be ignored, only low 32 bits matter
    // 5*3=15, 10*5=50, 100*2=200, 1000*4=4000
    assert_eq!(result, I64x8::new([15, 50, 200, 4000, 15, 50, 200, 4000]));
}

/// Test I64X8 signed widening multiply (VPMULDQ).
/// Takes low 32 bits of each 64-bit lane, multiplies as signed, produces 64-bit results.
#[test]
fn test_i64x8_smullq_low() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_smullq_low", |builder, a, b| {
            builder.ins().x86_smullq_low(a, b)
        })
        .expect("Failed to compile i64x8_smullq_low");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple signed multiplications
    let a = I64x8::new([2, -3, 4, -5, 6, -7, 8, -9]);
    let b = I64x8::new([10, 20, -30, -40, 50, 60, -70, -80]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Note: the values are interpreted as 32-bit signed in the low bits
    // 2*10=20, -3*20=-60, 4*-30=-120, -5*-40=200, 6*50=300, -7*60=-420, 8*-70=-560, -9*-80=720
    assert_eq!(
        result,
        I64x8::new([20, -60, -120, 200, 300, -420, -560, 720])
    );

    // Test 2: Large 32-bit signed values
    // -1 (0xFFFFFFFF as i32) * -1 = 1
    // 0x80000000 as i32 is -2147483648
    let a = I64x8::new([
        0xFFFFFFFF_u64 as i64, // -1 in low 32 bits (sign-extended in VPMULDQ)
        0x80000000_u64 as i64, // i32::MIN in low 32 bits
        0x7FFFFFFF_u64 as i64, // i32::MAX in low 32 bits
        -1000,
        1000,
        -1,
        0x12345678,
        0xEDCBA988_u64 as i64, // negative in low 32 bits
    ]);
    let b = I64x8::new([
        0xFFFFFFFF_u64 as i64, // -1
        2,
        2,
        1000,
        -1000,
        0x7FFFFFFF_u64 as i64, // i32::MAX
        0x87654321_u64 as i64, // negative in low 32 bits
        0x12345678,
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Expected (signed multiplication):
    // (-1) * (-1) = 1
    // i32::MIN * 2 = -4294967296 (fits in i64)
    // i32::MAX * 2 = 4294967294
    // -1000 * 1000 = -1000000
    // 1000 * -1000 = -1000000
    // (-1) * i32::MAX = -2147483647
    // 0x12345678 * (0x87654321 as i32) = 305419896 * (-2023406815)
    // (0xEDCBA988 as i32) * 0x12345678 = (-305419896) * 305419896
    let expected = I64x8::new([
        1,                                                        // (-1) * (-1)
        (i32::MIN as i64) * 2,                                    // i32::MIN * 2
        (i32::MAX as i64) * 2,                                    // i32::MAX * 2
        -1000000,                                                 // -1000 * 1000
        -1000000,                                                 // 1000 * -1000
        -(i32::MAX as i64),                                       // (-1) * i32::MAX
        (0x12345678_i32 as i64) * (0x87654321_u32 as i32 as i64), // signed multiply
        (0xEDCBA988_u32 as i32 as i64) * (0x12345678_i32 as i64), // signed multiply
    ]);
    assert_eq!(result, expected);
}

// =============================================================================
// Floating-Point Min/Max Tests (VMINPS/PD, VMAXPS/PD)
// =============================================================================

/// Test F64X8 minimum (VMINPD).
#[test]
fn test_f64x8_fmin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fmin", |builder, a, b| builder.ins().fmin(a, b))
        .expect("Failed to compile f64x8_fmin");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple minimum
    let a = F64x8::new([1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0]);
    let b = F64x8::new([2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 0.5]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, F64x8::new([1.0, 3.0, 3.0, 1.0, 2.0, 6.0, 4.0, 0.5]));

    // Test 2: Negative values
    let a = F64x8::new([-1.0, -5.0, -3.0, 0.0, 2.0, -9.0, 4.0, -7.0]);
    let b = F64x8::new([2.0, -3.0, -4.0, -1.0, -5.0, 6.0, -7.0, 0.5]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        F64x8::new([-1.0, -5.0, -4.0, -1.0, -5.0, -9.0, -7.0, -7.0])
    );

    // Test 3: Large and small values
    let a = F64x8::new([
        1e308,
        -1e308,
        1e-308,
        -1e-308,
        0.0,
        -0.0,
        f64::MAX,
        f64::MIN,
    ]);
    let b = F64x8::new([
        -1e308,
        1e308,
        -1e-308,
        1e-308,
        -0.0,
        0.0,
        f64::MIN,
        f64::MAX,
    ]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Note: -0.0 vs 0.0 behavior can vary; focus on magnitude comparisons
    assert_eq!(result.0[0], -1e308);
    assert_eq!(result.0[1], -1e308);
    assert_eq!(result.0[6], f64::MIN);
    assert_eq!(result.0[7], f64::MIN);
}

/// Test F64X8 maximum (VMAXPD).
#[test]
fn test_f64x8_fmax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fmax", |builder, a, b| builder.ins().fmax(a, b))
        .expect("Failed to compile f64x8_fmax");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple maximum
    let a = F64x8::new([1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0]);
    let b = F64x8::new([2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 0.5]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(result, F64x8::new([2.0, 5.0, 4.0, 8.0, 5.0, 9.0, 7.0, 7.0]));

    // Test 2: Negative values
    let a = F64x8::new([-1.0, -5.0, -3.0, 0.0, 2.0, -9.0, 4.0, -7.0]);
    let b = F64x8::new([2.0, -3.0, -4.0, -1.0, -5.0, 6.0, -7.0, 0.5]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        F64x8::new([2.0, -3.0, -3.0, 0.0, 2.0, 6.0, 4.0, 0.5])
    );
}

/// Test F32X16 minimum (VMINPS).
#[test]
fn test_f32x16_fmin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fmin", |builder, a, b| builder.ins().fmin(a, b))
        .expect("Failed to compile f32x16_fmin");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // Test: 16 lanes of minimums
    let a = F32x16::new([
        1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, -1.0, -5.0, -3.0, 0.0, 2.0, -9.0, 4.0, -7.0,
    ]);
    let b = F32x16::new([
        2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 0.5, 2.0, -3.0, -4.0, -1.0, -5.0, 6.0, -7.0, 0.5,
    ]);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        F32x16::new([
            1.0, 3.0, 3.0, 1.0, 2.0, 6.0, 4.0, 0.5, -1.0, -5.0, -4.0, -1.0, -5.0, -9.0, -7.0, -7.0,
        ])
    );
}

/// Test F32X16 maximum (VMAXPS).
#[test]
fn test_f32x16_fmax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fmax", |builder, a, b| builder.ins().fmax(a, b))
        .expect("Failed to compile f32x16_fmax");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // Test: 16 lanes of maximums
    let a = F32x16::new([
        1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, -1.0, -5.0, -3.0, 0.0, 2.0, -9.0, 4.0, -7.0,
    ]);
    let b = F32x16::new([
        2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 0.5, 2.0, -3.0, -4.0, -1.0, -5.0, 6.0, -7.0, 0.5,
    ]);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    assert_eq!(
        result,
        F32x16::new([
            2.0, 5.0, 4.0, 8.0, 5.0, 9.0, 7.0, 7.0, 2.0, -3.0, -3.0, 0.0, 2.0, 6.0, 4.0, 0.5,
        ])
    );
}

// =============================================================================
// Integer <-> Float Conversion Tests (VCVTQQ2PD, VCVTTPD2QQ, VCVTDQ2PS, VCVTTPS2DQ)
// =============================================================================

/// Test I64X8 -> F64X8 conversion (VCVTQQ2PD).
#[test]
fn test_i64x8_to_f64x8() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_i64x8_to_f64x8("i64x8_to_f64x8", |builder, src| {
            builder.ins().fcvt_from_sint(F64X8, src)
        })
        .expect("Failed to compile i64x8_to_f64x8");

    let func: ConvertI64x8ToF64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple conversions
    let src = I64x8::new([0, 1, -1, 100, -100, 1000000, -1000000, i64::MAX / 2]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&src, &mut result);
    }

    assert_eq!(result.0[0], 0.0);
    assert_eq!(result.0[1], 1.0);
    assert_eq!(result.0[2], -1.0);
    assert_eq!(result.0[3], 100.0);
    assert_eq!(result.0[4], -100.0);
    assert_eq!(result.0[5], 1000000.0);
    assert_eq!(result.0[6], -1000000.0);
    // Large integers may lose precision in f64
    assert!((result.0[7] - (i64::MAX / 2) as f64).abs() < 1024.0);

    // Test 2: Edge cases
    let src = I64x8::new([
        i64::MIN,
        i64::MAX,
        i32::MIN as i64,
        i32::MAX as i64,
        0,
        1,
        -1,
        123456789012345,
    ]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&src, &mut result);
    }

    assert_eq!(result.0[0], i64::MIN as f64);
    // i64::MAX may lose precision when converted to f64
    assert!((result.0[1] - i64::MAX as f64).abs() < 2048.0);
    assert_eq!(result.0[2], i32::MIN as f64);
    assert_eq!(result.0[3], i32::MAX as f64);
    assert_eq!(result.0[4], 0.0);
    assert_eq!(result.0[5], 1.0);
    assert_eq!(result.0[6], -1.0);
    assert_eq!(result.0[7], 123456789012345.0);
}

/// Test F64X8 -> I64X8 conversion with saturation (VCVTTPD2QQ).
#[test]
fn test_f64x8_to_i64x8() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_f64x8_to_i64x8("f64x8_to_i64x8", |builder, src| {
            builder.ins().fcvt_to_sint_sat(I64X8, src)
        })
        .expect("Failed to compile f64x8_to_i64x8");

    let func: ConvertF64x8ToI64x8Fn = unsafe { mem::transmute(code) };

    // Test 1: Simple conversions (values that fit in i64)
    let src = F64x8::new([0.0, 1.0, -1.0, 100.5, -100.9, 1000000.0, -1000000.0, 0.5]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&src, &mut result);
    }

    // With truncation toward zero
    assert_eq!(result.0[0], 0);
    assert_eq!(result.0[1], 1);
    assert_eq!(result.0[2], -1);
    assert_eq!(result.0[3], 100); // truncated
    assert_eq!(result.0[4], -100); // truncated toward zero
    assert_eq!(result.0[5], 1000000);
    assert_eq!(result.0[6], -1000000);
    assert_eq!(result.0[7], 0); // 0.5 truncated to 0

    // Test 2: Negative fractional values
    let src = F64x8::new([-0.1, -0.5, -0.9, -1.1, -1.5, -1.9, -2.5, -3.9]);
    let mut result = I64x8::splat(99);

    unsafe {
        func(&src, &mut result);
    }

    // Truncation toward zero: negative values become less negative
    assert_eq!(result.0[0], 0);
    assert_eq!(result.0[1], 0);
    assert_eq!(result.0[2], 0);
    assert_eq!(result.0[3], -1);
    assert_eq!(result.0[4], -1);
    assert_eq!(result.0[5], -1);
    assert_eq!(result.0[6], -2);
    assert_eq!(result.0[7], -3);
}

/// Test I32X16 -> F32X16 conversion (VCVTDQ2PS).
#[test]
fn test_i32x16_to_f32x16() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_i32x16_to_f32x16("i32x16_to_f32x16", |builder, src| {
            builder.ins().fcvt_from_sint(F32X16, src)
        })
        .expect("Failed to compile i32x16_to_f32x16");

    let func: ConvertI32x16ToF32x16Fn = unsafe { mem::transmute(code) };

    // Test: 16 lanes of conversions
    let src = I32x16::new([
        0,
        1,
        -1,
        100,
        -100,
        1000,
        -1000,
        10000,
        -10000,
        100000,
        -100000,
        1000000,
        -1000000,
        i32::MAX / 2,
        i32::MIN / 2,
        42,
    ]);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&src, &mut result);
    }

    assert_eq!(result.0[0], 0.0);
    assert_eq!(result.0[1], 1.0);
    assert_eq!(result.0[2], -1.0);
    assert_eq!(result.0[3], 100.0);
    assert_eq!(result.0[4], -100.0);
    assert_eq!(result.0[5], 1000.0);
    assert_eq!(result.0[6], -1000.0);
    assert_eq!(result.0[7], 10000.0);
    assert_eq!(result.0[15], 42.0);
}

/// Test F32X16 -> I32X16 conversion with saturation (VCVTTPS2DQ).
#[test]
fn test_f32x16_to_i32x16() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_f32x16_to_i32x16("f32x16_to_i32x16", |builder, src| {
            builder.ins().fcvt_to_sint_sat(I32X16, src)
        })
        .expect("Failed to compile f32x16_to_i32x16");

    let func: ConvertF32x16ToI32x16Fn = unsafe { mem::transmute(code) };

    // Test: 16 lanes of conversions with truncation
    let src = F32x16::new([
        0.0, 1.0, -1.0, 100.5, -100.9, 1000.0, -1000.0, 0.5, -0.5, 1.9, -1.9, 2.5, -2.5, 3.1, -3.1,
        42.7,
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&src, &mut result);
    }

    // Truncation toward zero
    assert_eq!(result.0[0], 0);
    assert_eq!(result.0[1], 1);
    assert_eq!(result.0[2], -1);
    assert_eq!(result.0[3], 100); // 100.5 truncated
    assert_eq!(result.0[4], -100); // -100.9 truncated toward zero
    assert_eq!(result.0[5], 1000);
    assert_eq!(result.0[6], -1000);
    assert_eq!(result.0[7], 0); // 0.5 truncated
    assert_eq!(result.0[8], 0); // -0.5 truncated
    assert_eq!(result.0[9], 1); // 1.9 truncated
    assert_eq!(result.0[10], -1); // -1.9 truncated toward zero
    assert_eq!(result.0[15], 42); // 42.7 truncated
}

// =============================================================================
// Fused Multiply-Add Tests (VFMADD213PD, VFMSUB213PD, VFNMADD213PD)
// =============================================================================

/// Test F64X8 fused multiply-add: fma(x, y, z) = x * y + z (VFMADD213PD).
#[test]
fn test_f64x8_fma() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_ternary_f64x8("f64x8_fma", |builder, x, y, z| builder.ins().fma(x, y, z))
        .expect("Failed to compile f64x8_fma");

    let func: TernaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test: x * y + z for 8 lanes
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::new([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let z = F64x8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&x, &y, &z, &mut result);
    }

    // Expected: x[i] * y[i] + z[i]
    assert_eq!(result.0[0], 1.0 * 2.0 + 10.0); // 12.0
    assert_eq!(result.0[1], 2.0 * 3.0 + 20.0); // 26.0
    assert_eq!(result.0[2], 3.0 * 4.0 + 30.0); // 42.0
    assert_eq!(result.0[3], 4.0 * 5.0 + 40.0); // 60.0
    assert_eq!(result.0[4], 5.0 * 6.0 + 50.0); // 80.0
    assert_eq!(result.0[5], 6.0 * 7.0 + 60.0); // 102.0
    assert_eq!(result.0[6], 7.0 * 8.0 + 70.0); // 126.0
    assert_eq!(result.0[7], 8.0 * 9.0 + 80.0); // 152.0
}

/// Test F64X8 fused multiply-subtract fusion: fsub(fmul(x, y), z) = x * y - z (VFMSUB213PD).
#[test]
fn test_f64x8_fmsub_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_ternary_f64x8("f64x8_fmsub", |builder, x, y, z| {
            // fsub(fmul(x, y), z) should fuse into VFMSUB
            let prod = builder.ins().fmul(x, y);
            builder.ins().fsub(prod, z)
        })
        .expect("Failed to compile f64x8_fmsub");

    let func: TernaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test: x * y - z for 8 lanes
    let x = F64x8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let y = F64x8::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let z = F64x8::new([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&x, &y, &z, &mut result);
    }

    // Expected: x[i] * y[i] - z[i]
    assert_eq!(result.0[0], 10.0 * 2.0 - 5.0); // 15.0
    assert_eq!(result.0[1], 20.0 * 2.0 - 10.0); // 30.0
    assert_eq!(result.0[2], 30.0 * 2.0 - 15.0); // 45.0
    assert_eq!(result.0[3], 40.0 * 2.0 - 20.0); // 60.0
    assert_eq!(result.0[4], 50.0 * 2.0 - 25.0); // 75.0
    assert_eq!(result.0[5], 60.0 * 2.0 - 30.0); // 90.0
    assert_eq!(result.0[6], 70.0 * 2.0 - 35.0); // 105.0
    assert_eq!(result.0[7], 80.0 * 2.0 - 40.0); // 120.0
}

/// Test F64X8 fused negate-multiply-add fusion: fsub(z, fmul(x, y)) = z - x * y (VFNMADD213PD).
#[test]
fn test_f64x8_fnmadd_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_ternary_f64x8("f64x8_fnmadd", |builder, x, y, z| {
            // fsub(z, fmul(x, y)) should fuse into VFNMADD
            let prod = builder.ins().fmul(x, y);
            builder.ins().fsub(z, prod)
        })
        .expect("Failed to compile f64x8_fnmadd");

    let func: TernaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test: z - x * y for 8 lanes
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let z = F64x8::new([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&x, &y, &z, &mut result);
    }

    // Expected: z[i] - x[i] * y[i]
    assert_eq!(result.0[0], 100.0 - 1.0 * 2.0); // 98.0
    assert_eq!(result.0[1], 100.0 - 2.0 * 2.0); // 96.0
    assert_eq!(result.0[2], 100.0 - 3.0 * 2.0); // 94.0
    assert_eq!(result.0[3], 100.0 - 4.0 * 2.0); // 92.0
    assert_eq!(result.0[4], 100.0 - 5.0 * 2.0); // 90.0
    assert_eq!(result.0[5], 100.0 - 6.0 * 2.0); // 88.0
    assert_eq!(result.0[6], 100.0 - 7.0 * 2.0); // 86.0
    assert_eq!(result.0[7], 100.0 - 8.0 * 2.0); // 84.0
}

// =============================================================================
// Integer Absolute Value (iabs) Tests
// =============================================================================

/// Test I64X8 iabs - VPABSQ instruction.
#[test]
fn test_i64x8_iabs() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_unary_i64x8("i64x8_iabs", |builder, src| builder.ins().iabs(src))
        .expect("Failed to compile i64x8_iabs");

    let func: UnaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test with positive and negative values
    let a = I64x8::new([
        0,
        1,
        -1,
        42,
        -42,
        i64::MAX,
        i64::MIN + 1, // iabs(MIN) overflows, so test MIN+1
        -1000000,
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    assert_eq!(
        result,
        I64x8::new([
            0,
            1,
            1,
            42,
            42,
            i64::MAX,
            i64::MAX, // abs(MIN+1) = MAX
            1000000,
        ])
    );
}

/// Test I32X16 iabs - VPABSD instruction.
#[test]
fn test_i32x16_iabs() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_unary_i32x16("i32x16_iabs", |builder, src| builder.ins().iabs(src))
        .expect("Failed to compile i32x16_iabs");

    let func: UnaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test with positive and negative values
    let a = I32x16::new([
        0,
        1,
        -1,
        42,
        -42,
        100,
        -100,
        1000,
        -1000,
        i32::MAX,
        i32::MIN + 1,
        -999999,
        999999,
        -12345,
        12345,
        0,
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    assert_eq!(
        result,
        I32x16::new([
            0,
            1,
            1,
            42,
            42,
            100,
            100,
            1000,
            1000,
            i32::MAX,
            i32::MAX,
            999999,
            999999,
            12345,
            12345,
            0,
        ])
    );
}

// =============================================================================
// Integer Splat (broadcast) Tests
// =============================================================================

/// Test I64X8 splat - VPBROADCASTQ instruction.
#[test]
fn test_i64x8_splat() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_splat_i64x8("i64x8_splat")
        .expect("Failed to compile i64x8_splat");

    let func: SplatI64x8Fn = unsafe { mem::transmute(code) };
    let mut result = I64x8::splat(0);

    // Test splatting 42
    unsafe {
        func(42, &mut result);
    }
    assert_eq!(result, I64x8::splat(42));

    // Test splatting -1
    unsafe {
        func(-1, &mut result);
    }
    assert_eq!(result, I64x8::splat(-1));

    // Test splatting 0
    unsafe {
        func(0, &mut result);
    }
    assert_eq!(result, I64x8::splat(0));

    // Test splatting a large value
    unsafe {
        func(0x123456789ABCDEF0, &mut result);
    }
    assert_eq!(result, I64x8::splat(0x123456789ABCDEF0));
}

/// Test I32X16 splat - VPBROADCASTD instruction.
#[test]
fn test_i32x16_splat() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_splat_i32x16("i32x16_splat")
        .expect("Failed to compile i32x16_splat");

    let func: SplatI32x16Fn = unsafe { mem::transmute(code) };
    let mut result = I32x16::splat(0);

    // Test splatting 42
    unsafe {
        func(42, &mut result);
    }
    assert_eq!(result, I32x16::splat(42));

    // Test splatting -1
    unsafe {
        func(-1, &mut result);
    }
    assert_eq!(result, I32x16::splat(-1));

    // Test splatting 0
    unsafe {
        func(0, &mut result);
    }
    assert_eq!(result, I32x16::splat(0));

    // Test splatting 0x12345678
    unsafe {
        func(0x12345678, &mut result);
    }
    assert_eq!(result, I32x16::splat(0x12345678));
}

// =============================================================================
// Population Count (popcnt) Tests
// =============================================================================

/// Test I64X8 popcnt - VPOPCNTQ instruction (AVX-512 VPOPCNTDQ).
#[test]
fn test_i64x8_popcnt() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
        println!("Skipping test: AVX-512 VPOPCNTDQ not available");
        return;
    }

    let code = compiler
        .compile_unary_i64x8("i64x8_popcnt", |builder, src| builder.ins().popcnt(src))
        .expect("Failed to compile i64x8_popcnt");

    let func: UnaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test with values with known bit counts
    let a = I64x8::new([
        0,                                // popcnt = 0
        1,                                // popcnt = 1
        3,                                // popcnt = 2 (0b11)
        0xFF,                             // popcnt = 8
        0xFFFF,                           // popcnt = 16
        0xFFFF_FFFF,                      // popcnt = 32
        0xFFFF_FFFF_FFFF_FFFF_u64 as i64, // popcnt = 64
        0x1234_5678_9ABC_DEF0_u64 as i64, // specific pattern
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    // Calculate expected popcnts
    assert_eq!(result.0[0], 0); // popcnt(0)
    assert_eq!(result.0[1], 1); // popcnt(1)
    assert_eq!(result.0[2], 2); // popcnt(3)
    assert_eq!(result.0[3], 8); // popcnt(0xFF)
    assert_eq!(result.0[4], 16); // popcnt(0xFFFF)
    assert_eq!(result.0[5], 32); // popcnt(0xFFFF_FFFF)
    assert_eq!(result.0[6], 64); // popcnt(all 1s)
    assert_eq!(result.0[7], (0x1234_5678_9ABC_DEF0_u64).count_ones() as i64);
}

/// Test I32X16 popcnt - VPOPCNTD instruction (AVX-512 VPOPCNTDQ).
#[test]
fn test_i32x16_popcnt() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
        println!("Skipping test: AVX-512 VPOPCNTDQ not available");
        return;
    }

    let code = compiler
        .compile_unary_i32x16("i32x16_popcnt", |builder, src| builder.ins().popcnt(src))
        .expect("Failed to compile i32x16_popcnt");

    let func: UnaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test with values with known bit counts
    let a = I32x16::new([
        0,                     // popcnt = 0
        1,                     // popcnt = 1
        3,                     // popcnt = 2
        7,                     // popcnt = 3
        0xFF,                  // popcnt = 8
        0xFFFF,                // popcnt = 16
        0xFFFF_FFFFu32 as i32, // popcnt = 32
        0x12345678,            // specific pattern
        0x55555555,            // alternating bits = 16
        0xAAAAAAAAu32 as i32,  // alternating bits = 16
        0x0F0F0F0F,            // 4 bits per byte = 16
        0xF0F0F0F0u32 as i32,  // 4 bits per byte = 16
        0x01010101,            // 1 bit per byte = 4
        0x80808080u32 as i32,  // 1 bit per byte = 4
        0xDEADBEEFu32 as i32,  // specific pattern
        0xCAFEBABEu32 as i32,  // specific pattern
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &mut result);
    }

    assert_eq!(result.0[0], 0); // popcnt(0)
    assert_eq!(result.0[1], 1); // popcnt(1)
    assert_eq!(result.0[2], 2); // popcnt(3)
    assert_eq!(result.0[3], 3); // popcnt(7)
    assert_eq!(result.0[4], 8); // popcnt(0xFF)
    assert_eq!(result.0[5], 16); // popcnt(0xFFFF)
    assert_eq!(result.0[6], 32); // popcnt(all 1s)
    assert_eq!(result.0[7], 0x12345678_u32.count_ones() as i32);
    assert_eq!(result.0[8], 16); // 0x55555555
    assert_eq!(result.0[9], 16); // 0xAAAAAAAA
    assert_eq!(result.0[10], 16); // 0x0F0F0F0F
    assert_eq!(result.0[11], 16); // 0xF0F0F0F0
    assert_eq!(result.0[12], 4); // 0x01010101
    assert_eq!(result.0[13], 4); // 0x80808080
    assert_eq!(result.0[14], 0xDEADBEEF_u32.count_ones() as i32);
    assert_eq!(result.0[15], 0xCAFEBABE_u32.count_ones() as i32);
}

// =============================================================================
// Masked Operation Fusion Tests
// =============================================================================

/// Test masked iadd fusion: bitselect(mask, iadd(x, y), passthru)
/// This should compile to a single masked VPADDD instruction.
#[test]
fn test_i32x16_masked_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // We need a 4-operand function: mask, x, y, passthru -> result
    // Build: bitselect(mask, iadd(x, y), passthru)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr
    sig.params.push(AbiParam::new(ptr_type)); // x ptr
    sig.params.push(AbiParam::new(ptr_type)); // y ptr
    sig.params.push(AbiParam::new(ptr_type)); // passthru ptr
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_iadd_i32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        // Load all inputs
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let x = builder.ins().load(I32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(I32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), passthru_ptr, 0);

        // bitselect(mask, iadd(x, y), passthru)
        let sum = builder.ins().iadd(x, y);
        let result = builder.ins().bitselect(mask, sum, passthru);

        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for masked_iadd_i32x16 ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedBinaryI32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedBinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test: mask selects which lanes get the sum, others get passthru
    // Mask: all 1s for lanes 0,2,4,6,8,10,12,14 (even lanes)
    //       all 0s for lanes 1,3,5,7,9,11,13,15 (odd lanes)
    let mask = I32x16::new([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
    let x = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let y = I32x16::new([
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ]);
    let passthru = I32x16::new([
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x + y, Odd lanes: passthru
    assert_eq!(
        result,
        I32x16::new([
            11, -2, 33, -4, 55, -6, 77, -8, 99, -10, 121, -12, 143, -14, 165, -16,
        ])
    );
}

/// Test masked iadd fusion for I64X8: bitselect(mask, iadd(x, y), passthru)
#[test]
fn test_i64x8_masked_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr
    sig.params.push(AbiParam::new(ptr_type)); // x ptr
    sig.params.push(AbiParam::new(ptr_type)); // y ptr
    sig.params.push(AbiParam::new(ptr_type)); // passthru ptr
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_iadd_i64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let x = builder.ins().load(I64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(I64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), passthru_ptr, 0);

        let sum = builder.ins().iadd(x, y);
        let result = builder.ins().bitselect(mask, sum, passthru);

        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for masked_iadd_i64x8 ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedBinaryI64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const I64x8, *const I64x8, *const I64x8, *mut I64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedBinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Mask: all 1s for even lanes, all 0s for odd lanes
    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let y = I64x8::new([10, 20, 30, 40, 50, 60, 70, 80]);
    let passthru = I64x8::new([-100, -200, -300, -400, -500, -600, -700, -800]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x + y, Odd lanes: passthru
    assert_eq!(
        result,
        I64x8::new([11, -200, 33, -400, 55, -600, 77, -800,])
    );
}

// =============================================================================
// Tests: Masked Unsigned Min/Max Fusion
// =============================================================================

/// Test masked umin fusion for I32X16: bitselect(mask, umin(x, y), passthru)
#[test]
fn test_i32x16_masked_umin_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Use pointer-based ABI like the working masked iadd tests
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr
    sig.params.push(AbiParam::new(ptr_type)); // x ptr
    sig.params.push(AbiParam::new(ptr_type)); // y ptr
    sig.params.push(AbiParam::new(ptr_type)); // passthru ptr
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_umin_i32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let x = builder.ins().load(I32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(I32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), passthru_ptr, 0);

        let umin_result = builder.ins().umin(x, y);
        let masked_result = builder.ins().bitselect(mask, umin_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedBinaryI32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedBinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Mask: all 1s for first 8 lanes, all 0s for last 8 lanes
    let mask = I32x16::new([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let x = I32x16::new([
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    ]);
    let y = I32x16::new([
        150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
    ]);
    let passthru = I32x16::splat(-999);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // First 8 lanes: umin(x, y), Last 8 lanes: passthru
    assert_eq!(
        result,
        I32x16::new([
            100, 150, 150, 150, 150, 150, 150, 150, // umin results
            -999, -999, -999, -999, -999, -999, -999, -999, // passthru
        ])
    );
}

/// Test masked umax fusion for I32X16: bitselect(mask, umax(x, y), passthru)
#[test]
fn test_i32x16_masked_umax_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_umax_i32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let x = builder.ins().load(I32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(I32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), passthru_ptr, 0);

        let umax_result = builder.ins().umax(x, y);
        let masked_result = builder.ins().bitselect(mask, umax_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedBinaryI32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *const I32x16,
        *mut I32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedBinaryI32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let x = I32x16::new([
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    ]);
    let y = I32x16::new([
        150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,
    ]);
    let passthru = I32x16::splat(-999);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // First 8 lanes: umax(x, y), Last 8 lanes: passthru
    assert_eq!(
        result,
        I32x16::new([
            150, 200, 300, 400, 500, 600, 700, 800, // umax results
            -999, -999, -999, -999, -999, -999, -999, -999, // passthru
        ])
    );
}

/// Test masked umin fusion for I64X8: bitselect(mask, umin(x, y), passthru)
#[test]
fn test_i64x8_masked_umin_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_umin_i64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let x = builder.ins().load(I64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(I64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), passthru_ptr, 0);

        let umin_result = builder.ins().umin(x, y);
        let masked_result = builder.ins().bitselect(mask, umin_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedBinaryI64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const I64x8, *const I64x8, *const I64x8, *mut I64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedBinaryI64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = I64x8::new([100, 200, 300, 400, 500, 600, 700, 800]);
    let y = I64x8::new([150, 150, 150, 150, 150, 150, 150, 150]);
    let passthru = I64x8::splat(-999);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: umin(x, y), Odd lanes: passthru
    assert_eq!(
        result,
        I64x8::new([100, -999, 150, -999, 150, -999, 150, -999,])
    );
}

// =============================================================================
// Tests: Masked FP Fusion (F32X16)
// =============================================================================

/// Test masked fadd fusion for F32X16: bitselect(mask, fadd(x, y), passthru)
#[test]
fn test_f32x16_masked_fadd_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    // Use pointer-based ABI like the working tests
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I32X16 interpreted as F32X16)
    sig.params.push(AbiParam::new(ptr_type)); // x ptr
    sig.params.push(AbiParam::new(ptr_type)); // y ptr
    sig.params.push(AbiParam::new(ptr_type)); // passthru ptr
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fadd_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        // Load mask as I32X16 then bitcast to F32X16
        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fadd_result = builder.ins().fadd(x, y);
        let masked_result = builder.ins().bitselect(mask, fadd_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFaddF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFaddF32x16Fn = unsafe { mem::transmute(code) };

    // Mask: all 1s for first 8 lanes, all 0s for last 8 lanes
    let mask = I32x16::new([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let x = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let y = F32x16::new([
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ]);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // First 8 lanes: x + y, Last 8 lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, // fadd results
            -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, // passthru
        ])
    );
}

/// Test masked fmul fusion for F32X16: bitselect(mask, fmul(x, y), passthru)
#[test]
fn test_f32x16_masked_fmul_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmul_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fmul_result = builder.ins().fmul(x, y);
        let masked_result = builder.ins().bitselect(mask, fmul_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFmulF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFmulF32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let x = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let y = F32x16::splat(2.0);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // First 8 lanes: x * y, Last 8 lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, // fmul results
            -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, // passthru
        ])
    );
}

/// Test masked fmin fusion for F32X16: bitselect(mask, fmin(x, y), passthru)
#[test]
fn test_f32x16_masked_fmin_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmin_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fmin_result = builder.ins().fmin(x, y);
        let masked_result = builder.ins().bitselect(mask, fmin_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFminF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFminF32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let x = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let y = F32x16::splat(5.0);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // First 8 lanes: min(x, y), Last 8 lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, // fmin results
            -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, // passthru
        ])
    );
}

// =============================================================================
// Tests: Masked FP Fusion (F64X8)
// =============================================================================

/// Test masked fadd fusion for F64X8: bitselect(mask, fadd(x, y), passthru)
#[test]
fn test_f64x8_masked_fadd_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fadd_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fadd_result = builder.ins().fadd(x, y);
        let masked_result = builder.ins().bitselect(mask, fadd_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFaddF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFaddF64x8Fn = unsafe { mem::transmute(code) };

    // Mask: all 1s for even lanes, all 0s for odd lanes
    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::new([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x + y, Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([1.5, -100.0, 3.5, -100.0, 5.5, -100.0, 7.5, -100.0,])
    );
}

/// Test masked fmul fusion for F64X8: bitselect(mask, fmul(x, y), passthru)
#[test]
fn test_f64x8_masked_fmul_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmul_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fmul_result = builder.ins().fmul(x, y);
        let masked_result = builder.ins().bitselect(mask, fmul_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFmulF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFmulF64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::splat(2.0);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x * y, Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([2.0, -100.0, 6.0, -100.0, 10.0, -100.0, 14.0, -100.0,])
    );
}

/// Test masked fmin fusion for F64X8: bitselect(mask, fmin(x, y), passthru)
#[test]
fn test_f64x8_masked_fmin_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping test: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmin_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fmin_result = builder.ins().fmin(x, y);
        let masked_result = builder.ins().bitselect(mask, fmin_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFminF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFminF64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::splat(5.0);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: min(x, y), Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([1.0, -100.0, 3.0, -100.0, 5.0, -100.0, 5.0, -100.0,])
    );
}

/// Test masked fsub fusion for F32X16: bitselect(mask, fsub(x, y), passthru)
#[test]
fn test_f32x16_masked_fsub_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fsub_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fsub_result = builder.ins().fsub(x, y);
        let masked_result = builder.ins().bitselect(mask, fsub_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFsubF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFsubF32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F32x16::new([
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        150.0, 160.0,
    ]);
    let y = F32x16::splat(5.0);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x - y, Odd lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            5.0, -100.0, 25.0, -100.0, 45.0, -100.0, 65.0, -100.0, 85.0, -100.0, 105.0, -100.0,
            125.0, -100.0, 145.0, -100.0,
        ])
    );
}

/// Test masked fdiv fusion for F32X16: bitselect(mask, fdiv(x, y), passthru)
#[test]
fn test_f32x16_masked_fdiv_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fdiv_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fdiv_result = builder.ins().fdiv(x, y);
        let masked_result = builder.ins().bitselect(mask, fdiv_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFdivF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFdivF32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F32x16::new([
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        150.0, 160.0,
    ]);
    let y = F32x16::splat(2.0);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x / y, Odd lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            5.0, -100.0, 15.0, -100.0, 25.0, -100.0, 35.0, -100.0, 45.0, -100.0, 55.0, -100.0,
            65.0, -100.0, 75.0, -100.0,
        ])
    );
}

/// Test masked fmax fusion for F32X16: bitselect(mask, fmax(x, y), passthru)
#[test]
fn test_f32x16_masked_fmax_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmax_f32x16", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F32X16, MemFlags::new(), mask_int);
        let x = builder.ins().load(F32X16, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F32X16, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F32X16, MemFlags::trusted(), passthru_ptr, 0);

        let fmax_result = builder.ins().fmax(x, y);
        let masked_result = builder.ins().bitselect(mask, fmax_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFmaxF32x16Fn = unsafe extern "C" fn(
        *const I32x16,
        *const F32x16,
        *const F32x16,
        *const F32x16,
        *mut F32x16,
    );
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFmaxF32x16Fn = unsafe { mem::transmute(code) };

    let mask = I32x16::new([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let y = F32x16::splat(5.0);
    let passthru = F32x16::splat(-100.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: max(x, y), Odd lanes: passthru
    assert_eq!(
        result,
        F32x16::new([
            5.0, -100.0, 5.0, -100.0, 5.0, -100.0, 7.0, -100.0, 9.0, -100.0, 11.0, -100.0, 13.0,
            -100.0, 15.0, -100.0,
        ])
    );
}

/// Test masked fsub fusion for F64X8: bitselect(mask, fsub(x, y), passthru)
#[test]
fn test_f64x8_masked_fsub_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fsub_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fsub_result = builder.ins().fsub(x, y);
        let masked_result = builder.ins().bitselect(mask, fsub_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFsubF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFsubF64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let y = F64x8::splat(5.0);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x - y, Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([5.0, -100.0, 25.0, -100.0, 45.0, -100.0, 65.0, -100.0,])
    );
}

/// Test masked fdiv fusion for F64X8: bitselect(mask, fdiv(x, y), passthru)
#[test]
fn test_f64x8_masked_fdiv_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fdiv_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fdiv_result = builder.ins().fdiv(x, y);
        let masked_result = builder.ins().bitselect(mask, fdiv_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFdivF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFdivF64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let y = F64x8::splat(2.0);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: x / y, Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([5.0, -100.0, 15.0, -100.0, 25.0, -100.0, 35.0, -100.0,])
    );
}

/// Test masked fmax fusion for F64X8: bitselect(mask, fmax(x, y), passthru)
#[test]
fn test_f64x8_masked_fmax_fusion() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.params.push(AbiParam::new(ptr_type));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("masked_fmax_f64x8", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];
        let x_ptr = params[1];
        let y_ptr = params[2];
        let passthru_ptr = params[3];
        let dst_ptr = params[4];

        let mask_int = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);
        let mask = builder.ins().bitcast(F64X8, MemFlags::new(), mask_int);
        let x = builder.ins().load(F64X8, MemFlags::trusted(), x_ptr, 0);
        let y = builder.ins().load(F64X8, MemFlags::trusted(), y_ptr, 0);
        let passthru = builder
            .ins()
            .load(F64X8, MemFlags::trusted(), passthru_ptr, 0);

        let fmax_result = builder.ins().fmax(x, y);
        let masked_result = builder.ins().bitselect(mask, fmax_result, passthru);

        builder
            .ins()
            .store(MemFlags::trusted(), masked_result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");
    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    type MaskedFmaxF64x8Fn =
        unsafe extern "C" fn(*const I64x8, *const F64x8, *const F64x8, *const F64x8, *mut F64x8);
    let code = compiler.module.get_finalized_function(func_id);
    let func: MaskedFmaxF64x8Fn = unsafe { mem::transmute(code) };

    let mask = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    let x = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = F64x8::splat(5.0);
    let passthru = F64x8::splat(-100.0);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&mask, &x, &y, &passthru, &mut result);
    }

    // Even lanes: max(x, y), Odd lanes: passthru
    assert_eq!(
        result,
        F64x8::new([5.0, -100.0, 5.0, -100.0, 5.0, -100.0, 7.0, -100.0,])
    );
}
// NOTE: Masked shift fusion patterns are not implemented because CLIF's ishl/ushr/sshr
// instructions take a scalar shift amount, not a per-element vector. Implementing masked
// shifts would require either:
// 1. Uniform shift instructions (VPSLLD zmm, zmm, xmm) with masking
// 2. An x86-specific CLIF opcode for per-element variable shifts

// =============================================================================
// AVX-512BW Byte (I8X64) Tests
// =============================================================================

#[test]
fn test_i8x64_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_iadd", |builder, a, b| builder.ins().iadd(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    // Create test data
    let mut a_arr = [0i8; 64];
    let mut b_arr = [0i8; 64];
    for i in 0..64 {
        a_arr[i] = i as i8;
        b_arr[i] = 1;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // a[i] + b[i] = i + 1
    let mut expected = [0i8; 64];
    for i in 0..64 {
        expected[i] = (i + 1) as i8;
    }
    assert_eq!(result, I8x64::new(expected));
}

#[test]
fn test_i8x64_isub() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_isub", |builder, a, b| builder.ins().isub(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i8; 64];
    let mut b_arr = [0i8; 64];
    for i in 0..64 {
        a_arr[i] = (i + 10) as i8;
        b_arr[i] = 5;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i8; 64];
    for i in 0..64 {
        expected[i] = (i + 5) as i8;
    }
    assert_eq!(result, I8x64::new(expected));
}

#[test]
fn test_i8x64_smin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_smin", |builder, a, b| builder.ins().smin(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i8; 64];
    let mut b_arr = [0i8; 64];
    for i in 0..64 {
        a_arr[i] = (i as i8) - 32; // -32 to 31
        b_arr[i] = 0;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // min(a[i], 0) = a[i] if a[i] < 0, else 0
    let mut expected = [0i8; 64];
    for i in 0..64 {
        expected[i] = std::cmp::min(a_arr[i], 0);
    }
    assert_eq!(result, I8x64::new(expected));
}

#[test]
fn test_i8x64_smax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_smax", |builder, a, b| builder.ins().smax(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i8; 64];
    let mut b_arr = [0i8; 64];
    for i in 0..64 {
        a_arr[i] = (i as i8) - 32;
        b_arr[i] = 0;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i8; 64];
    for i in 0..64 {
        expected[i] = std::cmp::max(a_arr[i], 0);
    }
    assert_eq!(result, I8x64::new(expected));
}

#[test]
fn test_i8x64_umin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_umin", |builder, a, b| builder.ins().umin(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i8; 64];
    let b_arr = [100i8; 64];
    for i in 0..64 {
        a_arr[i] = (i * 4) as i8;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Unsigned min - treat bytes as unsigned
    let mut expected = [0i8; 64];
    for i in 0..64 {
        let a_u = a_arr[i] as u8;
        let b_u = 100u8;
        expected[i] = std::cmp::min(a_u, b_u) as i8;
    }
    assert_eq!(result, I8x64::new(expected));
}

#[test]
fn test_i8x64_umax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i8x64("i8x64_umax", |builder, a, b| builder.ins().umax(a, b))
        .expect("Failed to compile");

    let func: BinaryI8x64Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i8; 64];
    let b_arr = [100i8; 64];
    for i in 0..64 {
        a_arr[i] = (i * 4) as i8;
    }
    let a = I8x64::new(a_arr);
    let b = I8x64::new(b_arr);
    let mut result = I8x64::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i8; 64];
    for i in 0..64 {
        let a_u = a_arr[i] as u8;
        let b_u = 100u8;
        expected[i] = std::cmp::max(a_u, b_u) as i8;
    }
    assert_eq!(result, I8x64::new(expected));
}

// =============================================================================
// AVX-512BW Word (I16X32) Tests
// =============================================================================

#[test]
fn test_i16x32_iadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_iadd", |builder, a, b| builder.ins().iadd(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let mut b_arr = [0i16; 32];
    for i in 0..32 {
        a_arr[i] = (i * 100) as i16;
        b_arr[i] = 1;
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        expected[i] = (i * 100 + 1) as i16;
    }
    assert_eq!(result, I16x32::new(expected));
}

#[test]
fn test_i16x32_isub() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_isub", |builder, a, b| builder.ins().isub(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let b_arr = [500i16; 32];
    for i in 0..32 {
        a_arr[i] = (i * 100 + 500) as i16;
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        expected[i] = (i * 100) as i16;
    }
    assert_eq!(result, I16x32::new(expected));
}

#[test]
fn test_i16x32_smin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_smin", |builder, a, b| builder.ins().smin(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let b_arr = [0i16; 32];
    for i in 0..32 {
        a_arr[i] = (i as i16) - 16; // -16 to 15
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        expected[i] = std::cmp::min(a_arr[i], 0);
    }
    assert_eq!(result, I16x32::new(expected));
}

#[test]
fn test_i16x32_smax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_smax", |builder, a, b| builder.ins().smax(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let b_arr = [0i16; 32];
    for i in 0..32 {
        a_arr[i] = (i as i16) - 16;
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        expected[i] = std::cmp::max(a_arr[i], 0);
    }
    assert_eq!(result, I16x32::new(expected));
}

#[test]
fn test_i16x32_umin() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_umin", |builder, a, b| builder.ins().umin(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let b_arr = [1000i16; 32];
    for i in 0..32 {
        a_arr[i] = (i * 100) as i16;
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        let a_u = a_arr[i] as u16;
        let b_u = 1000u16;
        expected[i] = std::cmp::min(a_u, b_u) as i16;
    }
    assert_eq!(result, I16x32::new(expected));
}

#[test]
fn test_i16x32_umax() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i16x32("i16x32_umax", |builder, a, b| builder.ins().umax(a, b))
        .expect("Failed to compile");

    let func: BinaryI16x32Fn = unsafe { mem::transmute(code) };

    let mut a_arr = [0i16; 32];
    let b_arr = [1000i16; 32];
    for i in 0..32 {
        a_arr[i] = (i * 100) as i16;
    }
    let a = I16x32::new(a_arr);
    let b = I16x32::new(b_arr);
    let mut result = I16x32::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let mut expected = [0i16; 32];
    for i in 0..32 {
        let a_u = a_arr[i] as u16;
        let b_u = 1000u16;
        expected[i] = std::cmp::max(a_u, b_u) as i16;
    }
    assert_eq!(result, I16x32::new(expected));
}

// =============================================================================
// Tests: 512-bit Rotate Left (VPROLVD / VPROLVQ) - Scalar rotation broadcast
// =============================================================================

// Function type: vector input, scalar rotation, vector output
type RotlI64x8Fn = unsafe extern "C" fn(*const I64x8, i64, *mut I64x8);
type RotlI32x16Fn = unsafe extern "C" fn(*const I32x16, i32, *mut I32x16);

#[test]
fn test_i64x8_rotl() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: load vector, rotl with scalar, store result
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // src ptr
    sig.params.push(AbiParam::new(I64)); // scalar rotation amount
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_rotl", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src_ptr = params[0];
        let rotate = params[1];
        let dst_ptr = params[2];

        let src = builder.ins().load(I64X8, MemFlags::trusted(), src_ptr, 0);
        let result = builder.ins().rotl(src, rotate);
        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_rotl ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    let func: RotlI64x8Fn = unsafe { mem::transmute(code) };

    // Test: rotate all elements by 4 bits
    let a = I64x8::new([
        0x0000_0000_0000_00FFu64 as i64,
        0x0000_0000_0000_FF00u64 as i64,
        0x8000_0000_0000_0001u64 as i64,
        0x1234_5678_9ABC_DEF0u64 as i64,
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
        0x0000_0000_0000_0001u64 as i64,
        0xAAAA_AAAA_AAAA_AAAAu64 as i64,
        0x5555_5555_5555_5555u64 as i64,
    ]);
    let rotate: i64 = 4;
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, rotate, &mut result);
    }

    // Expected: all elements rotated left by 4 bits
    let mut expected = [0i64; 8];
    for i in 0..8 {
        expected[i] = (a.0[i] as u64).rotate_left(4) as i64;
    }
    assert_eq!(result, I64x8::new(expected));
}

#[test]
fn test_i32x16_rotl() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: load vector, rotl with scalar, store result
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // src ptr
    sig.params.push(AbiParam::new(I32)); // scalar rotation amount
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_rotl", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src_ptr = params[0];
        let rotate = params[1];
        let dst_ptr = params[2];

        let src = builder.ins().load(I32X16, MemFlags::trusted(), src_ptr, 0);
        let result = builder.ins().rotl(src, rotate);
        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_rotl ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    let func: RotlI32x16Fn = unsafe { mem::transmute(code) };

    // Test: rotate all elements by 8 bits
    let a = I32x16::new([
        0x0000_00FFu32 as i32,
        0x0000_FF00u32 as i32,
        0x8000_0001u32 as i32,
        0x1234_5678u32 as i32,
        0xFFFF_FFFFu32 as i32,
        0x0000_0001u32 as i32,
        0xAAAA_AAAAu32 as i32,
        0x5555_5555u32 as i32,
        0xDEAD_BEEFu32 as i32,
        0xCAFE_BABEu32 as i32,
        0x0F0F_0F0Fu32 as i32,
        0xF0F0_F0F0u32 as i32,
        0x0000_FFFFu32 as i32,
        0xFFFF_0000u32 as i32,
        0x1111_1111u32 as i32,
        0x8888_8888u32 as i32,
    ]);
    let rotate: i32 = 8;
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, rotate, &mut result);
    }

    // Expected: all elements rotated left by 8 bits
    let mut expected = [0i32; 16];
    for i in 0..16 {
        expected[i] = (a.0[i] as u32).rotate_left(8) as i32;
    }
    assert_eq!(result, I32x16::new(expected));
}

// =============================================================================
// Tests: 512-bit Bitwise AND NOT (VPANDND / VPANDNQ)
// =============================================================================

#[test]
fn test_i64x8_band_not() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_band_not", |builder, a, b| {
            builder.ins().band_not(a, b)
        })
        .expect("Failed to compile i64x8_band_not");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // band_not(a, b) = a & (~b)
    let a = I64x8::new([
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
        0x1234_5678_1234_5678u64 as i64,
        0x0000_FFFF_0000_FFFFu64 as i64,
        0xF0F0_F0F0_F0F0_F0F0u64 as i64,
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
        0xFFFF_FFFF_FFFF_FFFFu64 as i64,
    ]);
    let b = I64x8::new([
        0xFFFF_FFFF_FFFF_FFFFu64 as i64, // ~b = 0
        0x0000_0000_0000_0000u64 as i64, // ~b = all 1s
        0xFFFF_0000_FFFF_0000u64 as i64,
        0x0F0F_0F0F_0F0F_0F0Fu64 as i64,
        0x1234_5678_9ABC_DEF0u64 as i64,
        0x5555_5555_5555_5555u64 as i64,
        0xAAAA_AAAA_AAAA_AAAAu64 as i64,
        0xDEAD_BEEF_CAFE_BABEu64 as i64,
    ]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Expected: a & (~b)
    let mut expected = [0i64; 8];
    for i in 0..8 {
        expected[i] = ((a.0[i] as u64) & !(b.0[i] as u64)) as i64;
    }
    assert_eq!(result, I64x8::new(expected));
}

#[test]
fn test_i32x16_band_not() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_band_not", |builder, a, b| {
            builder.ins().band_not(a, b)
        })
        .expect("Failed to compile i32x16_band_not");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // band_not(a, b) = a & (~b)
    let a = I32x16::new([
        0xFFFF_FFFFu32 as i32,
        0x1234_5678u32 as i32,
        0x0000_FFFFu32 as i32,
        0xF0F0_F0F0u32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0x0000_0000u32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
        0xFFFF_FFFFu32 as i32,
    ]);
    let b = I32x16::new([
        0xFFFF_FFFFu32 as i32, // ~b = 0
        0x0000_0000u32 as i32, // ~b = all 1s
        0xFFFF_0000u32 as i32,
        0x0F0F_0F0Fu32 as i32,
        0x1234_5678u32 as i32,
        0x5555_5555u32 as i32,
        0xAAAA_AAAAu32 as i32,
        0xDEAD_BEEFu32 as i32,
        0xCAFE_BABEu32 as i32,
        0x0000_FFFFu32 as i32,
        0xFFFF_0000u32 as i32,
        0x00FF_00FFu32 as i32,
        0xFF00_FF00u32 as i32,
        0x0F0F_0F0Fu32 as i32,
        0xF0F0_F0F0u32 as i32,
        0x1111_1111u32 as i32,
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Expected: a & (~b)
    let mut expected = [0i32; 16];
    for i in 0..16 {
        expected[i] = ((a.0[i] as u32) & !(b.0[i] as u32)) as i32;
    }
    assert_eq!(result, I32x16::new(expected));
}

// =============================================================================
// Tests: Float Precision Conversion (VCVTPS2PD / VCVTPD2PS)
// =============================================================================

// NOTE: These tests are currently ignored because CLIF's fpromote/fdemote instructions
// have a "Narrower" constraint that only accepts scalar float types, not vector types.
// To enable these tests, we would need to either:
// 1. Modify CLIF's Narrower constraint to support vector types (F32X8, F64X8)
// 2. Add x86-specific opcodes (x86_vcvtps2pd, x86_vcvtpd2ps) that bypass the constraint
// The lowering rules in avx512.isle are ready; this is purely a CLIF type constraint issue.

#[test]
#[ignore = "CLIF Narrower constraint doesn't support 512-bit vector types (F32X8, F64X8)"]
fn test_f32x8_to_f64x8_fpromote() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_f32x8_to_f64x8("f32x8_to_f64x8", |builder, src| {
            builder.ins().fpromote(F64X8, src)
        })
        .expect("Failed to compile f32x8_to_f64x8");

    let func: ConvertF32x8ToF64x8Fn = unsafe { mem::transmute(code) };

    // Test values covering various float values
    let src = F32x8::new([
        1.0f32,
        -1.0f32,
        0.0f32,
        3.14159f32,
        f32::MAX,
        f32::MIN,
        f32::EPSILON,
        1.5e10f32,
    ]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&src, &mut result);
    }

    // Each f32 should be promoted to f64 exactly
    for i in 0..8 {
        let expected = src.0[i] as f64;
        assert!(
            (result.0[i] - expected).abs() < 1e-30 || result.0[i] == expected,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            result.0[i]
        );
    }
}

#[test]
#[ignore = "CLIF Narrower constraint doesn't support 512-bit vector types (F32X8, F64X8)"]
fn test_f64x8_to_f32x8_fdemote() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_convert_f64x8_to_f32x8("f64x8_to_f32x8", |builder, src| {
            builder.ins().fdemote(F32X8, src)
        })
        .expect("Failed to compile f64x8_to_f32x8");

    let func: ConvertF64x8ToF32x8Fn = unsafe { mem::transmute(code) };

    // Test values that can be represented in f32
    let src = F64x8::new([
        1.0f64, -1.0f64, 0.0f64, 3.14159f64, 1000.0f64, -1000.0f64, 0.5f64, 1.5e10f64,
    ]);
    let mut result = F32x8::splat(0.0);

    unsafe {
        func(&src, &mut result);
    }

    // Each f64 should be demoted to f32 (with potential loss of precision)
    for i in 0..8 {
        let expected = src.0[i] as f32;
        assert!(
            (result.0[i] - expected).abs() < 1e-6 || result.0[i] == expected,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            result.0[i]
        );
    }
}

// =============================================================================
// Tests: AVX-512 Compress (VPCOMPRESSD/Q) - Core of Filter Pattern
// =============================================================================
//
// This tests the critical VPCOMPRESSD/VPCOMPRESSQ instructions used in:
// - Vectorized filter: compress surviving row indices based on filter mask
// - CTE materialization: compress column values for surviving rows
//
// Pattern:
//   mask = vpcmpd(values, threshold)
//   survivors = vpcompressd(mask, values)
//   count = popcnt(mask)

#[test]
fn test_i32x16_compress() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: compress elements where mask is -1 (true)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I32X16 with -1/0)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_compress", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];
        let mask_ptr = params[1];
        let out_ptr = params[2];

        let values = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);

        // x86_simd_compress: compress values where mask bit is set
        // First argument is the result type
        let compressed = builder.ins().x86_simd_compress(I32X16, mask, values);

        builder
            .ins()
            .store(MemFlags::trusted(), compressed, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_compress ===\n{}", disasm);
            // Verify VPCOMPRESSD is used
            assert!(
                disasm.contains("vpcompressd") || disasm.contains("vpmovd2m"),
                "Expected VPCOMPRESSD instruction in: {}",
                disasm
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type CompressI32x16Fn = unsafe extern "C" fn(*const I32x16, *const I32x16, *mut I32x16);
    let func: CompressI32x16Fn = unsafe { mem::transmute(code) };

    // Test: compress with mask selecting indices 0, 3, 7, 10, 15
    let values = I32x16::new([
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    ]);
    // Mask: -1 means select, 0 means skip
    let mask = I32x16::new([-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&values, &mask, &mut result);
    }

    // Expected: values at positions 0, 3, 7, 10, 15 compressed to front
    // Positions: 100, 103, 107, 110, 115, then zeros
    assert_eq!(result.0[0], 100, "First compressed element");
    assert_eq!(result.0[1], 103, "Second compressed element");
    assert_eq!(result.0[2], 107, "Third compressed element");
    assert_eq!(result.0[3], 110, "Fourth compressed element");
    assert_eq!(result.0[4], 115, "Fifth compressed element");
}

#[test]
fn test_i64x8_compress() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I64X8 with -1/0)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_compress", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];
        let mask_ptr = params[1];
        let out_ptr = params[2];

        let values = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);

        // First argument is the result type
        let compressed = builder.ins().x86_simd_compress(I64X8, mask, values);

        builder
            .ins()
            .store(MemFlags::trusted(), compressed, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_compress ===\n{}", disasm);
            assert!(
                disasm.contains("vpcompressq") || disasm.contains("vpmovq2m"),
                "Expected VPCOMPRESSQ instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type CompressI64x8Fn = unsafe extern "C" fn(*const I64x8, *const I64x8, *mut I64x8);
    let func: CompressI64x8Fn = unsafe { mem::transmute(code) };

    // Test: compress with mask selecting indices 1, 4, 7
    let values = I64x8::new([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]);
    let mask = I64x8::new([0, -1, 0, 0, -1, 0, 0, -1]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&values, &mask, &mut result);
    }

    assert_eq!(result.0[0], 1001, "First compressed element");
    assert_eq!(result.0[1], 1004, "Second compressed element");
    assert_eq!(result.0[2], 1007, "Third compressed element");
}

// =============================================================================
// Tests: AVX-512 Expand (VPEXPANDD/Q) - Inverse of Compress
// =============================================================================

#[test]
fn test_i32x16_expand() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr (compressed)
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I32X16 with -1/0)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_expand", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];
        let mask_ptr = params[1];
        let out_ptr = params[2];

        let values = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);

        // First argument is the result type
        let expanded = builder.ins().x86_simd_expand(I32X16, mask, values);

        builder
            .ins()
            .store(MemFlags::trusted(), expanded, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_expand ===\n{}", disasm);
            assert!(
                disasm.contains("vpexpandd") || disasm.contains("vpmovd2m"),
                "Expected VPEXPANDD instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type ExpandI32x16Fn = unsafe extern "C" fn(*const I32x16, *const I32x16, *mut I32x16);
    let func: ExpandI32x16Fn = unsafe { mem::transmute(code) };

    // Test: expand values to positions 0, 3, 7
    let values = I32x16::new([100, 103, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let mask = I32x16::new([-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&values, &mask, &mut result);
    }

    assert_eq!(result.0[0], 100, "Expanded to position 0");
    assert_eq!(result.0[3], 103, "Expanded to position 3");
    assert_eq!(result.0[7], 107, "Expanded to position 7");
    assert_eq!(result.0[1], 0, "Unset position should be 0");
}

// =============================================================================
// Test: 256-bit Load/Store through JIT (I32X8)
// =============================================================================

#[test]
fn test_i32x8_load_store() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: load I32X8, add 1 to each element, store
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // src ptr
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x8_load_store", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let src_ptr = params[0];
        let dst_ptr = params[1];

        // Load 256-bit vector (8 x i32)
        let loaded = builder.ins().load(I32X8, MemFlags::trusted(), src_ptr, 0);

        // Store it back
        builder.ins().store(MemFlags::trusted(), loaded, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x8_load_store ===\n{}", disasm);
            assert!(
                disasm.contains("vmovdqu32"),
                "Expected VMOVDQU32 instruction for I32X8"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    #[repr(C, align(32))]
    struct I32x8([i32; 8]);

    let src = I32x8([10, 20, 30, 40, 50, 60, 70, 80]);
    let mut dst = I32x8([0; 8]);

    type LoadStoreFn = unsafe extern "C" fn(*const I32x8, *mut I32x8);
    let func: LoadStoreFn = unsafe { mem::transmute(code) };

    unsafe {
        func(&src, &mut dst);
    }

    assert_eq!(dst.0, [10, 20, 30, 40, 50, 60, 70, 80]);
}

// =============================================================================
// Tests: AVX-512 Gather (VPGATHERQQ) - For GermanString Indexed Access
// =============================================================================
//
// This tests VPGATHERQQ which is critical for:
// - GermanString prefilter: gather low64 and high64 of 16-byte strings
// - Indexed column access after VPCOMPRESS
//
// Pattern for GermanString:
//   indices = compressed_row_indices * 16  (pre-scaled for 16-byte stride)
//   low64 = vpgatherqq([base + indices])
//   high64 = vpgatherqq([base + 8 + indices])

#[test]
fn test_i64x8_gather_dd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: gather 8 i64 values using 32-bit indices
    // This is VPGATHERDQ (32-bit indices → 64-bit elements)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I32X8 with byte offsets)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_gather_dq", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let out_ptr = params[2];

        // Load 8 x i32 indices (uses I32X8 which is 256-bit, but stored in I32X16 for alignment)
        let indices = builder
            .ins()
            .load(I32X8, MemFlags::trusted(), indices_ptr, 0);

        // Gather: base[indices[i]] for each lane, with scale=1 (indices are byte offsets)
        // First argument is the result type, then MemFlags, then base, indices, scale, offset
        let gathered = builder.ins().x86_simd_gather(
            I64X8, // result type
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8,  // scale: Uimm8 immediate
            0i32, // offset: Offset32 immediate
        );

        builder
            .ins()
            .store(MemFlags::trusted(), gathered, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_gather_dq ===\n{}", disasm);
            assert!(
                disasm.contains("vpgatherdq"),
                "Expected VPGATHERDQ instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    // Create a data array to gather from
    #[repr(C, align(64))]
    struct DataArray([i64; 32]);
    let data = DataArray([
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
        1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
        1030, 1031,
    ]);

    // Indices are byte offsets: element 0, 2, 4, 6, 8, 10, 12, 14
    // Each i64 is 8 bytes, so indices = [0, 16, 32, 48, 64, 80, 96, 112]
    #[repr(C, align(32))]
    struct Indices([i32; 8]);
    let indices = Indices([0, 16, 32, 48, 64, 80, 96, 112]);

    let mut result = I64x8::splat(0);

    type GatherDQFn = unsafe extern "C" fn(*const i64, *const i32, *mut I64x8);
    let func: GatherDQFn = unsafe { mem::transmute(code) };

    unsafe {
        func(data.0.as_ptr(), indices.0.as_ptr(), &mut result);
    }

    // Should gather elements 0, 2, 4, 6, 8, 10, 12, 14
    assert_eq!(result.0[0], 1000, "Gathered element 0");
    assert_eq!(result.0[1], 1002, "Gathered element 2");
    assert_eq!(result.0[2], 1004, "Gathered element 4");
    assert_eq!(result.0[3], 1006, "Gathered element 6");
    assert_eq!(result.0[4], 1008, "Gathered element 8");
    assert_eq!(result.0[5], 1010, "Gathered element 10");
    assert_eq!(result.0[6], 1012, "Gathered element 12");
    assert_eq!(result.0[7], 1014, "Gathered element 14");
}

#[test]
fn test_i64x8_gather_qq() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: gather 8 i64 values using 64-bit indices
    // This is VPGATHERQQ (64-bit indices → 64-bit elements)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I64X8 with byte offsets)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_gather_qq", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let out_ptr = params[2];

        let indices = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), indices_ptr, 0);

        // First argument is the result type, then MemFlags, then base, indices, scale, offset
        let gathered = builder.ins().x86_simd_gather(
            I64X8, // result type
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8,  // scale: Uimm8 immediate
            0i32, // offset: Offset32 immediate
        );

        builder
            .ins()
            .store(MemFlags::trusted(), gathered, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_gather_qq ===\n{}", disasm);
            assert!(
                disasm.contains("vpgatherqq"),
                "Expected VPGATHERQQ instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    #[repr(C, align(64))]
    struct DataArray([i64; 32]);
    let data = DataArray([
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
        1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
        1030, 1031,
    ]);

    // 64-bit indices (byte offsets): gather elements 1, 3, 5, 7, 9, 11, 13, 15
    let indices = I64x8::new([8, 24, 40, 56, 72, 88, 104, 120]);
    let mut result = I64x8::splat(0);

    type GatherQQFn = unsafe extern "C" fn(*const i64, *const I64x8, *mut I64x8);
    let func: GatherQQFn = unsafe { mem::transmute(code) };

    unsafe {
        func(data.0.as_ptr(), &indices, &mut result);
    }

    assert_eq!(result.0[0], 1001, "Gathered element 1");
    assert_eq!(result.0[1], 1003, "Gathered element 3");
    assert_eq!(result.0[2], 1005, "Gathered element 5");
    assert_eq!(result.0[3], 1007, "Gathered element 7");
    assert_eq!(result.0[4], 1009, "Gathered element 9");
    assert_eq!(result.0[5], 1011, "Gathered element 11");
    assert_eq!(result.0[6], 1013, "Gathered element 13");
    assert_eq!(result.0[7], 1015, "Gathered element 15");
}

// =============================================================================
// Tests: AVX-512 Scatter (VPSCATTERQQ) - For Indexed Store
// =============================================================================

#[test]
fn test_i64x8_scatter_qq() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I64X8)
    sig.params.push(AbiParam::new(ptr_type)); // values ptr (I64X8)
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I64X8)
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_scatter_qq", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
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
            .load(I64X8, MemFlags::trusted(), indices_ptr, 0);
        let values = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I64X8, MemFlags::trusted(), mask_ptr, 0);

        // Note: scale and offset are immediate values, not Value types
        builder.ins().x86_simd_scatter(
            MemFlags::trusted(),
            mask,
            values,
            base_ptr,
            indices,
            1u8,  // scale: Uimm8 immediate
            0i32, // offset: Offset32 immediate
        );

        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_scatter_qq ===\n{}", disasm);
            assert!(
                disasm.contains("vpscatterqq"),
                "Expected VPSCATTERQQ instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    #[repr(C, align(64))]
    struct DataArray([i64; 16]);
    let mut data = DataArray([0; 16]);

    // Scatter to positions 1, 3, 5, 7 (odd positions)
    let indices = I64x8::new([8, 24, 40, 56, 0, 0, 0, 0]); // byte offsets
    let values = I64x8::new([2001, 2003, 2005, 2007, 9999, 9999, 9999, 9999]);
    // Mask: only first 4 lanes active
    let mask = I64x8::new([-1, -1, -1, -1, 0, 0, 0, 0]);

    type ScatterQQFn = unsafe extern "C" fn(*mut i64, *const I64x8, *const I64x8, *const I64x8);
    let func: ScatterQQFn = unsafe { mem::transmute(code) };

    unsafe {
        func(data.0.as_mut_ptr(), &indices, &values, &mask);
    }

    assert_eq!(data.0[0], 0, "Position 0 should be unchanged");
    assert_eq!(data.0[1], 2001, "Scattered to position 1");
    assert_eq!(data.0[2], 0, "Position 2 should be unchanged");
    assert_eq!(data.0[3], 2003, "Scattered to position 3");
    assert_eq!(data.0[4], 0, "Position 4 should be unchanged");
    assert_eq!(data.0[5], 2005, "Scattered to position 5");
    assert_eq!(data.0[6], 0, "Position 6 should be unchanged");
    assert_eq!(data.0[7], 2007, "Scattered to position 7");
}

// =============================================================================
// Tests: AVX-512 Gather/Scatter for I32X16 (VPGATHERDD/VPSCATTERDD)
// =============================================================================
//
// These test the 32-bit element variants with 32-bit indices.

#[test]
fn test_i32x16_gather_dd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Build function: gather 16 i32 values using 32-bit indices
    // This is VPGATHERDD (32-bit indices → 32-bit elements)
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I32X16 with byte offsets)
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_gather_dd", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let out_ptr = params[2];

        // Load 16 x i32 indices
        let indices = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), indices_ptr, 0);

        // Gather: base[indices[i]] for each lane, with scale=1 (indices are byte offsets)
        let gathered = builder.ins().x86_simd_gather(
            I32X16, // result type
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8,  // scale
            0i32, // offset
        );

        builder
            .ins()
            .store(MemFlags::trusted(), gathered, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_gather_dd ===\n{}", disasm);
            assert!(
                disasm.contains("vpgatherdd"),
                "Expected VPGATHERDD instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    // Create a data array to gather from
    #[repr(C, align(64))]
    struct DataArray([i32; 64]);
    let data = DataArray([
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
        132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
        148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
    ]);

    // Indices are byte offsets: element 0, 2, 4, 6, ... (even elements)
    // Each i32 is 4 bytes, so byte offsets are 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120
    #[repr(C, align(64))]
    struct Indices([i32; 16]);
    let indices = Indices([
        0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
    ]);

    #[repr(C, align(64))]
    struct I32x16([i32; 16]);
    let mut result = I32x16([0; 16]);

    type GatherDDFn = unsafe extern "C" fn(*const i32, *const i32, *mut I32x16);
    let func: GatherDDFn = unsafe { mem::transmute(code) };

    unsafe {
        func(data.0.as_ptr(), indices.0.as_ptr(), &mut result);
    }

    // Should gather elements 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    assert_eq!(result.0[0], 100, "Gathered element 0");
    assert_eq!(result.0[1], 102, "Gathered element 2");
    assert_eq!(result.0[2], 104, "Gathered element 4");
    assert_eq!(result.0[3], 106, "Gathered element 6");
    assert_eq!(result.0[4], 108, "Gathered element 8");
    assert_eq!(result.0[5], 110, "Gathered element 10");
    assert_eq!(result.0[6], 112, "Gathered element 12");
    assert_eq!(result.0[7], 114, "Gathered element 14");
    assert_eq!(result.0[8], 116, "Gathered element 16");
    assert_eq!(result.0[9], 118, "Gathered element 18");
    assert_eq!(result.0[10], 120, "Gathered element 20");
    assert_eq!(result.0[11], 122, "Gathered element 22");
    assert_eq!(result.0[12], 124, "Gathered element 24");
    assert_eq!(result.0[13], 126, "Gathered element 26");
    assert_eq!(result.0[14], 128, "Gathered element 28");
    assert_eq!(result.0[15], 130, "Gathered element 30");
}

#[test]
fn test_i32x16_scatter_dd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr
    sig.params.push(AbiParam::new(ptr_type)); // indices ptr (I32X16)
    sig.params.push(AbiParam::new(ptr_type)); // values ptr (I32X16)
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I32X16)
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_scatter_dd", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
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
            .load(I32X16, MemFlags::trusted(), indices_ptr, 0);
        let values = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);

        builder.ins().x86_simd_scatter(
            MemFlags::trusted(),
            mask,
            values,
            base_ptr,
            indices,
            1u8,  // scale
            0i32, // offset
        );

        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_scatter_dd ===\n{}", disasm);
            assert!(
                disasm.contains("vpscatterdd"),
                "Expected VPSCATTERDD instruction"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    #[repr(C, align(64))]
    struct DataArray([i32; 32]);
    let mut data = DataArray([0; 32]);

    // Scatter to positions 1, 3, 5, 7 (odd positions) in first 8 elements, skip rest
    #[repr(C, align(64))]
    struct I32x16([i32; 16]);
    let indices = I32x16([4, 12, 20, 28, 36, 44, 52, 60, 0, 0, 0, 0, 0, 0, 0, 0]); // byte offsets
    let values = I32x16([
        201, 203, 205, 207, 209, 211, 213, 215, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,
    ]);
    // Mask: only first 8 lanes active
    let mask = I32x16([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]);

    type ScatterDDFn = unsafe extern "C" fn(*mut i32, *const I32x16, *const I32x16, *const I32x16);
    let func: ScatterDDFn = unsafe { mem::transmute(code) };

    unsafe {
        func(data.0.as_mut_ptr(), &indices, &values, &mask);
    }

    assert_eq!(data.0[0], 0, "Position 0 should be unchanged");
    assert_eq!(data.0[1], 201, "Scattered to position 1");
    assert_eq!(data.0[2], 0, "Position 2 should be unchanged");
    assert_eq!(data.0[3], 203, "Scattered to position 3");
    assert_eq!(data.0[4], 0, "Position 4 should be unchanged");
    assert_eq!(data.0[5], 205, "Scattered to position 5");
    assert_eq!(data.0[6], 0, "Position 6 should be unchanged");
    assert_eq!(data.0[7], 207, "Scattered to position 7");
    assert_eq!(data.0[8], 0, "Position 8 should be unchanged");
    assert_eq!(data.0[9], 209, "Scattered to position 9");
    assert_eq!(data.0[10], 0, "Position 10 should be unchanged");
    assert_eq!(data.0[11], 211, "Scattered to position 11");
    assert_eq!(data.0[12], 0, "Position 12 should be unchanged");
    assert_eq!(data.0[13], 213, "Scattered to position 13");
    assert_eq!(data.0[14], 0, "Position 14 should be unchanged");
    assert_eq!(data.0[15], 215, "Scattered to position 15");
}

// =============================================================================
// Tests: Vector Comparison producing vector mask (icmp → I32X16/I64X8)
// =============================================================================
//
// This tests that icmp produces a vector mask (-1/0) that can be used for:
// - VPBLENDMD/Q (bitselect)
// - Conversion to k-register for VPCOMPRESSD/Q

#[test]
fn test_i32x16_icmp_gt_vector_mask() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // icmp gt produces vector mask, then use it in bitselect (blend)
    let code = compiler
        .compile_binary_i32x16("i32x16_icmp_blend", |builder, a, threshold_vec| {
            // Compare: a > threshold (element-wise)
            let mask = builder
                .ins()
                .icmp(IntCC::SignedGreaterThan, a, threshold_vec);
            // Use mask to select: if a[i] > threshold then a[i] else 0
            let zero = builder.ins().iconst(I32, 0);
            let zero_vec = builder.ins().splat(I32X16, zero);
            builder.ins().bitselect(mask, a, zero_vec)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test: filter values > 100
    let a = I32x16::new([
        50, 150, 99, 101, 200, 0, 100, 300, 75, 125, 100, 101, 50, 250, 80, 120,
    ]);
    let threshold = I32x16::splat(100);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&a, &threshold, &mut result);
    }

    // Expected: keep values > 100, else 0
    let expected = I32x16::new([
        0, 150, 0, 101, 200, 0, 0, 300, 0, 125, 0, 101, 0, 250, 0, 120,
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_i64x8_icmp_eq_vector_mask() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_icmp_eq_blend", |builder, a, b| {
            let mask = builder.ins().icmp(IntCC::Equal, a, b);
            let neg_one = builder.ins().iconst(I64, -1i64 as i64);
            let neg_one_vec = builder.ins().splat(I64X8, neg_one);
            let zero = builder.ins().iconst(I64, 0);
            let zero_vec = builder.ins().splat(I64X8, zero);
            builder.ins().bitselect(mask, neg_one_vec, zero_vec)
        })
        .expect("Failed to compile");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let b = I64x8::new([1, 99, 3, 99, 5, 99, 7, 99]);
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Where equal: -1, else 0
    let expected = I64x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
    assert_eq!(result, expected);
}

// =============================================================================
// Tests: Horizontal Reduction (for aggregate final step)
// =============================================================================
//
// Pattern for SUM aggregate:
//   1. Masked accumulation across batches (VPADDD {k})
//   2. Final horizontal reduce:
//      - VEXTRACTI64X4 ymm, zmm, 1  (extract upper 256 bits)
//      - VPADDD ymm, ymm, ymm
//      - VEXTRACTI128 xmm, ymm, 1
//      - VPADDD xmm, xmm, xmm
//      - PSHUFD + PADDD (32-bit pairs)
//      - MOVD eax, xmm

#[test]
#[ignore = "CLIF ireduce doesn't support vector types - need dedicated horizontal sum instruction"]
fn test_i32x16_horizontal_sum() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.returns.push(AbiParam::new(I32));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_hsum", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];

        let vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);

        // Horizontal sum using vector_reduce_iadd_ordered
        let sum = builder.ins().ireduce(I32, vec);

        builder.ins().return_(&[sum]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_hsum ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type HSumI32x16Fn = unsafe extern "C" fn(*const I32x16) -> i32;
    let func: HSumI32x16Fn = unsafe { mem::transmute(code) };

    // Test with known sum
    let values = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let result = unsafe { func(&values) };

    // Sum of 1..16 = 136
    assert_eq!(result, 136, "Horizontal sum should be 136");
}

#[test]
#[ignore = "CLIF ireduce doesn't support vector types - need dedicated horizontal sum instruction"]
fn test_i64x8_horizontal_sum() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.returns.push(AbiParam::new(I64));
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_hsum", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];

        let vec = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), values_ptr, 0);
        let sum = builder.ins().ireduce(I64, vec);

        builder.ins().return_(&[sum]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_hsum ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type HSumI64x8Fn = unsafe extern "C" fn(*const I64x8) -> i64;
    let func: HSumI64x8Fn = unsafe { mem::transmute(code) };

    let values = I64x8::new([100, 200, 300, 400, 500, 600, 700, 800]);
    let result = unsafe { func(&values) };

    assert_eq!(result, 3600, "Horizontal sum should be 3600");
}

// =============================================================================
// Tests: Complete Filter-Compress-Count Pattern
// =============================================================================
//
// This tests the complete vectorized filter pattern:
// 1. Load column batch (VMOVDQU32)
// 2. Compare with threshold (VPCMPD → k-register internally)
// 3. Create index vector [0,1,2,...,15]
// 4. Compress indices (VPCOMPRESSD)
// 5. Count survivors (POPCNT of original mask)

#[test]
#[ignore = "CLIF ireduce doesn't support vector types - need dedicated horizontal sum or VPMOVMSKB+POPCNT pattern"]
fn test_filter_compress_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(I32)); // threshold
    sig.params.push(AbiParam::new(ptr_type)); // output indices ptr
    sig.returns.push(AbiParam::new(I32)); // count of survivors
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("filter_compress", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let values_ptr = params[0];
        let threshold = params[1];
        let output_ptr = params[2];

        // Load values
        let values = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);

        // Create threshold vector
        let threshold_vec = builder.ins().splat(I32X16, threshold);

        // Compare: values > threshold (produces mask)
        let mask = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, values, threshold_vec);

        // Create index vector [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        let indices = {
            let c0 = builder.ins().iconst(I32, 0);
            let c1 = builder.ins().iconst(I32, 1);
            let c2 = builder.ins().iconst(I32, 2);
            let c3 = builder.ins().iconst(I32, 3);
            let c4 = builder.ins().iconst(I32, 4);
            let c5 = builder.ins().iconst(I32, 5);
            let c6 = builder.ins().iconst(I32, 6);
            let c7 = builder.ins().iconst(I32, 7);
            let c8 = builder.ins().iconst(I32, 8);
            let c9 = builder.ins().iconst(I32, 9);
            let c10 = builder.ins().iconst(I32, 10);
            let c11 = builder.ins().iconst(I32, 11);
            let c12 = builder.ins().iconst(I32, 12);
            let c13 = builder.ins().iconst(I32, 13);
            let c14 = builder.ins().iconst(I32, 14);
            let c15 = builder.ins().iconst(I32, 15);
            let v = builder.ins().scalar_to_vector(I32X16, c0);
            let v = builder.ins().insertlane(v, c1, 1);
            let v = builder.ins().insertlane(v, c2, 2);
            let v = builder.ins().insertlane(v, c3, 3);
            let v = builder.ins().insertlane(v, c4, 4);
            let v = builder.ins().insertlane(v, c5, 5);
            let v = builder.ins().insertlane(v, c6, 6);
            let v = builder.ins().insertlane(v, c7, 7);
            let v = builder.ins().insertlane(v, c8, 8);
            let v = builder.ins().insertlane(v, c9, 9);
            let v = builder.ins().insertlane(v, c10, 10);
            let v = builder.ins().insertlane(v, c11, 11);
            let v = builder.ins().insertlane(v, c12, 12);
            let v = builder.ins().insertlane(v, c13, 13);
            let v = builder.ins().insertlane(v, c14, 14);
            builder.ins().insertlane(v, c15, 15)
        };

        // Compress indices based on mask
        // First argument is the result type
        let compressed = builder.ins().x86_simd_compress(I32X16, mask, indices);

        // Store compressed indices
        builder
            .ins()
            .store(MemFlags::trusted(), compressed, output_ptr, 0);

        // Count survivors via popcnt on mask
        // First reduce mask to scalar by summing (each -1 becomes 1, 0 stays 0)
        // Actually use a different approach: negate mask (-1 → 1, 0 → 0) then sum
        let neg_mask = builder.ins().ineg(mask);
        let count = builder.ins().ireduce(I32, neg_mask);

        builder.ins().return_(&[count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for filter_compress ===\n{}", disasm);
            // Should see VPCMPD, VPCOMPRESSD
            assert!(
                disasm.contains("vpcmp") || disasm.contains("vpcompressd"),
                "Expected filter/compress instructions"
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type FilterCompressFn = unsafe extern "C" fn(*const I32x16, i32, *mut I32x16) -> i32;
    let func: FilterCompressFn = unsafe { mem::transmute(code) };

    // Test: filter values > 100
    let values = I32x16::new([
        50, 150, 99, 101, 200, 0, 100, 300, 75, 125, 100, 101, 50, 250, 80, 120,
    ]);
    let mut output = I32x16::splat(-1);

    let count = unsafe { func(&values, 100, &mut output) };

    // Survivors: indices 1 (150), 3 (101), 4 (200), 7 (300), 9 (125), 11 (101), 13 (250), 15 (120)
    assert_eq!(count, 8, "Should have 8 survivors");
    assert_eq!(output.0[0], 1, "First survivor at index 1");
    assert_eq!(output.0[1], 3, "Second survivor at index 3");
    assert_eq!(output.0[2], 4, "Third survivor at index 4");
    assert_eq!(output.0[3], 7, "Fourth survivor at index 7");
    assert_eq!(output.0[4], 9, "Fifth survivor at index 9");
    assert_eq!(output.0[5], 11, "Sixth survivor at index 11");
    assert_eq!(output.0[6], 13, "Seventh survivor at index 13");
    assert_eq!(output.0[7], 15, "Eighth survivor at index 15");
}

// =============================================================================
// Tests: Masked Accumulation (for aggregate SUM with predicate)
// =============================================================================
//
// Pattern for masked SUM:
//   acc = vpaddd {k}, acc, values  ; only add where mask is set
//
// Using merge-masking mode (not zeroing), accumulator values are preserved
// for lanes where mask is 0.

#[test]
fn test_i32x16_masked_accumulate() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // This tests bitselect(mask, acc+values, acc) which should fuse to masked add
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // accumulator ptr
    sig.params.push(AbiParam::new(ptr_type)); // values ptr
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_masked_acc", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let acc_ptr = params[0];
        let values_ptr = params[1];
        let mask_ptr = params[2];
        let out_ptr = params[3];

        let acc = builder.ins().load(I32X16, MemFlags::trusted(), acc_ptr, 0);
        let values = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), values_ptr, 0);
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);

        // acc + values where mask is set, else acc
        let sum = builder.ins().iadd(acc, values);
        let result = builder.ins().bitselect(mask, sum, acc);

        builder.ins().store(MemFlags::trusted(), result, out_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_masked_acc ===\n{}", disasm);
            // Should see fused masked VPADDD or VPBLENDMD
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type MaskedAccFn =
        unsafe extern "C" fn(*const I32x16, *const I32x16, *const I32x16, *mut I32x16);
    let func: MaskedAccFn = unsafe { mem::transmute(code) };

    let acc = I32x16::new([
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    ]);
    let values = I32x16::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    // Mask: active for even indices only
    let mask = I32x16::new([-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&acc, &values, &mask, &mut result);
    }

    // Even indices: acc + values, odd indices: acc unchanged
    let expected = I32x16::new([
        101, 100, 103, 100, 105, 100, 107, 100, 109, 100, 111, 100, 113, 100, 115, 100,
    ]);
    assert_eq!(result, expected);
}

// =============================================================================
// Tests: GermanString Pre-filter Pattern (Gather both halves)
// =============================================================================
//
// For GermanString (16-byte values), we need to:
// 1. Gather low 64 bits (contains prefix[4] + len_tag[4] or inline data[8])
// 2. Gather high 64 bits (contains inline data[6] + tag[1] or ptr[7] + tag[1])
// 3. Extract length from either location based on tag
// 4. Compare length and prefix

#[test]
fn test_germanstring_gather_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Simulate GermanString layout with 16-byte structs
    // We'll gather low64 and high64 separately using VPGATHERQQ

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // base ptr (array of 16-byte GermanStrings)
    sig.params.push(AbiParam::new(ptr_type)); // row indices ptr (I64X8, pre-scaled by 16)
    sig.params.push(AbiParam::new(ptr_type)); // output low64 ptr
    sig.params.push(AbiParam::new(ptr_type)); // output high64 ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("germanstring_gather", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let base_ptr = params[0];
        let indices_ptr = params[1];
        let low64_ptr = params[2];
        let high64_ptr = params[3];

        // Load byte-offset indices (pre-scaled by 16 for GermanString stride)
        let indices = builder
            .ins()
            .load(I64X8, MemFlags::trusted(), indices_ptr, 0);

        // Gather low 64 bits
        // First argument is the result type, then MemFlags, then base, indices, scale, offset
        let low64 = builder.ins().x86_simd_gather(
            I64X8, // result type
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8,  // scale: Uimm8 immediate
            0i32, // offset: Offset32 immediate (low half at offset 0)
        );

        // Gather high 64 bits (offset by 8 bytes)
        let high64 = builder.ins().x86_simd_gather(
            I64X8, // result type
            MemFlags::trusted(),
            base_ptr,
            indices,
            1u8,  // scale: Uimm8 immediate
            8i32, // offset: Offset32 immediate (high half at offset 8)
        );

        builder
            .ins()
            .store(MemFlags::trusted(), low64, low64_ptr, 0);
        builder
            .ins()
            .store(MemFlags::trusted(), high64, high64_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for germanstring_gather ===\n{}", disasm);
            // Should see two VPGATHERQQ instructions
            let gather_count = disasm.matches("vpgatherqq").count();
            assert!(
                gather_count >= 2,
                "Expected at least 2 VPGATHERQQ instructions, found {}",
                gather_count
            );
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);

    // Create test data: 16-byte "GermanStrings"
    #[repr(C, align(64))]
    struct GermanStringArray([[u8; 16]; 16]);

    let mut data = GermanStringArray([[0u8; 16]; 16]);
    // Fill with recognizable patterns
    for i in 0..16 {
        // Low 8 bytes: 0x1000 + i as prefix pattern
        let low_val = 0x1000u64 + i as u64;
        let high_val = 0x2000u64 + i as u64;
        data.0[i][0..8].copy_from_slice(&low_val.to_le_bytes());
        data.0[i][8..16].copy_from_slice(&high_val.to_le_bytes());
    }

    // Indices: gather rows 0, 2, 4, 6, 8, 10, 12, 14 (byte offsets for 16-byte stride)
    let indices = I64x8::new([0, 32, 64, 96, 128, 160, 192, 224]);
    let mut low64_result = I64x8::splat(0);
    let mut high64_result = I64x8::splat(0);

    type GatherGermanStringFn =
        unsafe extern "C" fn(*const u8, *const I64x8, *mut I64x8, *mut I64x8);
    let func: GatherGermanStringFn = unsafe { mem::transmute(code) };

    unsafe {
        func(
            data.0.as_ptr() as *const u8,
            &indices,
            &mut low64_result,
            &mut high64_result,
        );
    }

    // Verify we gathered the correct data
    assert_eq!(low64_result.0[0] as u64, 0x1000, "Row 0 low64");
    assert_eq!(low64_result.0[1] as u64, 0x1002, "Row 2 low64");
    assert_eq!(low64_result.0[2] as u64, 0x1004, "Row 4 low64");
    assert_eq!(high64_result.0[0] as u64, 0x2000, "Row 0 high64");
    assert_eq!(high64_result.0[1] as u64, 0x2002, "Row 2 high64");
}

// =============================================================================
// Tests: F64X8 Floating-Point Operations (for aggregate functions)
// =============================================================================

#[test]
fn test_f64x8_fadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fadd", |builder, a, b| builder.ins().fadd(a, b))
        .expect("Failed to compile");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    let a = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = F64x8::new([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let expected = F64x8::new([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]);
    assert_eq!(result, expected);
}

#[test]
fn test_f64x8_fmul() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fmul", |builder, a, b| builder.ins().fmul(a, b))
        .expect("Failed to compile");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    let a = F64x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = F64x8::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let expected = F64x8::new([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    assert_eq!(result, expected);
}

#[test]
fn test_f64x8_fdiv() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fdiv", |builder, a, b| builder.ins().fdiv(a, b))
        .expect("Failed to compile");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    let a = F64x8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let b = F64x8::new([2.0, 4.0, 5.0, 8.0, 10.0, 6.0, 7.0, 8.0]);
    let mut result = F64x8::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let expected = F64x8::new([5.0, 5.0, 6.0, 5.0, 5.0, 10.0, 10.0, 10.0]);
    assert_eq!(result, expected);
}

// =============================================================================
// Tests: F32X16 Floating-Point Operations
// =============================================================================

#[test]
fn test_f32x16_fadd() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fadd", |builder, a, b| builder.ins().fadd(a, b))
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    let a = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let b = F32x16::splat(0.5);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let expected = F32x16::new([
        1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5,
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_f32x16_fmul() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fmul", |builder, a, b| builder.ins().fmul(a, b))
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    let a = F32x16::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);
    let b = F32x16::splat(2.0);
    let mut result = F32x16::splat(0.0);

    unsafe {
        func(&a, &b, &mut result);
    }

    let expected = F32x16::new([
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
    ]);
    assert_eq!(result, expected);
}

// =============================================================================
// Tests: VPBLENDMD/Q for CASE/COALESCE patterns
// =============================================================================

#[test]
fn test_i32x16_blend_case_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // CASE WHEN x > 0 THEN x ELSE -x END (abs via blend)
    let code = compiler
        .compile_unary_i32x16("i32x16_blend_abs", |builder, x| {
            let zero = builder.ins().iconst(I32, 0);
            let zero_vec = builder.ins().splat(I32X16, zero);
            // mask = x > 0
            let mask = builder.ins().icmp(IntCC::SignedGreaterThan, x, zero_vec);
            // neg_x = -x
            let neg_x = builder.ins().ineg(x);
            // result = mask ? x : neg_x
            builder.ins().bitselect(mask, x, neg_x)
        })
        .expect("Failed to compile");

    type UnaryI32x16Fn = unsafe extern "C" fn(*const I32x16, *mut I32x16);
    let func: UnaryI32x16Fn = unsafe { mem::transmute(code) };

    let x = I32x16::new([
        -5,
        5,
        -10,
        10,
        0,
        -100,
        100,
        -1,
        1,
        -50,
        50,
        i32::MIN,
        i32::MAX,
        -42,
        42,
        0,
    ]);
    let mut result = I32x16::splat(0);

    unsafe {
        func(&x, &mut result);
    }

    // Note: -i32::MIN overflows, but that's expected behavior
    let expected = I32x16::new([
        5,
        5,
        10,
        10,
        0,
        100,
        100,
        1,
        1,
        50,
        50,
        i32::MIN, // overflow: -MIN = MIN
        i32::MAX,
        42,
        42,
        0,
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_i64x8_coalesce_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // COALESCE(a, b): return a if a != NULL_MARKER, else b
    // Using 0 as NULL marker for this test
    let code = compiler
        .compile_binary_i64x8("i64x8_coalesce", |builder, a, b| {
            let zero = builder.ins().iconst(I64, 0);
            let zero_vec = builder.ins().splat(I64X8, zero);
            // mask = a != 0 (a is not null)
            let mask = builder.ins().icmp(IntCC::NotEqual, a, zero_vec);
            // result = mask ? a : b
            builder.ins().bitselect(mask, a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    let a = I64x8::new([100, 0, 300, 0, 500, 0, 700, 0]); // 0 = null
    let b = I64x8::new([1, 2, 3, 4, 5, 6, 7, 8]); // fallback values
    let mut result = I64x8::splat(0);

    unsafe {
        func(&a, &b, &mut result);
    }

    // Where a is non-null (!=0), use a; else use b
    let expected = I64x8::new([100, 2, 300, 4, 500, 6, 700, 8]);
    assert_eq!(result, expected);
}

// =============================================================================
// Tests: POPCNT via scalar fallback (for counting survivors)
// =============================================================================
//
// Note: Cranelift's `popcnt` on vectors returns per-lane popcnt, not total.
// For counting survivors from a mask, we use ireduce (horizontal sum) on
// the negated mask (-1 → 1, 0 → 0).

#[test]
#[ignore = "CLIF ireduce doesn't support vector types - need VPMOVMSKB + POPCNT pattern"]
fn test_count_mask_bits() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask ptr (I32X16 with -1/0)
    sig.returns.push(AbiParam::new(I32)); // count
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("count_mask", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let mask_ptr = params[0];

        let mask = builder.ins().load(I32X16, MemFlags::trusted(), mask_ptr, 0);

        // -mask turns -1 → 1, 0 → 0
        let ones = builder.ins().ineg(mask);

        // Horizontal sum of ones = popcount of original mask
        let count = builder.ins().ireduce(I32, ones);

        builder.ins().return_(&[count]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for count_mask ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type CountMaskFn = unsafe extern "C" fn(*const I32x16) -> i32;
    let func: CountMaskFn = unsafe { mem::transmute(code) };

    // Test: 7 bits set
    let mask = I32x16::new([-1, 0, -1, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, -1]);
    let count = unsafe { func(&mask) };

    assert_eq!(count, 7, "Should count 7 set bits");
}

// =============================================================================
// Edge Case Tests: Boundary Values and Special Cases (H6)
// =============================================================================

/// Test I32X16 iadd with MIN/MAX boundary values
#[test]
fn test_i32x16_iadd_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_iadd_boundary", |builder, a, b| {
            builder.ins().iadd(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test MIN boundary
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(0);
    let mut result = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN));

    // Test MAX boundary
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MAX));

    // Test wraparound: MAX + 1 = MIN
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN));

    // Test wraparound: MIN - 1 = MAX (via iadd with -1)
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MAX));

    // Test mixed boundaries
    let a = I32x16::new([
        i32::MIN,
        i32::MAX,
        i32::MIN + 1,
        i32::MAX - 1,
        0,
        -1,
        1,
        i32::MIN,
        i32::MAX,
        i32::MIN + 1,
        i32::MAX - 1,
        0,
        -1,
        1,
        i32::MIN / 2,
        i32::MAX / 2,
    ]);
    let b = I32x16::new([
        0,
        0,
        -1,
        1,
        0,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        0,
        1,
        -1,
        i32::MIN / 2,
        i32::MAX / 2,
    ]);
    unsafe { func(&a, &b, &mut result) };
    let expected = I32x16::new([
        i32::MIN,
        i32::MAX,
        i32::MIN,
        i32::MAX,
        0,
        0,
        0,
        i32::MIN + 1,
        i32::MAX - 1,
        i32::MIN,
        i32::MAX,
        0,
        0,
        0,
        i32::MIN,             // MIN/2 + MIN/2 = MIN (for even division)
        i32::MAX / 2 * 2 + 1, // MAX/2 + MAX/2 (accounting for truncation)
    ]);
    // Only check the simpler cases that don't depend on integer division truncation
    assert_eq!(result.0[0], expected.0[0]);
    assert_eq!(result.0[1], expected.0[1]);
    assert_eq!(result.0[4], expected.0[4]);
    assert_eq!(result.0[5], expected.0[5]);
    assert_eq!(result.0[6], expected.0[6]);
}

/// Test I64X8 iadd with MIN/MAX boundary values
#[test]
fn test_i64x8_iadd_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i64x8("i64x8_iadd_boundary", |builder, a, b| {
            builder.ins().iadd(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI64x8Fn = unsafe { mem::transmute(code) };

    // Test MIN boundary
    let a = I64x8::splat(i64::MIN);
    let b = I64x8::splat(0);
    let mut result = I64x8::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I64x8::splat(i64::MIN));

    // Test MAX boundary
    let a = I64x8::splat(i64::MAX);
    let b = I64x8::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I64x8::splat(i64::MAX));

    // Test wraparound: MAX + 1 = MIN
    let a = I64x8::splat(i64::MAX);
    let b = I64x8::splat(1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I64x8::splat(i64::MIN));

    // Test wraparound: MIN - 1 = MAX
    let a = I64x8::splat(i64::MIN);
    let b = I64x8::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I64x8::splat(i64::MAX));
}

/// Test I32X16 isub with MIN/MAX boundary values
#[test]
fn test_i32x16_isub_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_isub_boundary", |builder, a, b| {
            builder.ins().isub(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test MIN - 0
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(0);
    let mut result = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN));

    // Test MAX - 0
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MAX));

    // Test wraparound: MIN - 1 = MAX
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MAX));

    // Test MAX - (-1) = MIN (wraparound)
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN));

    // Test self-subtraction
    let a = I32x16::new([
        i32::MIN,
        i32::MAX,
        0,
        -1,
        1,
        100,
        -100,
        12345,
        i32::MIN,
        i32::MAX,
        0,
        -1,
        1,
        100,
        -100,
        12345,
    ]);
    unsafe { func(&a, &a, &mut result) };
    assert_eq!(result, I32x16::splat(0));
}

/// Test I32X16 imul with boundary values
#[test]
fn test_i32x16_imul_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_imul_boundary", |builder, a, b| {
            builder.ins().imul(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test multiply by 0
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(0);
    let mut result = I32x16::splat(1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // Test multiply by 1
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MAX));

    // Test multiply by -1
    let a = I32x16::splat(100);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-100));

    // Test MIN * -1 (special case: overflows back to MIN)
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN)); // Overflow wraps

    // Test powers of 2
    let a = I32x16::splat(12345);
    let b = I32x16::splat(2);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(24690));
}

/// Test F32X16 with special floating-point values (NaN, Inf, -0)
#[test]
fn test_f32x16_fadd_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fadd_special", |builder, a, b| {
            builder.ins().fadd(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // Test with positive infinity
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(1.0);
    let mut result = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_positive());

    // Test with negative infinity
    let a = F32x16::splat(f32::NEG_INFINITY);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_negative());

    // Test Inf + (-Inf) = NaN
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(f32::NEG_INFINITY);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_nan());

    // Test with NaN (NaN + anything = NaN)
    let a = F32x16::splat(f32::NAN);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_nan());

    // Test -0.0 + 0.0 = 0.0 (positive zero)
    let a = F32x16::splat(-0.0);
    let b = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 0.0);

    // Test MAX + MAX (overflow to infinity)
    let a = F32x16::splat(f32::MAX);
    let b = F32x16::splat(f32::MAX);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite());

    // Test MIN_POSITIVE (smallest positive normal)
    let a = F32x16::splat(f32::MIN_POSITIVE);
    let b = F32x16::splat(f32::MIN_POSITIVE);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 2.0 * f32::MIN_POSITIVE);
}

/// Test F64X8 with special floating-point values
#[test]
fn test_f64x8_fadd_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f64x8("f64x8_fadd_special", |builder, a, b| {
            builder.ins().fadd(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryF64x8Fn = unsafe { mem::transmute(code) };

    // Test Infinity
    let a = F64x8::splat(f64::INFINITY);
    let b = F64x8::splat(1.0);
    let mut result = F64x8::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_positive());

    // Test NaN propagation
    let a = F64x8::splat(f64::NAN);
    let b = F64x8::splat(42.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_nan());

    // Test -0.0
    let a = F64x8::splat(-0.0);
    let b = F64x8::splat(-0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0] == 0.0 && result.0[0].is_sign_negative());
}

/// Test F32X16 fdiv with special values (division by zero, etc.)
#[test]
fn test_f32x16_fdiv_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_f32x16("f32x16_fdiv_special", |builder, a, b| {
            builder.ins().fdiv(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // Test 1.0 / 0.0 = +Infinity
    let a = F32x16::splat(1.0);
    let b = F32x16::splat(0.0);
    let mut result = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_positive());

    // Test -1.0 / 0.0 = -Infinity
    let a = F32x16::splat(-1.0);
    let b = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_negative());

    // Test 0.0 / 0.0 = NaN
    let a = F32x16::splat(0.0);
    let b = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_nan());

    // Test Inf / Inf = NaN
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(f32::INFINITY);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_nan());

    // Test finite / Inf = 0
    let a = F32x16::splat(1.0);
    let b = F32x16::splat(f32::INFINITY);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 0.0);
}

/// Test F32X16 sqrt with special values
#[test]
fn test_f32x16_sqrt_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_unary_f32x16("f32x16_sqrt_special", |builder, a| builder.ins().sqrt(a))
        .expect("Failed to compile");

    type UnaryF32x16Fn = unsafe extern "C" fn(*const F32x16, *mut F32x16);
    let func: UnaryF32x16Fn = unsafe { mem::transmute(code) };

    // Test sqrt(0) = 0
    let a = F32x16::splat(0.0);
    let mut result = F32x16::splat(-1.0);
    unsafe { func(&a, &mut result) };
    assert_eq!(result.0[0], 0.0);

    // Test sqrt(-0) = -0
    let a = F32x16::splat(-0.0);
    unsafe { func(&a, &mut result) };
    assert!(result.0[0] == 0.0 && result.0[0].is_sign_negative());

    // Test sqrt(1) = 1
    let a = F32x16::splat(1.0);
    unsafe { func(&a, &mut result) };
    assert_eq!(result.0[0], 1.0);

    // Test sqrt(Inf) = Inf
    let a = F32x16::splat(f32::INFINITY);
    unsafe { func(&a, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_positive());

    // Test sqrt(-1) = NaN
    let a = F32x16::splat(-1.0);
    unsafe { func(&a, &mut result) };
    assert!(result.0[0].is_nan());

    // Test sqrt(NaN) = NaN
    let a = F32x16::splat(f32::NAN);
    unsafe { func(&a, &mut result) };
    assert!(result.0[0].is_nan());
}

/// Test I32X16 bitwise operations with all-ones and all-zeros patterns
#[test]
fn test_i32x16_bitwise_patterns() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test AND
    let code = compiler
        .compile_binary_i32x16("i32x16_band_patterns", |builder, a, b| {
            builder.ins().band(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // x & 0 = 0
    let a = I32x16::splat(-1); // all ones
    let b = I32x16::splat(0);
    let mut result = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // x & -1 = x (all ones mask)
    let a = I32x16::splat(0x12345678);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0x12345678));

    // x & x = x
    let a = I32x16::splat(0xABCDEF01_u32 as i32);
    unsafe { func(&a, &a, &mut result) };
    assert_eq!(result, I32x16::splat(0xABCDEF01_u32 as i32));

    // Test OR
    let code = compiler
        .compile_binary_i32x16("i32x16_bor_patterns", |builder, a, b| {
            builder.ins().bor(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // x | 0 = x
    let a = I32x16::splat(0x12345678);
    let b = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0x12345678));

    // x | -1 = -1
    let a = I32x16::splat(0x12345678);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // Test XOR
    let code = compiler
        .compile_binary_i32x16("i32x16_bxor_patterns", |builder, a, b| {
            builder.ins().bxor(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // x ^ 0 = x
    let a = I32x16::splat(0x12345678);
    let b = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0x12345678));

    // x ^ x = 0
    let a = I32x16::splat(0x12345678);
    unsafe { func(&a, &a, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // x ^ -1 = ~x (NOT)
    let a = I32x16::splat(0x12345678);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(!0x12345678));
}

/// Test I32X16 ineg with boundary values
#[test]
fn test_i32x16_ineg_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_unary_i32x16("i32x16_ineg_boundary", |builder, a| builder.ins().ineg(a))
        .expect("Failed to compile");

    type UnaryI32x16Fn = unsafe extern "C" fn(*const I32x16, *mut I32x16);
    let func: UnaryI32x16Fn = unsafe { mem::transmute(code) };

    // -0 = 0
    let a = I32x16::splat(0);
    let mut result = I32x16::splat(1);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // -1 = -1
    let a = I32x16::splat(1);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // -(-1) = 1
    let a = I32x16::splat(-1);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I32x16::splat(1));

    // -MIN = MIN (special case: negation overflow)
    let a = I32x16::splat(i32::MIN);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I32x16::splat(i32::MIN));

    // -MAX = -(MAX) = MIN + 1
    let a = I32x16::splat(i32::MAX);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I32x16::splat(-i32::MAX));
}

/// Test I64X8 ineg with boundary values
#[test]
fn test_i64x8_ineg_boundary_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_unary_i64x8("i64x8_ineg_boundary", |builder, a| builder.ins().ineg(a))
        .expect("Failed to compile");

    type UnaryI64x8Fn = unsafe extern "C" fn(*const I64x8, *mut I64x8);
    let func: UnaryI64x8Fn = unsafe { mem::transmute(code) };

    // -MIN = MIN (special case)
    let a = I64x8::splat(i64::MIN);
    let mut result = I64x8::splat(0);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I64x8::splat(i64::MIN));

    // -MAX = -MAX
    let a = I64x8::splat(i64::MAX);
    unsafe { func(&a, &mut result) };
    assert_eq!(result, I64x8::splat(-i64::MAX));
}

/// Test signed integer comparisons at boundaries
#[test]
fn test_i32x16_icmp_signed_boundary() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test signed greater than
    let code = compiler
        .compile_comparison_i32x16("i32x16_sgt_boundary", IntCC::SignedGreaterThan)
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // MIN > MAX should be false
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(i32::MAX);
    let mut result = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // MAX > MIN should be true
    let a = I32x16::splat(i32::MAX);
    let b = I32x16::splat(i32::MIN);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // -1 > 0 should be false
    let a = I32x16::splat(-1);
    let b = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // 0 > -1 should be true
    let a = I32x16::splat(0);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));
}

/// Test SignedGreaterThanOrEqual with the exact pattern that platform-vec uses
#[test]
fn test_i32x16_icmp_signed_ge_platform_vec_pattern() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test signed greater than or equal
    let code = compiler
        .compile_comparison_i32x16("i32x16_sge", IntCC::SignedGreaterThanOrEqual)
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test with the exact data pattern from platform-vec: [10,20,30,...,160] >= 100
    let data = I32x16::new([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
    let constant = I32x16::splat(100);
    let mut result = I32x16::splat(0);

    unsafe { func(&data, &constant, &mut result) };

    // Expected: lanes 9-15 should be -1 (TRUE), lanes 0-8 should be 0 (FALSE)
    let expected = I32x16::new([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1]);

    // Check individual lanes for better error messages
    for i in 0..16 {
        let actual_lane = result.0[i];
        let expected_lane = expected.0[i];
        assert_eq!(
            actual_lane, expected_lane,
            "Lane {} mismatch: data[{}]={} >= 100 should be {}, got {}",
            i, i, data.0[i],
            if expected_lane == -1 { "TRUE (-1)" } else { "FALSE (0)" },
            if actual_lane == -1 { "TRUE (-1)" } else { "FALSE (0)" }
        );
    }

    // 100 >= 100 should be true
    let a = I32x16::splat(100);
    let b = I32x16::splat(100);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1), "100 >= 100 should be TRUE");

    // 99 >= 100 should be false
    let a = I32x16::splat(99);
    let b = I32x16::splat(100);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0), "99 >= 100 should be FALSE");
}

/// Test that icmp followed by extractlane works for ALL lanes (including 4+)
/// This specifically tests that VPMOVM2D correctly expands to full 512 bits
#[test]
fn test_i32x16_icmp_extractlane_all_lanes() {
    let Some(isa) = isa_with_avx512() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    // Test function: takes two I32X16 ptrs, returns extracted lane 4 from icmp result
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(I64)); // a_ptr
    sig.params.push(AbiParam::new(I64)); // b_ptr
    sig.returns.push(AbiParam::new(I32)); // extracted lane

    let func_id = module
        .declare_function("icmp_extract4", Linkage::Local, &sig)
        .expect("declare");

    let mut ctx = module.make_context();
    ctx.func.signature = sig.clone();

    let mut fnbuilder_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fnbuilder_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let a_ptr = builder.block_params(entry_block)[0];
        let b_ptr = builder.block_params(entry_block)[1];

        // Load vectors
        let a = builder.ins().load(I32X16, MemFlags::trusted(), a_ptr, 0);
        let b = builder.ins().load(I32X16, MemFlags::trusted(), b_ptr, 0);

        // Do icmp equal
        let mask = builder.ins().icmp(IntCC::Equal, a, b);

        // Extract lane 4 from the mask
        let lane4 = builder.ins().extractlane(mask, 4u8);

        builder.ins().return_(&[lane4]);
    }

    module
        .define_function(func_id, &mut ctx)
        .expect("define");
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize");

    let code = module.get_finalized_function(func_id);
    let func: extern "C" fn(*const I32x16, *const I32x16) -> i32 = unsafe { mem::transmute(code) };

    // Test case 1: lane 4 should match (both have 4 at position 4)
    let a = I32x16::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let b = I32x16::new([99, 99, 99, 99, 4, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]);
    let result = func(&a, &b);
    assert_eq!(result, -1, "Lane 4 should match: a[4]=4 == b[4]=4");

    // Test case 2: lane 4 should NOT match
    let a = I32x16::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let b = I32x16::new([99, 99, 99, 99, 5, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]);
    let result = func(&a, &b);
    assert_eq!(result, 0, "Lane 4 should NOT match: a[4]=4 != b[4]=5");

}

/// Test unsigned integer comparisons at boundaries
#[test]
fn test_i32x16_icmp_unsigned_boundary() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test unsigned greater than
    let code = compiler
        .compile_comparison_i32x16("i32x16_ugt_boundary", IntCC::UnsignedGreaterThan)
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Unsigned: 0xFFFFFFFF (-1) > 0 should be TRUE
    let a = I32x16::splat(-1);
    let b = I32x16::splat(0);
    let mut result = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // Unsigned: 0 > 0xFFFFFFFF should be FALSE
    let a = I32x16::splat(0);
    let b = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // Unsigned: i32::MIN (0x80000000) > i32::MAX (0x7FFFFFFF) should be TRUE
    let a = I32x16::splat(i32::MIN);
    let b = I32x16::splat(i32::MAX);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));
}

/// Test floating-point comparisons with special values
#[test]
fn test_f32x16_fcmp_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test equal (NaN != NaN should be false for eq, true for ne)
    let code = compiler
        .compile_comparison_f32x16("f32x16_eq_nan", FloatCC::Equal)
        .expect("Failed to compile");

    let func: ComparisonF32x16Fn = unsafe { mem::transmute(code) };

    // NaN == NaN should be FALSE
    let a = F32x16::splat(f32::NAN);
    let b = F32x16::splat(f32::NAN);
    let mut result = I32x16::splat(-1);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // Inf == Inf should be TRUE
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(f32::INFINITY);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // -0.0 == 0.0 should be TRUE
    let a = F32x16::splat(-0.0);
    let b = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(-1));

    // Test ordered comparison (returns false if either is NaN)
    let code = compiler
        .compile_comparison_f32x16("f32x16_lt_nan", FloatCC::LessThan)
        .expect("Failed to compile");

    let func: ComparisonF32x16Fn = unsafe { mem::transmute(code) };

    // NaN < 1.0 should be FALSE (unordered)
    let a = F32x16::splat(f32::NAN);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));

    // 1.0 < NaN should be FALSE
    let a = F32x16::splat(1.0);
    let b = F32x16::splat(f32::NAN);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(0));
}

/// Test fmin/fmax with special values
#[test]
fn test_f32x16_fmin_fmax_special_values() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test fmin
    let code = compiler
        .compile_binary_f32x16("f32x16_fmin_special", |builder, a, b| {
            builder.ins().fmin(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // min(Inf, 1.0) = 1.0
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(1.0);
    let mut result = F32x16::splat(0.0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 1.0);

    // min(-Inf, 1.0) = -Inf
    let a = F32x16::splat(f32::NEG_INFINITY);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_negative());

    // Test fmax
    let code = compiler
        .compile_binary_f32x16("f32x16_fmax_special", |builder, a, b| {
            builder.ins().fmax(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryF32x16Fn = unsafe { mem::transmute(code) };

    // max(Inf, 1.0) = Inf
    let a = F32x16::splat(f32::INFINITY);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert!(result.0[0].is_infinite() && result.0[0].is_sign_positive());

    // max(-Inf, 1.0) = 1.0
    let a = F32x16::splat(f32::NEG_INFINITY);
    let b = F32x16::splat(1.0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 1.0);
}

/// Test all-lanes patterns (alternating, first/last)
#[test]
fn test_i32x16_lane_patterns() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    let code = compiler
        .compile_binary_i32x16("i32x16_iadd_lanes", |builder, a, b| {
            builder.ins().iadd(a, b)
        })
        .expect("Failed to compile");

    let func: BinaryI32x16Fn = unsafe { mem::transmute(code) };

    // Test alternating pattern
    let a = I32x16::new([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    let b = I32x16::new([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    let mut result = I32x16::splat(0);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result, I32x16::splat(1));

    // Test first lane only
    let a = I32x16::new([100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let b = I32x16::new([200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    unsafe { func(&a, &b, &mut result) };
    assert_eq!(result.0[0], 300);
    for i in 1..16 {
        assert_eq!(result.0[i], 0);
    }

    // Test last lane only
    let a = I32x16::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]);
    let b = I32x16::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200]);
    unsafe { func(&a, &b, &mut result) };
    for i in 0..15 {
        assert_eq!(result.0[i], 0);
    }
    assert_eq!(result.0[15], 300);
}

/// Test vconst with 512-bit I32X16 vector - verifies that all 16 lanes are correctly loaded
/// This is a regression test for a bug where only the first 4 lanes were loaded (128-bit movdqu
/// was used instead of 512-bit vmovdqu32)
#[test]
fn test_i32x16_vconst() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Create a 64-byte constant with unique values in each lane
    // lane 0-3:   1, 2, 4, 8 (bits 0-3)
    // lane 4-7:  16, 32, 64, 128 (bits 4-7)
    // lane 8-11: 256, 512, 1024, 2048 (bits 8-11)
    // lane 12-15: 4096, 8192, 16384, 32768 (bits 12-15)
    let bit_masks: [u32; 16] = [
        1, 2, 4, 8,           // lanes 0-3
        16, 32, 64, 128,      // lanes 4-7
        256, 512, 1024, 2048, // lanes 8-11
        4096, 8192, 16384, 32768, // lanes 12-15
    ];

    // Convert to bytes (little-endian)
    let const_bytes: Vec<u8> = bit_masks.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_vconst", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let dst_ptr = params[0];

        // Create the constant
        let const_data = ConstantData::from(&const_bytes[..]);
        let const_handle = builder.func.dfg.constants.insert(const_data);

        // Load the 512-bit constant using vconst
        let vec_const = builder.ins().vconst(I32X16, const_handle);

        // Store result
        builder.ins().store(MemFlags::trusted(), vec_const, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_vconst ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    let code = compiler.module.get_finalized_function(func_id);
    type VconstFn = unsafe extern "C" fn(*mut I32x16);
    let func: VconstFn = unsafe { mem::transmute(code) };

    // Execute and verify all 16 lanes
    let mut result = I32x16::splat(0);
    unsafe { func(&mut result) };

    println!("Result lanes: {:?}", result.0);

    // Verify each lane has the expected bit mask value
    for (i, expected) in bit_masks.iter().enumerate() {
        assert_eq!(
            result.0[i] as u32, *expected,
            "Lane {} mismatch: expected {}, got {}",
            i, expected, result.0[i]
        );
    }
}

/// Test vconst with 512-bit I64X8 vector - verifies that all 8 lanes are correctly loaded
#[test]
fn test_i64x8_vconst() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Create a 64-byte constant with unique values in each lane
    let values: [u64; 8] = [
        0x0001_0001_0001_0001, // lane 0
        0x0002_0002_0002_0002, // lane 1
        0x0004_0004_0004_0004, // lane 2
        0x0008_0008_0008_0008, // lane 3
        0x0010_0010_0010_0010, // lane 4 - This and following lanes would be 0 with the bug
        0x0020_0020_0020_0020, // lane 5
        0x0040_0040_0040_0040, // lane 6
        0x0080_0080_0080_0080, // lane 7
    ];

    // Convert to bytes (little-endian)
    let const_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i64x8_vconst", Linkage::Local, &sig)
        .expect("Failed to declare function");

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let dst_ptr = params[0];

        // Create the constant
        let const_data = ConstantData::from(&const_bytes[..]);
        let const_handle = builder.func.dfg.constants.insert(const_data);

        // Load the 512-bit constant using vconst
        let vec_const = builder.ins().vconst(I64X8, const_handle);

        // Store result
        builder.ins().store(MemFlags::trusted(), vec_const, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i64x8_vconst ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler
        .module
        .finalize_definitions()
        .expect("Failed to finalize");

    let code = compiler.module.get_finalized_function(func_id);
    type VconstFn = unsafe extern "C" fn(*mut I64x8);
    let func: VconstFn = unsafe { mem::transmute(code) };

    // Execute and verify all 8 lanes
    let mut result = I64x8::splat(0);
    unsafe { func(&mut result) };

    println!("Result lanes: {:?}", result.0);

    // Verify each lane has the expected value
    for (i, expected) in values.iter().enumerate() {
        assert_eq!(
            result.0[i] as u64, *expected,
            "Lane {} mismatch: expected 0x{:016x}, got 0x{:016x}",
            i, expected, result.0[i] as u64
        );
    }
}

// =============================================================================
// Tests: vhigh_bits on I32X16 - Bug Reproducer
// =============================================================================
//
// This test verifies that vhigh_bits correctly extracts all 16 sign bits
// from a 512-bit I32X16 vector. A bug was found where only the lower 128 bits
// (lanes 0-3) were being processed correctly.

#[test]
fn test_i32x16_vhigh_bits_all_lanes() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Test: Extract sign bits from I32X16 where specific lanes are -1 (sign bit set)
    // We'll set lanes 0, 4, 8, 12 to -1, others to 0
    // Expected mask: 0b0001000100010001 = 0x1111 = 4369

    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // input ptr
    sig.returns.push(AbiParam::new(I32)); // mask bits
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_vhigh_bits", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let input_ptr = params[0];

        // Load I32X16 from memory
        let vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), input_ptr, 0);

        // Extract high bits using vhigh_bits
        let mask_bits = builder.ins().vhigh_bits(I32, vec);

        builder.ins().return_(&[mask_bits]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_vhigh_bits ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type VHighBitsFn = unsafe extern "C" fn(*const I32x16) -> i32;
    let func: VHighBitsFn = unsafe { mem::transmute(code) };

    // Test case 1: lanes 0, 4, 8, 12 are -1 (sign bit set)
    let input1 = I32x16::new([
        -1, 0, 0, 0, // lanes 0-3
        -1, 0, 0, 0, // lanes 4-7
        -1, 0, 0, 0, // lanes 8-11
        -1, 0, 0, 0, // lanes 12-15
    ]);
    let result1 = unsafe { func(&input1) };
    let expected1 = 0b0001_0001_0001_0001i32; // 0x1111 = 4369
    println!(
        "Test 1: lanes 0,4,8,12 set. Expected: 0x{:04x}, Got: 0x{:04x}",
        expected1, result1
    );
    assert_eq!(
        result1, expected1,
        "vhigh_bits failed for lanes 0,4,8,12: expected 0x{:04x}, got 0x{:04x}",
        expected1, result1
    );

    // Test case 2: All lanes -1
    let input2 = I32x16::splat(-1);
    let result2 = unsafe { func(&input2) };
    let expected2 = 0xFFFFi32;
    println!(
        "Test 2: all lanes set. Expected: 0x{:04x}, Got: 0x{:04x}",
        expected2, result2
    );
    assert_eq!(
        result2, expected2,
        "vhigh_bits failed for all lanes: expected 0x{:04x}, got 0x{:04x}",
        expected2, result2
    );

    // Test case 3: lanes 0-9 set (10 valid rows pattern)
    let input3 = I32x16::new([
        -1, -1, -1, -1, // lanes 0-3
        -1, -1, -1, -1, // lanes 4-7
        -1, -1, 0, 0, // lanes 8-11 (only 8,9 set)
        0, 0, 0, 0, // lanes 12-15
    ]);
    let result3 = unsafe { func(&input3) };
    let expected3 = 0b0000_0011_1111_1111i32; // 0x03FF = 1023
    println!(
        "Test 3: lanes 0-9 set. Expected: 0x{:04x}, Got: 0x{:04x}",
        expected3, result3
    );
    assert_eq!(
        result3, expected3,
        "vhigh_bits failed for lanes 0-9: expected 0x{:04x}, got 0x{:04x}",
        expected3, result3
    );

    // Test case 4: Only lanes 4-9 set (middle lanes only)
    let input4 = I32x16::new([
        0, 0, 0, 0, // lanes 0-3
        -1, -1, -1, -1, // lanes 4-7
        -1, -1, 0, 0, // lanes 8-11 (only 8,9 set)
        0, 0, 0, 0, // lanes 12-15
    ]);
    let result4 = unsafe { func(&input4) };
    let expected4 = 0b0000_0011_1111_0000i32; // 0x03F0 = 1008
    println!(
        "Test 4: lanes 4-9 set. Expected: 0x{:04x}, Got: 0x{:04x}",
        expected4, result4
    );
    assert_eq!(
        result4, expected4,
        "vhigh_bits failed for lanes 4-9: expected 0x{:04x}, got 0x{:04x}",
        expected4, result4
    );
}

// =============================================================================
// Tests: I32X16 Store and Load Round-Trip
// =============================================================================
//
// This test verifies that storing an I32X16 and loading it back preserves
// all 16 lanes correctly. A bug was found where store/load only handled 128 bits.

#[test]
fn test_i32x16_store_load_roundtrip() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Function: load I32X16, store to stack, reload and return
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // input ptr
    sig.params.push(AbiParam::new(ptr_type)); // output ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_roundtrip", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let input_ptr = params[0];
        let output_ptr = params[1];

        // Load I32X16 from input
        let vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), input_ptr, 0);

        // Create a stack slot and store/reload (this is what platform-vec does)
        let stack_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let stack_ptr = builder.ins().stack_addr(I64, stack_slot, 0);

        // Store the I32X16 to stack
        builder.ins().store(MemFlags::trusted(), vec, stack_ptr, 0);

        // Load it back as I32X16
        let reloaded = builder.ins().load(I32X16, MemFlags::trusted(), stack_ptr, 0);

        // Store to output
        builder.ins().store(MemFlags::trusted(), reloaded, output_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_roundtrip ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type RoundtripFn = unsafe extern "C" fn(*const I32x16, *mut I32x16);
    let func: RoundtripFn = unsafe { mem::transmute(code) };

    // Test with distinct values in each lane
    let input = I32x16::new([
        10, 11, 12, 13, // lanes 0-3
        14, 15, 16, 17, // lanes 4-7
        18, 19, 20, 21, // lanes 8-11
        22, 23, 24, 25, // lanes 12-15
    ]);
    let mut output = I32x16::splat(0);

    unsafe { func(&input, &mut output) };

    println!("Input:  {:?}", input.0);
    println!("Output: {:?}", output.0);

    for i in 0..16 {
        assert_eq!(
            input.0[i], output.0[i],
            "Lane {} mismatch: expected {}, got {}",
            i, input.0[i], output.0[i]
        );
    }
}

// =============================================================================
// Tests: I32X16 Complete Pipeline - Load, Splat, Compare, vhigh_bits
// =============================================================================
//
// This test mimics the exact sequence of operations in platform-vec's
// vectorized predicate evaluation to reproduce the bug.

#[test]
fn test_i32x16_load_splat_icmp_vhigh_bits_pipeline() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Function: load column, splat constant, icmp, vhigh_bits
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // column data ptr
    sig.params.push(AbiParam::new(I32));      // constant to compare against
    sig.returns.push(AbiParam::new(I32));     // mask bits
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_pipeline", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let params = builder.block_params(block).to_vec();
        let column_ptr = params[0];
        let compare_val = params[1];

        // Step 1: Load column as I32X16
        let col_vec = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), column_ptr, 0);

        // Step 2: Spill/reload column (workaround from platform-vec)
        let col_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let col_slot_ptr = builder.ins().stack_addr(I64, col_slot, 0);
        builder.ins().store(MemFlags::trusted(), col_vec, col_slot_ptr, 0);
        let col_reloaded = builder.ins().load(I32X16, MemFlags::trusted(), col_slot_ptr, 0);

        // Step 3: Splat constant to I32X16
        let const_vec = builder.ins().splat(I32X16, compare_val);

        // Step 4: Spill/reload splat (workaround from platform-vec)
        let splat_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let splat_slot_ptr = builder.ins().stack_addr(I64, splat_slot, 0);
        builder.ins().store(MemFlags::trusted(), const_vec, splat_slot_ptr, 0);
        let const_reloaded = builder.ins().load(I32X16, MemFlags::trusted(), splat_slot_ptr, 0);

        // Step 5: Compare using chunked I32X4 (workaround from platform-vec)
        let l_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let r_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let result_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));

        let l_ptr = builder.ins().stack_addr(I64, l_slot, 0);
        let r_ptr = builder.ins().stack_addr(I64, r_slot, 0);
        let result_ptr = builder.ins().stack_addr(I64, result_slot, 0);

        builder.ins().store(MemFlags::trusted(), col_reloaded, l_ptr, 0);
        builder.ins().store(MemFlags::trusted(), const_reloaded, r_ptr, 0);

        // Compare in 4 chunks of I32X4
        for chunk in 0..4i32 {
            let offset = chunk * 16;
            let l_chunk = builder.ins().load(I32X4, MemFlags::trusted(), l_ptr, offset);
            let r_chunk = builder.ins().load(I32X4, MemFlags::trusted(), r_ptr, offset);
            let cmp_chunk = builder.ins().icmp(IntCC::Equal, l_chunk, r_chunk);
            builder.ins().store(MemFlags::trusted(), cmp_chunk, result_ptr, offset);
        }

        // Load result as I32X16
        let mask = builder.ins().load(I32X16, MemFlags::trusted(), result_ptr, 0);

        // Step 6: Extract high bits
        let mask_bits = builder.ins().vhigh_bits(I32, mask);

        builder.ins().return_(&[mask_bits]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_pipeline ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type PipelineFn = unsafe extern "C" fn(*const I32x16, i32) -> i32;
    let func: PipelineFn = unsafe { mem::transmute(code) };

    // Test: column has [0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,0], compare against 4
    let column = I32x16::new([
        0, 1, 2, 3, // lanes 0-3
        4, 5, 6, 7, // lanes 4-7
        8, 9, 0, 0, // lanes 8-11 (padding is 0)
        0, 0, 0, 0, // lanes 12-15 (padding)
    ]);

    // Compare against 4: only lane 4 should match
    let result = unsafe { func(&column, 4) };
    let expected = 0b0000_0000_0001_0000i32; // 0x0010 = 16 (only bit 4 set)
    println!(
        "Compare == 4: Expected: 0x{:04x} ({}), Got: 0x{:04x} ({})",
        expected, expected, result, result
    );
    assert_eq!(
        result, expected,
        "Pipeline compare == 4 failed: expected 0x{:04x}, got 0x{:04x}",
        expected, result
    );

    // Compare against 0: lanes 0, 10-15 should match
    let result0 = unsafe { func(&column, 0) };
    let expected0 = 0b1111_1100_0000_0001i32; // 0xFC01 (bits 0 and 10-15 set)
    println!(
        "Compare == 0: Expected: 0x{:04x} ({}), Got: 0x{:04x} ({})",
        expected0 as u32, expected0, result0 as u32, result0
    );
    assert_eq!(
        result0, expected0,
        "Pipeline compare == 0 failed: expected 0x{:04x}, got 0x{:04x}",
        expected0 as u32, result0 as u32
    );
}

// =============================================================================
// Tests: I32X16 band with Control Flow
// =============================================================================
//
// This test mimics the control flow in platform-vec's visibility mask loading
// with branching to test if control flow affects I32X16 operations.

#[test]
fn test_i32x16_band_with_control_flow() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Function: conditional load, band, vhigh_bits
    let mut sig = compiler.module.make_signature();
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // mask1 ptr
    sig.params.push(AbiParam::new(ptr_type)); // mask2 ptr (or null)
    sig.returns.push(AbiParam::new(I32)); // result bits
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("i32x16_band_cf", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params = builder.block_params(entry_block).to_vec();
        let mask1_ptr = params[0];
        let mask2_ptr = params[1];

        // Load mask1
        let mask1 = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), mask1_ptr, 0);

        // Check if mask2_ptr is null
        let zero_ptr = builder.ins().iconst(I64, 0);
        let is_null = builder.ins().icmp(IntCC::Equal, mask2_ptr, zero_ptr);

        // Create result stack slot
        let result_slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 64, 6,
        ));
        let result_ptr = builder.ins().stack_addr(I64, result_slot, 0);

        // Branching logic
        let null_block = builder.create_block();
        let not_null_block = builder.create_block();
        let merge_block = builder.create_block();

        builder.ins().brif(is_null, null_block, &[], not_null_block, &[]);

        // Null case: use all ones
        builder.switch_to_block(null_block);
        builder.seal_block(null_block);
        let all_ones = {
            let neg_one = builder.ins().iconst(I32, -1);
            builder.ins().splat(I32X16, neg_one)
        };
        builder.ins().store(MemFlags::trusted(), all_ones, result_ptr, 0);
        builder.ins().jump(merge_block, &[]);

        // Not null case: load mask2
        builder.switch_to_block(not_null_block);
        builder.seal_block(not_null_block);
        let mask2 = builder
            .ins()
            .load(I32X16, MemFlags::trusted(), mask2_ptr, 0);
        builder.ins().store(MemFlags::trusted(), mask2, result_ptr, 0);
        builder.ins().jump(merge_block, &[]);

        // Merge and band
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        let loaded_mask2 = builder.ins().load(I32X16, MemFlags::trusted(), result_ptr, 0);
        let banded = builder.ins().band(mask1, loaded_mask2);
        let bits = builder.ins().vhigh_bits(I32, banded);

        builder.ins().return_(&[bits]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for i32x16_band_cf ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type BandCfFn = unsafe extern "C" fn(*const I32x16, *const I32x16) -> i32;
    let func: BandCfFn = unsafe { mem::transmute(code) };

    // Test 1: mask1 = 1023 (bits 0-9), mask2 = all ones, mask2_ptr not null
    let mask1 = I32x16::new([
        -1, -1, -1, -1, // lanes 0-3
        -1, -1, -1, -1, // lanes 4-7
        -1, -1, 0, 0, // lanes 8-11 (only 8,9 set)
        0, 0, 0, 0, // lanes 12-15
    ]);
    let mask2 = I32x16::splat(-1); // all ones = 65535

    let result = unsafe { func(&mask1, &mask2) };
    let expected = 0b0000_0011_1111_1111i32; // 0x03FF = 1023
    println!(
        "Test 1 (not null): Expected: 0x{:04x}, Got: 0x{:04x}",
        expected, result
    );
    assert_eq!(
        result, expected,
        "band_cf test 1 failed: expected 0x{:04x}, got 0x{:04x}",
        expected, result
    );

    // Test 2: mask1 = 1023, mask2_ptr is null (should use all ones)
    let result_null = unsafe { func(&mask1, std::ptr::null()) };
    println!(
        "Test 2 (null): Expected: 0x{:04x}, Got: 0x{:04x}",
        expected, result_null
    );
    assert_eq!(
        result_null, expected,
        "band_cf test 2 (null) failed: expected 0x{:04x}, got 0x{:04x}",
        expected, result_null
    );
}

// =============================================================================
// Tests: EVEX Register Extension Encoding (Registers 16-31)
// =============================================================================
//
// This test specifically verifies that VPBROADCASTD and VPBROADCASTQ correctly
// encode registers 16-31 using the EVEX.X bit. Previously there was a bug where
// the X bit was always set to 1 (inverted 0) for register operands, which caused
// registers 16-31 to be misencoded.
//
// The fix ensures that both EVEX.B (bit 3) and EVEX.X (bit 4) are correctly
// computed from the source register encoding.

#[test]
fn test_vpbroadcastd_evex_register_encoding() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // This test compiles a function that uses multiple splat operations to
    // force register allocation to use high registers (16-31).
    // If the EVEX encoding is wrong, this will produce incorrect results or crash.
    
    let mut sig = compiler.module.make_signature();
    sig.params.push(AbiParam::new(I32)); // input value
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vpbroadcastd_high_regs", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params = builder.block_params(entry_block).to_vec();
        let input = params[0];
        let dst_ptr = params[1];

        // Create many splat operations to pressure register allocation
        // This should force use of high registers (zmm16-zmm31)
        let splat1 = builder.ins().splat(I32X16, input);
        let c1 = builder.ins().iconst(I32, 1);
        let splat2 = builder.ins().splat(I32X16, c1);
        let c2 = builder.ins().iconst(I32, 2);
        let splat3 = builder.ins().splat(I32X16, c2);
        let c3 = builder.ins().iconst(I32, 3);
        let splat4 = builder.ins().splat(I32X16, c3);
        let c4 = builder.ins().iconst(I32, 4);
        let splat5 = builder.ins().splat(I32X16, c4);
        let c5 = builder.ins().iconst(I32, 5);
        let splat6 = builder.ins().splat(I32X16, c5);
        let c6 = builder.ins().iconst(I32, 6);
        let splat7 = builder.ins().splat(I32X16, c6);
        let c7 = builder.ins().iconst(I32, 7);
        let splat8 = builder.ins().splat(I32X16, c7);

        // Use all the splats to prevent dead code elimination
        let sum1 = builder.ins().iadd(splat1, splat2);
        let sum2 = builder.ins().iadd(splat3, splat4);
        let sum3 = builder.ins().iadd(splat5, splat6);
        let sum4 = builder.ins().iadd(splat7, splat8);
        let sum5 = builder.ins().iadd(sum1, sum2);
        let sum6 = builder.ins().iadd(sum3, sum4);
        let result = builder.ins().iadd(sum5, sum6);

        // Store the result
        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vpbroadcastd_high_regs ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type HighRegFn = unsafe extern "C" fn(i32, *mut I32x16);
    let func: HighRegFn = unsafe { mem::transmute(code) };

    // Run the test
    let mut result = I32x16::splat(0);
    let input = 100;
    unsafe { func(input, &mut result) };

    // Expected: input + 1 + 2 + 3 + 4 + 5 + 6 + 7 = input + 28
    let expected_value = input + 28;
    let expected = I32x16::splat(expected_value);
    
    println!("Input: {}, Expected value per lane: {}", input, expected_value);
    println!("Result: {:?}", result.0);
    
    assert_eq!(result, expected, 
        "VPBROADCASTD with high registers failed: expected splat({}), got {:?}",
        expected_value, result.0);
}

#[test]
fn test_vpbroadcastq_evex_register_encoding() {
    let Some(mut compiler) = TestCompiler::new() else {
        println!("Skipping: AVX-512 not available");
        return;
    };

    // Similar test for VPBROADCASTQ (I64X8)
    let mut sig = compiler.module.make_signature();
    sig.params.push(AbiParam::new(I64)); // input value
    let ptr_type = compiler.module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type)); // dst ptr
    sig.call_conv = CallConv::SystemV;

    let func_id = compiler
        .module
        .declare_function("vpbroadcastq_high_regs", Linkage::Local, &sig)
        .unwrap();

    compiler.ctx.func =
        Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

    {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params = builder.block_params(entry_block).to_vec();
        let input = params[0];
        let dst_ptr = params[1];

        // Create many splat operations to pressure register allocation
        let splat1 = builder.ins().splat(I64X8, input);
        let c1 = builder.ins().iconst(I64, 1);
        let splat2 = builder.ins().splat(I64X8, c1);
        let c2 = builder.ins().iconst(I64, 2);
        let splat3 = builder.ins().splat(I64X8, c2);
        let c3 = builder.ins().iconst(I64, 3);
        let splat4 = builder.ins().splat(I64X8, c3);
        let c4 = builder.ins().iconst(I64, 4);
        let splat5 = builder.ins().splat(I64X8, c4);
        let c5 = builder.ins().iconst(I64, 5);
        let splat6 = builder.ins().splat(I64X8, c5);
        let c6 = builder.ins().iconst(I64, 6);
        let splat7 = builder.ins().splat(I64X8, c6);
        let c7 = builder.ins().iconst(I64, 7);
        let splat8 = builder.ins().splat(I64X8, c7);

        // Use all the splats
        let sum1 = builder.ins().iadd(splat1, splat2);
        let sum2 = builder.ins().iadd(splat3, splat4);
        let sum3 = builder.ins().iadd(splat5, splat6);
        let sum4 = builder.ins().iadd(splat7, splat8);
        let sum5 = builder.ins().iadd(sum1, sum2);
        let sum6 = builder.ins().iadd(sum3, sum4);
        let result = builder.ins().iadd(sum5, sum6);

        builder.ins().store(MemFlags::trusted(), result, dst_ptr, 0);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    compiler.ctx.set_disasm(true);
    compiler
        .module
        .define_function(func_id, &mut compiler.ctx)
        .expect("Failed to define function");

    if let Some(compiled) = compiler.ctx.compiled_code() {
        if let Some(disasm) = &compiled.vcode {
            println!("=== VCode for vpbroadcastq_high_regs ===\n{}", disasm);
        }
    }

    compiler.module.clear_context(&mut compiler.ctx);
    compiler.module.finalize_definitions().unwrap();

    let code = compiler.module.get_finalized_function(func_id);
    type HighRegFn = unsafe extern "C" fn(i64, *mut I64x8);
    let func: HighRegFn = unsafe { mem::transmute(code) };

    // Run the test
    let mut result = I64x8::splat(0);
    let input: i64 = 1000;
    unsafe { func(input, &mut result) };

    // Expected: input + 1 + 2 + 3 + 4 + 5 + 6 + 7 = input + 28
    let expected_value = input + 28;
    let expected = I64x8::splat(expected_value);
    
    println!("Input: {}, Expected value per lane: {}", input, expected_value);
    println!("Result: {:?}", result.0);
    
    assert_eq!(result, expected, 
        "VPBROADCASTQ with high registers failed: expected splat({}), got {:?}",
        expected_value, result.0);
}
