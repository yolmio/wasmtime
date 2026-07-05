//! X86_64-bit Instruction Set Architecture.

pub use self::inst::{AtomicRmwSeqOp, EmitInfo, EmitState, Inst, args, external};

use super::{OwnedTargetIsa, TargetIsa};
use crate::dominator_tree::DominatorTree;
use crate::ir::{self, Function, Type, types};
#[cfg(feature = "unwind")]
use crate::isa::unwind::systemv;
use crate::isa::x64::settings as x64_settings;
use crate::isa::{Builder as IsaBuilder, FunctionAlignment, IsaFlagsHashKey};
use crate::machinst::{
    CompiledCodeStencil, MachInst, MachTextSectionBuilder, Reg, SigSet, TextSectionBuilder, VCode,
    compile,
};
use crate::result::{CodegenError, CodegenResult};
use crate::settings::{self as shared_settings, Flags};
use crate::{Final, MachBufferFinalized};
use alloc::string::String;
use alloc::{borrow::ToOwned, boxed::Box, vec::Vec};
use core::fmt;
use cranelift_control::ControlPlane;
use target_lexicon::Triple;

mod abi;
mod inst;
mod lower;
mod pcc;
pub mod settings;

#[cfg(feature = "unwind")]
pub use inst::unwind::systemv::create_cie;

/// An X64 backend.
pub(crate) struct X64Backend {
    triple: Triple,
    flags: Flags,
    x64_flags: x64_settings::Flags,
}

impl X64Backend {
    /// Create a new X64 backend with the given (shared) flags.
    fn new_with_flags(
        triple: Triple,
        flags: Flags,
        x64_flags: x64_settings::Flags,
    ) -> CodegenResult<Self> {
        if triple.pointer_width().unwrap() != target_lexicon::PointerWidth::U64 {
            return Err(CodegenError::Unsupported(
                "the x32 ABI is not supported".to_owned(),
            ));
        }

        // The AVX-512 support in this backend assumes that `has_avx512f`
        // implies the BW, DQ, and VL extensions as well: k-mask register
        // spills/reloads use `kmovq` (AVX512BW), the vector icmp/fcmp/
        // bitselect lowerings use `vpmovm2*`/`vpmov*2m` (AVX512BW/DQ), and
        // 256-bit loads/stores use the VL-encoded forms of `vmovdqu32`.
        // Rather than guarding every lowering rule individually, enforce the
        // feature-set invariant once at backend construction.
        if x64_flags.has_avx512f()
            && !(x64_flags.has_avx512bw() && x64_flags.has_avx512dq() && x64_flags.has_avx512vl())
        {
            return Err(CodegenError::Unsupported(
                "has_avx512f requires has_avx512bw, has_avx512dq, and has_avx512vl: \
                 enable all four AVX-512 flags or disable has_avx512f"
                    .to_owned(),
            ));
        }

        Ok(Self {
            triple,
            flags,
            x64_flags,
        })
    }

    fn compile_vcode(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<(VCode<inst::Inst>, regalloc2::Output)> {
        // Win64 (windows_fastcall) preserves only the low 128 bits of the
        // callee-saved registers xmm6-xmm15 across calls: the upper bits of
        // the corresponding ymm/zmm registers are volatile. With AVX-512
        // enabled, 512-bit values may live in any function body (not just in
        // signatures), and keeping one in xmm6-xmm15 across a
        // fastcall-convention call would silently truncate it to 128 bits.
        // The clobber and callee-save sets are convention-keyed constants
        // with no access to ISA flags, so we conservatively reject the
        // combination here: no function may itself use windows_fastcall, or
        // reference a windows_fastcall signature (i.e. make such a call),
        // while AVX-512 is enabled.
        if self.x64_flags.has_avx512f() {
            let uses_fastcall = func.signature.call_conv == crate::isa::CallConv::WindowsFastcall
                || func
                    .dfg
                    .signatures
                    .values()
                    .any(|sig| sig.call_conv == crate::isa::CallConv::WindowsFastcall);
            if uses_fastcall {
                return Err(CodegenError::Unsupported(
                    "the windows_fastcall calling convention is not supported \
                     with AVX-512 (has_avx512f): Win64 only preserves the low \
                     128 bits of xmm6-xmm15 across calls"
                        .to_owned(),
                ));
            }

            // The check above only sees signatures present in the IR. Libcall
            // signatures are synthesized during lowering (`emit_vm_call`
            // builds them via `CallConv::for_libcall(flags,
            // CallConv::triple_default(triple))`) and never enter
            // `func.dfg.signatures`, so a windows_fastcall libcall (selected
            // by the `libcall_call_conv` flag, or by default on Windows
            // triples) would slip past it while a 512-bit value is live in
            // xmm6-xmm15 across the call. Any function might emit a libcall
            // (fma, floor, ceil, mem*, ...), and predicting which lowerings
            // will is fragile, so this is coarse but sound: reject whenever
            // the resolved libcall convention for this backend is
            // windows_fastcall. This mirrors exactly how `emit_vm_call`
            // computes the libcall convention.
            let libcall_conv = crate::isa::CallConv::for_libcall(
                &self.flags,
                crate::isa::CallConv::triple_default(&self.triple),
            );
            if libcall_conv == crate::isa::CallConv::WindowsFastcall {
                return Err(CodegenError::Unsupported(
                    "libcalls resolve to the windows_fastcall calling \
                     convention (via the libcall_call_conv flag or the \
                     target triple's default), which is not supported with \
                     AVX-512 (has_avx512f): Win64 only preserves the low 128 \
                     bits of xmm6-xmm15 across calls"
                        .to_owned(),
                ));
            }
        } else {
            // Without has_avx512f the backend has no instructions that can
            // move vectors wider than 128 bits: the register-passing ABI
            // code would still assign a 512-bit argument or return value to
            // an XMM register, and any spill/reload of it would use a
            // 16-byte `movdqu`, silently truncating the upper 384 bits. Wide
            // vector values cannot originate inside a function body without
            // the AVX-512 lowering rules (those lowerings have no non-AVX-512
            // fallback and fail cleanly), so signatures are the only way such
            // a value can enter a function; reject them here. Scalar types
            // are at most 16 bytes (i128), so only fixed-width vectors can
            // exceed this limit.
            let has_wide_vector = |sig: &ir::Signature| {
                sig.params
                    .iter()
                    .chain(sig.returns.iter())
                    .any(|param| param.value_type.bytes() > 16)
            };
            if has_wide_vector(&func.signature) || func.dfg.signatures.values().any(has_wide_vector)
            {
                return Err(CodegenError::Unsupported(
                    "vector types wider than 128 bits in function signatures \
                     require AVX-512: enable has_avx512f, has_avx512bw, \
                     has_avx512dq, and has_avx512vl"
                        .to_owned(),
                ));
            }
        }

        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let emit_info = EmitInfo::new(self.flags.clone(), self.x64_flags.clone());
        let sigs = SigSet::new::<abi::X64ABIMachineSpec>(func, &self.flags)?;
        let abi = abi::X64Callee::new(func, self, &self.x64_flags, &sigs)?;
        compile::compile::<Self>(func, domtree, self, abi, emit_info, sigs, ctrl_plane)
    }
}

impl TargetIsa for X64Backend {
    fn compile_function(
        &self,
        func: &Function,
        domtree: &DominatorTree,
        want_disasm: bool,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        let (vcode, regalloc_result) = self.compile_vcode(func, domtree, ctrl_plane)?;

        let emit_result = vcode.emit(&regalloc_result, want_disasm, &self.flags, ctrl_plane);
        let value_labels_ranges = emit_result.value_labels_ranges;
        let buffer = emit_result.buffer;

        if let Some(disasm) = emit_result.disasm.as_ref() {
            crate::trace!("disassembly:\n{}", disasm);
        }

        Ok(CompiledCodeStencil {
            buffer,
            vcode: emit_result.disasm,
            value_labels_ranges,
            bb_starts: emit_result.bb_offsets,
            bb_edges: emit_result.bb_edges,
        })
    }

    fn flags(&self) -> &Flags {
        &self.flags
    }

    fn isa_flags(&self) -> Vec<shared_settings::Value> {
        self.x64_flags.iter().collect()
    }

    fn isa_flags_hash_key(&self) -> IsaFlagsHashKey<'_> {
        IsaFlagsHashKey(self.x64_flags.hash_key())
    }

    fn dynamic_vector_bytes(&self, _dyn_ty: Type) -> u32 {
        16
    }

    fn name(&self) -> &'static str {
        "x64"
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    #[cfg(feature = "unwind")]
    fn emit_unwind_info(
        &self,
        result: &crate::machinst::CompiledCode,
        kind: crate::isa::unwind::UnwindInfoKind,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        emit_unwind_info(&result.buffer, kind)
    }

    #[cfg(feature = "unwind")]
    fn create_systemv_cie(&self) -> Option<gimli::write::CommonInformationEntry> {
        Some(inst::unwind::systemv::create_cie())
    }

    #[cfg(feature = "unwind")]
    fn map_regalloc_reg_to_dwarf(&self, reg: Reg) -> Result<u16, systemv::RegisterMappingError> {
        inst::unwind::systemv::map_reg(reg).map(|reg| reg.0)
    }

    fn text_section_builder(&self, num_funcs: usize) -> Box<dyn TextSectionBuilder> {
        Box::new(MachTextSectionBuilder::<inst::Inst>::new(num_funcs))
    }

    fn function_alignment(&self) -> FunctionAlignment {
        Inst::function_alignment()
    }

    fn page_size_align_log2(&self) -> u8 {
        debug_assert_eq!(1 << 12, 0x1000);
        12
    }

    #[cfg(feature = "disas")]
    fn to_capstone(&self) -> Result<capstone::Capstone, capstone::Error> {
        use capstone::prelude::*;
        Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .syntax(arch::x86::ArchSyntax::Att)
            .detail(true)
            .build()
    }

    fn pretty_print_reg(&self, reg: Reg, size: u8) -> String {
        inst::regs::pretty_print_reg(reg, size)
    }

    fn has_native_fma(&self) -> bool {
        self.x64_flags.has_avx() && self.x64_flags.has_fma()
    }

    fn has_round(&self) -> bool {
        self.x64_flags.has_sse41()
    }

    fn has_blendv_lowering(&self, ty: Type) -> bool {
        // The `blendvpd`, `blendvps`, and `pblendvb` instructions are all only
        // available from SSE 4.1 and onwards. Otherwise the i16x8 type has no
        // equivalent instruction which only looks at the top bit for a select
        // operation, so that always returns `false`
        self.x64_flags.has_sse41() && ty != types::I16X8
    }

    fn has_x86_pshufb_lowering(&self) -> bool {
        self.x64_flags.has_ssse3()
    }

    fn has_x86_pmulhrsw_lowering(&self) -> bool {
        self.x64_flags.has_ssse3()
    }

    fn has_x86_pmaddubsw_lowering(&self) -> bool {
        self.x64_flags.has_ssse3()
    }

    fn default_argument_extension(&self) -> ir::ArgumentExtension {
        // This is copied/carried over from a historical piece of code in
        // Wasmtime:
        //
        // https://github.com/bytecodealliance/wasmtime/blob/a018a5a9addb77d5998021a0150192aa955c71bf/crates/cranelift/src/lib.rs#L366-L374
        //
        // Whether or not it is still applicable here is unsure, but it's left
        // the same as-is for now to reduce the likelihood of problems arising.
        ir::ArgumentExtension::Uext
    }
}

/// Emit unwind info for an x86 target.
pub fn emit_unwind_info(
    buffer: &MachBufferFinalized<Final>,
    kind: crate::isa::unwind::UnwindInfoKind,
) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
    #[cfg(feature = "unwind")]
    use crate::isa::unwind::{UnwindInfo, UnwindInfoKind};
    #[cfg(not(feature = "unwind"))]
    let _ = buffer;
    Ok(match kind {
        #[cfg(feature = "unwind")]
        UnwindInfoKind::SystemV => {
            let mapper = self::inst::unwind::systemv::RegisterMapper;
            Some(UnwindInfo::SystemV(
                crate::isa::unwind::systemv::create_unwind_info_from_insts(
                    &buffer.unwind_info[..],
                    buffer.data().len(),
                    &mapper,
                )?,
            ))
        }
        #[cfg(feature = "unwind")]
        UnwindInfoKind::Windows => Some(UnwindInfo::WindowsX64(
            crate::isa::unwind::winx64::create_unwind_info_from_insts::<
                self::inst::unwind::winx64::RegisterMapper,
            >(&buffer.unwind_info[..])?,
        )),
        _ => None,
    })
}

impl fmt::Display for X64Backend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MachBackend")
            .field("name", &self.name())
            .field("triple", &self.triple())
            .field("flags", &format!("{}", self.flags()))
            .finish()
    }
}

/// Create a new `isa::Builder`.
pub(crate) fn isa_builder(triple: Triple) -> IsaBuilder {
    IsaBuilder {
        triple,
        setup: x64_settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    triple: Triple,
    shared_flags: Flags,
    builder: &shared_settings::Builder,
) -> CodegenResult<OwnedTargetIsa> {
    let isa_flags = x64_settings::Flags::new(&shared_flags, builder);
    let backend = X64Backend::new_with_flags(triple, shared_flags, isa_flags)?;
    Ok(backend.wrapped())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::{InstBuilder, Signature, UserFuncName};
    use crate::isa::CallConv;
    use crate::settings::Configurable;
    use alloc::string::ToString;

    fn triple() -> Triple {
        "x86_64".parse().unwrap()
    }

    fn shared_flags() -> Flags {
        Flags::new(shared_settings::builder())
    }

    const AVX512_FLAGS: [&str; 4] = [
        "has_avx512f",
        "has_avx512bw",
        "has_avx512dq",
        "has_avx512vl",
    ];

    /// B5: `has_avx512f` without any one of BW/DQ/VL must be rejected at
    /// backend construction, since the lowering rules assume F implies all
    /// three.
    #[test]
    fn avx512f_requires_bw_dq_vl() {
        for missing in &AVX512_FLAGS[1..] {
            let mut builder = isa_builder(triple());
            builder.enable("has_avx512f").unwrap();
            for flag in &AVX512_FLAGS[1..] {
                if flag != missing {
                    builder.enable(flag).unwrap();
                }
            }
            match builder.finish(shared_flags()) {
                Err(CodegenError::Unsupported(msg)) => {
                    assert!(
                        msg.contains("has_avx512f requires"),
                        "unexpected error message: {msg}"
                    );
                }
                Err(err) => panic!("unexpected error kind: {err:?}"),
                Ok(_) => panic!("expected has_avx512f without {missing} to be rejected"),
            }
        }
    }

    #[test]
    fn avx512f_with_all_extensions_is_accepted() {
        let mut builder = isa_builder(triple());
        for flag in AVX512_FLAGS {
            builder.enable(flag).unwrap();
        }
        builder
            .finish(shared_flags())
            .expect("all four AVX-512 flags together must be accepted");
    }

    fn empty_function(call_conv: CallConv) -> Function {
        let mut func =
            Function::with_name_signature(UserFuncName::default(), Signature::new(call_conv));
        let block0 = func.dfg.make_block();
        let mut cur = FuncCursor::new(&mut func);
        cur.insert_block(block0);
        cur.ins().return_(&[]);
        func
    }

    /// B3/B4: with AVX-512 enabled, windows_fastcall must be rejected for
    /// *any* function using that convention, even when no wide vector type
    /// appears in its signature, because 512-bit values live across a call
    /// would be kept in xmm6-xmm15, of which Win64 preserves only the low
    /// 128 bits.
    #[test]
    fn fastcall_rejected_with_avx512() {
        let mut builder = isa_builder(triple());
        for flag in AVX512_FLAGS {
            builder.enable(flag).unwrap();
        }
        let isa = builder.finish(shared_flags()).unwrap();

        let mut ctx = Context::for_function(empty_function(CallConv::WindowsFastcall));
        match ctx.compile(&*isa, &mut Default::default()) {
            Err(err) => {
                let msg = err.inner.to_string();
                assert!(msg.contains("windows_fastcall"), "unexpected error: {msg}");
            }
            Ok(_) => panic!("expected windows_fastcall + has_avx512f to be rejected"),
        }
    }

    /// Sanity check for the above: the same function compiles fine when
    /// AVX-512 is not enabled.
    #[test]
    fn fastcall_accepted_without_avx512() {
        let isa = isa_builder(triple()).finish(shared_flags()).unwrap();
        let mut ctx = Context::for_function(empty_function(CallConv::WindowsFastcall));
        ctx.compile(&*isa, &mut Default::default())
            .expect("windows_fastcall without AVX-512 must still compile");
    }

    fn avx512_isa_for(triple: Triple, shared: Flags) -> OwnedTargetIsa {
        let mut builder = isa_builder(triple);
        for flag in AVX512_FLAGS {
            builder.enable(flag).unwrap();
        }
        builder.finish(shared).unwrap()
    }

    fn expect_unsupported(isa: &dyn TargetIsa, func: Function, needle: &str) {
        let mut ctx = Context::for_function(func);
        match ctx.compile(isa, &mut Default::default()) {
            Err(err) => {
                let msg = err.inner.to_string();
                assert!(msg.contains(needle), "unexpected error: {msg}");
            }
            Ok(_) => panic!("expected compilation to be rejected ({needle})"),
        }
    }

    /// Without AVX-512, a register-passed 512-bit argument would be assigned
    /// to an XMM register and spilled/reloaded with 16-byte `movdqu`,
    /// silently truncating it. Such signatures must be rejected cleanly.
    #[test]
    fn wide_vector_param_rejected_without_avx512() {
        let isa = isa_builder(triple()).finish(shared_flags()).unwrap();

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(ir::AbiParam::new(types::I32X16));
        let mut func = Function::with_name_signature(UserFuncName::default(), sig);
        let block0 = func.dfg.make_block();
        func.dfg.append_block_param(block0, types::I32X16);
        let mut cur = FuncCursor::new(&mut func);
        cur.insert_block(block0);
        cur.ins().return_(&[]);

        expect_unsupported(&*isa, func, "wider than 128 bits");
    }

    /// Same for a 512-bit return value.
    #[test]
    fn wide_vector_return_rejected_without_avx512() {
        let isa = isa_builder(triple()).finish(shared_flags()).unwrap();

        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(ir::AbiParam::new(types::F64X8));
        let mut func = Function::with_name_signature(UserFuncName::default(), sig);
        let block0 = func.dfg.make_block();
        let mut cur = FuncCursor::new(&mut func);
        cur.insert_block(block0);
        let zero = cur.ins().f64const(0.0);
        let vec = cur.ins().splat(types::F64X8, zero);
        cur.ins().return_(&[vec]);

        expect_unsupported(&*isa, func, "wider than 128 bits");
    }

    /// ... and for a merely *referenced* signature containing a wide vector
    /// (i.e. calling such a function), even when the caller's own signature
    /// is all-scalar.
    #[test]
    fn wide_vector_in_referenced_sig_rejected_without_avx512() {
        let isa = isa_builder(triple()).finish(shared_flags()).unwrap();

        let mut func = empty_function(CallConv::SystemV);
        let mut sig0 = Signature::new(CallConv::SystemV);
        sig0.params.push(ir::AbiParam::new(types::I16X32));
        func.import_signature(sig0);

        expect_unsupported(&*isa, func, "wider than 128 bits");
    }

    /// Sanity check for the above: all-scalar signatures (including the
    /// 16-byte i128, which sits exactly on the limit) still compile without
    /// AVX-512.
    #[test]
    fn scalar_signature_accepted_without_avx512() {
        // i128 args/returns additionally require the LLVM ABI extensions.
        let mut shared = shared_settings::builder();
        shared.enable("enable_llvm_abi_extensions").unwrap();
        let isa = isa_builder(triple()).finish(Flags::new(shared)).unwrap();

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(ir::AbiParam::new(types::I64));
        sig.params.push(ir::AbiParam::new(types::I128));
        sig.params.push(ir::AbiParam::new(types::F64X2));
        let mut func = Function::with_name_signature(UserFuncName::default(), sig);
        let block0 = func.dfg.make_block();
        func.dfg.append_block_param(block0, types::I64);
        func.dfg.append_block_param(block0, types::I128);
        func.dfg.append_block_param(block0, types::F64X2);
        let mut cur = FuncCursor::new(&mut func);
        cur.insert_block(block0);
        cur.ins().return_(&[]);

        let mut ctx = Context::for_function(func);
        ctx.compile(&*isa, &mut Default::default())
            .expect("scalar/128-bit signatures without AVX-512 must compile");
    }

    /// With AVX-512 enabled, libcalls resolving to windows_fastcall (via the
    /// `libcall_call_conv` flag) must be rejected: libcall signatures are
    /// synthesized during lowering and never enter `dfg.signatures`, so the
    /// signature-based fastcall check cannot see them.
    #[test]
    fn fastcall_libcall_conv_rejected_with_avx512() {
        let mut shared = shared_settings::builder();
        shared.set("libcall_call_conv", "windows_fastcall").unwrap();
        let isa = avx512_isa_for(triple(), Flags::new(shared));

        expect_unsupported(&*isa, empty_function(CallConv::SystemV), "windows_fastcall");
    }

    /// Same via a Windows target triple, whose *default* libcall convention
    /// is windows_fastcall (with `libcall_call_conv=isa_default`).
    #[test]
    fn windows_triple_default_libcall_conv_rejected_with_avx512() {
        let isa = avx512_isa_for("x86_64-pc-windows-msvc".parse().unwrap(), shared_flags());

        expect_unsupported(&*isa, empty_function(CallConv::SystemV), "windows_fastcall");
    }

    /// Sanity check for the above: explicitly routing libcalls to system_v
    /// makes the same Windows-triple configuration compile again.
    #[test]
    fn system_v_libcall_conv_accepted_with_avx512() {
        let mut shared = shared_settings::builder();
        shared.set("libcall_call_conv", "system_v").unwrap();
        let isa = avx512_isa_for(
            "x86_64-pc-windows-msvc".parse().unwrap(),
            Flags::new(shared),
        );

        let mut ctx = Context::for_function(empty_function(CallConv::SystemV));
        ctx.compile(&*isa, &mut Default::default())
            .expect("system_v libcalls with AVX-512 must compile");
    }
}
