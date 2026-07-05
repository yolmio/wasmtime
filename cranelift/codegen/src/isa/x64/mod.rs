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
}
