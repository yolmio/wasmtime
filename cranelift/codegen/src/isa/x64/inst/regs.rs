//! Register definitions for regalloc2.
//!
//! We define 16 GPRs, with indices equal to the hardware encoding,
//! and up to 32 XMM/YMM/ZMM registers.
//!
//! Note also that we make use of pinned VRegs to refer to PRegs.
//!
//! ## AVX-512 Register Support
//!
//! AVX-512 provides 32 ZMM registers (zmm0-zmm31). When AVX-512 is enabled,
//! all 32 registers are exposed to the register allocator:
//!
//! - **xmm0-xmm15**: Available with SSE, AVX (VEX), and AVX-512 (EVEX) encoding
//! - **xmm16-xmm31**: Available only with AVX-512 (EVEX) encoding
//!
//! The extended registers (16-31) are conditionally added to the allocatable
//! register pool when `has_avx512f` is enabled. This is safe because all
//! 512-bit vector operations use EVEX encoding, which supports the full
//! 32-register file.

use crate::machinst::Reg;
use alloc::string::ToString;
use cranelift_assembler_x64::{gpr, xmm};
use regalloc2::{PReg, RegClass, VReg};
use std::string::String;

// Constructors for Regs.

const fn gpr(enc: u8) -> Reg {
    let preg = gpr_preg(enc);
    Reg::from_virtual_reg(VReg::new(preg.index(), RegClass::Int))
}
pub(crate) const fn gpr_preg(enc: u8) -> PReg {
    PReg::new(enc as usize, RegClass::Int)
}

pub(crate) const fn rax() -> Reg {
    gpr(gpr::enc::RAX)
}
pub(crate) const fn rcx() -> Reg {
    gpr(gpr::enc::RCX)
}
pub(crate) const fn rdx() -> Reg {
    gpr(gpr::enc::RDX)
}
pub(crate) const fn rbx() -> Reg {
    gpr(gpr::enc::RBX)
}
pub(crate) const fn rsp() -> Reg {
    gpr(gpr::enc::RSP)
}
pub(crate) const fn rbp() -> Reg {
    gpr(gpr::enc::RBP)
}
pub(crate) const fn rsi() -> Reg {
    gpr(gpr::enc::RSI)
}
pub(crate) const fn rdi() -> Reg {
    gpr(gpr::enc::RDI)
}
pub(crate) const fn r8() -> Reg {
    gpr(gpr::enc::R8)
}
pub(crate) const fn r9() -> Reg {
    gpr(gpr::enc::R9)
}
pub(crate) const fn r10() -> Reg {
    gpr(gpr::enc::R10)
}
pub(crate) const fn r11() -> Reg {
    gpr(gpr::enc::R11)
}
pub(crate) const fn r12() -> Reg {
    gpr(gpr::enc::R12)
}
pub(crate) const fn r13() -> Reg {
    gpr(gpr::enc::R13)
}
pub(crate) const fn r14() -> Reg {
    gpr(gpr::enc::R14)
}
pub(crate) const fn r15() -> Reg {
    gpr(gpr::enc::R15)
}

/// The pinned register on this architecture.
/// It must be the same as Spidermonkey's HeapReg, as found in this file.
/// https://searchfox.org/mozilla-central/source/js/src/jit/x64/Assembler-x64.h#99
pub(crate) const fn pinned_reg() -> Reg {
    r15()
}

const fn fpr(enc: u8) -> Reg {
    let preg = fpr_preg(enc);
    Reg::from_virtual_reg(VReg::new(preg.index(), RegClass::Float))
}

pub(crate) const fn fpr_preg(enc: u8) -> PReg {
    PReg::new(enc as usize, RegClass::Float)
}

pub(crate) const fn xmm0() -> Reg {
    fpr(xmm::enc::XMM0)
}
pub(crate) const fn xmm1() -> Reg {
    fpr(xmm::enc::XMM1)
}
pub(crate) const fn xmm2() -> Reg {
    fpr(xmm::enc::XMM2)
}
pub(crate) const fn xmm3() -> Reg {
    fpr(xmm::enc::XMM3)
}
pub(crate) const fn xmm4() -> Reg {
    fpr(xmm::enc::XMM4)
}
pub(crate) const fn xmm5() -> Reg {
    fpr(xmm::enc::XMM5)
}
pub(crate) const fn xmm6() -> Reg {
    fpr(xmm::enc::XMM6)
}
pub(crate) const fn xmm7() -> Reg {
    fpr(xmm::enc::XMM7)
}
pub(crate) const fn xmm8() -> Reg {
    fpr(xmm::enc::XMM8)
}
pub(crate) const fn xmm9() -> Reg {
    fpr(xmm::enc::XMM9)
}
pub(crate) const fn xmm10() -> Reg {
    fpr(xmm::enc::XMM10)
}
pub(crate) const fn xmm11() -> Reg {
    fpr(xmm::enc::XMM11)
}
pub(crate) const fn xmm12() -> Reg {
    fpr(xmm::enc::XMM12)
}
pub(crate) const fn xmm13() -> Reg {
    fpr(xmm::enc::XMM13)
}
pub(crate) const fn xmm14() -> Reg {
    fpr(xmm::enc::XMM14)
}
pub(crate) const fn xmm15() -> Reg {
    fpr(xmm::enc::XMM15)
}

// AVX-512 extended registers (xmm16-xmm31)
// These require EVEX encoding and are only available with AVX-512.
pub(crate) const fn xmm16() -> Reg {
    fpr(xmm::enc::XMM16)
}
pub(crate) const fn xmm17() -> Reg {
    fpr(xmm::enc::XMM17)
}
pub(crate) const fn xmm18() -> Reg {
    fpr(xmm::enc::XMM18)
}
pub(crate) const fn xmm19() -> Reg {
    fpr(xmm::enc::XMM19)
}
pub(crate) const fn xmm20() -> Reg {
    fpr(xmm::enc::XMM20)
}
pub(crate) const fn xmm21() -> Reg {
    fpr(xmm::enc::XMM21)
}
pub(crate) const fn xmm22() -> Reg {
    fpr(xmm::enc::XMM22)
}
pub(crate) const fn xmm23() -> Reg {
    fpr(xmm::enc::XMM23)
}
pub(crate) const fn xmm24() -> Reg {
    fpr(xmm::enc::XMM24)
}
pub(crate) const fn xmm25() -> Reg {
    fpr(xmm::enc::XMM25)
}
pub(crate) const fn xmm26() -> Reg {
    fpr(xmm::enc::XMM26)
}
pub(crate) const fn xmm27() -> Reg {
    fpr(xmm::enc::XMM27)
}
pub(crate) const fn xmm28() -> Reg {
    fpr(xmm::enc::XMM28)
}
pub(crate) const fn xmm29() -> Reg {
    fpr(xmm::enc::XMM29)
}
pub(crate) const fn xmm30() -> Reg {
    fpr(xmm::enc::XMM30)
}
pub(crate) const fn xmm31() -> Reg {
    fpr(xmm::enc::XMM31)
}

// K-registers are AVX-512 mask registers.
// These are physical registers with hardware encodings 0-7 (k0-k7).
pub(crate) const fn k_preg(enc: u8) -> PReg {
    PReg::new(enc as usize, RegClass::Vector)
}

const fn k_reg(enc: u8) -> Reg {
    let preg = k_preg(enc);
    // Create as a real register, not a virtual register
    // This allows reg_def/reg_use to properly identify it as fixed/nonallocatable
    Reg::from_real_reg(preg)
}

pub(crate) const fn k0() -> Reg {
    k_reg(0)
}
pub(crate) const fn k1() -> Reg {
    k_reg(1)
}
pub(crate) const fn k2() -> Reg {
    k_reg(2)
}
pub(crate) const fn k3() -> Reg {
    k_reg(3)
}
pub(crate) const fn k4() -> Reg {
    k_reg(4)
}
pub(crate) const fn k5() -> Reg {
    k_reg(5)
}
pub(crate) const fn k6() -> Reg {
    k_reg(6)
}
pub(crate) const fn k7() -> Reg {
    k_reg(7)
}

// N.B.: this is not an `impl PrettyPrint for Reg` because it is
// specific to x64; other backends have analogous functions. The
// disambiguation happens statically by virtue of higher-level,
// x64-specific, types calling the right `pretty_print_reg`. (In other
// words, we can't pretty-print a `Reg` all by itself in a build that
// may have multiple backends; but we can pretty-print one as part of
// an x64 Inst or x64 RegMemImm.)
pub fn pretty_print_reg(reg: Reg, size: u8) -> String {
    if let Some(rreg) = reg.to_real_reg() {
        let enc = rreg.hw_enc();
        match rreg.class() {
            RegClass::Int => {
                let size = match size {
                    8 => gpr::Size::Quadword,
                    4 => gpr::Size::Doubleword,
                    2 => gpr::Size::Word,
                    1 => gpr::Size::Byte,
                    _ => unreachable!("invalid size"),
                };
                gpr::enc::to_string(enc, size).to_string()
            }
            RegClass::Float => xmm::enc::to_string(enc).to_string(),
            RegClass::Vector => format!("%k{enc}"),
        }
    } else {
        let mut name = format!("%{reg:?}");
        // Add size suffixes to GPR virtual registers at narrower widths.
        if reg.class() == RegClass::Int && size != 8 {
            name.push_str(match size {
                4 => "l",
                2 => "w",
                1 => "b",
                _ => unreachable!("invalid size"),
            });
        }
        name
    }
}
