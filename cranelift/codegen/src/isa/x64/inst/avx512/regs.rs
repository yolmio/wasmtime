// cranelift/codegen/src/isa/x64/inst/avx512/regs.rs
//
// AVX-512 k-register utilities.
//
// This module provides utilities for working with AVX-512 mask registers (k-registers).
// The actual k-register definitions are in the main regs.rs module at:
//   cranelift/codegen/src/isa/x64/inst/regs.rs
//
// Use `regs::k1()`, `regs::k2()`, etc. for k-register construction.

/// Returns the hardware encoding for a k-register (0-7).
#[inline]
#[allow(dead_code, reason = "k-register utility reserved for future use")]
pub fn k_hw_enc(kreg: u8) -> u8 {
    debug_assert!(kreg < 8, "k-register index must be 0-7");
    kreg
}

/// Check if a k-register index is valid for masking (k1-k7).
/// k0 is special and means "no masking" - using k0 as a writemask
/// is architecturally undefined and should be avoided.
#[inline]
#[allow(dead_code, reason = "k-register utility reserved for future use")]
pub fn is_valid_mask_kreg(kreg: u8) -> bool {
    kreg >= 1 && kreg <= 7
}
