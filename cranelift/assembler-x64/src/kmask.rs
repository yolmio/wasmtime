//! K-mask register operands for AVX-512 predicated operations.
//!
//! AVX-512 introduces 8 opmask registers (k0-k7) for predicated (masked)
//! operations. Each bit in a mask register corresponds to one vector element.
//!
//! **Special case**: k0 is hardcoded as "no masking" - when used as a mask,
//! all elements are processed unconditionally.
//!
//! See [`Kmask`] for the register type.

use crate::{AsReg, CodeSink, rex::encode_modrm};

/// An x64 AVX-512 mask register (e.g., `%k1`).
#[derive(Clone, Copy, Debug)]
pub struct Kmask<R: AsReg = u8>(pub(crate) R);

impl<R: AsReg> Kmask<R> {
    /// Create a new [`Kmask`] register.
    #[must_use]
    pub fn new(reg: R) -> Self {
        Self(reg)
    }

    /// Return the register's hardware encoding; the underlying type `R` _must_
    /// be a real register at this point.
    ///
    /// # Panics
    ///
    /// Panics if the register is not a valid Kmask register.
    #[must_use]
    pub fn enc(&self) -> u8 {
        let enc = self.0.enc();
        assert!(enc < 8, "invalid k-register: {enc}");
        enc
    }

    /// Return the register name.
    #[must_use]
    pub fn to_string(&self) -> String {
        enc::to_string(self.enc()).to_owned()
    }

    /// Emit this register as the `r/m` field of a ModR/M byte.
    pub(crate) fn encode_modrm(&self, sink: &mut impl CodeSink, enc_reg: u8) {
        sink.put1(encode_modrm(0b11, enc_reg & 0b111, self.enc() & 0b111));
    }

    /// Return the registers for encoding the `b` and `x` bits (e.g., in a VEX
    /// prefix).
    ///
    /// For k-registers, only the `b` bit is set by the topmost bits (bits 3+)
    /// of this register. We expect this register to be in the `rm` slot.
    /// Note: k-registers only use 3 bits (k0-k7), so this typically returns
    /// an encoding < 8.
    pub(crate) fn encode_bx_regs(&self) -> (Option<u8>, Option<u8>) {
        (Some(self.enc()), None)
    }
}

impl<R: AsReg> AsRef<R> for Kmask<R> {
    fn as_ref(&self) -> &R {
        &self.0
    }
}

impl<R: AsReg> AsMut<R> for Kmask<R> {
    fn as_mut(&mut self) -> &mut R {
        &mut self.0
    }
}

impl<R: AsReg> From<R> for Kmask<R> {
    fn from(reg: R) -> Kmask<R> {
        Kmask(reg)
    }
}

/// Encode k-mask registers.
pub mod enc {
    pub const K0: u8 = 0;
    pub const K1: u8 = 1;
    pub const K2: u8 = 2;
    pub const K3: u8 = 3;
    pub const K4: u8 = 4;
    pub const K5: u8 = 5;
    pub const K6: u8 = 6;
    pub const K7: u8 = 7;

    /// Return the name of a K-mask encoding (`enc`).
    ///
    /// # Panics
    ///
    /// This function will panic if the encoding is not a valid k-register.
    pub fn to_string(enc: u8) -> &'static str {
        match enc {
            K0 => "%k0",
            K1 => "%k1",
            K2 => "%k2",
            K3 => "%k3",
            K4 => "%k4",
            K5 => "%k5",
            K6 => "%k6",
            K7 => "%k7",
            _ => panic!("%invalid{enc}"),
        }
    }
}
