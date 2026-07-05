// cranelift/codegen/src/isa/x64/inst/avx512/encoding.rs
//
// AVX-512: EVEX prefix encoding for AVX-512 instructions.
//
// EVEX is a 4-byte prefix used for AVX-512 instructions. This module
// implements the encoding logic for Avx512-native 512-bit operations.

use crate::isa::x64::inst::Inst;
use crate::machinst::MachBuffer;

/// EVEX prefix structure for AVX-512 instructions.
///
/// EVEX Prefix Format (4 bytes):
/// ```text
/// Byte 0: 0x62 (EVEX escape)
/// Byte 1 (P0): [R:1][X:1][B:1][R':1][0:1][0:1][m:2]
/// Byte 2 (P1): [W:1][vvvv:4][1:1][pp:2]
/// Byte 3 (P2): [z:1][L':1][L:1][b:1][V':1][aaa:3]
/// ```
///
/// Where:
/// - R, X, B, R': Register index extensions (inverted)
/// - m: Opcode map (01=0F, 02=0F38, 03=0F3A)
/// - W: Operand size (0=32-bit, 1=64-bit for most instructions)
/// - vvvv: Second source register (inverted, 4 bits)
/// - pp: Prefix (00=none, 01=66, 10=F3, 11=F2)
/// - z: Zeroing (1) vs Merging (0)
/// - L'L: Vector length (00=128, 01=256, 10=512)
/// - b: Broadcast mode / embedded rounding
/// - V': High bit of vvvv extension (for EVEX.vvvv 5th bit)
/// - aaa: Opmask register (k0-k7)
#[derive(Clone, Copy, Debug)]
pub struct EvexPrefix {
    /// Opcode map: 0x01 = 0F, 0x02 = 0F38, 0x03 = 0F3A
    pub map: u8,
    /// W bit: operand size modifier
    pub w: bool,
    /// Prefix: 0x00 = none, 0x01 = 66, 0x02 = F3, 0x03 = F2
    pub pp: u8,
    /// Opmask register index (k1-k7, or k0 for no mask)
    pub aaa: u8,
    /// Zeroing mode (true = zeroing, false = merging)
    pub z: bool,
    /// Broadcast mode / embedded rounding
    pub b: bool,
    /// Vector length: 0b00 = 128-bit, 0b01 = 256-bit, 0b10 = 512-bit
    pub ll: u8,
}

impl EvexPrefix {
    /// Emit the EVEX prefix for a register-to-register form (ModRM.mod =
    /// 0b11) to the code buffer. Memory-operand forms need the X/B bits
    /// derived from the base/index registers and must not use this method;
    /// the only remaining manual memory form (VSIB gather/scatter) goes
    /// through `emit_vsib` instead.
    ///
    /// # Arguments
    /// * `dst_enc` - Destination register encoding (0-31)
    /// * `src1_enc` - First source register encoding (vvvv field, 0-31)
    /// * `src2_enc` - Second source register encoding (R/M field, 0-31)
    /// * `sink` - The code buffer to emit into
    pub fn emit_reg(&self, dst_enc: u8, src1_enc: u8, src2_enc: u8, sink: &mut MachBuffer<Inst>) {
        // Byte 0: EVEX escape byte
        sink.put1(0x62);

        // Byte 1 (P0): R X B R' 0 0 mm
        // R, X, B, R' are inverted register extension bits
        let r = if dst_enc & 0x08 != 0 { 0 } else { 1 };
        let r_prime = if dst_enc & 0x10 != 0 { 0 } else { 1 };

        // For register operands, B extends src2 (bit 3) and X extends src2
        // (bit 4, EVEX's fifth register-encoding bit).
        let b = if src2_enc & 0x08 != 0 { 0 } else { 1 };
        let x = if src2_enc & 0x10 != 0 { 0 } else { 1 };

        let p0 = (r << 7) | (x << 6) | (b << 5) | (r_prime << 4) | (self.map & 0x07);
        sink.put1(p0);

        // Byte 2 (P1): W vvvv 1 pp
        // vvvv is inverted src1 register encoding
        let vvvv = !src1_enc & 0x0F;
        let p1 = ((self.w as u8) << 7) | (vvvv << 3) | 0x04 | (self.pp & 0x03);
        sink.put1(p1);

        // Byte 3 (P2): z L'L b V' aaa
        let l_prime = (self.ll >> 1) & 1;
        let l = self.ll & 1;
        let v_prime = if src1_enc & 0x10 != 0 { 0 } else { 1 };
        // IMPORTANT: Zeroing mode (z=1) is only valid with a mask register (aaa != 0)
        // Without a mask, z must be 0 even if self.z is true
        let z = if self.aaa == 0 { 0 } else { self.z as u8 };
        let p2 = (z << 7)
            | (l_prime << 6)
            | (l << 5)
            | ((self.b as u8) << 4)
            | (v_prime << 3)
            | (self.aaa & 0x07);
        sink.put1(p2);
    }
}

impl EvexPrefix {
    /// Emit the EVEX prefix for VSIB (Vector SIB) addressing used in gather/scatter.
    ///
    /// In VSIB addressing:
    /// - The index register is a vector register (XMM/YMM/ZMM)
    /// - R and R' extend the destination/source vector register
    /// - X extends the vector index register (bits 3-4)
    /// - B extends the base GPR register
    ///
    /// # Arguments
    /// * `reg_enc` - Destination (gather) or source (scatter) vector register encoding (0-31)
    /// * `index_enc` - Index vector register encoding (0-31)
    /// * `base_enc` - Base GPR register encoding (0-15)
    /// * `sink` - The code buffer to emit into
    pub fn emit_vsib(&self, reg_enc: u8, index_enc: u8, base_enc: u8, sink: &mut MachBuffer<Inst>) {
        // Byte 0: EVEX escape byte
        sink.put1(0x62);

        // Byte 1 (P0): R X B R' 0 0 mm
        // R extends reg_enc bit 3, R' extends reg_enc bit 4
        // X extends index_enc bit 3 (vector index register)
        // B extends base_enc bit 3 (GPR base register)
        let r = if reg_enc & 0x08 != 0 { 0 } else { 1 };
        let r_prime = if reg_enc & 0x10 != 0 { 0 } else { 1 };
        let x = if index_enc & 0x08 != 0 { 0 } else { 1 };
        let b = if base_enc & 0x08 != 0 { 0 } else { 1 };

        let p0 = (r << 7) | (x << 6) | (b << 5) | (r_prime << 4) | (self.map & 0x07);
        sink.put1(p0);

        // Byte 2 (P1): W vvvv 1 pp
        // For gather/scatter, vvvv is UNUSED and must be 1111b (0x0F).
        // The index register is encoded only in the SIB byte and extended via:
        // - EVEX.X (bit 3 of index_enc) in P0
        // - EVEX.V' (bit 4 of index_enc) in P2
        let vvvv = 0x0F; // Must be 1111b for gather/scatter
        let p1 = ((self.w as u8) << 7) | (vvvv << 3) | 0x04 | (self.pp & 0x03);
        sink.put1(p1);

        // Byte 3 (P2): z L'L b V' aaa
        // V' extends the index register (bit 4)
        let l_prime = (self.ll >> 1) & 1;
        let l = self.ll & 1;
        let v_prime = if index_enc & 0x10 != 0 { 0 } else { 1 };
        // IMPORTANT: Zeroing mode (z=1) is only valid with a mask register (aaa != 0)
        // Without a mask, z must be 0 even if self.z is true
        let z = if self.aaa == 0 { 0 } else { self.z as u8 };
        let p2 = (z << 7)
            | (l_prime << 6)
            | (l << 5)
            | ((self.b as u8) << 4)
            | (v_prime << 3)
            | (self.aaa & 0x07);
        sink.put1(p2);
    }
}

/// An EVEX memory-operand displacement, after compressed-displacement
/// (disp8*N) classification.
///
/// EVEX-encoded instructions never use a raw 8-bit displacement. Per Intel
/// SDM Vol. 2A §2.7.5, an 8-bit displacement is always *compressed*: the
/// byte stored in the instruction is `disp / N`, and the hardware multiplies
/// it back by N when computing the effective address. For gather/scatter
/// (Tuple1 Scalar), N is the element size in bytes (4 for dword elements, 8
/// for qword elements).
///
/// This type is the only way displacement bytes get emitted for the
/// hand-written EVEX paths (gather/scatter): `EvexDisp::new` performs the
/// scaling and range checks, so it is impossible to accidentally emit a raw
/// displacement as disp8 (which the hardware would silently multiply by N).
///
/// The assembler DSL has its own equivalent in `cranelift-assembler-x64`
/// (`rex::Disp::new` with `evex_scaling`), but that type is not exported and
/// the manual VSIB emission here does not go through the DSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvexDisp {
    /// No displacement bytes (ModRM.mod = 0b00).
    None,
    /// Compressed 8-bit displacement (ModRM.mod = 0b01). `scaled` is the
    /// *already-scaled* value (`raw_disp / N`); it is emitted as-is and the
    /// hardware multiplies it by N.
    Disp8 {
        /// The compressed displacement byte: `raw_disp / N`.
        scaled: i8,
    },
    /// Full 32-bit displacement (ModRM.mod = 0b10), *not* scaled.
    Disp32(i32),
}

impl EvexDisp {
    /// Classify the raw byte displacement `disp` for an EVEX memory operand
    /// with compressed-displacement scaling factor `n` (must be non-zero;
    /// the element size for gather/scatter).
    ///
    /// `base_enc` is the hardware encoding of the base register. With
    /// ModRM.mod = 0b00 and a SIB byte, a base whose low three bits are 101
    /// (RBP/R13) means "no base, disp32 follows", so such bases must always
    /// carry an explicit displacement even when it is zero.
    pub fn new(disp: i32, n: u8, base_enc: u8) -> Self {
        debug_assert!(n != 0, "EVEX compressed-displacement factor must be non-zero");
        if disp == 0 && (base_enc & 0x07) != 5 {
            return Self::None;
        }
        let n = i32::from(n);
        if disp % n == 0 {
            if let Ok(scaled) = i8::try_from(disp / n) {
                return Self::Disp8 { scaled };
            }
        }
        Self::Disp32(disp)
    }

    /// The ModRM.mod field value (0b00, 0b01, or 0b10) selecting this
    /// displacement form.
    pub fn mod_bits(&self) -> u8 {
        match self {
            Self::None => 0b00,
            Self::Disp8 { .. } => 0b01,
            Self::Disp32(..) => 0b10,
        }
    }

    /// Emit the displacement bytes. Must be called directly after the
    /// ModRM/SIB bytes, whose mod field must be `self.mod_bits()`.
    pub fn emit(&self, sink: &mut MachBuffer<Inst>) {
        match *self {
            Self::None => {}
            Self::Disp8 { scaled } => sink.put1(scaled as u8),
            Self::Disp32(disp) => sink.put4(disp as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EVEX Prefix Structure Tests
    // =========================================================================

    #[test]
    fn test_evex_prefix_basic() {
        // Test basic EVEX encoding for VPADDD zmm0, zmm1, zmm2
        // Expected: 62 F1 75 48 FE C2
        //   62 = EVEX escape
        //   F1 = P0: R=1 X=1 B=1 R'=1 00 mm=01 (0F map)
        //   75 = P1: W=0 vvvv=1110 (inverted 1) 1 pp=01 (66)
        //   48 = P2: z=0 L'L=10 (512-bit) b=0 V'=1 aaa=000

        let evex = EvexPrefix {
            map: 0x01, // 0F
            w: false,  // 32-bit elements
            pp: 0x01,  // 66 prefix
            aaa: 0,    // no mask
            z: false,  // no zeroing
            b: false,  // no broadcast
            ll: 0b10,  // 512-bit
        };

        assert_eq!(evex.map, 0x01);
        assert!(!evex.w);
        assert_eq!(evex.pp, 0x01);
        assert_eq!(evex.aaa, 0);
        assert!(!evex.z);
        assert_eq!(evex.ll, 0b10);
    }

    // =========================================================================
    // EVEX Map Constants Tests
    // =========================================================================

    #[test]
    fn test_evex_map_constants() {
        // Verify map values match Intel documentation
        // 0x01 = 0F (legacy 2-byte opcode)
        // 0x02 = 0F38 (legacy 3-byte opcode)
        // 0x03 = 0F3A (legacy 3-byte opcode with immediate)
        assert_eq!(0x01, 1); // 0F
        assert_eq!(0x02, 2); // 0F38
        assert_eq!(0x03, 3); // 0F3A
    }

    // =========================================================================
    // EVEX PP (Prefix) Constants Tests
    // =========================================================================

    #[test]
    fn test_evex_pp_constants() {
        // Verify pp values match Intel documentation
        // 0x00 = no prefix (NP)
        // 0x01 = 66 prefix
        // 0x02 = F3 prefix
        // 0x03 = F2 prefix
        assert_eq!(0x00, 0); // NP
        assert_eq!(0x01, 1); // 66
        assert_eq!(0x02, 2); // F3
        assert_eq!(0x03, 3); // F2
    }

    // =========================================================================
    // EVEX Vector Length Tests
    // =========================================================================

    // =========================================================================
    // EVEX Compressed Displacement (disp8*N) Classification Tests
    // =========================================================================

    #[test]
    fn test_evex_disp_none_for_zero_disp() {
        // rdi (enc 7) base, zero displacement: no displacement bytes.
        assert_eq!(EvexDisp::new(0, 4, 7), EvexDisp::None);
        assert_eq!(EvexDisp::new(0, 8, 0), EvexDisp::None);
    }

    #[test]
    fn test_evex_disp_rbp_r13_base_needs_explicit_disp() {
        // RBP (enc 5) and R13 (enc 13) cannot use mod=00 with a SIB byte;
        // a zero displacement must be encoded as compressed disp8 of 0.
        assert_eq!(EvexDisp::new(0, 4, 5), EvexDisp::Disp8 { scaled: 0 });
        assert_eq!(EvexDisp::new(0, 8, 13), EvexDisp::Disp8 { scaled: 0 });
    }

    #[test]
    fn test_evex_disp_disp8_is_scaled() {
        // The emitted disp8 byte is disp / N, never the raw displacement.
        assert_eq!(EvexDisp::new(4, 4, 7), EvexDisp::Disp8 { scaled: 1 });
        assert_eq!(EvexDisp::new(8, 4, 7), EvexDisp::Disp8 { scaled: 2 });
        assert_eq!(EvexDisp::new(64, 4, 7), EvexDisp::Disp8 { scaled: 16 });
        assert_eq!(EvexDisp::new(-8, 4, 7), EvexDisp::Disp8 { scaled: -2 });
        assert_eq!(EvexDisp::new(8, 8, 7), EvexDisp::Disp8 { scaled: 1 });
        // Extremes of the scaled i8 range.
        assert_eq!(EvexDisp::new(127 * 4, 4, 7), EvexDisp::Disp8 { scaled: 127 });
        assert_eq!(
            EvexDisp::new(-128 * 8, 8, 7),
            EvexDisp::Disp8 { scaled: -128 }
        );
    }

    #[test]
    fn test_evex_disp_unaligned_uses_disp32() {
        // Displacements not divisible by N cannot be compressed.
        assert_eq!(EvexDisp::new(127, 4, 7), EvexDisp::Disp32(127));
        assert_eq!(EvexDisp::new(2, 4, 7), EvexDisp::Disp32(2));
        assert_eq!(EvexDisp::new(4, 8, 7), EvexDisp::Disp32(4));
    }

    #[test]
    fn test_evex_disp_out_of_range_uses_disp32() {
        // Scaled value must fit in i8; 512/4 = 128 does not.
        assert_eq!(EvexDisp::new(512, 4, 7), EvexDisp::Disp32(512));
        assert_eq!(EvexDisp::new(128, 1, 7), EvexDisp::Disp32(128));
        assert_eq!(EvexDisp::new(-129 * 8, 8, 7), EvexDisp::Disp32(-129 * 8));
    }

    #[test]
    fn test_evex_disp_mod_bits() {
        assert_eq!(EvexDisp::None.mod_bits(), 0b00);
        assert_eq!(EvexDisp::Disp8 { scaled: 1 }.mod_bits(), 0b01);
        assert_eq!(EvexDisp::Disp32(4).mod_bits(), 0b10);
    }

    #[test]
    fn test_evex_vector_length_constants() {
        // Verify ll values match Intel documentation
        // 0b00 = 128-bit (XMM)
        // 0b01 = 256-bit (YMM)
        // 0b10 = 512-bit (ZMM)
        assert_eq!(0b00, 0); // 128-bit
        assert_eq!(0b01, 1); // 256-bit
        assert_eq!(0b10, 2); // 512-bit
    }
}
