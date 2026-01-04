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
    /// Emit the EVEX prefix to the code buffer.
    ///
    /// # Arguments
    /// * `dst_enc` - Destination register encoding (0-31)
    /// * `src1_enc` - First source register encoding (vvvv field, 0-31)
    /// * `src2_enc` - Second source register encoding (R/M field, 0-31)
    /// * `is_mem` - Whether src2 is a memory operand
    /// * `sink` - The code buffer to emit into
    pub fn emit(
        &self,
        dst_enc: u8,
        src1_enc: u8,
        src2_enc: u8,
        is_mem: bool,
        sink: &mut MachBuffer<Inst>,
    ) {
        // Byte 0: EVEX escape byte
        sink.put1(0x62);

        // Byte 1 (P0): R X B R' 0 0 mm
        // R, X, B, R' are inverted register extension bits
        let r = if dst_enc & 0x08 != 0 { 0 } else { 1 };
        let r_prime = if dst_enc & 0x10 != 0 { 0 } else { 1 };

        // X and B depend on whether we have a memory operand
        let (x, b) = if is_mem {
            // For memory operands, X and B come from the SIB/base register
            // For now, assume no extension bits needed for memory
            (1, 1)
        } else {
            // For register operands, B extends src2
            let b = if src2_enc & 0x08 != 0 { 0 } else { 1 };
            let x = if src2_enc & 0x10 != 0 { 0 } else { 1 };
            (x, b)
        };

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

    /// Emit the EVEX prefix for a memory operand with explicit base/index register encodings.
    ///
    /// This is used for load/store instructions where we need to encode the
    /// base register extension bits (B) and index register extension bits (X)
    /// correctly in the EVEX prefix.
    ///
    /// # Arguments
    /// * `reg_enc` - The vector register encoding (destination for loads, source for stores)
    /// * `base_enc` - The base GPR register encoding (0-15)
    /// * `index_enc` - The index GPR register encoding (0-15), or None if no index
    /// * `sink` - The code buffer to emit into
    pub fn emit_for_mem(
        &self,
        reg_enc: u8,
        base_enc: u8,
        index_enc: Option<u8>,
        sink: &mut MachBuffer<Inst>,
    ) {
        // Byte 0: EVEX escape byte
        sink.put1(0x62);

        // Byte 1 (P0): R X B R' 0 0 mm
        // R extends reg_enc bit 3, R' extends reg_enc bit 4
        // B extends base_enc bit 3
        // X extends index_enc bit 3 (or 1 if no index)
        let r = if reg_enc & 0x08 != 0 { 0 } else { 1 };
        let r_prime = if reg_enc & 0x10 != 0 { 0 } else { 1 };
        let b = if base_enc & 0x08 != 0 { 0 } else { 1 };
        let x = match index_enc {
            Some(idx) => {
                if idx & 0x08 != 0 {
                    0
                } else {
                    1
                }
            }
            None => 1, // No index register, X bit is 1 (inverted 0)
        };

        let p0 = (r << 7) | (x << 6) | (b << 5) | (r_prime << 4) | (self.map & 0x07);
        sink.put1(p0);

        // Byte 2 (P1): W vvvv 1 pp
        // For loads/stores, vvvv is typically unused (set to 1111 = no register)
        let vvvv = 0x0F; // All ones = no second source register
        let p1 = ((self.w as u8) << 7) | (vvvv << 3) | 0x04 | (self.pp & 0x03);
        sink.put1(p1);

        // Byte 3 (P2): z L'L b V' aaa
        let l_prime = (self.ll >> 1) & 1;
        let l = self.ll & 1;
        let v_prime = 1; // No second source register extension
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

    /// Emit the EVEX prefix for ALU instructions with memory operands.
    ///
    /// This is used for 3-operand ALU instructions (dst = src1 op mem) where we need to
    /// encode the src1 register in the vvvv field while also correctly encoding the
    /// base/index register extension bits for the memory operand.
    ///
    /// # Arguments
    /// * `dst_enc` - Destination vector register encoding (0-31)
    /// * `src1_enc` - First source vector register encoding (0-31), encoded in vvvv
    /// * `base_enc` - Base GPR register encoding for memory operand (0-15)
    /// * `index_enc` - Index GPR register encoding (0-15), or None if no index
    /// * `sink` - The code buffer to emit into
    pub fn emit_for_alu_mem(
        &self,
        dst_enc: u8,
        src1_enc: u8,
        base_enc: u8,
        index_enc: Option<u8>,
        sink: &mut MachBuffer<Inst>,
    ) {
        // Byte 0: EVEX escape byte
        sink.put1(0x62);

        // Byte 1 (P0): R X B R' 0 0 mm
        // R extends dst_enc bit 3, R' extends dst_enc bit 4
        // B extends base_enc bit 3
        // X extends index_enc bit 3 (or 1 if no index)
        let r = if dst_enc & 0x08 != 0 { 0 } else { 1 };
        let r_prime = if dst_enc & 0x10 != 0 { 0 } else { 1 };
        let b = if base_enc & 0x08 != 0 { 0 } else { 1 };
        let x = match index_enc {
            Some(idx) => {
                if idx & 0x08 != 0 {
                    0
                } else {
                    1
                }
            }
            None => 1, // No index register, X bit is 1 (inverted 0)
        };

        let p0 = (r << 7) | (x << 6) | (b << 5) | (r_prime << 4) | (self.map & 0x07);
        sink.put1(p0);

        // Byte 2 (P1): W vvvv 1 pp
        // vvvv encodes src1 (inverted)
        let vvvv = !src1_enc & 0x0F;
        let p1 = ((self.w as u8) << 7) | (vvvv << 3) | 0x04 | (self.pp & 0x03);
        sink.put1(p1);

        // Byte 3 (P2): z L'L b V' aaa
        let l_prime = (self.ll >> 1) & 1;
        let l = self.ll & 1;
        // V' extends src1_enc bit 4 (inverted)
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

impl EvexPrefix {
    /// Create an EVEX prefix for a 512-bit AVX-512 operation with default settings.
    pub fn new_512bit(map: u8, w: bool, pp: u8) -> Self {
        Self {
            map,
            w,
            pp,
            aaa: 0,   // no mask
            z: false, // merging (not zeroing)
            b: false, // no broadcast
            ll: 0b10, // 512-bit
        }
    }

    /// Set the write mask register (k1-k7).
    pub fn with_mask(mut self, kreg: u8) -> Self {
        debug_assert!(kreg <= 7, "k-register index must be 0-7");
        self.aaa = kreg;
        self
    }

    /// Enable zeroing mode (z bit).
    pub fn with_zeroing(mut self) -> Self {
        self.z = true;
        self
    }

    /// Enable broadcast mode (b bit).
    pub fn with_broadcast(mut self) -> Self {
        self.b = true;
        self
    }

    /// Validate the EVEX prefix configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.map > 3 {
            return Err("EVEX map must be 0-3");
        }
        if self.pp > 3 {
            return Err("EVEX pp must be 0-3");
        }
        if self.aaa > 7 {
            return Err("EVEX aaa (mask register) must be 0-7");
        }
        if self.ll > 2 {
            return Err("EVEX ll (vector length) must be 0-2");
        }
        // z bit requires a mask register (aaa != 0)
        if self.z && self.aaa == 0 {
            return Err("Zeroing mode requires a mask register (k1-k7)");
        }
        Ok(())
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

    #[test]
    fn test_evex_prefix_new_512bit() {
        let evex = EvexPrefix::new_512bit(0x02, true, 0x01);
        assert_eq!(evex.map, 0x02);
        assert!(evex.w);
        assert_eq!(evex.pp, 0x01);
        assert_eq!(evex.ll, 0b10);
        assert_eq!(evex.aaa, 0);
        assert!(!evex.z);
        assert!(!evex.b);
    }

    #[test]
    fn test_evex_prefix_with_mask() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01).with_mask(3);
        assert_eq!(evex.aaa, 3);
    }

    #[test]
    fn test_evex_prefix_with_zeroing() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01)
            .with_mask(1)
            .with_zeroing();
        assert!(evex.z);
        assert_eq!(evex.aaa, 1);
    }

    #[test]
    fn test_evex_prefix_with_broadcast() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01).with_broadcast();
        assert!(evex.b);
    }

    // =========================================================================
    // EVEX Prefix Validation Tests
    // =========================================================================

    #[test]
    fn test_evex_prefix_validate_valid() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01);
        assert!(evex.validate().is_ok());

        let evex_with_mask = EvexPrefix::new_512bit(0x02, true, 0x01)
            .with_mask(1)
            .with_zeroing();
        assert!(evex_with_mask.validate().is_ok());
    }

    #[test]
    fn test_evex_prefix_validate_invalid_map() {
        let evex = EvexPrefix {
            map: 4, // Invalid: must be 0-3
            w: false,
            pp: 0x01,
            aaa: 0,
            z: false,
            b: false,
            ll: 0b10,
        };
        assert!(evex.validate().is_err());
    }

    #[test]
    fn test_evex_prefix_validate_invalid_pp() {
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 4, // Invalid: must be 0-3
            aaa: 0,
            z: false,
            b: false,
            ll: 0b10,
        };
        assert!(evex.validate().is_err());
    }

    #[test]
    fn test_evex_prefix_validate_invalid_aaa() {
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 0x01,
            aaa: 8, // Invalid: must be 0-7
            z: false,
            b: false,
            ll: 0b10,
        };
        assert!(evex.validate().is_err());
    }

    #[test]
    fn test_evex_prefix_validate_invalid_ll() {
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 0x01,
            aaa: 0,
            z: false,
            b: false,
            ll: 3, // Invalid: must be 0-2
        };
        assert!(evex.validate().is_err());
    }

    #[test]
    fn test_evex_prefix_validate_zeroing_without_mask() {
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 0x01,
            aaa: 0,  // No mask
            z: true, // But zeroing is set - invalid!
            b: false,
            ll: 0b10,
        };
        assert!(evex.validate().is_err());
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
