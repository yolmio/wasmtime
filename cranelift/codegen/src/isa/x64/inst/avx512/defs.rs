// cranelift/codegen/src/isa/x64/inst/avx512/defs.rs
//
// AVX-512 instruction definitions.
//
// This module defines the AVX-512 instruction opcodes, merge modes,
// and related types used for 512-bit SIMD operations.

/// Merge mode for AVX-512 masked operations.
///
/// AVX-512 supports two merge modes when using mask registers:
/// - **Merging**: Elements where the mask bit is 0 retain their original value
/// - **Zeroing**: Elements where the mask bit is 0 are set to zero
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Preserve destination elements where mask bit is 0.
    Merging,
    /// Zero destination elements where mask bit is 0.
    #[default]
    Zeroing,
}

/// Mask register ALU operations.
///
/// These operate on k-registers directly (k1-k7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAluOp {
    /// KAND - Bitwise AND of mask registers
    Kand,
    /// KOR - Bitwise OR of mask registers
    Kor,
    /// KXOR - Bitwise XOR of mask registers
    Kxor,
    /// KXNOR - Bitwise XNOR of mask registers (all 1s when src1 == src2)
    Kxnor,
    /// KNOT - Bitwise NOT of mask register
    Knot,
    /// KANDN - Bitwise AND NOT of mask registers
    Kandn,
}

/// AVX-512 comparison conditions (for VPCMPD, VPCMPQ, etc.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512Cond {
    /// Equal
    Eq = 0,
    /// Less than (signed)
    Lt = 1,
    /// Less than or equal (signed)
    Le = 2,
    /// Not equal
    Neq = 4,
    /// Greater than or equal (signed)
    Ge = 5,
    /// Greater than (signed)
    Gt = 6,
}

/// AVX-512 ALU operations.
///
/// These operations use standard AVX-512 EVEX encoding for 512-bit vector operations.
/// All operations use 512-bit vectors (ZMM registers) with EVEX encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512AluOp {
    // =========================================
    // Integer Arithmetic Operations
    // =========================================
    /// VPADDD - Packed doubleword (32-bit) integer add
    Vpaddd,
    /// VPADDQ - Packed quadword (64-bit) integer add
    Vpaddq,
    /// VPSUBD - Packed doubleword (32-bit) integer subtract
    Vpsubd,
    /// VPSUBQ - Packed quadword (64-bit) integer subtract
    Vpsubq,
    /// VPMULLD - Packed 32-bit multiply low (low 32 bits of 32x32 result)
    Vpmulld,
    /// VPMULLQ - Packed 64-bit multiply low (low 64 bits of 64x64 result)
    Vpmullq,
    /// VPMULUDQ - Packed 32x32→64 unsigned multiply (uses low 32 bits of each 64-bit element)
    Vpmuludq,
    /// VPMULDQ - Packed 32x32→64 signed multiply (uses low 32 bits of each 64-bit element)
    Vpmuldq,

    // =========================================
    // Byte/Word Arithmetic Operations (AVX-512BW)
    // =========================================
    /// VPADDB - Packed byte (8-bit) integer add
    Vpaddb,
    /// VPADDW - Packed word (16-bit) integer add
    Vpaddw,
    /// VPSUBB - Packed byte (8-bit) integer subtract
    Vpsubb,
    /// VPSUBW - Packed word (16-bit) integer subtract
    Vpsubw,
    /// VPMINSB - Packed minimum signed (8-bit)
    Vpminsb,
    /// VPMINUB - Packed minimum unsigned (8-bit)
    Vpminub,
    /// VPMINSW - Packed minimum signed (16-bit)
    Vpminsw,
    /// VPMINUW - Packed minimum unsigned (16-bit)
    Vpminuw,
    /// VPMAXSB - Packed maximum signed (8-bit)
    Vpmaxsb,
    /// VPMAXUB - Packed maximum unsigned (8-bit)
    Vpmaxub,
    /// VPMAXSW - Packed maximum signed (16-bit)
    Vpmaxsw,
    /// VPMAXUW - Packed maximum unsigned (16-bit)
    Vpmaxuw,
    /// VPABSB - Packed absolute value (8-bit)
    Vpabsb,
    /// VPABSW - Packed absolute value (16-bit)
    Vpabsw,
    /// VPCMPEQB - Packed compare equal (8-bit) - all 1s if equal, else 0
    Vpcmpeqb,
    /// VPCMPEQW - Packed compare equal (16-bit) - all 1s if equal, else 0
    Vpcmpeqw,
    /// VPCMPGTB - Packed compare greater than (8-bit signed)
    Vpcmpgtb,
    /// VPCMPGTW - Packed compare greater than (16-bit signed)
    Vpcmpgtw,

    // =========================================
    // Bitwise Logical Operations
    // =========================================
    /// VPANDD - Packed bitwise AND (32-bit elements)
    Vpandd,
    /// VPANDQ - Packed bitwise AND (64-bit elements)
    Vpandq,
    /// VPORD - Packed bitwise OR (32-bit elements)
    Vpord,
    /// VPORQ - Packed bitwise OR (64-bit elements)
    Vporq,
    /// VPXORD - Packed bitwise XOR (32-bit elements)
    Vpxord,
    /// VPXORQ - Packed bitwise XOR (64-bit elements)
    Vpxorq,
    /// VPANDND - Packed bitwise AND NOT (32-bit elements)
    Vpandnd,
    /// VPANDNQ - Packed bitwise AND NOT (64-bit elements)
    Vpandnq,

    // =========================================
    // Shift Operations
    // =========================================
    /// VPSLLD - Packed shift left logical (32-bit, uniform shift)
    Vpslld,
    /// VPSLLQ - Packed shift left logical (64-bit, uniform shift)
    Vpsllq,
    /// VPSRLD - Packed shift right logical (32-bit, uniform shift)
    Vpsrld,
    /// VPSRLQ - Packed shift right logical (64-bit, uniform shift)
    Vpsrlq,
    /// VPSRAD - Packed shift right arithmetic (32-bit, uniform shift)
    Vpsrad,
    /// VPSRAQ - Packed shift right arithmetic (64-bit, uniform shift)
    Vpsraq,
    /// VPSLLVD - Packed shift left logical variable (32-bit, per-element shift)
    Vpsllvd,
    /// VPSLLVQ - Packed shift left logical variable (64-bit, per-element shift)
    Vpsllvq,
    /// VPSRLVD - Packed shift right logical variable (32-bit, per-element shift)
    Vpsrlvd,
    /// VPSRLVQ - Packed shift right logical variable (64-bit, per-element shift)
    Vpsrlvq,
    /// VPSRAVD - Packed shift right arithmetic variable (32-bit, per-element shift)
    Vpsravd,
    /// VPSRAVQ - Packed shift right arithmetic variable (64-bit, per-element shift)
    Vpsravq,

    // =========================================
    // Rotate Operations
    // =========================================
    /// VPROLVD - Packed rotate left variable (32-bit, per-element rotation)
    Vprolvd,
    /// VPROLVQ - Packed rotate left variable (64-bit, per-element rotation)
    Vprolvq,

    // =========================================
    // Min/Max Operations
    // =========================================
    /// VPMINSD - Packed minimum signed (32-bit)
    Vpminsd,
    /// VPMINSQ - Packed minimum signed (64-bit)
    Vpminsq,
    /// VPMAXSD - Packed maximum signed (32-bit)
    Vpmaxsd,
    /// VPMAXSQ - Packed maximum signed (64-bit)
    Vpmaxsq,
    /// VPMINUD - Packed minimum unsigned (32-bit)
    Vpminud,
    /// VPMINUQ - Packed minimum unsigned (64-bit)
    Vpminuq,
    /// VPMAXUD - Packed maximum unsigned (32-bit)
    Vpmaxud,
    /// VPMAXUQ - Packed maximum unsigned (64-bit)
    Vpmaxuq,

    // =========================================
    // Absolute Value Operations
    // =========================================
    /// VPABSD - Packed absolute value (32-bit)
    Vpabsd,
    /// VPABSQ - Packed absolute value (64-bit)
    Vpabsq,

    // =========================================
    // Population Count Operations (AVX-512 VPOPCNTDQ)
    // =========================================
    /// VPOPCNTD - Population count (count 1 bits) for each 32-bit element
    Vpopcntd,
    /// VPOPCNTQ - Population count (count 1 bits) for each 64-bit element
    Vpopcntq,

    // =========================================
    // Broadcast Operations
    // =========================================
    /// VPBROADCASTB - Broadcast 8-bit element to all lanes (AVX-512BW)
    Vpbroadcastb,
    /// VPBROADCASTW - Broadcast 16-bit element to all lanes (AVX-512BW)
    Vpbroadcastw,
    /// VPBROADCASTD - Broadcast 32-bit element to all lanes
    Vpbroadcastd,
    /// VPBROADCASTQ - Broadcast 64-bit element to all lanes
    Vpbroadcastq,

    // =========================================
    // Blend Operations
    // =========================================
    /// VPBLENDMD - Blend 32-bit elements using mask
    Vpblendmd,
    /// VPBLENDMQ - Blend 64-bit elements using mask
    Vpblendmq,

    // =========================================
    // Permute Operations (critical for columnar operations)
    // =========================================
    /// VPERMD - Permute 32-bit elements
    Vpermd,
    /// VPERMQ - Permute 64-bit elements
    Vpermq,
    /// VPERMI2D - Permute from two sources (32-bit)
    Vpermi2d,
    /// VPERMI2Q - Permute from two sources (64-bit)
    Vpermi2q,
    /// VPERMT2D - Permute from two sources with table (32-bit)
    Vpermt2d,
    /// VPERMT2Q - Permute from two sources with table (64-bit)
    Vpermt2q,

    // =========================================
    // Conflict Detection (for parallel histograms)
    // =========================================
    /// VPCONFLICTD - Detect 32-bit element conflicts
    Vpconflictd,
    /// VPCONFLICTQ - Detect 64-bit element conflicts
    Vpconflictq,

    // =========================================
    // Leading Zero Count (for find-first, log2)
    // =========================================
    /// VPLZCNTD - Count leading zeros per 32-bit element
    Vplzcntd,
    /// VPLZCNTQ - Count leading zeros per 64-bit element
    Vplzcntq,

    // =========================================
    // Ternary Logic (powerful for complex operations)
    // =========================================
    /// VPTERNLOGD - Ternary logic on 32-bit elements
    Vpternlogd,
    /// VPTERNLOGQ - Ternary logic on 64-bit elements
    Vpternlogq,

    // =========================================
    // Pack/Unpack Operations (type narrowing/widening)
    // =========================================
    /// VPACKSSDW - Pack 32-bit signed to 16-bit signed with saturation
    Vpackssdw,
    /// VPACKUSDW - Pack 32-bit unsigned to 16-bit unsigned with saturation
    Vpackusdw,
    /// VPACKSSWB - Pack 16-bit signed to 8-bit signed with saturation
    Vpacksswb,
    /// VPACKUSWB - Pack 16-bit unsigned to 8-bit unsigned with saturation
    Vpackuswb,

    /// VPUNPCKLBW - Unpack low bytes
    Vpunpcklbw,
    /// VPUNPCKHBW - Unpack high bytes
    Vpunpckhbw,
    /// VPUNPCKLWD - Unpack low words
    Vpunpcklwd,
    /// VPUNPCKHWD - Unpack high words
    Vpunpckhwd,
    /// VPUNPCKLDQ - Unpack low doublewords
    Vpunpckldq,
    /// VPUNPCKHDQ - Unpack high doublewords
    Vpunpckhdq,
    /// VPUNPCKLQDQ - Unpack low quadwords
    Vpunpcklqdq,
    /// VPUNPCKHQDQ - Unpack high quadwords
    Vpunpckhqdq,

    // =========================================
    // Shuffle Operations (data rearrangement)
    // =========================================
    /// VPSHUFB - Shuffle bytes using indices from another vector
    Vpshufb,
    /// VPSHUFD - Shuffle dwords using immediate control
    Vpshufd,
    /// VPSHUFHW - Shuffle high words
    Vpshufhw,
    /// VPSHUFLW - Shuffle low words
    Vpshuflw,

    // =========================================
    // Multiply-Add Operations (dot products)
    // =========================================
    /// VPMADDWD - Multiply-add packed words to dwords
    Vpmaddwd,
    /// VPMADDUBSW - Multiply-add packed unsigned/signed bytes to words
    Vpmaddubsw,
}

/// AVX-512 Floating-Point ALU operations.
///
/// These operations handle single (32-bit) and double (64-bit) precision
/// floating-point arithmetic. F32X16 uses PS instructions, F64X8 uses PD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512FpAluOp {
    // =========================================
    // Floating-Point Arithmetic
    // =========================================
    /// VADDPS - Packed single-precision add
    Vaddps,
    /// VADDPD - Packed double-precision add
    Vaddpd,
    /// VSUBPS - Packed single-precision subtract
    Vsubps,
    /// VSUBPD - Packed double-precision subtract
    Vsubpd,
    /// VMULPS - Packed single-precision multiply
    Vmulps,
    /// VMULPD - Packed double-precision multiply
    Vmulpd,
    /// VDIVPS - Packed single-precision divide
    Vdivps,
    /// VDIVPD - Packed double-precision divide
    Vdivpd,

    // =========================================
    // Floating-Point Min/Max
    // =========================================
    /// VMINPS - Packed single-precision minimum
    Vminps,
    /// VMINPD - Packed double-precision minimum
    Vminpd,
    /// VMAXPS - Packed single-precision maximum
    Vmaxps,
    /// VMAXPD - Packed double-precision maximum
    Vmaxpd,

    // =========================================
    // Floating-Point Square Root
    // =========================================
    /// VSQRTPS - Packed single-precision square root
    Vsqrtps,
    /// VSQRTPD - Packed double-precision square root
    Vsqrtpd,

    // =========================================
    // Floating-Point Absolute/Negate (via bitwise)
    // =========================================
    /// VANDPS - Packed single-precision AND (for abs via mask)
    Vandps,
    /// VANDPD - Packed double-precision AND (for abs via mask)
    Vandpd,
    /// VXORPS - Packed single-precision XOR (for negate)
    Vxorps,
    /// VXORPD - Packed double-precision XOR (for negate)
    Vxorpd,
    /// VORPS - Packed single-precision OR
    Vorps,
    /// VORPD - Packed double-precision OR
    Vorpd,
    /// VANDNPS - Packed single-precision AND NOT
    Vandnps,
    /// VANDNPD - Packed double-precision AND NOT
    Vandnpd,
}

impl Avx512FpAluOp {
    /// Returns the primary opcode byte for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Arithmetic
            Avx512FpAluOp::Vaddps | Avx512FpAluOp::Vaddpd => 0x58,
            Avx512FpAluOp::Vsubps | Avx512FpAluOp::Vsubpd => 0x5C,
            Avx512FpAluOp::Vmulps | Avx512FpAluOp::Vmulpd => 0x59,
            Avx512FpAluOp::Vdivps | Avx512FpAluOp::Vdivpd => 0x5E,

            // Min/Max
            Avx512FpAluOp::Vminps | Avx512FpAluOp::Vminpd => 0x5D,
            Avx512FpAluOp::Vmaxps | Avx512FpAluOp::Vmaxpd => 0x5F,

            // Square root (unary)
            Avx512FpAluOp::Vsqrtps | Avx512FpAluOp::Vsqrtpd => 0x51,

            // Bitwise (for abs/negate)
            Avx512FpAluOp::Vandps | Avx512FpAluOp::Vandpd => 0x54,
            Avx512FpAluOp::Vorps | Avx512FpAluOp::Vorpd => 0x56,
            Avx512FpAluOp::Vxorps | Avx512FpAluOp::Vxorpd => 0x57,
            Avx512FpAluOp::Vandnps | Avx512FpAluOp::Vandnpd => 0x55,
        }
    }

    /// Returns the EVEX opcode map for this operation.
    /// All these FP ops are in the 0F map.
    pub fn evex_map(&self) -> u8 {
        0x01 // 0F map
    }

    /// Returns the EVEX.pp (prefix) field for this operation.
    ///
    /// - Single-precision (PS): no prefix (0x00)
    /// - Double-precision (PD): 66 prefix (0x01)
    pub fn evex_pp(&self) -> u8 {
        match self {
            // Single-precision: no prefix
            Avx512FpAluOp::Vaddps
            | Avx512FpAluOp::Vsubps
            | Avx512FpAluOp::Vmulps
            | Avx512FpAluOp::Vdivps
            | Avx512FpAluOp::Vminps
            | Avx512FpAluOp::Vmaxps
            | Avx512FpAluOp::Vsqrtps
            | Avx512FpAluOp::Vandps
            | Avx512FpAluOp::Vorps
            | Avx512FpAluOp::Vxorps
            | Avx512FpAluOp::Vandnps => 0x00,

            // Double-precision: 66 prefix
            Avx512FpAluOp::Vaddpd
            | Avx512FpAluOp::Vsubpd
            | Avx512FpAluOp::Vmulpd
            | Avx512FpAluOp::Vdivpd
            | Avx512FpAluOp::Vminpd
            | Avx512FpAluOp::Vmaxpd
            | Avx512FpAluOp::Vsqrtpd
            | Avx512FpAluOp::Vandpd
            | Avx512FpAluOp::Vorpd
            | Avx512FpAluOp::Vxorpd
            | Avx512FpAluOp::Vandnpd => 0x01,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    ///
    /// - Single-precision (PS): W=0
    /// - Double-precision (PD): W=1
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512FpAluOp::Vaddps
            | Avx512FpAluOp::Vsubps
            | Avx512FpAluOp::Vmulps
            | Avx512FpAluOp::Vdivps
            | Avx512FpAluOp::Vminps
            | Avx512FpAluOp::Vmaxps
            | Avx512FpAluOp::Vsqrtps
            | Avx512FpAluOp::Vandps
            | Avx512FpAluOp::Vorps
            | Avx512FpAluOp::Vxorps
            | Avx512FpAluOp::Vandnps => false,

            Avx512FpAluOp::Vaddpd
            | Avx512FpAluOp::Vsubpd
            | Avx512FpAluOp::Vmulpd
            | Avx512FpAluOp::Vdivpd
            | Avx512FpAluOp::Vminpd
            | Avx512FpAluOp::Vmaxpd
            | Avx512FpAluOp::Vsqrtpd
            | Avx512FpAluOp::Vandpd
            | Avx512FpAluOp::Vorpd
            | Avx512FpAluOp::Vxorpd
            | Avx512FpAluOp::Vandnpd => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512FpAluOp::Vaddps => "vaddps",
            Avx512FpAluOp::Vaddpd => "vaddpd",
            Avx512FpAluOp::Vsubps => "vsubps",
            Avx512FpAluOp::Vsubpd => "vsubpd",
            Avx512FpAluOp::Vmulps => "vmulps",
            Avx512FpAluOp::Vmulpd => "vmulpd",
            Avx512FpAluOp::Vdivps => "vdivps",
            Avx512FpAluOp::Vdivpd => "vdivpd",
            Avx512FpAluOp::Vminps => "vminps",
            Avx512FpAluOp::Vminpd => "vminpd",
            Avx512FpAluOp::Vmaxps => "vmaxps",
            Avx512FpAluOp::Vmaxpd => "vmaxpd",
            Avx512FpAluOp::Vsqrtps => "vsqrtps",
            Avx512FpAluOp::Vsqrtpd => "vsqrtpd",
            Avx512FpAluOp::Vandps => "vandps",
            Avx512FpAluOp::Vandpd => "vandpd",
            Avx512FpAluOp::Vorps => "vorps",
            Avx512FpAluOp::Vorpd => "vorpd",
            Avx512FpAluOp::Vxorps => "vxorps",
            Avx512FpAluOp::Vxorpd => "vxorpd",
            Avx512FpAluOp::Vandnps => "vandnps",
            Avx512FpAluOp::Vandnpd => "vandnpd",
        }
    }

    /// Returns true if this is a unary operation (single source).
    pub fn is_unary(&self) -> bool {
        matches!(self, Avx512FpAluOp::Vsqrtps | Avx512FpAluOp::Vsqrtpd)
    }

    /// Returns true if this is a double-precision operation.
    pub fn is_double(&self) -> bool {
        self.evex_w()
    }
}

/// AVX-512 Fused Multiply-Add operations.
///
/// FMA operations compute `(a * b) + c` or `(a * b) - c` in a single instruction
/// with only one rounding at the end, providing better precision than separate
/// multiply and add operations.
///
/// The three number variants (132, 213, 231) indicate operand ordering:
/// - 132: result = src1 * src3 ± src2  (src1 is destination, destroyed)
/// - 213: result = src2 * src1 ± src3  (src1 is destination, destroyed)
/// - 231: result = src2 * src3 ± src1  (src1 is destination, destroyed)
///
/// For Cranelift's `fma(x, y, z)` = x * y + z, we use:
/// - VFMADD213: dst/src1=x, src2=y, src3=z → result = y * x + z = x * y + z ✓
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512FmaOp {
    // =========================================
    // VFMADD - (a * b) + c
    // =========================================
    /// VFMADD132PS - Fused multiply-add single-precision (132 form)
    Vfmadd132ps,
    /// VFMADD132PD - Fused multiply-add double-precision (132 form)
    Vfmadd132pd,
    /// VFMADD213PS - Fused multiply-add single-precision (213 form)
    Vfmadd213ps,
    /// VFMADD213PD - Fused multiply-add double-precision (213 form)
    Vfmadd213pd,
    /// VFMADD231PS - Fused multiply-add single-precision (231 form)
    Vfmadd231ps,
    /// VFMADD231PD - Fused multiply-add double-precision (231 form)
    Vfmadd231pd,

    // =========================================
    // VFMSUB - (a * b) - c
    // =========================================
    /// VFMSUB132PS - Fused multiply-subtract single-precision (132 form)
    Vfmsub132ps,
    /// VFMSUB132PD - Fused multiply-subtract double-precision (132 form)
    Vfmsub132pd,
    /// VFMSUB213PS - Fused multiply-subtract single-precision (213 form)
    Vfmsub213ps,
    /// VFMSUB213PD - Fused multiply-subtract double-precision (213 form)
    Vfmsub213pd,
    /// VFMSUB231PS - Fused multiply-subtract single-precision (231 form)
    Vfmsub231ps,
    /// VFMSUB231PD - Fused multiply-subtract double-precision (231 form)
    Vfmsub231pd,

    // =========================================
    // VFNMADD - -(a * b) + c = c - (a * b)
    // =========================================
    /// VFNMADD132PS - Fused negate-multiply-add single-precision (132 form)
    Vfnmadd132ps,
    /// VFNMADD132PD - Fused negate-multiply-add double-precision (132 form)
    Vfnmadd132pd,
    /// VFNMADD213PS - Fused negate-multiply-add single-precision (213 form)
    Vfnmadd213ps,
    /// VFNMADD213PD - Fused negate-multiply-add double-precision (213 form)
    Vfnmadd213pd,
    /// VFNMADD231PS - Fused negate-multiply-add single-precision (231 form)
    Vfnmadd231ps,
    /// VFNMADD231PD - Fused negate-multiply-add double-precision (231 form)
    Vfnmadd231pd,

    // =========================================
    // VFNMSUB - -(a * b) - c
    // =========================================
    /// VFNMSUB132PS - Fused negate-multiply-subtract single-precision (132 form)
    Vfnmsub132ps,
    /// VFNMSUB132PD - Fused negate-multiply-subtract double-precision (132 form)
    Vfnmsub132pd,
    /// VFNMSUB213PS - Fused negate-multiply-subtract single-precision (213 form)
    Vfnmsub213ps,
    /// VFNMSUB213PD - Fused negate-multiply-subtract double-precision (213 form)
    Vfnmsub213pd,
    /// VFNMSUB231PS - Fused negate-multiply-subtract single-precision (231 form)
    Vfnmsub231ps,
    /// VFNMSUB231PD - Fused negate-multiply-subtract double-precision (231 form)
    Vfnmsub231pd,
}

impl Avx512FmaOp {
    /// Returns the primary opcode byte for this operation.
    ///
    /// FMA opcodes follow this pattern in 0F38 map:
    /// - VFMADD:  0x98 (132), 0xA8 (213), 0xB8 (231)
    /// - VFMSUB:  0x9A (132), 0xAA (213), 0xBA (231)
    /// - VFNMADD: 0x9C (132), 0xAC (213), 0xBC (231)
    /// - VFNMSUB: 0x9E (132), 0xAE (213), 0xBE (231)
    pub fn opcode(&self) -> u8 {
        match self {
            // VFMADD
            Avx512FmaOp::Vfmadd132ps | Avx512FmaOp::Vfmadd132pd => 0x98,
            Avx512FmaOp::Vfmadd213ps | Avx512FmaOp::Vfmadd213pd => 0xA8,
            Avx512FmaOp::Vfmadd231ps | Avx512FmaOp::Vfmadd231pd => 0xB8,

            // VFMSUB
            Avx512FmaOp::Vfmsub132ps | Avx512FmaOp::Vfmsub132pd => 0x9A,
            Avx512FmaOp::Vfmsub213ps | Avx512FmaOp::Vfmsub213pd => 0xAA,
            Avx512FmaOp::Vfmsub231ps | Avx512FmaOp::Vfmsub231pd => 0xBA,

            // VFNMADD
            Avx512FmaOp::Vfnmadd132ps | Avx512FmaOp::Vfnmadd132pd => 0x9C,
            Avx512FmaOp::Vfnmadd213ps | Avx512FmaOp::Vfnmadd213pd => 0xAC,
            Avx512FmaOp::Vfnmadd231ps | Avx512FmaOp::Vfnmadd231pd => 0xBC,

            // VFNMSUB
            Avx512FmaOp::Vfnmsub132ps | Avx512FmaOp::Vfnmsub132pd => 0x9E,
            Avx512FmaOp::Vfnmsub213ps | Avx512FmaOp::Vfnmsub213pd => 0xAE,
            Avx512FmaOp::Vfnmsub231ps | Avx512FmaOp::Vfnmsub231pd => 0xBE,
        }
    }

    /// Returns the EVEX opcode map (always 0F38 for FMA).
    pub fn evex_map(&self) -> u8 {
        0x02 // 0F38 map
    }

    /// Returns the EVEX.pp (prefix) field.
    /// All FMA ops use 66 prefix.
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit.
    /// - Single-precision (PS): W=0
    /// - Double-precision (PD): W=1
    pub fn evex_w(&self) -> bool {
        match self {
            // Single-precision: W=0
            Avx512FmaOp::Vfmadd132ps
            | Avx512FmaOp::Vfmadd213ps
            | Avx512FmaOp::Vfmadd231ps
            | Avx512FmaOp::Vfmsub132ps
            | Avx512FmaOp::Vfmsub213ps
            | Avx512FmaOp::Vfmsub231ps
            | Avx512FmaOp::Vfnmadd132ps
            | Avx512FmaOp::Vfnmadd213ps
            | Avx512FmaOp::Vfnmadd231ps
            | Avx512FmaOp::Vfnmsub132ps
            | Avx512FmaOp::Vfnmsub213ps
            | Avx512FmaOp::Vfnmsub231ps => false,

            // Double-precision: W=1
            Avx512FmaOp::Vfmadd132pd
            | Avx512FmaOp::Vfmadd213pd
            | Avx512FmaOp::Vfmadd231pd
            | Avx512FmaOp::Vfmsub132pd
            | Avx512FmaOp::Vfmsub213pd
            | Avx512FmaOp::Vfmsub231pd
            | Avx512FmaOp::Vfnmadd132pd
            | Avx512FmaOp::Vfnmadd213pd
            | Avx512FmaOp::Vfnmadd231pd
            | Avx512FmaOp::Vfnmsub132pd
            | Avx512FmaOp::Vfnmsub213pd
            | Avx512FmaOp::Vfnmsub231pd => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512FmaOp::Vfmadd132ps => "vfmadd132ps",
            Avx512FmaOp::Vfmadd132pd => "vfmadd132pd",
            Avx512FmaOp::Vfmadd213ps => "vfmadd213ps",
            Avx512FmaOp::Vfmadd213pd => "vfmadd213pd",
            Avx512FmaOp::Vfmadd231ps => "vfmadd231ps",
            Avx512FmaOp::Vfmadd231pd => "vfmadd231pd",
            Avx512FmaOp::Vfmsub132ps => "vfmsub132ps",
            Avx512FmaOp::Vfmsub132pd => "vfmsub132pd",
            Avx512FmaOp::Vfmsub213ps => "vfmsub213ps",
            Avx512FmaOp::Vfmsub213pd => "vfmsub213pd",
            Avx512FmaOp::Vfmsub231ps => "vfmsub231ps",
            Avx512FmaOp::Vfmsub231pd => "vfmsub231pd",
            Avx512FmaOp::Vfnmadd132ps => "vfnmadd132ps",
            Avx512FmaOp::Vfnmadd132pd => "vfnmadd132pd",
            Avx512FmaOp::Vfnmadd213ps => "vfnmadd213ps",
            Avx512FmaOp::Vfnmadd213pd => "vfnmadd213pd",
            Avx512FmaOp::Vfnmadd231ps => "vfnmadd231ps",
            Avx512FmaOp::Vfnmadd231pd => "vfnmadd231pd",
            Avx512FmaOp::Vfnmsub132ps => "vfnmsub132ps",
            Avx512FmaOp::Vfnmsub132pd => "vfnmsub132pd",
            Avx512FmaOp::Vfnmsub213ps => "vfnmsub213ps",
            Avx512FmaOp::Vfnmsub213pd => "vfnmsub213pd",
            Avx512FmaOp::Vfnmsub231ps => "vfnmsub231ps",
            Avx512FmaOp::Vfnmsub231pd => "vfnmsub231pd",
        }
    }

    /// Returns true if this is a double-precision operation.
    pub fn is_double(&self) -> bool {
        self.evex_w()
    }
}

/// AVX-512 VNNI (Vector Neural Network Instructions) operations.
///
/// These are accumulating dot-product operations for ML/analytics workloads.
/// All operations accumulate: dst = dst + operation(src1, src2).
/// The destination register is tied to both input and output.
///
/// These instructions provide highly efficient dot product computations for:
/// - Neural network inference (INT8/INT16 quantized models)
/// - Similarity computations
/// - Analytics aggregations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512VnniOp {
    /// VPDPBUSD - Multiply unsigned bytes by signed bytes, add adjacent pairs to dwords
    /// dst[i] = dst[i] + (src1[4i]*src2[4i] + src1[4i+1]*src2[4i+1] + src1[4i+2]*src2[4i+2] + src1[4i+3]*src2[4i+3])
    Vpdpbusd,
    /// VPDPBUSDS - Same as VPDPBUSD but with signed saturation
    Vpdpbusds,
    /// VPDPWSSD - Multiply signed words, add adjacent pairs to dwords
    /// dst[i] = dst[i] + (src1[2i]*src2[2i] + src1[2i+1]*src2[2i+1])
    Vpdpwssd,
    /// VPDPWSSDS - Same as VPDPWSSD but with signed saturation
    Vpdpwssds,
}

impl Avx512VnniOp {
    /// Returns the primary opcode byte for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            Avx512VnniOp::Vpdpbusd => 0x50,
            Avx512VnniOp::Vpdpbusds => 0x51,
            Avx512VnniOp::Vpdpwssd => 0x52,
            Avx512VnniOp::Vpdpwssds => 0x53,
        }
    }

    /// Returns the EVEX opcode map (always 0F38 for VNNI).
    pub fn evex_map(&self) -> u8 {
        0x02 // 0F38 map
    }

    /// Returns the EVEX.pp (prefix) encoding (always 66 for VNNI).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit (always 0 for VNNI - dword operations).
    pub fn evex_w(&self) -> bool {
        false
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512VnniOp::Vpdpbusd => "vpdpbusd",
            Avx512VnniOp::Vpdpbusds => "vpdpbusds",
            Avx512VnniOp::Vpdpwssd => "vpdpwssd",
            Avx512VnniOp::Vpdpwssds => "vpdpwssds",
        }
    }

    /// Returns true if this is a saturating operation.
    pub fn is_saturating(&self) -> bool {
        matches!(self, Avx512VnniOp::Vpdpbusds | Avx512VnniOp::Vpdpwssds)
    }
}

impl Avx512AluOp {
    /// Returns the primary opcode byte for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Integer arithmetic
            Avx512AluOp::Vpaddd => 0xFE,
            Avx512AluOp::Vpaddq => 0xD4,
            Avx512AluOp::Vpsubd => 0xFA,
            Avx512AluOp::Vpsubq => 0xFB,
            Avx512AluOp::Vpmulld => 0x40,
            Avx512AluOp::Vpmullq => 0x40,
            Avx512AluOp::Vpmuludq => 0xF4, // 0F map
            Avx512AluOp::Vpmuldq => 0x28,  // 0F38 map

            // Byte/word arithmetic (AVX-512BW)
            Avx512AluOp::Vpaddb => 0xFC,
            Avx512AluOp::Vpaddw => 0xFD,
            Avx512AluOp::Vpsubb => 0xF8,
            Avx512AluOp::Vpsubw => 0xF9,
            Avx512AluOp::Vpminsb => 0x38, // 0F38 map
            Avx512AluOp::Vpminub => 0xDA,
            Avx512AluOp::Vpminsw => 0xEA,
            Avx512AluOp::Vpminuw => 0x3A, // 0F38 map
            Avx512AluOp::Vpmaxsb => 0x3C, // 0F38 map
            Avx512AluOp::Vpmaxub => 0xDE,
            Avx512AluOp::Vpmaxsw => 0xEE,
            Avx512AluOp::Vpmaxuw => 0x3E, // 0F38 map
            Avx512AluOp::Vpabsb => 0x1C,  // 0F38 map
            Avx512AluOp::Vpabsw => 0x1D,  // 0F38 map
            Avx512AluOp::Vpcmpeqb => 0x74,
            Avx512AluOp::Vpcmpeqw => 0x75,
            Avx512AluOp::Vpcmpgtb => 0x64,
            Avx512AluOp::Vpcmpgtw => 0x65,

            // Bitwise logical
            Avx512AluOp::Vpandd => 0xDB,
            Avx512AluOp::Vpandq => 0xDB,
            Avx512AluOp::Vpord => 0xEB,
            Avx512AluOp::Vporq => 0xEB,
            Avx512AluOp::Vpxord => 0xEF,
            Avx512AluOp::Vpxorq => 0xEF,
            Avx512AluOp::Vpandnd => 0xDF,
            Avx512AluOp::Vpandnq => 0xDF,

            // Shifts (uniform)
            Avx512AluOp::Vpslld => 0xF2,
            Avx512AluOp::Vpsllq => 0xF3,
            Avx512AluOp::Vpsrld => 0xD2,
            Avx512AluOp::Vpsrlq => 0xD3,
            Avx512AluOp::Vpsrad => 0xE2,
            Avx512AluOp::Vpsraq => 0xE2, // Same opcode, different W bit

            // Shifts (variable)
            Avx512AluOp::Vpsllvd => 0x47,
            Avx512AluOp::Vpsllvq => 0x47,
            Avx512AluOp::Vpsrlvd => 0x45,
            Avx512AluOp::Vpsrlvq => 0x45,
            Avx512AluOp::Vpsravd => 0x46,
            Avx512AluOp::Vpsravq => 0x46,

            // Rotates (variable)
            Avx512AluOp::Vprolvd => 0x15,
            Avx512AluOp::Vprolvq => 0x15,

            // Min/Max
            Avx512AluOp::Vpminsd => 0x39,
            Avx512AluOp::Vpminsq => 0x39,
            Avx512AluOp::Vpmaxsd => 0x3D,
            Avx512AluOp::Vpmaxsq => 0x3D,
            Avx512AluOp::Vpminud => 0x3B,
            Avx512AluOp::Vpminuq => 0x3B,
            Avx512AluOp::Vpmaxud => 0x3F,
            Avx512AluOp::Vpmaxuq => 0x3F,

            // Absolute value
            Avx512AluOp::Vpabsd => 0x1E,
            Avx512AluOp::Vpabsq => 0x1F,

            // Population count (AVX-512 VPOPCNTDQ)
            Avx512AluOp::Vpopcntd => 0x55,
            Avx512AluOp::Vpopcntq => 0x55,

            // Broadcast
            Avx512AluOp::Vpbroadcastb => 0x78,
            Avx512AluOp::Vpbroadcastw => 0x79,
            Avx512AluOp::Vpbroadcastd => 0x58,
            Avx512AluOp::Vpbroadcastq => 0x59,

            // Blend
            Avx512AluOp::Vpblendmd => 0x64,
            Avx512AluOp::Vpblendmq => 0x64,

            // Permute
            Avx512AluOp::Vpermd => 0x36,
            Avx512AluOp::Vpermq => 0x36,
            Avx512AluOp::Vpermi2d => 0x76,
            Avx512AluOp::Vpermi2q => 0x76,
            Avx512AluOp::Vpermt2d => 0x7E,
            Avx512AluOp::Vpermt2q => 0x7E,

            // Conflict detection
            Avx512AluOp::Vpconflictd => 0xC4,
            Avx512AluOp::Vpconflictq => 0xC4,

            // Leading zero count
            Avx512AluOp::Vplzcntd => 0x44,
            Avx512AluOp::Vplzcntq => 0x44,

            // Ternary logic
            Avx512AluOp::Vpternlogd => 0x25,
            Avx512AluOp::Vpternlogq => 0x25,

            // Pack operations
            Avx512AluOp::Vpackssdw => 0x6B,
            Avx512AluOp::Vpackusdw => 0x2B, // 0F38 map
            Avx512AluOp::Vpacksswb => 0x63,
            Avx512AluOp::Vpackuswb => 0x67,

            // Unpack operations
            Avx512AluOp::Vpunpcklbw => 0x60,
            Avx512AluOp::Vpunpckhbw => 0x68,
            Avx512AluOp::Vpunpcklwd => 0x61,
            Avx512AluOp::Vpunpckhwd => 0x69,
            Avx512AluOp::Vpunpckldq => 0x62,
            Avx512AluOp::Vpunpckhdq => 0x6A,
            Avx512AluOp::Vpunpcklqdq => 0x6C,
            Avx512AluOp::Vpunpckhqdq => 0x6D,

            // Shuffle operations
            Avx512AluOp::Vpshufb => 0x00,  // 0F38 map
            Avx512AluOp::Vpshufd => 0x70,  // 0F map
            Avx512AluOp::Vpshufhw => 0x70, // 0F map
            Avx512AluOp::Vpshuflw => 0x70, // 0F map

            // Multiply-add operations
            Avx512AluOp::Vpmaddwd => 0xF5,   // 0F map
            Avx512AluOp::Vpmaddubsw => 0x04, // 0F38 map
        }
    }

    /// Returns the EVEX opcode map for this operation.
    ///
    /// - 0x01 = 0F map
    /// - 0x02 = 0F38 map
    /// - 0x03 = 0F3A map
    pub fn evex_map(&self) -> u8 {
        match self {
            // 0F map (simple arithmetic/logical)
            Avx512AluOp::Vpaddd
            | Avx512AluOp::Vpaddq
            | Avx512AluOp::Vpsubd
            | Avx512AluOp::Vpsubq
            | Avx512AluOp::Vpmuludq  // 32x32->64 unsigned multiply
            | Avx512AluOp::Vpandd
            | Avx512AluOp::Vpandq
            | Avx512AluOp::Vpord
            | Avx512AluOp::Vporq
            | Avx512AluOp::Vpxord
            | Avx512AluOp::Vpxorq
            | Avx512AluOp::Vpandnd
            | Avx512AluOp::Vpandnq
            | Avx512AluOp::Vpslld
            | Avx512AluOp::Vpsllq
            | Avx512AluOp::Vpsrld
            | Avx512AluOp::Vpsrlq
            | Avx512AluOp::Vpsrad
            | Avx512AluOp::Vpsraq
            // Pack operations (except VPACKUSDW which is 0F38)
            | Avx512AluOp::Vpackssdw
            | Avx512AluOp::Vpacksswb
            | Avx512AluOp::Vpackuswb
            // Unpack operations
            | Avx512AluOp::Vpunpcklbw
            | Avx512AluOp::Vpunpckhbw
            | Avx512AluOp::Vpunpcklwd
            | Avx512AluOp::Vpunpckhwd
            | Avx512AluOp::Vpunpckldq
            | Avx512AluOp::Vpunpckhdq
            | Avx512AluOp::Vpunpcklqdq
            | Avx512AluOp::Vpunpckhqdq
            // Shuffle operations (0F map)
            | Avx512AluOp::Vpshufd
            | Avx512AluOp::Vpshufhw
            | Avx512AluOp::Vpshuflw
            // Multiply-add (0F map)
            | Avx512AluOp::Vpmaddwd
            // Byte/word ops (0F map)
            | Avx512AluOp::Vpaddb
            | Avx512AluOp::Vpaddw
            | Avx512AluOp::Vpsubb
            | Avx512AluOp::Vpsubw
            | Avx512AluOp::Vpminub
            | Avx512AluOp::Vpminsw
            | Avx512AluOp::Vpmaxub
            | Avx512AluOp::Vpmaxsw
            | Avx512AluOp::Vpcmpeqb
            | Avx512AluOp::Vpcmpeqw
            | Avx512AluOp::Vpcmpgtb
            | Avx512AluOp::Vpcmpgtw => 0x01,

            // 0F38 map (more complex operations)
            Avx512AluOp::Vpmulld
            | Avx512AluOp::Vpshufb
            | Avx512AluOp::Vpmaddubsw
            | Avx512AluOp::Vpmullq
            | Avx512AluOp::Vpmuldq  // 32x32->64 signed multiply
            | Avx512AluOp::Vpsllvd
            | Avx512AluOp::Vpsllvq
            | Avx512AluOp::Vpsrlvd
            | Avx512AluOp::Vpsrlvq
            | Avx512AluOp::Vpsravd
            | Avx512AluOp::Vpsravq
            | Avx512AluOp::Vprolvd
            | Avx512AluOp::Vprolvq
            | Avx512AluOp::Vpminsd
            | Avx512AluOp::Vpminsq
            | Avx512AluOp::Vpmaxsd
            | Avx512AluOp::Vpmaxsq
            | Avx512AluOp::Vpminud
            | Avx512AluOp::Vpminuq
            | Avx512AluOp::Vpmaxud
            | Avx512AluOp::Vpmaxuq
            | Avx512AluOp::Vpabsd
            | Avx512AluOp::Vpabsq
            | Avx512AluOp::Vpopcntd
            | Avx512AluOp::Vpopcntq
            | Avx512AluOp::Vpbroadcastb
            | Avx512AluOp::Vpbroadcastw
            | Avx512AluOp::Vpbroadcastd
            | Avx512AluOp::Vpbroadcastq
            | Avx512AluOp::Vpblendmd
            | Avx512AluOp::Vpblendmq
            | Avx512AluOp::Vpermd
            | Avx512AluOp::Vpermq
            | Avx512AluOp::Vpermi2d
            | Avx512AluOp::Vpermi2q
            | Avx512AluOp::Vpermt2d
            | Avx512AluOp::Vpermt2q
            | Avx512AluOp::Vpconflictd
            | Avx512AluOp::Vpconflictq
            | Avx512AluOp::Vplzcntd
            | Avx512AluOp::Vplzcntq
            | Avx512AluOp::Vpackusdw
            // Byte/word ops (0F38 map)
            | Avx512AluOp::Vpminsb
            | Avx512AluOp::Vpminuw
            | Avx512AluOp::Vpmaxsb
            | Avx512AluOp::Vpmaxuw
            | Avx512AluOp::Vpabsb
            | Avx512AluOp::Vpabsw => 0x02,

            // 0F3A map (ternary logic)
            Avx512AluOp::Vpternlogd | Avx512AluOp::Vpternlogq => 0x03,
        }
    }

    /// Returns the EVEX.pp (prefix) field for this operation.
    ///
    /// - 0x00 = no prefix
    /// - 0x01 = 66 prefix
    /// - 0x02 = F3 prefix
    /// - 0x03 = F2 prefix
    pub fn evex_pp(&self) -> u8 {
        // All our integer operations use 66 prefix
        0x01
    }

    /// Returns the EVEX.W bit for this operation.
    ///
    /// - false (0) = 32-bit operation
    /// - true (1) = 64-bit operation
    pub fn evex_w(&self) -> bool {
        match self {
            // 64-bit element operations
            Avx512AluOp::Vpaddq
            | Avx512AluOp::Vpsubq
            | Avx512AluOp::Vpmullq
            | Avx512AluOp::Vpmuludq  // 32x32->64, uses W=1 for 64-bit result
            | Avx512AluOp::Vpmuldq   // 32x32->64, uses W=1 for 64-bit result
            | Avx512AluOp::Vpandq
            | Avx512AluOp::Vporq
            | Avx512AluOp::Vpxorq
            | Avx512AluOp::Vpandnq
            | Avx512AluOp::Vpsllq
            | Avx512AluOp::Vpsrlq
            | Avx512AluOp::Vpsraq
            | Avx512AluOp::Vpsllvq
            | Avx512AluOp::Vpsrlvq
            | Avx512AluOp::Vpsravq
            | Avx512AluOp::Vprolvq
            | Avx512AluOp::Vpminsq
            | Avx512AluOp::Vpmaxsq
            | Avx512AluOp::Vpminuq
            | Avx512AluOp::Vpmaxuq
            | Avx512AluOp::Vpabsq
            | Avx512AluOp::Vpopcntq
            | Avx512AluOp::Vpbroadcastq
            | Avx512AluOp::Vpblendmq
            | Avx512AluOp::Vpermq
            | Avx512AluOp::Vpermi2q
            | Avx512AluOp::Vpermt2q
            | Avx512AluOp::Vpconflictq
            | Avx512AluOp::Vplzcntq
            | Avx512AluOp::Vpternlogq => true,

            // 32-bit element operations
            _ => false,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512AluOp::Vpaddd => "vpaddd",
            Avx512AluOp::Vpaddq => "vpaddq",
            Avx512AluOp::Vpsubd => "vpsubd",
            Avx512AluOp::Vpsubq => "vpsubq",
            Avx512AluOp::Vpmulld => "vpmulld",
            Avx512AluOp::Vpmullq => "vpmullq",
            Avx512AluOp::Vpmuludq => "vpmuludq",
            Avx512AluOp::Vpmuldq => "vpmuldq",
            // Byte/word arithmetic
            Avx512AluOp::Vpaddb => "vpaddb",
            Avx512AluOp::Vpaddw => "vpaddw",
            Avx512AluOp::Vpsubb => "vpsubb",
            Avx512AluOp::Vpsubw => "vpsubw",
            Avx512AluOp::Vpminsb => "vpminsb",
            Avx512AluOp::Vpminub => "vpminub",
            Avx512AluOp::Vpminsw => "vpminsw",
            Avx512AluOp::Vpminuw => "vpminuw",
            Avx512AluOp::Vpmaxsb => "vpmaxsb",
            Avx512AluOp::Vpmaxub => "vpmaxub",
            Avx512AluOp::Vpmaxsw => "vpmaxsw",
            Avx512AluOp::Vpmaxuw => "vpmaxuw",
            Avx512AluOp::Vpabsb => "vpabsb",
            Avx512AluOp::Vpabsw => "vpabsw",
            Avx512AluOp::Vpcmpeqb => "vpcmpeqb",
            Avx512AluOp::Vpcmpeqw => "vpcmpeqw",
            Avx512AluOp::Vpcmpgtb => "vpcmpgtb",
            Avx512AluOp::Vpcmpgtw => "vpcmpgtw",
            Avx512AluOp::Vpandd => "vpandd",
            Avx512AluOp::Vpandq => "vpandq",
            Avx512AluOp::Vpord => "vpord",
            Avx512AluOp::Vporq => "vporq",
            Avx512AluOp::Vpxord => "vpxord",
            Avx512AluOp::Vpxorq => "vpxorq",
            Avx512AluOp::Vpandnd => "vpandnd",
            Avx512AluOp::Vpandnq => "vpandnq",
            Avx512AluOp::Vpslld => "vpslld",
            Avx512AluOp::Vpsllq => "vpsllq",
            Avx512AluOp::Vpsrld => "vpsrld",
            Avx512AluOp::Vpsrlq => "vpsrlq",
            Avx512AluOp::Vpsrad => "vpsrad",
            Avx512AluOp::Vpsraq => "vpsraq",
            Avx512AluOp::Vpsllvd => "vpsllvd",
            Avx512AluOp::Vpsllvq => "vpsllvq",
            Avx512AluOp::Vpsrlvd => "vpsrlvd",
            Avx512AluOp::Vpsrlvq => "vpsrlvq",
            Avx512AluOp::Vpsravd => "vpsravd",
            Avx512AluOp::Vpsravq => "vpsravq",
            Avx512AluOp::Vprolvd => "vprolvd",
            Avx512AluOp::Vprolvq => "vprolvq",
            Avx512AluOp::Vpminsd => "vpminsd",
            Avx512AluOp::Vpminsq => "vpminsq",
            Avx512AluOp::Vpmaxsd => "vpmaxsd",
            Avx512AluOp::Vpmaxsq => "vpmaxsq",
            Avx512AluOp::Vpminud => "vpminud",
            Avx512AluOp::Vpminuq => "vpminuq",
            Avx512AluOp::Vpmaxud => "vpmaxud",
            Avx512AluOp::Vpmaxuq => "vpmaxuq",
            Avx512AluOp::Vpabsd => "vpabsd",
            Avx512AluOp::Vpabsq => "vpabsq",
            Avx512AluOp::Vpopcntd => "vpopcntd",
            Avx512AluOp::Vpopcntq => "vpopcntq",
            Avx512AluOp::Vpbroadcastb => "vpbroadcastb",
            Avx512AluOp::Vpbroadcastw => "vpbroadcastw",
            Avx512AluOp::Vpbroadcastd => "vpbroadcastd",
            Avx512AluOp::Vpbroadcastq => "vpbroadcastq",
            Avx512AluOp::Vpblendmd => "vpblendmd",
            Avx512AluOp::Vpblendmq => "vpblendmq",
            Avx512AluOp::Vpermd => "vpermd",
            Avx512AluOp::Vpermq => "vpermq",
            Avx512AluOp::Vpermi2d => "vpermi2d",
            Avx512AluOp::Vpermi2q => "vpermi2q",
            Avx512AluOp::Vpermt2d => "vpermt2d",
            Avx512AluOp::Vpermt2q => "vpermt2q",
            Avx512AluOp::Vpconflictd => "vpconflictd",
            Avx512AluOp::Vpconflictq => "vpconflictq",
            Avx512AluOp::Vplzcntd => "vplzcntd",
            Avx512AluOp::Vplzcntq => "vplzcntq",
            Avx512AluOp::Vpternlogd => "vpternlogd",
            Avx512AluOp::Vpternlogq => "vpternlogq",
            // Pack operations
            Avx512AluOp::Vpackssdw => "vpackssdw",
            Avx512AluOp::Vpackusdw => "vpackusdw",
            Avx512AluOp::Vpacksswb => "vpacksswb",
            Avx512AluOp::Vpackuswb => "vpackuswb",
            // Unpack operations
            Avx512AluOp::Vpunpcklbw => "vpunpcklbw",
            Avx512AluOp::Vpunpckhbw => "vpunpckhbw",
            Avx512AluOp::Vpunpcklwd => "vpunpcklwd",
            Avx512AluOp::Vpunpckhwd => "vpunpckhwd",
            Avx512AluOp::Vpunpckldq => "vpunpckldq",
            Avx512AluOp::Vpunpckhdq => "vpunpckhdq",
            Avx512AluOp::Vpunpcklqdq => "vpunpcklqdq",
            Avx512AluOp::Vpunpckhqdq => "vpunpckhqdq",
            // Shuffle operations
            Avx512AluOp::Vpshufb => "vpshufb",
            Avx512AluOp::Vpshufd => "vpshufd",
            Avx512AluOp::Vpshufhw => "vpshufhw",
            Avx512AluOp::Vpshuflw => "vpshuflw",
            // Multiply-add operations
            Avx512AluOp::Vpmaddwd => "vpmaddwd",
            Avx512AluOp::Vpmaddubsw => "vpmaddubsw",
        }
    }

    /// Returns true if this is a 64-bit element operation.
    pub fn is_64bit(&self) -> bool {
        self.evex_w()
    }

    /// Returns true if this operation requires a third operand (ternary logic).
    pub fn is_ternary(&self) -> bool {
        matches!(self, Avx512AluOp::Vpternlogd | Avx512AluOp::Vpternlogq)
    }

    /// Returns true if this is a unary operation (single source).
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Avx512AluOp::Vpabsd
                | Avx512AluOp::Vpabsq
                | Avx512AluOp::Vpopcntd
                | Avx512AluOp::Vpopcntq
                | Avx512AluOp::Vpbroadcastb
                | Avx512AluOp::Vpbroadcastw
                | Avx512AluOp::Vpbroadcastd
                | Avx512AluOp::Vpbroadcastq
                | Avx512AluOp::Vpconflictd
                | Avx512AluOp::Vpconflictq
                | Avx512AluOp::Vplzcntd
                | Avx512AluOp::Vplzcntq
        )
    }
}

impl MaskAluOp {
    /// Returns true if this is a unary operation (KNOT).
    pub fn is_unary(&self) -> bool {
        matches!(self, MaskAluOp::Knot)
    }

    /// Returns the VEX opcode for this mask operation.
    pub fn vex_opcode(&self) -> u8 {
        match self {
            MaskAluOp::Kand => 0x41,
            MaskAluOp::Kor => 0x45,
            MaskAluOp::Kxor => 0x47,
            MaskAluOp::Kxnor => 0x46, // KXNORW is VEX.L0.0F.W0 46 /r
            MaskAluOp::Knot => 0x44,
            MaskAluOp::Kandn => 0x42,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskAluOp::Kand => "kandw",
            MaskAluOp::Kor => "korw",
            MaskAluOp::Kxor => "kxorw",
            MaskAluOp::Kxnor => "kxnorw",
            MaskAluOp::Knot => "knotw",
            MaskAluOp::Kandn => "kandnw",
        }
    }
}

/// K-register shift operations.
///
/// These shift the bits in a mask register by an immediate count.
/// Shift count is modulo the register width (8/16/32/64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskShiftOp {
    /// KSHIFTLB - Shift k-register left by immediate (8-bit register)
    Kshiftlb,
    /// KSHIFTLW - Shift k-register left by immediate (16-bit register)
    Kshiftlw,
    /// KSHIFTLD - Shift k-register left by immediate (32-bit register)
    Kshiftld,
    /// KSHIFTLQ - Shift k-register left by immediate (64-bit register)
    Kshiftlq,
    /// KSHIFTRB - Shift k-register right by immediate (8-bit register)
    Kshiftrb,
    /// KSHIFTRW - Shift k-register right by immediate (16-bit register)
    Kshiftrw,
    /// KSHIFTRD - Shift k-register right by immediate (32-bit register)
    Kshiftrd,
    /// KSHIFTRQ - Shift k-register right by immediate (64-bit register)
    Kshiftrq,
}

impl MaskShiftOp {
    /// Returns the VEX opcode for this shift operation.
    /// KSHIFTL* uses 0x32/0x33, KSHIFTR* uses 0x30/0x31
    pub fn vex_opcode(&self) -> u8 {
        match self {
            // KSHIFTL: 0x32 for byte/word, 0x33 for dword/qword
            MaskShiftOp::Kshiftlb | MaskShiftOp::Kshiftlw => 0x32,
            MaskShiftOp::Kshiftld | MaskShiftOp::Kshiftlq => 0x33,
            // KSHIFTR: 0x30 for byte/word, 0x31 for dword/qword
            MaskShiftOp::Kshiftrb | MaskShiftOp::Kshiftrw => 0x30,
            MaskShiftOp::Kshiftrd | MaskShiftOp::Kshiftrq => 0x31,
        }
    }

    /// Returns the VEX.W bit for this operation.
    /// W=0 for byte, W=1 for word, W=0 for dword, W=1 for qword
    pub fn vex_w(&self) -> bool {
        match self {
            MaskShiftOp::Kshiftlb
            | MaskShiftOp::Kshiftld
            | MaskShiftOp::Kshiftrb
            | MaskShiftOp::Kshiftrd => false,
            MaskShiftOp::Kshiftlw
            | MaskShiftOp::Kshiftlq
            | MaskShiftOp::Kshiftrw
            | MaskShiftOp::Kshiftrq => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskShiftOp::Kshiftlb => "kshiftlb",
            MaskShiftOp::Kshiftlw => "kshiftlw",
            MaskShiftOp::Kshiftld => "kshiftld",
            MaskShiftOp::Kshiftlq => "kshiftlq",
            MaskShiftOp::Kshiftrb => "kshiftrb",
            MaskShiftOp::Kshiftrw => "kshiftrw",
            MaskShiftOp::Kshiftrd => "kshiftrd",
            MaskShiftOp::Kshiftrq => "kshiftrq",
        }
    }
}

/// K-register unpack operations.
///
/// These unpack and interleave low halves of two mask registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskUnpackOp {
    /// KUNPCKBW - Unpack 8-bit to 16-bit (low bytes of two 8-bit regs -> 16-bit)
    Kunpckbw,
    /// KUNPCKWD - Unpack 16-bit to 32-bit (low words of two 16-bit regs -> 32-bit)
    Kunpckwd,
    /// KUNPCKDQ - Unpack 32-bit to 64-bit (low dwords of two 32-bit regs -> 64-bit)
    Kunpckdq,
}

impl MaskUnpackOp {
    /// Returns the VEX opcode for this unpack operation.
    pub fn vex_opcode(&self) -> u8 {
        // KUNPCK* all use opcode 0x4B in 0F map
        0x4B
    }

    /// Returns the VEX.W bit for this operation.
    pub fn vex_w(&self) -> bool {
        match self {
            MaskUnpackOp::Kunpckbw => false, // W=0
            MaskUnpackOp::Kunpckwd => false, // W=0 (L=1 distinguishes from BW)
            MaskUnpackOp::Kunpckdq => true,  // W=1
        }
    }

    /// Returns the VEX.L bit for this operation.
    pub fn vex_l(&self) -> bool {
        match self {
            MaskUnpackOp::Kunpckbw => false, // L=0
            MaskUnpackOp::Kunpckwd => true,  // L=1
            MaskUnpackOp::Kunpckdq => true,  // L=1
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskUnpackOp::Kunpckbw => "kunpckbw",
            MaskUnpackOp::Kunpckwd => "kunpckwd",
            MaskUnpackOp::Kunpckdq => "kunpckdq",
        }
    }
}

/// K-register add operations.
///
/// These add two mask registers element-wise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAddOp {
    /// KADDB - Add 8-bit mask registers
    Kaddb,
    /// KADDW - Add 16-bit mask registers
    Kaddw,
    /// KADDD - Add 32-bit mask registers
    Kaddd,
    /// KADDQ - Add 64-bit mask registers
    Kaddq,
}

impl MaskAddOp {
    /// Returns the VEX opcode for this add operation.
    pub fn vex_opcode(&self) -> u8 {
        // KADD* all use opcode 0x4A in 0F map
        0x4A
    }

    /// Returns the VEX.W bit for this operation.
    pub fn vex_w(&self) -> bool {
        match self {
            MaskAddOp::Kaddb | MaskAddOp::Kaddd => false,
            MaskAddOp::Kaddw | MaskAddOp::Kaddq => true,
        }
    }

    /// Returns the VEX.L bit for this operation.
    pub fn vex_l(&self) -> bool {
        match self {
            MaskAddOp::Kaddb | MaskAddOp::Kaddw => false, // L=0
            MaskAddOp::Kaddd | MaskAddOp::Kaddq => true,  // L=1
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskAddOp::Kaddb => "kaddb",
            MaskAddOp::Kaddw => "kaddw",
            MaskAddOp::Kaddd => "kaddd",
            MaskAddOp::Kaddq => "kaddq",
        }
    }
}

/// K-register test operations.
///
/// These test two mask registers and set CPU flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskTestOp {
    /// KTESTB - Test 8-bit mask registers
    Ktestb,
    /// KTESTW - Test 16-bit mask registers
    Ktestw,
    /// KTESTD - Test 32-bit mask registers
    Ktestd,
    /// KTESTQ - Test 64-bit mask registers
    Ktestq,
}

impl MaskTestOp {
    /// Returns the VEX opcode for this test operation.
    pub fn vex_opcode(&self) -> u8 {
        // KTEST* all use opcode 0x99 in 0F map
        0x99
    }

    /// Returns the VEX.W bit for this operation.
    pub fn vex_w(&self) -> bool {
        match self {
            MaskTestOp::Ktestb | MaskTestOp::Ktestd => false,
            MaskTestOp::Ktestw | MaskTestOp::Ktestq => true,
        }
    }

    /// Returns the VEX.L bit for this operation.
    pub fn vex_l(&self) -> bool {
        match self {
            MaskTestOp::Ktestb | MaskTestOp::Ktestw => false, // L=0
            MaskTestOp::Ktestd | MaskTestOp::Ktestq => true,  // L=1
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskTestOp::Ktestb => "ktestb",
            MaskTestOp::Ktestw => "ktestw",
            MaskTestOp::Ktestd => "ktestd",
            MaskTestOp::Ktestq => "ktestq",
        }
    }
}

impl Avx512Cond {
    /// Returns a human-readable name for this condition.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512Cond::Eq => "eq",
            Avx512Cond::Lt => "lt",
            Avx512Cond::Le => "le",
            Avx512Cond::Neq => "neq",
            Avx512Cond::Ge => "ge",
            Avx512Cond::Gt => "gt",
        }
    }

    /// Returns the immediate byte encoding for this condition.
    pub fn imm(&self) -> u8 {
        *self as u8
    }
}

/// AVX-512 Gather operations.
///
/// Gather loads non-contiguous memory elements into a vector register
/// using index values from another vector register. Essential for
/// columnar database operations where row indices are used to access
/// scattered column data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherOp {
    /// VPGATHERDD - Gather 32-bit elements using 32-bit indices
    /// zmm1 {k1}, [base + zmm_index*scale]
    Vpgatherdd,
    /// VPGATHERDQ - Gather 64-bit elements using 32-bit indices
    /// zmm1 {k1}, [base + ymm_index*scale]
    Vpgatherdq,
    /// VPGATHERQD - Gather 32-bit elements using 64-bit indices
    /// ymm1 {k1}, [base + zmm_index*scale]
    Vpgatherqd,
    /// VPGATHERQQ - Gather 64-bit elements using 64-bit indices
    /// zmm1 {k1}, [base + zmm_index*scale]
    Vpgatherqq,
}

impl GatherOp {
    /// Returns the opcode for this gather operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Opcode depends on index size, NOT element size:
            // - 0x90 for 32-bit indices (D): VPGATHERDD, VPGATHERDQ
            // - 0x91 for 64-bit indices (Q): VPGATHERQD, VPGATHERQQ
            // The W bit (element size) is a separate encoding field.
            GatherOp::Vpgatherdd | GatherOp::Vpgatherdq => 0x90,
            GatherOp::Vpgatherqd | GatherOp::Vpgatherqq => 0x91,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            GatherOp::Vpgatherdd | GatherOp::Vpgatherqd => false, // W=0 for 32-bit elements
            GatherOp::Vpgatherdq | GatherOp::Vpgatherqq => true,  // W=1 for 64-bit elements
        }
    }

    /// Returns true if this uses 64-bit indices (Q index).
    pub fn uses_64bit_indices(&self) -> bool {
        matches!(self, GatherOp::Vpgatherqd | GatherOp::Vpgatherqq)
    }

    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u8 {
        match self {
            GatherOp::Vpgatherdd | GatherOp::Vpgatherqd => 4,
            GatherOp::Vpgatherdq | GatherOp::Vpgatherqq => 8,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            GatherOp::Vpgatherdd => "vpgatherdd",
            GatherOp::Vpgatherdq => "vpgatherdq",
            GatherOp::Vpgatherqd => "vpgatherqd",
            GatherOp::Vpgatherqq => "vpgatherqq",
        }
    }
}

/// AVX-512 Type Conversion operations.
///
/// These operations convert between integer and floating-point types.
/// Essential for columnar database operations where data types need
/// to be converted for arithmetic operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512CvtOp {
    // =========================================
    // Integer to Float Conversions
    // =========================================
    /// VCVTDQ2PS - Convert packed i32 to packed f32 (I32X16 -> F32X16)
    Vcvtdq2ps,
    /// VCVTQQ2PD - Convert packed i64 to packed f64 (I64X8 -> F64X8)
    Vcvtqq2pd,
    /// VCVTQQ2PS - Convert packed i64 to packed f32 (I64X8 -> F32X8)
    Vcvtqq2ps,

    // =========================================
    // Float to Integer Conversions (with rounding)
    // =========================================
    /// VCVTPS2DQ - Convert packed f32 to packed i32 with rounding (F32X16 -> I32X16)
    Vcvtps2dq,
    /// VCVTPD2QQ - Convert packed f64 to packed i64 with rounding (F64X8 -> I64X8)
    Vcvtpd2qq,
    /// VCVTPS2QQ - Convert packed f32 to packed i64 with rounding (F32X8 -> I64X8)
    Vcvtps2qq,

    // =========================================
    // Float to Integer Conversions (with truncation)
    // =========================================
    /// VCVTTPS2DQ - Convert packed f32 to packed i32 with truncation (F32X16 -> I32X16)
    Vcvttps2dq,
    /// VCVTTPD2QQ - Convert packed f64 to packed i64 with truncation (F64X8 -> I64X8)
    Vcvttpd2qq,
    /// VCVTTPS2QQ - Convert packed f32 to packed i64 with truncation (F32X8 -> I64X8)
    Vcvttps2qq,

    // =========================================
    // Float to Float Conversions
    // =========================================
    /// VCVTPS2PD - Convert packed f32 to packed f64 (F32X8 -> F64X8)
    Vcvtps2pd,
    /// VCVTPD2PS - Convert packed f64 to packed f32 (F64X8 -> F32X8)
    Vcvtpd2ps,

    // =========================================
    // Integer to Float/Float to Integer (widening/narrowing)
    // =========================================
    /// VCVTDQ2PD - Convert packed i32 to packed f64 (I32X8 -> F64X8)
    Vcvtdq2pd,
    /// VCVTPD2DQ - Convert packed f64 to packed i32 with rounding (F64X8 -> I32X8)
    Vcvtpd2dq,
    /// VCVTTPD2DQ - Convert packed f64 to packed i32 with truncation (F64X8 -> I32X8)
    Vcvttpd2dq,

    // =========================================
    // Integer Zero/Sign Extension (VPMOVZX/VPMOVSX)
    // =========================================
    /// VPMOVZXBD - Zero-extend packed bytes to dwords (I8X16 low -> I32X16)
    Vpmovzxbd,
    /// VPMOVZXBQ - Zero-extend packed bytes to qwords (I8X8 low -> I64X8)
    Vpmovzxbq,
    /// VPMOVZXWD - Zero-extend packed words to dwords (I16X16 low -> I32X16)
    Vpmovzxwd,
    /// VPMOVZXWQ - Zero-extend packed words to qwords (I16X8 low -> I64X8)
    Vpmovzxwq,
    /// VPMOVZXDQ - Zero-extend packed dwords to qwords (I32X8 low -> I64X8)
    Vpmovzxdq,
    /// VPMOVSXBD - Sign-extend packed bytes to dwords (I8X16 low -> I32X16)
    Vpmovsxbd,
    /// VPMOVSXBQ - Sign-extend packed bytes to qwords (I8X8 low -> I64X8)
    Vpmovsxbq,
    /// VPMOVSXWD - Sign-extend packed words to dwords (I16X16 low -> I32X16)
    Vpmovsxwd,
    /// VPMOVSXWQ - Sign-extend packed words to qwords (I16X8 low -> I64X8)
    Vpmovsxwq,
    /// VPMOVSXDQ - Sign-extend packed dwords to qwords (I32X8 low -> I64X8)
    Vpmovsxdq,

    // =========================================
    // Integer Truncation (VPMOV - down-conversion)
    // =========================================
    /// VPMOVDB - Truncate packed dwords to bytes (I32X16 -> I8X16 low)
    Vpmovdb,
    /// VPMOVDW - Truncate packed dwords to words (I32X16 -> I16X16 low)
    Vpmovdw,
    /// VPMOVQB - Truncate packed qwords to bytes (I64X8 -> I8X8 low)
    Vpmovqb,
    /// VPMOVQD - Truncate packed qwords to dwords (I64X8 -> I32X8 low)
    Vpmovqd,
    /// VPMOVQW - Truncate packed qwords to words (I64X8 -> I16X8 low)
    Vpmovqw,
    /// VPMOVWB - Truncate packed words to bytes (I16X32 -> I8X32 low)
    Vpmovwb,

    /// VPMOVSDB - Truncate packed dwords to bytes with saturation (signed)
    Vpmovsdb,
    /// VPMOVSDW - Truncate packed dwords to words with saturation (signed)
    Vpmovsdw,
    /// VPMOVSQB - Truncate packed qwords to bytes with saturation (signed)
    Vpmovsqb,
    /// VPMOVSQD - Truncate packed qwords to dwords with saturation (signed)
    Vpmovsqd,
    /// VPMOVSQW - Truncate packed qwords to words with saturation (signed)
    Vpmovsqw,
    /// VPMOVSWB - Truncate packed words to bytes with saturation (signed)
    Vpmovswb,

    /// VPMOVUSDB - Truncate packed dwords to bytes with saturation (unsigned)
    Vpmovusdb,
    /// VPMOVUSDW - Truncate packed dwords to words with saturation (unsigned)
    Vpmovusdw,
    /// VPMOVUSQB - Truncate packed qwords to bytes with saturation (unsigned)
    Vpmovusqb,
    /// VPMOVUSQD - Truncate packed qwords to dwords with saturation (unsigned)
    Vpmovusqd,
    /// VPMOVUSQW - Truncate packed qwords to words with saturation (unsigned)
    Vpmovusqw,
    /// VPMOVUSWB - Truncate packed words to bytes with saturation (unsigned)
    Vpmovuswb,
}

impl Avx512CvtOp {
    /// Returns the primary opcode byte for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Integer to Float - all use opcode 0x5B except VCVTQQ2PD/PS
            Avx512CvtOp::Vcvtdq2ps => 0x5B,
            Avx512CvtOp::Vcvtqq2pd => 0xE6,
            Avx512CvtOp::Vcvtqq2ps => 0x5B,

            // Float to Integer with rounding
            Avx512CvtOp::Vcvtps2dq => 0x5B,
            Avx512CvtOp::Vcvtpd2qq => 0x7B,
            Avx512CvtOp::Vcvtps2qq => 0x7B,

            // Float to Integer with truncation
            Avx512CvtOp::Vcvttps2dq => 0x5B,
            Avx512CvtOp::Vcvttpd2qq => 0x7A,
            Avx512CvtOp::Vcvttps2qq => 0x7A,

            // Float to Float
            Avx512CvtOp::Vcvtps2pd => 0x5A,
            Avx512CvtOp::Vcvtpd2ps => 0x5A,

            // Integer to Double / Double to Integer
            Avx512CvtOp::Vcvtdq2pd => 0xE6,
            Avx512CvtOp::Vcvtpd2dq => 0xE6,
            Avx512CvtOp::Vcvttpd2dq => 0xE6,

            // Zero-extend
            Avx512CvtOp::Vpmovzxbd => 0x31,
            Avx512CvtOp::Vpmovzxbq => 0x32,
            Avx512CvtOp::Vpmovzxwd => 0x33,
            Avx512CvtOp::Vpmovzxwq => 0x34,
            Avx512CvtOp::Vpmovzxdq => 0x35,

            // Sign-extend
            Avx512CvtOp::Vpmovsxbd => 0x21,
            Avx512CvtOp::Vpmovsxbq => 0x22,
            Avx512CvtOp::Vpmovsxwd => 0x23,
            Avx512CvtOp::Vpmovsxwq => 0x24,
            Avx512CvtOp::Vpmovsxdq => 0x25,

            // Truncation (no saturation)
            Avx512CvtOp::Vpmovdb => 0x31,
            Avx512CvtOp::Vpmovdw => 0x33,
            Avx512CvtOp::Vpmovqb => 0x32,
            Avx512CvtOp::Vpmovqd => 0x35,
            Avx512CvtOp::Vpmovqw => 0x34,
            Avx512CvtOp::Vpmovwb => 0x30,

            // Truncation with signed saturation
            Avx512CvtOp::Vpmovsdb => 0x21,
            Avx512CvtOp::Vpmovsdw => 0x23,
            Avx512CvtOp::Vpmovsqb => 0x22,
            Avx512CvtOp::Vpmovsqd => 0x25,
            Avx512CvtOp::Vpmovsqw => 0x24,
            Avx512CvtOp::Vpmovswb => 0x20,

            // Truncation with unsigned saturation
            Avx512CvtOp::Vpmovusdb => 0x11,
            Avx512CvtOp::Vpmovusdw => 0x13,
            Avx512CvtOp::Vpmovusqb => 0x12,
            Avx512CvtOp::Vpmovusqd => 0x15,
            Avx512CvtOp::Vpmovusqw => 0x14,
            Avx512CvtOp::Vpmovuswb => 0x10,
        }
    }

    /// Returns the EVEX opcode map for this operation.
    pub fn evex_map(&self) -> u8 {
        match self {
            // Most are in 0F map
            Avx512CvtOp::Vcvtdq2ps
            | Avx512CvtOp::Vcvtqq2pd
            | Avx512CvtOp::Vcvtqq2ps
            | Avx512CvtOp::Vcvtps2dq
            | Avx512CvtOp::Vcvttps2dq
            | Avx512CvtOp::Vcvtps2pd
            | Avx512CvtOp::Vcvtpd2ps
            // QQ conversions are also in 0F map (not 0F38!)
            | Avx512CvtOp::Vcvtpd2qq
            | Avx512CvtOp::Vcvtps2qq
            | Avx512CvtOp::Vcvttpd2qq
            | Avx512CvtOp::Vcvttps2qq => 0x01, // 0F map

            // Zero/Sign-extend are all in 0F38 map
            Avx512CvtOp::Vpmovzxbd
            | Avx512CvtOp::Vpmovzxbq
            | Avx512CvtOp::Vpmovzxwd
            | Avx512CvtOp::Vpmovzxwq
            | Avx512CvtOp::Vpmovzxdq
            | Avx512CvtOp::Vpmovsxbd
            | Avx512CvtOp::Vpmovsxbq
            | Avx512CvtOp::Vpmovsxwd
            | Avx512CvtOp::Vpmovsxwq
            | Avx512CvtOp::Vpmovsxdq
            // Truncation operations are all in 0F38 map
            | Avx512CvtOp::Vpmovdb
            | Avx512CvtOp::Vpmovdw
            | Avx512CvtOp::Vpmovqb
            | Avx512CvtOp::Vpmovqd
            | Avx512CvtOp::Vpmovqw
            | Avx512CvtOp::Vpmovwb
            | Avx512CvtOp::Vpmovsdb
            | Avx512CvtOp::Vpmovsdw
            | Avx512CvtOp::Vpmovsqb
            | Avx512CvtOp::Vpmovsqd
            | Avx512CvtOp::Vpmovsqw
            | Avx512CvtOp::Vpmovswb
            | Avx512CvtOp::Vpmovusdb
            | Avx512CvtOp::Vpmovusdw
            | Avx512CvtOp::Vpmovusqb
            | Avx512CvtOp::Vpmovusqd
            | Avx512CvtOp::Vpmovusqw
            | Avx512CvtOp::Vpmovuswb => 0x02, // 0F38 map

            // DQ2PD and PD2DQ are in 0F map
            Avx512CvtOp::Vcvtdq2pd
            | Avx512CvtOp::Vcvtpd2dq
            | Avx512CvtOp::Vcvttpd2dq => 0x01, // 0F map
        }
    }

    /// Returns the EVEX.pp (prefix) field for this operation.
    pub fn evex_pp(&self) -> u8 {
        match self {
            // No prefix
            Avx512CvtOp::Vcvtdq2ps => 0x00,

            // 66 prefix
            Avx512CvtOp::Vcvtps2dq
            | Avx512CvtOp::Vcvtpd2qq
            | Avx512CvtOp::Vcvttpd2qq
            | Avx512CvtOp::Vcvtpd2ps
            | Avx512CvtOp::Vcvtps2qq
            | Avx512CvtOp::Vcvttps2qq => 0x01,

            // F3 prefix
            Avx512CvtOp::Vcvttps2dq
            | Avx512CvtOp::Vcvtqq2pd
            | Avx512CvtOp::Vcvtqq2ps
            | Avx512CvtOp::Vcvtps2pd
            | Avx512CvtOp::Vcvtdq2pd
            // Truncation operations use F3 prefix
            | Avx512CvtOp::Vpmovdb
            | Avx512CvtOp::Vpmovdw
            | Avx512CvtOp::Vpmovqb
            | Avx512CvtOp::Vpmovqd
            | Avx512CvtOp::Vpmovqw
            | Avx512CvtOp::Vpmovwb
            | Avx512CvtOp::Vpmovsdb
            | Avx512CvtOp::Vpmovsdw
            | Avx512CvtOp::Vpmovsqb
            | Avx512CvtOp::Vpmovsqd
            | Avx512CvtOp::Vpmovsqw
            | Avx512CvtOp::Vpmovswb
            | Avx512CvtOp::Vpmovusdb
            | Avx512CvtOp::Vpmovusdw
            | Avx512CvtOp::Vpmovusqb
            | Avx512CvtOp::Vpmovusqd
            | Avx512CvtOp::Vpmovusqw
            | Avx512CvtOp::Vpmovuswb => 0x02,

            // 66 prefix for zero/sign-extend
            Avx512CvtOp::Vpmovzxbd
            | Avx512CvtOp::Vpmovzxbq
            | Avx512CvtOp::Vpmovzxwd
            | Avx512CvtOp::Vpmovzxwq
            | Avx512CvtOp::Vpmovzxdq
            | Avx512CvtOp::Vpmovsxbd
            | Avx512CvtOp::Vpmovsxbq
            | Avx512CvtOp::Vpmovsxwd
            | Avx512CvtOp::Vpmovsxwq
            | Avx512CvtOp::Vpmovsxdq
            | Avx512CvtOp::Vcvttpd2dq => 0x01,

            // F2 prefix
            Avx512CvtOp::Vcvtpd2dq => 0x03,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            // W=0 for 32-bit source/destination operations
            Avx512CvtOp::Vcvtdq2ps
            | Avx512CvtOp::Vcvtps2dq
            | Avx512CvtOp::Vcvttps2dq
            | Avx512CvtOp::Vcvtps2pd
            | Avx512CvtOp::Vcvtps2qq
            | Avx512CvtOp::Vcvttps2qq
            | Avx512CvtOp::Vcvtdq2pd
            // Zero/sign-extend all use W=0
            | Avx512CvtOp::Vpmovzxbd
            | Avx512CvtOp::Vpmovzxbq
            | Avx512CvtOp::Vpmovzxwd
            | Avx512CvtOp::Vpmovzxwq
            | Avx512CvtOp::Vpmovzxdq
            | Avx512CvtOp::Vpmovsxbd
            | Avx512CvtOp::Vpmovsxbq
            | Avx512CvtOp::Vpmovsxwd
            | Avx512CvtOp::Vpmovsxwq
            | Avx512CvtOp::Vpmovsxdq
            // Truncation operations all use W=0
            | Avx512CvtOp::Vpmovdb
            | Avx512CvtOp::Vpmovdw
            | Avx512CvtOp::Vpmovqb
            | Avx512CvtOp::Vpmovqd
            | Avx512CvtOp::Vpmovqw
            | Avx512CvtOp::Vpmovwb
            | Avx512CvtOp::Vpmovsdb
            | Avx512CvtOp::Vpmovsdw
            | Avx512CvtOp::Vpmovsqb
            | Avx512CvtOp::Vpmovsqd
            | Avx512CvtOp::Vpmovsqw
            | Avx512CvtOp::Vpmovswb
            | Avx512CvtOp::Vpmovusdb
            | Avx512CvtOp::Vpmovusdw
            | Avx512CvtOp::Vpmovusqb
            | Avx512CvtOp::Vpmovusqd
            | Avx512CvtOp::Vpmovusqw
            | Avx512CvtOp::Vpmovuswb => false,

            // W=1 for 64-bit source/destination operations
            Avx512CvtOp::Vcvtqq2pd
            | Avx512CvtOp::Vcvtqq2ps
            | Avx512CvtOp::Vcvtpd2qq
            | Avx512CvtOp::Vcvttpd2qq
            | Avx512CvtOp::Vcvtpd2ps
            | Avx512CvtOp::Vcvtpd2dq
            | Avx512CvtOp::Vcvttpd2dq => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512CvtOp::Vcvtdq2ps => "vcvtdq2ps",
            Avx512CvtOp::Vcvtqq2pd => "vcvtqq2pd",
            Avx512CvtOp::Vcvtqq2ps => "vcvtqq2ps",
            Avx512CvtOp::Vcvtps2dq => "vcvtps2dq",
            Avx512CvtOp::Vcvtpd2qq => "vcvtpd2qq",
            Avx512CvtOp::Vcvtps2qq => "vcvtps2qq",
            Avx512CvtOp::Vcvttps2dq => "vcvttps2dq",
            Avx512CvtOp::Vcvttpd2qq => "vcvttpd2qq",
            Avx512CvtOp::Vcvttps2qq => "vcvttps2qq",
            Avx512CvtOp::Vcvtps2pd => "vcvtps2pd",
            Avx512CvtOp::Vcvtpd2ps => "vcvtpd2ps",
            Avx512CvtOp::Vcvtdq2pd => "vcvtdq2pd",
            Avx512CvtOp::Vcvtpd2dq => "vcvtpd2dq",
            Avx512CvtOp::Vcvttpd2dq => "vcvttpd2dq",
            Avx512CvtOp::Vpmovzxbd => "vpmovzxbd",
            Avx512CvtOp::Vpmovzxbq => "vpmovzxbq",
            Avx512CvtOp::Vpmovzxwd => "vpmovzxwd",
            Avx512CvtOp::Vpmovzxwq => "vpmovzxwq",
            Avx512CvtOp::Vpmovzxdq => "vpmovzxdq",
            Avx512CvtOp::Vpmovsxbd => "vpmovsxbd",
            Avx512CvtOp::Vpmovsxbq => "vpmovsxbq",
            Avx512CvtOp::Vpmovsxwd => "vpmovsxwd",
            Avx512CvtOp::Vpmovsxwq => "vpmovsxwq",
            Avx512CvtOp::Vpmovsxdq => "vpmovsxdq",
            // Truncation (no saturation)
            Avx512CvtOp::Vpmovdb => "vpmovdb",
            Avx512CvtOp::Vpmovdw => "vpmovdw",
            Avx512CvtOp::Vpmovqb => "vpmovqb",
            Avx512CvtOp::Vpmovqd => "vpmovqd",
            Avx512CvtOp::Vpmovqw => "vpmovqw",
            Avx512CvtOp::Vpmovwb => "vpmovwb",
            // Truncation with signed saturation
            Avx512CvtOp::Vpmovsdb => "vpmovsdb",
            Avx512CvtOp::Vpmovsdw => "vpmovsdw",
            Avx512CvtOp::Vpmovsqb => "vpmovsqb",
            Avx512CvtOp::Vpmovsqd => "vpmovsqd",
            Avx512CvtOp::Vpmovsqw => "vpmovsqw",
            Avx512CvtOp::Vpmovswb => "vpmovswb",
            // Truncation with unsigned saturation
            Avx512CvtOp::Vpmovusdb => "vpmovusdb",
            Avx512CvtOp::Vpmovusdw => "vpmovusdw",
            Avx512CvtOp::Vpmovusqb => "vpmovusqb",
            Avx512CvtOp::Vpmovusqd => "vpmovusqd",
            Avx512CvtOp::Vpmovusqw => "vpmovusqw",
            Avx512CvtOp::Vpmovuswb => "vpmovuswb",
        }
    }
}

/// AVX-512 Vector Alignment operations (with immediate byte).
///
/// These operations concatenate two vectors and extract an aligned portion
/// based on an immediate byte index. Essential for sliding window operations
/// and cross-vector boundary data access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512AlignOp {
    /// VALIGND - Align 32-bit elements from concatenation of two vectors
    /// dst = src1[imm8..imm8+16] || src2[imm8..imm8+16] (16 elements)
    Valignd,
    /// VALIGNQ - Align 64-bit elements from concatenation of two vectors
    /// dst = src1[imm8..imm8+8] || src2[imm8..imm8+8] (8 elements)
    Valignq,
}

impl Avx512AlignOp {
    /// Returns the primary opcode byte for this operation.
    /// Both VALIGND and VALIGNQ use opcode 0x03 in the 0F3A map.
    pub fn opcode(&self) -> u8 {
        0x03
    }

    /// Returns the EVEX opcode map for this operation.
    /// VALIGND/Q are in the 0F3A map (0x03).
    pub fn evex_map(&self) -> u8 {
        0x03 // 0F3A map
    }

    /// Returns the EVEX.pp (prefix) field for this operation.
    /// Both use 66 prefix (pp=01).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512AlignOp::Valignd => false, // W=0 for 32-bit elements
            Avx512AlignOp::Valignq => true,  // W=1 for 64-bit elements
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512AlignOp::Valignd => "valignd",
            Avx512AlignOp::Valignq => "valignq",
        }
    }
}

/// AVX-512 Scatter operations.
///
/// Scatter stores vector elements to non-contiguous memory locations
/// using index values from another vector register. Essential for
/// columnar database operations where results need to be written to
/// scattered positions based on row indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterOp {
    /// VPSCATTERDD - Scatter 32-bit elements using 32-bit indices
    /// [base + zmm_index*scale] {k1}, zmm1
    Vpscatterdd,
    /// VPSCATTERDQ - Scatter 64-bit elements using 32-bit indices
    /// [base + ymm_index*scale] {k1}, zmm1
    Vpscatterdq,
    /// VPSCATTERQD - Scatter 32-bit elements using 64-bit indices
    /// [base + zmm_index*scale] {k1}, ymm1
    Vpscatterqd,
    /// VPSCATTERQQ - Scatter 64-bit elements using 64-bit indices
    /// [base + zmm_index*scale] {k1}, zmm1
    Vpscatterqq,
}

impl ScatterOp {
    /// Returns the opcode for this scatter operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Opcode depends on index size, NOT element size:
            // - 0xA0 for 32-bit indices (D): VPSCATTERDD, VPSCATTERDQ
            // - 0xA1 for 64-bit indices (Q): VPSCATTERQD, VPSCATTERQQ
            // The W bit (element size) is a separate encoding field.
            ScatterOp::Vpscatterdd | ScatterOp::Vpscatterdq => 0xA0,
            ScatterOp::Vpscatterqd | ScatterOp::Vpscatterqq => 0xA1,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            ScatterOp::Vpscatterdd | ScatterOp::Vpscatterqd => false, // W=0 for 32-bit elements
            ScatterOp::Vpscatterdq | ScatterOp::Vpscatterqq => true,  // W=1 for 64-bit elements
        }
    }

    /// Returns true if this uses 64-bit indices (Q index).
    pub fn uses_64bit_indices(&self) -> bool {
        matches!(self, ScatterOp::Vpscatterqd | ScatterOp::Vpscatterqq)
    }

    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u8 {
        match self {
            ScatterOp::Vpscatterdd | ScatterOp::Vpscatterqd => 4,
            ScatterOp::Vpscatterdq | ScatterOp::Vpscatterqq => 8,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            ScatterOp::Vpscatterdd => "vpscatterdd",
            ScatterOp::Vpscatterdq => "vpscatterdq",
            ScatterOp::Vpscatterqd => "vpscatterqd",
            ScatterOp::Vpscatterqq => "vpscatterqq",
        }
    }
}

// =============================================================================
// Validation Functions
// =============================================================================

/// Validates that a k-register index is valid for use as a write mask (k1-k7).
/// k0 is reserved and means "no masking".
#[inline]
#[allow(dead_code, reason = "validation utility reserved for future use")]
pub fn validate_mask_register(kreg: u8) -> Result<(), &'static str> {
    if kreg > 7 {
        Err("k-register index must be 0-7")
    } else {
        Ok(())
    }
}

/// Validates that a k-register index is valid for use as an active mask (k1-k7).
/// Returns an error if k0 is used where a real mask is required.
#[inline]
#[allow(dead_code, reason = "validation utility reserved for future use")]
pub fn validate_active_mask_register(kreg: u8) -> Result<(), &'static str> {
    if kreg == 0 {
        Err("k0 cannot be used as an active write mask; use k1-k7")
    } else if kreg > 7 {
        Err("k-register index must be 1-7 for active masking")
    } else {
        Ok(())
    }
}

/// Validates that OperandSize is appropriate for AVX-512 integer operations.
#[inline]
#[allow(dead_code, reason = "validation utility reserved for future use")]
pub fn validate_x64_512_operand_size(
    size: &crate::isa::x64::inst::args::OperandSize,
) -> Result<(), &'static str> {
    use crate::isa::x64::inst::args::OperandSize;
    match size {
        OperandSize::Size32 | OperandSize::Size64 => Ok(()),
        _ => Err("AVX-512 integer operations only support 32-bit or 64-bit element sizes"),
    }
}

/// AVX-512 Immediate Shuffle operations.
///
/// These operations shuffle elements according to an 8-bit immediate value.
/// All use opcode 0x70 with different prefixes:
/// - VPSHUFD: 66 prefix (shuffles dwords)
/// - VPSHUFHW: F3 prefix (shuffles high words in each qword)
/// - VPSHUFLW: F2 prefix (shuffles low words in each qword)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avx512ImmShuffleOp {
    /// VPSHUFD - Shuffle packed dwords according to immediate
    /// Each 2-bit field in imm8 selects one of 4 dwords within each 128-bit lane
    Vpshufd,
    /// VPSHUFHW - Shuffle packed high words according to immediate
    /// Shuffles the high 4 words (bits 64-127) of each 128-bit lane
    Vpshufhw,
    /// VPSHUFLW - Shuffle packed low words according to immediate
    /// Shuffles the low 4 words (bits 0-63) of each 128-bit lane
    Vpshuflw,
}

impl Avx512ImmShuffleOp {
    /// Returns the opcode for this shuffle operation.
    /// All immediate shuffle ops use opcode 0x70.
    pub fn opcode(&self) -> u8 {
        0x70
    }

    /// Returns the EVEX map encoding (0x01 = 0F map).
    pub fn evex_map(&self) -> u8 {
        0x01 // All use 0F map
    }

    /// Returns the EVEX pp (prefix) encoding.
    pub fn evex_pp(&self) -> u8 {
        match self {
            Avx512ImmShuffleOp::Vpshufd => 0x01,  // 66 prefix
            Avx512ImmShuffleOp::Vpshufhw => 0x02, // F3 prefix
            Avx512ImmShuffleOp::Vpshuflw => 0x03, // F2 prefix
        }
    }

    /// Returns the EVEX.W bit (always 0 for these operations).
    pub fn evex_w(&self) -> bool {
        false // All use W=0
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512ImmShuffleOp::Vpshufd => "vpshufd",
            Avx512ImmShuffleOp::Vpshufhw => "vpshufhw",
            Avx512ImmShuffleOp::Vpshuflw => "vpshuflw",
        }
    }
}

/// AVX-512 lane shuffle operations.
///
/// These instructions shuffle entire 128-bit lanes within a 512-bit register.
/// Each lane can be selected from any of the 4 lanes in src1 or src2.
/// The immediate byte encodes the selection pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avx512LaneShuffleOp {
    /// VSHUFF32X4 - Shuffle 128-bit lanes of packed single-precision FP
    /// Interprets each 128-bit lane as 4 x F32
    Vshuff32x4,
    /// VSHUFF64X2 - Shuffle 128-bit lanes of packed double-precision FP
    /// Interprets each 128-bit lane as 2 x F64
    Vshuff64x2,
    /// VSHUFI32X4 - Shuffle 128-bit lanes of packed 32-bit integers
    /// Interprets each 128-bit lane as 4 x I32
    Vshufi32x4,
    /// VSHUFI64X2 - Shuffle 128-bit lanes of packed 64-bit integers
    /// Interprets each 128-bit lane as 2 x I64
    Vshufi64x2,
}

impl Avx512LaneShuffleOp {
    /// Returns the opcode for this lane shuffle operation.
    /// F variants use 0x23, I variants use 0x43.
    pub fn opcode(&self) -> u8 {
        match self {
            Avx512LaneShuffleOp::Vshuff32x4 | Avx512LaneShuffleOp::Vshuff64x2 => 0x23,
            Avx512LaneShuffleOp::Vshufi32x4 | Avx512LaneShuffleOp::Vshufi64x2 => 0x43,
        }
    }

    /// Returns the EVEX map encoding (0x03 = 0F3A map).
    pub fn evex_map(&self) -> u8 {
        0x03 // All use 0F3A map
    }

    /// Returns the EVEX pp (prefix) encoding.
    /// All use 66 prefix (pp=0x01).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit.
    /// 64x2 variants use W=1, 32x4 variants use W=0.
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512LaneShuffleOp::Vshuff64x2 | Avx512LaneShuffleOp::Vshufi64x2 => true,
            Avx512LaneShuffleOp::Vshuff32x4 | Avx512LaneShuffleOp::Vshufi32x4 => false,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512LaneShuffleOp::Vshuff32x4 => "vshuff32x4",
            Avx512LaneShuffleOp::Vshuff64x2 => "vshuff64x2",
            Avx512LaneShuffleOp::Vshufi32x4 => "vshufi32x4",
            Avx512LaneShuffleOp::Vshufi64x2 => "vshufi64x2",
        }
    }
}

/// AVX-512 VP2INTERSECT operations for hash join acceleration.
///
/// VP2INTERSECT compares two vectors element-by-element and produces TWO mask
/// register outputs (consecutive k-registers):
/// - k[dst]: For each element in src1, set bit if it matches ANY element in src2
/// - k[dst+1]: For each element in src2, set bit if it matches ANY element in src1
///
/// This is crucial for accelerating hash join operations in databases:
/// - Build phase: Load hash table buckets into ZMM register
/// - Probe phase: Load probe keys, use VP2INTERSECT to find matches
/// - Both output masks indicate which elements found matches
///
/// Encoding: EVEX.512.F2.0F38.W0/W1 68 /r
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp2IntersectOp {
    /// VP2INTERSECTD - Compare 32-bit elements, produces two 16-bit masks
    /// For 16-element vectors (I32X16)
    Vp2intersectd,
    /// VP2INTERSECTQ - Compare 64-bit elements, produces two 8-bit masks
    /// For 8-element vectors (I64X8)
    Vp2intersectq,
}

impl Vp2IntersectOp {
    /// Returns the opcode for VP2INTERSECT (always 0x68).
    pub fn opcode(&self) -> u8 {
        0x68
    }

    /// Returns the EVEX map encoding (0x02 = 0F38 map).
    pub fn evex_map(&self) -> u8 {
        0x02 // 0F38 map
    }

    /// Returns the EVEX pp (prefix) encoding (F2 prefix).
    pub fn evex_pp(&self) -> u8 {
        0x03 // F2 prefix
    }

    /// Returns the EVEX.W bit.
    pub fn evex_w(&self) -> bool {
        match self {
            Vp2IntersectOp::Vp2intersectd => false, // W=0 for dword
            Vp2IntersectOp::Vp2intersectq => true,  // W=1 for qword
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Vp2IntersectOp::Vp2intersectd => "vp2intersectd",
            Vp2IntersectOp::Vp2intersectq => "vp2intersectq",
        }
    }

    /// Returns the number of elements in the vector for this element size.
    pub fn num_elements(&self) -> u8 {
        match self {
            Vp2IntersectOp::Vp2intersectd => 16, // 512/32 = 16 elements
            Vp2IntersectOp::Vp2intersectq => 8,  // 512/64 = 8 elements
        }
    }
}

/// AVX-512 extract operations for extracting 128-bit or 256-bit lanes from ZMM registers.
///
/// These are essential for interoperability between 512-bit and 128/256-bit code paths.
///
/// Encoding:
/// - VEXTRACTI32X4: EVEX.512.66.0F3A.W0 39 /r imm8
/// - VEXTRACTI64X2: EVEX.512.66.0F3A.W1 39 /r imm8
/// - VEXTRACTI32X8: EVEX.512.66.0F3A.W0 3B /r imm8
/// - VEXTRACTI64X4: EVEX.512.66.0F3A.W1 3B /r imm8
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avx512ExtractOp {
    /// VEXTRACTI32X4 - Extract 128-bit lane (as 4x32-bit) from 512-bit vector
    /// imm8[1:0] selects which 128-bit lane (0-3)
    Vextracti32x4,
    /// VEXTRACTI64X2 - Extract 128-bit lane (as 2x64-bit) from 512-bit vector
    /// imm8[1:0] selects which 128-bit lane (0-3)
    Vextracti64x2,
    /// VEXTRACTI32X8 - Extract 256-bit lane (as 8x32-bit) from 512-bit vector
    /// imm8[0] selects which 256-bit lane (0-1)
    Vextracti32x8,
    /// VEXTRACTI64X4 - Extract 256-bit lane (as 4x64-bit) from 512-bit vector
    /// imm8[0] selects which 256-bit lane (0-1)
    Vextracti64x4,
}

impl Avx512ExtractOp {
    /// Returns the opcode for this extract operation.
    pub fn opcode(&self) -> u8 {
        match self {
            Avx512ExtractOp::Vextracti32x4 | Avx512ExtractOp::Vextracti64x2 => 0x39,
            Avx512ExtractOp::Vextracti32x8 | Avx512ExtractOp::Vextracti64x4 => 0x3B,
        }
    }

    /// Returns the EVEX map encoding (0x03 = 0F3A map).
    pub fn evex_map(&self) -> u8 {
        0x03 // 0F3A map
    }

    /// Returns the EVEX pp (prefix) encoding (always 0x01 = 66 prefix).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit.
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512ExtractOp::Vextracti32x4 | Avx512ExtractOp::Vextracti32x8 => false,
            Avx512ExtractOp::Vextracti64x2 | Avx512ExtractOp::Vextracti64x4 => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512ExtractOp::Vextracti32x4 => "vextracti32x4",
            Avx512ExtractOp::Vextracti64x2 => "vextracti64x2",
            Avx512ExtractOp::Vextracti32x8 => "vextracti32x8",
            Avx512ExtractOp::Vextracti64x4 => "vextracti64x4",
        }
    }

    /// Returns true if this extracts a 256-bit result (ymm), false for 128-bit (xmm).
    pub fn is_256bit(&self) -> bool {
        match self {
            Avx512ExtractOp::Vextracti32x4 | Avx512ExtractOp::Vextracti64x2 => false,
            Avx512ExtractOp::Vextracti32x8 | Avx512ExtractOp::Vextracti64x4 => true,
        }
    }
}

/// AVX-512 insert operations (VINSERTI32X4, VINSERTI64X2, VINSERTI32X8, VINSERTI64X4).
///
/// These insert a 128-bit or 256-bit lane into a 512-bit vector.
///
/// Encoding:
/// - VINSERTI32X4: EVEX.512.66.0F3A.W0 38 /r imm8
/// - VINSERTI64X2: EVEX.512.66.0F3A.W1 38 /r imm8
/// - VINSERTI32X8: EVEX.512.66.0F3A.W0 3A /r imm8
/// - VINSERTI64X4: EVEX.512.66.0F3A.W1 3A /r imm8
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avx512InsertOp {
    /// VINSERTI32X4 - Insert 128-bit lane (as 4x32-bit) into 512-bit vector
    /// imm8[1:0] selects which 128-bit lane (0-3) to insert at
    Vinserti32x4,
    /// VINSERTI64X2 - Insert 128-bit lane (as 2x64-bit) into 512-bit vector
    /// imm8[1:0] selects which 128-bit lane (0-3) to insert at
    Vinserti64x2,
    /// VINSERTI32X8 - Insert 256-bit lane (as 8x32-bit) into 512-bit vector
    /// imm8[0] selects which 256-bit lane (0-1) to insert at
    Vinserti32x8,
    /// VINSERTI64X4 - Insert 256-bit lane (as 4x64-bit) into 512-bit vector
    /// imm8[0] selects which 256-bit lane (0-1) to insert at
    Vinserti64x4,
}

impl Avx512InsertOp {
    /// Returns the opcode for this insert operation.
    pub fn opcode(&self) -> u8 {
        match self {
            Avx512InsertOp::Vinserti32x4 | Avx512InsertOp::Vinserti64x2 => 0x38,
            Avx512InsertOp::Vinserti32x8 | Avx512InsertOp::Vinserti64x4 => 0x3A,
        }
    }

    /// Returns the EVEX map encoding (0x03 = 0F3A map).
    pub fn evex_map(&self) -> u8 {
        0x03 // 0F3A map
    }

    /// Returns the EVEX pp (prefix) encoding (always 0x01 = 66 prefix).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit.
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512InsertOp::Vinserti32x4 | Avx512InsertOp::Vinserti32x8 => false,
            Avx512InsertOp::Vinserti64x2 | Avx512InsertOp::Vinserti64x4 => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512InsertOp::Vinserti32x4 => "vinserti32x4",
            Avx512InsertOp::Vinserti64x2 => "vinserti64x2",
            Avx512InsertOp::Vinserti32x8 => "vinserti32x8",
            Avx512InsertOp::Vinserti64x4 => "vinserti64x4",
        }
    }

    /// Returns true if this inserts a 256-bit source (ymm), false for 128-bit (xmm).
    pub fn is_256bit(&self) -> bool {
        match self {
            Avx512InsertOp::Vinserti32x4 | Avx512InsertOp::Vinserti64x2 => false,
            Avx512InsertOp::Vinserti32x8 | Avx512InsertOp::Vinserti64x4 => true,
        }
    }
}

/// AVX-512 FP special operations (reciprocal, reciprocal sqrt, etc.)
///
/// These provide fast approximations useful for graphics and scientific computing.
///
/// Encoding:
/// - VRCP14PS: EVEX.512.66.0F38.W0 4C /r
/// - VRCP14PD: EVEX.512.66.0F38.W1 4C /r
/// - VRSQRT14PS: EVEX.512.66.0F38.W0 4E /r
/// - VRSQRT14PD: EVEX.512.66.0F38.W1 4E /r
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avx512FpSpecialOp {
    /// VRCP14PS - Reciprocal approximation of F32 elements (1/x)
    /// ~14 bits of precision
    Vrcp14ps,
    /// VRCP14PD - Reciprocal approximation of F64 elements (1/x)
    /// ~14 bits of precision
    Vrcp14pd,
    /// VRSQRT14PS - Reciprocal square root approximation of F32 elements (1/sqrt(x))
    /// ~14 bits of precision
    Vrsqrt14ps,
    /// VRSQRT14PD - Reciprocal square root approximation of F64 elements (1/sqrt(x))
    /// ~14 bits of precision
    Vrsqrt14pd,
}

impl Avx512FpSpecialOp {
    /// Returns the opcode for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            Avx512FpSpecialOp::Vrcp14ps | Avx512FpSpecialOp::Vrcp14pd => 0x4C,
            Avx512FpSpecialOp::Vrsqrt14ps | Avx512FpSpecialOp::Vrsqrt14pd => 0x4E,
        }
    }

    /// Returns the EVEX map encoding (0x02 = 0F38 map).
    pub fn evex_map(&self) -> u8 {
        0x02 // 0F38 map
    }

    /// Returns the EVEX pp (prefix) encoding (always 0x01 = 66 prefix).
    pub fn evex_pp(&self) -> u8 {
        0x01 // 66 prefix
    }

    /// Returns the EVEX.W bit.
    pub fn evex_w(&self) -> bool {
        match self {
            Avx512FpSpecialOp::Vrcp14ps | Avx512FpSpecialOp::Vrsqrt14ps => false,
            Avx512FpSpecialOp::Vrcp14pd | Avx512FpSpecialOp::Vrsqrt14pd => true,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512FpSpecialOp::Vrcp14ps => "vrcp14ps",
            Avx512FpSpecialOp::Vrcp14pd => "vrcp14pd",
            Avx512FpSpecialOp::Vrsqrt14ps => "vrsqrt14ps",
            Avx512FpSpecialOp::Vrsqrt14pd => "vrsqrt14pd",
        }
    }

    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u8 {
        match self {
            Avx512FpSpecialOp::Vrcp14ps | Avx512FpSpecialOp::Vrsqrt14ps => 4,
            Avx512FpSpecialOp::Vrcp14pd | Avx512FpSpecialOp::Vrsqrt14pd => 8,
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Avx512AluOp Tests
    // =========================================================================

    #[test]
    fn test_x64_512_alu_op_opcodes() {
        // Verify critical opcodes match Intel documentation
        assert_eq!(Avx512AluOp::Vpaddd.opcode(), 0xFE);
        assert_eq!(Avx512AluOp::Vpaddq.opcode(), 0xD4);
        assert_eq!(Avx512AluOp::Vpsubd.opcode(), 0xFA);
        assert_eq!(Avx512AluOp::Vpsubq.opcode(), 0xFB);
        assert_eq!(Avx512AluOp::Vpmulld.opcode(), 0x40);
        assert_eq!(Avx512AluOp::Vpandd.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpord.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vpxord.opcode(), 0xEF);
    }

    #[test]
    fn test_x64_512_alu_op_evex_map() {
        // 0F map operations
        assert_eq!(Avx512AluOp::Vpaddd.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpaddq.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpandd.evex_map(), 0x01);

        // 0F38 map operations
        assert_eq!(Avx512AluOp::Vpmulld.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpminsd.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpermd.evex_map(), 0x02);

        // 0F3A map operations
        assert_eq!(Avx512AluOp::Vpternlogd.evex_map(), 0x03);
        assert_eq!(Avx512AluOp::Vpternlogq.evex_map(), 0x03);
    }

    #[test]
    fn test_x64_512_alu_op_evex_w() {
        // 32-bit operations should have W=0
        assert!(!Avx512AluOp::Vpaddd.evex_w());
        assert!(!Avx512AluOp::Vpsubd.evex_w());
        assert!(!Avx512AluOp::Vpandd.evex_w());

        // 64-bit operations should have W=1
        assert!(Avx512AluOp::Vpaddq.evex_w());
        assert!(Avx512AluOp::Vpsubq.evex_w());
        assert!(Avx512AluOp::Vpandq.evex_w());
    }

    #[test]
    fn test_x64_512_alu_op_evex_pp() {
        // All integer ops use 66 prefix
        assert_eq!(Avx512AluOp::Vpaddd.evex_pp(), 0x01);
        assert_eq!(Avx512AluOp::Vpaddq.evex_pp(), 0x01);
        assert_eq!(Avx512AluOp::Vpermd.evex_pp(), 0x01);
    }

    #[test]
    fn test_x64_512_alu_op_is_64bit() {
        assert!(!Avx512AluOp::Vpaddd.is_64bit());
        assert!(Avx512AluOp::Vpaddq.is_64bit());
        assert!(!Avx512AluOp::Vpmulld.is_64bit());
        assert!(Avx512AluOp::Vpmullq.is_64bit());
    }

    #[test]
    fn test_x64_512_alu_op_is_unary() {
        assert!(Avx512AluOp::Vpabsd.is_unary());
        assert!(Avx512AluOp::Vpbroadcastd.is_unary());
        assert!(!Avx512AluOp::Vpaddd.is_unary());
        assert!(!Avx512AluOp::Vpermd.is_unary());
    }

    #[test]
    fn test_x64_512_alu_op_is_ternary() {
        assert!(Avx512AluOp::Vpternlogd.is_ternary());
        assert!(Avx512AluOp::Vpternlogq.is_ternary());
        assert!(!Avx512AluOp::Vpaddd.is_ternary());
    }

    #[test]
    fn test_x64_512_alu_op_names() {
        assert_eq!(Avx512AluOp::Vpaddd.name(), "vpaddd");
        assert_eq!(Avx512AluOp::Vpaddq.name(), "vpaddq");
        assert_eq!(Avx512AluOp::Vpternlogd.name(), "vpternlogd");
    }

    // =========================================================================
    // MergeMode Tests
    // =========================================================================

    #[test]
    fn test_merge_mode_default() {
        assert_eq!(MergeMode::default(), MergeMode::Zeroing);
    }

    // =========================================================================
    // MaskAluOp Tests
    // =========================================================================

    #[test]
    fn test_mask_alu_op_is_unary() {
        assert!(MaskAluOp::Knot.is_unary());
        assert!(!MaskAluOp::Kand.is_unary());
        assert!(!MaskAluOp::Kor.is_unary());
        assert!(!MaskAluOp::Kxor.is_unary());
        assert!(!MaskAluOp::Kandn.is_unary());
    }

    #[test]
    fn test_mask_alu_op_vex_opcode() {
        assert_eq!(MaskAluOp::Kand.vex_opcode(), 0x41);
        assert_eq!(MaskAluOp::Kor.vex_opcode(), 0x45);
        assert_eq!(MaskAluOp::Kxor.vex_opcode(), 0x47);
        assert_eq!(MaskAluOp::Knot.vex_opcode(), 0x44);
        assert_eq!(MaskAluOp::Kandn.vex_opcode(), 0x42);
    }

    #[test]
    fn test_mask_alu_op_names() {
        assert_eq!(MaskAluOp::Kand.name(), "kandw");
        assert_eq!(MaskAluOp::Kor.name(), "korw");
        assert_eq!(MaskAluOp::Kxor.name(), "kxorw");
        assert_eq!(MaskAluOp::Knot.name(), "knotw");
        assert_eq!(MaskAluOp::Kandn.name(), "kandnw");
    }

    // =========================================================================
    // Avx512Cond Tests
    // =========================================================================

    #[test]
    fn test_x64_512_cond_imm() {
        assert_eq!(Avx512Cond::Eq.imm(), 0);
        assert_eq!(Avx512Cond::Lt.imm(), 1);
        assert_eq!(Avx512Cond::Le.imm(), 2);
        assert_eq!(Avx512Cond::Neq.imm(), 4);
        assert_eq!(Avx512Cond::Ge.imm(), 5);
        assert_eq!(Avx512Cond::Gt.imm(), 6);
    }

    #[test]
    fn test_x64_512_cond_names() {
        assert_eq!(Avx512Cond::Eq.name(), "eq");
        assert_eq!(Avx512Cond::Lt.name(), "lt");
        assert_eq!(Avx512Cond::Le.name(), "le");
        assert_eq!(Avx512Cond::Neq.name(), "neq");
        assert_eq!(Avx512Cond::Ge.name(), "ge");
        assert_eq!(Avx512Cond::Gt.name(), "gt");
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_mask_register() {
        // Valid k-registers
        for i in 0..=7 {
            assert!(validate_mask_register(i).is_ok());
        }
        // Invalid k-registers
        assert!(validate_mask_register(8).is_err());
        assert!(validate_mask_register(255).is_err());
    }

    #[test]
    fn test_validate_active_mask_register() {
        // k0 is not valid for active masking
        assert!(validate_active_mask_register(0).is_err());
        // k1-k7 are valid
        for i in 1..=7 {
            assert!(validate_active_mask_register(i).is_ok());
        }
        // Invalid k-registers
        assert!(validate_active_mask_register(8).is_err());
    }
}
