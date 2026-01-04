//! AVX-512 instruction definitions using the DSL.
//!
//! This module defines AVX-512 (512-bit SIMD) instructions using the instruction DSL,
//! replacing the manual definitions in `cranelift/codegen/src/isa/x64/inst/avx512/`.
//!
//! All instructions use EVEX encoding with 512-bit vector length (L=10b).
//!
//! ## Format Naming Convention
//!
//! Format names uniquely identify instruction operand patterns. The naming scheme:
//!
//! ### Primary Prefixes (Vector Size)
//! - `Z`  - 512-bit ZMM operations
//! - `Y`  - 256-bit YMM operations
//! - `K`  - K-mask register operations
//! - `x`, `y` - Lower-case for cross-width conversions (e.g., `xZ` = XMM to ZMM)
//!
//! ### Suffixes (Operation Type)
//! - (none)    - Binary: `zmm1 = op(zmm2, zmm3/mem)` - most common
//! - `_unary`  - Unary: `zmm1 = op(zmm2/mem)`
//! - `l`       - Load: `zmm1 = load(mem)` (register-first destination)
//! - `s`       - Store: `store(mem, zmm1)` (memory-first destination)
//! - `_km`     - With k-mask: includes mask register operand
//! - `_fma`    - Fused multiply-add: 3-source accumulator pattern
//! - `_i`      - With immediate: includes imm8 operand
//!
//! ### Specialized Patterns
//! - `Z_ternlog`    - Ternary logic (3 sources + imm8)
//! - `Z_perm2`      - Two-source permute with immediate
//! - `Z_laneshuffle`- Lane shuffle operations
//! - `Z_immrotate`  - Immediate rotate operations
//! - `Z_align`      - Alignment operations
//! - `ZC`           - Compare to k-mask output
//! - `Zk`           - Operations with k-mask as operand
//!
//! ### Examples
//! - `Z`        = `vpaddd zmm1, zmm2, zmm3/m512`
//! - `Z_unary`  = `vsqrtps zmm1, zmm2/m512`
//! - `Zl`       = `vmovaps zmm1, m512` (load)
//! - `Zs`       = `vmovaps m512, zmm1` (store)
//! - `Z_km`     = `vaddps zmm1{k1}, zmm2, zmm3/m512`
//! - `K`        = `kmovq k1, k2/m64`
//! - `ZC`       = `vpcmpd k1, zmm2, zmm3/m512, imm8`

use crate::dsl::{Feature::*, Inst, Length::*, Location::*, TupleType::*};
use crate::dsl::{evex, fmt, inst, r, rw, vex, w};

#[rustfmt::skip]
pub fn list() -> Vec<Inst> {
    vec![
        // =========================================
        // Integer Arithmetic Operations
        // =========================================

        // VPADDD - Packed 32-bit integer add
        // EVEX.512.66.0F.W0 FE /r
        inst("vpaddd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xFE).r(), (_64b | compat) & avx512f),

        // VPADDQ - Packed 64-bit integer add
        // EVEX.512.66.0F.W1 D4 /r
        inst("vpaddq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xD4).r(), (_64b | compat) & avx512f),

        // VPSUBD - Packed 32-bit integer subtract
        // EVEX.512.66.0F.W0 FA /r
        inst("vpsubd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xFA).r(), (_64b | compat) & avx512f),

        // VPSUBQ - Packed 64-bit integer subtract
        // EVEX.512.66.0F.W1 FB /r
        inst("vpsubq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xFB).r(), (_64b | compat) & avx512f),

        // VPMULLD - Packed 32-bit multiply low
        // EVEX.512.66.0F38.W0 40 /r
        inst("vpmulld", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x40).r(), (_64b | compat) & avx512f),

        // VPMULLQ - Packed 64-bit multiply low (AVX-512DQ)
        // EVEX.512.66.0F38.W1 40 /r
        inst("vpmullq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x40).r(), (_64b | compat) & avx512dq),

        // =========================================
        // Bitwise Logical Operations
        // =========================================

        // VPANDD - Packed bitwise AND (32-bit elements)
        // EVEX.512.66.0F.W0 DB /r
        inst("vpandd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xDB).r(), (_64b | compat) & avx512f),

        // VPANDQ - Packed bitwise AND (64-bit elements)
        // EVEX.512.66.0F.W1 DB /r
        inst("vpandq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xDB).r(), (_64b | compat) & avx512f),

        // VPORD - Packed bitwise OR (32-bit elements)
        // EVEX.512.66.0F.W0 EB /r
        inst("vpord", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xEB).r(), (_64b | compat) & avx512f),

        // VPORQ - Packed bitwise OR (64-bit elements)
        // EVEX.512.66.0F.W1 EB /r
        inst("vporq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xEB).r(), (_64b | compat) & avx512f),

        // VPXORD - Packed bitwise XOR (32-bit elements)
        // EVEX.512.66.0F.W0 EF /r
        inst("vpxord", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xEF).r(), (_64b | compat) & avx512f),

        // VPXORQ - Packed bitwise XOR (64-bit elements)
        // EVEX.512.66.0F.W1 EF /r
        inst("vpxorq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xEF).r(), (_64b | compat) & avx512f),

        // VPANDND - Packed bitwise AND NOT (32-bit elements)
        // EVEX.512.66.0F.W0 DF /r
        inst("vpandnd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xDF).r(), (_64b | compat) & avx512f),

        // VPANDNQ - Packed bitwise AND NOT (64-bit elements)
        // EVEX.512.66.0F.W1 DF /r
        inst("vpandnq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xDF).r(), (_64b | compat) & avx512f),

        // =========================================
        // Min/Max Operations
        // =========================================

        // VPMINSD - Packed minimum signed (32-bit)
        // EVEX.512.66.0F38.W0 39 /r
        inst("vpminsd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x39).r(), (_64b | compat) & avx512f),

        // VPMINSQ - Packed minimum signed (64-bit)
        // EVEX.512.66.0F38.W1 39 /r
        inst("vpminsq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x39).r(), (_64b | compat) & avx512f),

        // VPMAXSD - Packed maximum signed (32-bit)
        // EVEX.512.66.0F38.W0 3D /r
        inst("vpmaxsd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3D).r(), (_64b | compat) & avx512f),

        // VPMAXSQ - Packed maximum signed (64-bit)
        // EVEX.512.66.0F38.W1 3D /r
        inst("vpmaxsq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3D).r(), (_64b | compat) & avx512f),

        // VPMINUD - Packed minimum unsigned (32-bit)
        // EVEX.512.66.0F38.W0 3B /r
        inst("vpminud", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3B).r(), (_64b | compat) & avx512f),

        // VPMINUQ - Packed minimum unsigned (64-bit)
        // EVEX.512.66.0F38.W1 3B /r
        inst("vpminuq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3B).r(), (_64b | compat) & avx512f),

        // VPMAXUD - Packed maximum unsigned (32-bit)
        // EVEX.512.66.0F38.W0 3F /r
        inst("vpmaxud", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3F).r(), (_64b | compat) & avx512f),

        // VPMAXUQ - Packed maximum unsigned (64-bit)
        // EVEX.512.66.0F38.W1 3F /r
        inst("vpmaxuq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3F).r(), (_64b | compat) & avx512f),

        // =========================================
        // Floating-Point Arithmetic Operations
        // =========================================

        // VADDPS - Packed single-precision FP add
        // EVEX.512.0F.W0 58 /r
        inst("vaddps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x58).r(), (_64b | compat) & avx512f),

        // VADDPD - Packed double-precision FP add
        // EVEX.512.66.0F.W1 58 /r
        inst("vaddpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x58).r(), (_64b | compat) & avx512f),

        // VSUBPS - Packed single-precision FP subtract
        // EVEX.512.0F.W0 5C /r
        inst("vsubps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5C).r(), (_64b | compat) & avx512f),

        // VSUBPD - Packed double-precision FP subtract
        // EVEX.512.66.0F.W1 5C /r
        inst("vsubpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5C).r(), (_64b | compat) & avx512f),

        // VMULPS - Packed single-precision FP multiply
        // EVEX.512.0F.W0 59 /r
        inst("vmulps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x59).r(), (_64b | compat) & avx512f),

        // VMULPD - Packed double-precision FP multiply
        // EVEX.512.66.0F.W1 59 /r
        inst("vmulpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x59).r(), (_64b | compat) & avx512f),

        // VDIVPS - Packed single-precision FP divide
        // EVEX.512.0F.W0 5E /r
        inst("vdivps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5E).r(), (_64b | compat) & avx512f),

        // VDIVPD - Packed double-precision FP divide
        // EVEX.512.66.0F.W1 5E /r
        inst("vdivpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5E).r(), (_64b | compat) & avx512f),

        // VMINPS - Packed single-precision FP minimum
        // EVEX.512.0F.W0 5D /r
        inst("vminps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5D).r(), (_64b | compat) & avx512f),

        // VMINPD - Packed double-precision FP minimum
        // EVEX.512.66.0F.W1 5D /r
        inst("vminpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5D).r(), (_64b | compat) & avx512f),

        // VMAXPS - Packed single-precision FP maximum
        // EVEX.512.0F.W0 5F /r
        inst("vmaxps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5F).r(), (_64b | compat) & avx512f),

        // VMAXPD - Packed double-precision FP maximum
        // EVEX.512.66.0F.W1 5F /r
        inst("vmaxpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5F).r(), (_64b | compat) & avx512f),

        // =========================================
        // FP Arithmetic with Merge-Masking
        // =========================================

        // VADDPS with merge-mask
        inst("vaddps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x58).r().merge_mask(), (_64b | compat) & avx512f),
        // VADDPD with merge-mask
        inst("vaddpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x58).r().merge_mask(), (_64b | compat) & avx512f),

        // VSUBPS with merge-mask
        inst("vsubps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5C).r().merge_mask(), (_64b | compat) & avx512f),
        // VSUBPD with merge-mask
        inst("vsubpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5C).r().merge_mask(), (_64b | compat) & avx512f),

        // VMULPS with merge-mask
        inst("vmulps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x59).r().merge_mask(), (_64b | compat) & avx512f),
        // VMULPD with merge-mask
        inst("vmulpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x59).r().merge_mask(), (_64b | compat) & avx512f),

        // VDIVPS with merge-mask
        inst("vdivps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5E).r().merge_mask(), (_64b | compat) & avx512f),
        // VDIVPD with merge-mask
        inst("vdivpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5E).r().merge_mask(), (_64b | compat) & avx512f),

        // VMINPS with merge-mask
        inst("vminps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5D).r().merge_mask(), (_64b | compat) & avx512f),
        // VMINPD with merge-mask
        inst("vminpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5D).r().merge_mask(), (_64b | compat) & avx512f),

        // VMAXPS with merge-mask
        inst("vmaxps", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5F).r().merge_mask(), (_64b | compat) & avx512f),
        // VMAXPD with merge-mask
        inst("vmaxpd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5F).r().merge_mask(), (_64b | compat) & avx512f),

        // VSQRTPS - Packed single-precision FP square root
        // EVEX.512.0F.W0 51 /r
        inst("vsqrtps", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x51).r(), (_64b | compat) & avx512f),

        // VSQRTPD - Packed double-precision FP square root
        // EVEX.512.66.0F.W1 51 /r
        inst("vsqrtpd", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x51).r(), (_64b | compat) & avx512f),

        // =========================================
        // Absolute Value Operations
        // =========================================

        // VPABSD - Packed absolute value (32-bit)
        // EVEX.512.66.0F38.W0 1E /r
        inst("vpabsd", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x1E).r(), (_64b | compat) & avx512f),

        // VPABSQ - Packed absolute value (64-bit)
        // EVEX.512.66.0F38.W1 1F /r
        inst("vpabsq", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x1F).r(), (_64b | compat) & avx512f),

        // =========================================
        // Shift Operations (uniform shift amount)
        // =========================================

        // VPSLLD - Packed shift left logical (32-bit)
        // EVEX.512.66.0F.W0 F2 /r
        inst("vpslld", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w0().op(0xF2).r(), (_64b | compat) & avx512f),

        // VPSLLQ - Packed shift left logical (64-bit)
        // EVEX.512.66.0F.W1 F3 /r
        inst("vpsllq", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w1().op(0xF3).r(), (_64b | compat) & avx512f),

        // VPSRLD - Packed shift right logical (32-bit)
        // EVEX.512.66.0F.W0 D2 /r
        inst("vpsrld", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w0().op(0xD2).r(), (_64b | compat) & avx512f),

        // VPSRLQ - Packed shift right logical (64-bit)
        // EVEX.512.66.0F.W1 D3 /r
        inst("vpsrlq", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w1().op(0xD3).r(), (_64b | compat) & avx512f),

        // VPSRAD - Packed shift right arithmetic (32-bit)
        // EVEX.512.66.0F.W0 E2 /r
        inst("vpsrad", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w0().op(0xE2).r(), (_64b | compat) & avx512f),

        // VPSRAQ - Packed shift right arithmetic (64-bit)
        // EVEX.512.66.0F.W1 E2 /r
        inst("vpsraq", fmt("Z", [w(zmm1), r(zmm2), r(xmm_m128)]), evex(L512, Mem128)._66()._0f().w1().op(0xE2).r(), (_64b | compat) & avx512f),

        // =========================================
        // Variable Shift Operations (per-element shift)
        // =========================================

        // VPSLLVD - Packed shift left logical variable (32-bit)
        // EVEX.512.66.0F38.W0 47 /r
        inst("vpsllvd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x47).r(), (_64b | compat) & avx512f),

        // VPSLLVQ - Packed shift left logical variable (64-bit)
        // EVEX.512.66.0F38.W1 47 /r
        inst("vpsllvq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x47).r(), (_64b | compat) & avx512f),

        // VPSRLVD - Packed shift right logical variable (32-bit)
        // EVEX.512.66.0F38.W0 45 /r
        inst("vpsrlvd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x45).r(), (_64b | compat) & avx512f),

        // VPSRLVQ - Packed shift right logical variable (64-bit)
        // EVEX.512.66.0F38.W1 45 /r
        inst("vpsrlvq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x45).r(), (_64b | compat) & avx512f),

        // VPSRAVD - Packed shift right arithmetic variable (32-bit)
        // EVEX.512.66.0F38.W0 46 /r
        inst("vpsravd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x46).r(), (_64b | compat) & avx512f),

        // VPSRAVQ - Packed shift right arithmetic variable (64-bit)
        // EVEX.512.66.0F38.W1 46 /r
        inst("vpsravq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x46).r(), (_64b | compat) & avx512f),

        // Masked variable shift operations (Z_km format)

        // VPSLLVD with merge-mask
        inst("vpsllvd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x47).r().merge_mask(), (_64b | compat) & avx512f),
        // VPSLLVQ with merge-mask
        inst("vpsllvq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x47).r().merge_mask(), (_64b | compat) & avx512f),
        // VPSRLVD with merge-mask
        inst("vpsrlvd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x45).r().merge_mask(), (_64b | compat) & avx512f),
        // VPSRLVQ with merge-mask
        inst("vpsrlvq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x45).r().merge_mask(), (_64b | compat) & avx512f),
        // VPSRAVD with merge-mask
        inst("vpsravd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x46).r().merge_mask(), (_64b | compat) & avx512f),
        // VPSRAVQ with merge-mask
        inst("vpsravq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x46).r().merge_mask(), (_64b | compat) & avx512f),

        // =========================================
        // Rotate Operations
        // =========================================

        // VPROLVD - Packed rotate left variable (32-bit)
        // EVEX.512.66.0F38.W0 15 /r
        inst("vprolvd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x15).r(), (_64b | compat) & avx512f),

        // VPROLVQ - Packed rotate left variable (64-bit)
        // EVEX.512.66.0F38.W1 15 /r
        inst("vprolvq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x15).r(), (_64b | compat) & avx512f),

        // VPRORVD - Packed rotate right variable (32-bit)
        // EVEX.512.66.0F38.W0 14 /r
        inst("vprorvd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x14).r(), (_64b | compat) & avx512f),

        // VPRORVQ - Packed rotate right variable (64-bit)
        // EVEX.512.66.0F38.W1 14 /r
        inst("vprorvq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x14).r(), (_64b | compat) & avx512f),

        // =========================================
        // Byte/Word Operations (AVX-512BW)
        // =========================================

        // VPADDB - Packed 8-bit integer add
        // EVEX.512.66.0F.WIG FC /r
        inst("vpaddb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xFC).r(), (_64b | compat) & avx512bw),

        // VPADDW - Packed 16-bit integer add
        // EVEX.512.66.0F.WIG FD /r
        inst("vpaddw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xFD).r(), (_64b | compat) & avx512bw),

        // VPSUBB - Packed 8-bit integer subtract
        // EVEX.512.66.0F.WIG F8 /r
        inst("vpsubb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xF8).r(), (_64b | compat) & avx512bw),

        // VPSUBW - Packed 16-bit integer subtract
        // EVEX.512.66.0F.WIG F9 /r
        inst("vpsubw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xF9).r(), (_64b | compat) & avx512bw),

        // VPMINSB - Packed minimum signed (8-bit)
        // EVEX.512.66.0F38.WIG 38 /r
        inst("vpminsb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x38).r(), (_64b | compat) & avx512bw),

        // VPMINUB - Packed minimum unsigned (8-bit)
        // EVEX.512.66.0F.WIG DA /r
        inst("vpminub", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xDA).r(), (_64b | compat) & avx512bw),

        // VPMINSW - Packed minimum signed (16-bit)
        // EVEX.512.66.0F.WIG EA /r
        inst("vpminsw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xEA).r(), (_64b | compat) & avx512bw),

        // VPMINUW - Packed minimum unsigned (16-bit)
        // EVEX.512.66.0F38.WIG 3A /r
        inst("vpminuw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x3A).r(), (_64b | compat) & avx512bw),

        // VPMAXSB - Packed maximum signed (8-bit)
        // EVEX.512.66.0F38.WIG 3C /r
        inst("vpmaxsb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x3C).r(), (_64b | compat) & avx512bw),

        // VPMAXUB - Packed maximum unsigned (8-bit)
        // EVEX.512.66.0F.WIG DE /r
        inst("vpmaxub", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xDE).r(), (_64b | compat) & avx512bw),

        // VPMAXSW - Packed maximum signed (16-bit)
        // EVEX.512.66.0F.WIG EE /r
        inst("vpmaxsw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xEE).r(), (_64b | compat) & avx512bw),

        // VPMAXUW - Packed maximum unsigned (16-bit)
        // EVEX.512.66.0F38.WIG 3E /r
        inst("vpmaxuw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x3E).r(), (_64b | compat) & avx512bw),

        // VPABSB - Packed absolute value (8-bit)
        // EVEX.512.66.0F38.WIG 1C /r
        inst("vpabsb", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x1C).r(), (_64b | compat) & avx512bw),

        // VPABSW - Packed absolute value (16-bit)
        // EVEX.512.66.0F38.WIG 1D /r
        inst("vpabsw", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x1D).r(), (_64b | compat) & avx512bw),

        // VPCMPEQB - Packed compare equal (8-bit), outputs to k-mask register
        // EVEX.512.66.0F.WIG 74 /r: k1 {k2}, zmm2, zmm3/m512
        inst("vpcmpeqb", fmt("kZZ", [w(k1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x74).r(), (_64b | compat) & avx512bw),

        // VPCMPEQW - Packed compare equal (16-bit), outputs to k-mask register
        // EVEX.512.66.0F.WIG 75 /r: k1 {k2}, zmm2, zmm3/m512
        inst("vpcmpeqw", fmt("kZZ", [w(k1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x75).r(), (_64b | compat) & avx512bw),

        // VPCMPGTB - Packed compare greater than (8-bit signed), outputs to k-mask register
        // EVEX.512.66.0F.WIG 64 /r: k1 {k2}, zmm2, zmm3/m512
        inst("vpcmpgtb", fmt("kZZ", [w(k1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x64).r(), (_64b | compat) & avx512bw),

        // VPCMPGTW - Packed compare greater than (16-bit signed), outputs to k-mask register
        // EVEX.512.66.0F.WIG 65 /r: k1 {k2}, zmm2, zmm3/m512
        inst("vpcmpgtw", fmt("kZZ", [w(k1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x65).r(), (_64b | compat) & avx512bw),

        // =========================================
        // Widening Multiply Operations
        // =========================================

        // VPMULUDQ - Packed 32x32→64 unsigned multiply
        // EVEX.512.66.0F.W1 F4 /r
        inst("vpmuludq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xF4).r(), (_64b | compat) & avx512f),

        // VPMULDQ - Packed 32x32→64 signed multiply
        // EVEX.512.66.0F38.W1 28 /r
        inst("vpmuldq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x28).r(), (_64b | compat) & avx512f),

        // =========================================
        // FP Bitwise Operations
        // =========================================

        // VANDPS - Packed single-precision AND
        // EVEX.512.0F.W0 54 /r
        inst("vandps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x54).r(), (_64b | compat) & avx512dq),

        // VANDPD - Packed double-precision AND
        // EVEX.512.66.0F.W1 54 /r
        inst("vandpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x54).r(), (_64b | compat) & avx512dq),

        // VORPS - Packed single-precision OR
        // EVEX.512.0F.W0 56 /r
        inst("vorps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x56).r(), (_64b | compat) & avx512dq),

        // VORPD - Packed double-precision OR
        // EVEX.512.66.0F.W1 56 /r
        inst("vorpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x56).r(), (_64b | compat) & avx512dq),

        // VXORPS - Packed single-precision XOR
        // EVEX.512.0F.W0 57 /r
        inst("vxorps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x57).r(), (_64b | compat) & avx512dq),

        // VXORPD - Packed double-precision XOR
        // EVEX.512.66.0F.W1 57 /r
        inst("vxorpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x57).r(), (_64b | compat) & avx512dq),

        // VANDNPS - Packed single-precision AND NOT
        // EVEX.512.0F.W0 55 /r
        inst("vandnps", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x55).r(), (_64b | compat) & avx512dq),

        // VANDNPD - Packed double-precision AND NOT
        // EVEX.512.66.0F.W1 55 /r
        inst("vandnpd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x55).r(), (_64b | compat) & avx512dq),

        // =========================================
        // Broadcast Operations
        // =========================================

        // VPBROADCASTB - Broadcast 8-bit element (AVX-512BW)
        // EVEX.512.66.0F38.W0 78 /r
        inst("vpbroadcastb", fmt("Z_unary", [w(zmm1), r(xmm_m8)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x78).r(), (_64b | compat) & avx512bw),

        // VPBROADCASTW - Broadcast 16-bit element (AVX-512BW)
        // EVEX.512.66.0F38.W0 79 /r
        inst("vpbroadcastw", fmt("Z_unary", [w(zmm1), r(xmm_m16)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x79).r(), (_64b | compat) & avx512bw),

        // VPBROADCASTD - Broadcast 32-bit element
        // EVEX.512.66.0F38.W0 58 /r
        inst("vpbroadcastd", fmt("Z_unary", [w(zmm1), r(xmm_m32)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x58).r(), (_64b | compat) & avx512f),

        // VPBROADCASTQ - Broadcast 64-bit element
        // EVEX.512.66.0F38.W1 59 /r
        inst("vpbroadcastq", fmt("Z_unary", [w(zmm1), r(xmm_m64)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x59).r(), (_64b | compat) & avx512f),

        // =========================================
        // Blend Operations
        // =========================================

        // VPBLENDMD - Blend 32-bit elements using mask
        // EVEX.512.66.0F38.W0 64 /r
        inst("vpblendmd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x64).r(), (_64b | compat) & avx512f),

        // VPBLENDMQ - Blend 64-bit elements using mask
        // EVEX.512.66.0F38.W1 64 /r
        inst("vpblendmq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x64).r(), (_64b | compat) & avx512f),

        // =========================================
        // Permute Operations
        // =========================================

        // VPERMD - Permute 32-bit elements
        // EVEX.512.66.0F38.W0 36 /r
        inst("vpermd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x36).r(), (_64b | compat) & avx512f),

        // VPERMQ - Permute 64-bit elements
        // EVEX.512.66.0F38.W1 36 /r
        inst("vpermq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x36).r(), (_64b | compat) & avx512f),

        // VPERMI2D - Permute from two sources (32-bit, indices in zmm1 is read-write)
        // EVEX.512.66.0F38.W0 76 /r
        inst("vpermi2d", fmt("Z_perm2", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x76).r(), (_64b | compat) & avx512f),

        // VPERMI2Q - Permute from two sources (64-bit, indices in zmm1 is read-write)
        // EVEX.512.66.0F38.W1 76 /r
        inst("vpermi2q", fmt("Z_perm2", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x76).r(), (_64b | compat) & avx512f),

        // VPERMT2D - Permute from two sources with table (32-bit, zmm1 is read-write)
        // EVEX.512.66.0F38.W0 7E /r
        inst("vpermt2d", fmt("Z_perm2", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x7E).r(), (_64b | compat) & avx512f),

        // VPERMT2Q - Permute from two sources with table (64-bit, zmm1 is read-write)
        // EVEX.512.66.0F38.W1 7E /r
        inst("vpermt2q", fmt("Z_perm2", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x7E).r(), (_64b | compat) & avx512f),

        // =========================================
        // Population Count (AVX-512VPOPCNTDQ)
        // =========================================

        // VPOPCNTD - Population count for 32-bit elements
        // EVEX.512.66.0F38.W0 55 /r
        inst("vpopcntd", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x55).r(), (_64b | compat) & avx512vpopcntdq),

        // VPOPCNTQ - Population count for 64-bit elements
        // EVEX.512.66.0F38.W1 55 /r
        inst("vpopcntq", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x55).r(), (_64b | compat) & avx512vpopcntdq),

        // =========================================
        // Leading Zero Count / Conflict Detection (AVX-512CD)
        // =========================================

        // VPLZCNTD - Count leading zeros for 32-bit elements
        // EVEX.512.66.0F38.W0 44 /r
        inst("vplzcntd", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x44).r(), (_64b | compat) & avx512cd),

        // VPLZCNTQ - Count leading zeros for 64-bit elements
        // EVEX.512.66.0F38.W1 44 /r
        inst("vplzcntq", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x44).r(), (_64b | compat) & avx512cd),

        // VPCONFLICTD - Detect conflicts in 32-bit elements
        // EVEX.512.66.0F38.W0 C4 /r
        inst("vpconflictd", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xC4).r(), (_64b | compat) & avx512cd),

        // VPCONFLICTQ - Detect conflicts in 64-bit elements
        // EVEX.512.66.0F38.W1 C4 /r
        inst("vpconflictq", fmt("Z_unary", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xC4).r(), (_64b | compat) & avx512cd),

        // =========================================
        // Ternary Logic (powerful for complex operations)
        // =========================================

        // VPTERNLOGD - Ternary logic on 32-bit elements (tied operand: zmm1 is both input and output)
        // EVEX.512.66.0F3A.W0 25 /r ib
        inst("vpternlogd", fmt("Z_ternlog", [rw(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x25).r().ib(), (_64b | compat) & avx512f),

        // VPTERNLOGQ - Ternary logic on 64-bit elements (tied operand: zmm1 is both input and output)
        // EVEX.512.66.0F3A.W1 25 /r ib
        inst("vpternlogq", fmt("Z_ternlog", [rw(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x25).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Alignment Operations
        // =========================================

        // VALIGND - Align 32-bit elements (concatenate and shift right by imm8 dwords)
        // EVEX.512.66.0F3A.W0 03 /r ib
        inst("valignd", fmt("Z_align", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x03).r().ib(), (_64b | compat) & avx512f),

        // VALIGNQ - Align 64-bit elements (concatenate and shift right by imm8 qwords)
        // EVEX.512.66.0F3A.W1 03 /r ib
        inst("valignq", fmt("Z_align", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x03).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Lane Shuffle Operations
        // =========================================

        // VSHUFF32X4 - Shuffle 128-bit lanes of 32-bit elements
        // EVEX.512.66.0F3A.W0 23 /r ib
        inst("vshuff32x4", fmt("Z_laneshuffle", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x23).r().ib(), (_64b | compat) & avx512f),

        // VSHUFF64X2 - Shuffle 128-bit lanes of 64-bit elements
        // EVEX.512.66.0F3A.W1 23 /r ib
        inst("vshuff64x2", fmt("Z_laneshuffle", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x23).r().ib(), (_64b | compat) & avx512f),

        // VSHUFI32X4 - Shuffle 128-bit lanes of 32-bit integers
        // EVEX.512.66.0F3A.W0 43 /r ib
        inst("vshufi32x4", fmt("Z_laneshuffle", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x43).r().ib(), (_64b | compat) & avx512f),

        // VSHUFI64X2 - Shuffle 128-bit lanes of 64-bit integers
        // EVEX.512.66.0F3A.W1 43 /r ib
        inst("vshufi64x2", fmt("Z_laneshuffle", [w(zmm1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x43).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Immediate Rotate Operations
        // =========================================

        // VPROLD - Rotate left 32-bit elements by immediate
        // EVEX.512.66.0F.W0 72 /1 ib
        inst("vprold", fmt("Z_immrotate", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w0().op(0x72).digit(1).ib(), (_64b | compat) & avx512f),

        // VPROLQ - Rotate left 64-bit elements by immediate
        // EVEX.512.66.0F.W1 72 /1 ib
        inst("vprolq", fmt("Z_immrotate", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w1().op(0x72).digit(1).ib(), (_64b | compat) & avx512f),

        // VPRORD - Rotate right 32-bit elements by immediate
        // EVEX.512.66.0F.W0 72 /0 ib
        inst("vprord", fmt("Z_immrotate", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w0().op(0x72).digit(0).ib(), (_64b | compat) & avx512f),

        // VPRORQ - Rotate right 64-bit elements by immediate
        // EVEX.512.66.0F.W1 72 /0 ib
        inst("vprorq", fmt("Z_immrotate", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w1().op(0x72).digit(0).ib(), (_64b | compat) & avx512f),

        // =========================================
        // Pack/Unpack Operations
        // =========================================

        // VPACKSSDW - Pack 32-bit signed to 16-bit signed with saturation
        // EVEX.512.66.0F.W0 6B /r
        inst("vpackssdw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0x6B).r(), (_64b | compat) & avx512bw),

        // VPACKUSDW - Pack 32-bit unsigned to 16-bit unsigned with saturation
        // EVEX.512.66.0F38.W0 2B /r
        inst("vpackusdw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x2B).r(), (_64b | compat) & avx512bw),

        // VPACKSSWB - Pack 16-bit signed to 8-bit signed with saturation
        // EVEX.512.66.0F.W0 63 /r
        inst("vpacksswb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x63).r(), (_64b | compat) & avx512bw),

        // VPACKUSWB - Pack 16-bit unsigned to 8-bit unsigned with saturation
        // EVEX.512.66.0F.W0 67 /r
        inst("vpackuswb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x67).r(), (_64b | compat) & avx512bw),

        // VPUNPCKLBW - Unpack low bytes
        // EVEX.512.66.0F.WIG 60 /r
        inst("vpunpcklbw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x60).r(), (_64b | compat) & avx512bw),

        // VPUNPCKHBW - Unpack high bytes
        // EVEX.512.66.0F.WIG 68 /r
        inst("vpunpckhbw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x68).r(), (_64b | compat) & avx512bw),

        // VPUNPCKLWD - Unpack low words
        // EVEX.512.66.0F.WIG 61 /r
        inst("vpunpcklwd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x61).r(), (_64b | compat) & avx512bw),

        // VPUNPCKHWD - Unpack high words
        // EVEX.512.66.0F.WIG 69 /r
        inst("vpunpckhwd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0x69).r(), (_64b | compat) & avx512bw),

        // VPUNPCKLDQ - Unpack low doublewords
        // EVEX.512.66.0F.W0 62 /r
        inst("vpunpckldq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0x62).r(), (_64b | compat) & avx512f),

        // VPUNPCKHDQ - Unpack high doublewords
        // EVEX.512.66.0F.W0 6A /r
        inst("vpunpckhdq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0x6A).r(), (_64b | compat) & avx512f),

        // VPUNPCKLQDQ - Unpack low quadwords
        // EVEX.512.66.0F.W1 6C /r
        inst("vpunpcklqdq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x6C).r(), (_64b | compat) & avx512f),

        // VPUNPCKHQDQ - Unpack high quadwords
        // EVEX.512.66.0F.W1 6D /r
        inst("vpunpckhqdq", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x6D).r(), (_64b | compat) & avx512f),

        // =========================================
        // Shuffle Operations
        // =========================================

        // VPSHUFB - Shuffle bytes using indices
        // EVEX.512.66.0F38.WIG 00 /r
        inst("vpshufb", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x00).r(), (_64b | compat) & avx512bw),

        // VPSHUFD - Shuffle dwords using immediate
        // EVEX.512.66.0F.W0 70 /r ib
        inst("vpshufd", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w0().op(0x70).r().ib(), (_64b | compat) & avx512f),

        // VPSHUFHW - Shuffle high words
        // EVEX.512.F3.0F.WIG 70 /r ib
        inst("vpshufhw", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, FullMem)._f3()._0f().w0().op(0x70).r().ib(), (_64b | compat) & avx512bw),

        // VPSHUFLW - Shuffle low words
        // EVEX.512.F2.0F.WIG 70 /r ib
        inst("vpshuflw", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, FullMem)._f2()._0f().w0().op(0x70).r().ib(), (_64b | compat) & avx512bw),

        // =========================================
        // Multiply-Add Operations
        // =========================================

        // VPMADDWD - Multiply-add packed words to dwords
        // EVEX.512.66.0F.WIG F5 /r
        inst("vpmaddwd", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f().w0().op(0xF5).r(), (_64b | compat) & avx512bw),

        // VPMADDUBSW - Multiply-add packed unsigned/signed bytes to words
        // EVEX.512.66.0F38.WIG 04 /r
        inst("vpmaddubsw", fmt("Z", [w(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, FullMem)._66()._0f38().w0().op(0x04).r(), (_64b | compat) & avx512bw),

        // =========================================
        // FMA (Fused Multiply-Add) Operations
        // =========================================
        // Note: FMA instructions have the first operand as both read and write (tied register)

        // VFMADD213PS - Fused multiply-add single-precision (213 form)
        // EVEX.512.66.0F38.W0 A8 /r
        inst("vfmadd213ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xA8).r(), (_64b | compat) & avx512f),

        // VFMADD213PD - Fused multiply-add double-precision (213 form)
        // EVEX.512.66.0F38.W1 A8 /r
        inst("vfmadd213pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xA8).r(), (_64b | compat) & avx512f),

        // VFMADD132PS - Fused multiply-add single-precision (132 form)
        // EVEX.512.66.0F38.W0 98 /r
        inst("vfmadd132ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x98).r(), (_64b | compat) & avx512f),

        // VFMADD132PD - Fused multiply-add double-precision (132 form)
        // EVEX.512.66.0F38.W1 98 /r
        inst("vfmadd132pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x98).r(), (_64b | compat) & avx512f),

        // VFMADD231PS - Fused multiply-add single-precision (231 form)
        // EVEX.512.66.0F38.W0 B8 /r
        inst("vfmadd231ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xB8).r(), (_64b | compat) & avx512f),

        // VFMADD231PD - Fused multiply-add double-precision (231 form)
        // EVEX.512.66.0F38.W1 B8 /r
        inst("vfmadd231pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xB8).r(), (_64b | compat) & avx512f),

        // VFMSUB213PS - Fused multiply-subtract single-precision (213 form)
        // EVEX.512.66.0F38.W0 AA /r
        inst("vfmsub213ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xAA).r(), (_64b | compat) & avx512f),

        // VFMSUB213PD - Fused multiply-subtract double-precision (213 form)
        // EVEX.512.66.0F38.W1 AA /r
        inst("vfmsub213pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xAA).r(), (_64b | compat) & avx512f),

        // VFMSUB132PS - Fused multiply-subtract single-precision (132 form)
        // EVEX.512.66.0F38.W0 9A /r
        inst("vfmsub132ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x9A).r(), (_64b | compat) & avx512f),

        // VFMSUB132PD - Fused multiply-subtract double-precision (132 form)
        // EVEX.512.66.0F38.W1 9A /r
        inst("vfmsub132pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x9A).r(), (_64b | compat) & avx512f),

        // VFMSUB231PS - Fused multiply-subtract single-precision (231 form)
        // EVEX.512.66.0F38.W0 BA /r
        inst("vfmsub231ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xBA).r(), (_64b | compat) & avx512f),

        // VFMSUB231PD - Fused multiply-subtract double-precision (231 form)
        // EVEX.512.66.0F38.W1 BA /r
        inst("vfmsub231pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xBA).r(), (_64b | compat) & avx512f),

        // VFNMADD213PS - Fused negate-multiply-add single-precision (213 form)
        // EVEX.512.66.0F38.W0 AC /r
        inst("vfnmadd213ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xAC).r(), (_64b | compat) & avx512f),

        // VFNMADD213PD - Fused negate-multiply-add double-precision (213 form)
        // EVEX.512.66.0F38.W1 AC /r
        inst("vfnmadd213pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xAC).r(), (_64b | compat) & avx512f),

        // VFNMADD132PS - Fused negate-multiply-add single-precision (132 form)
        // EVEX.512.66.0F38.W0 9C /r
        inst("vfnmadd132ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x9C).r(), (_64b | compat) & avx512f),

        // VFNMADD132PD - Fused negate-multiply-add double-precision (132 form)
        // EVEX.512.66.0F38.W1 9C /r
        inst("vfnmadd132pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x9C).r(), (_64b | compat) & avx512f),

        // VFNMADD231PS - Fused negate-multiply-add single-precision (231 form)
        // EVEX.512.66.0F38.W0 BC /r
        inst("vfnmadd231ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xBC).r(), (_64b | compat) & avx512f),

        // VFNMADD231PD - Fused negate-multiply-add double-precision (231 form)
        // EVEX.512.66.0F38.W1 BC /r
        inst("vfnmadd231pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xBC).r(), (_64b | compat) & avx512f),

        // VFNMSUB213PS - Fused negate-multiply-subtract single-precision (213 form)
        // EVEX.512.66.0F38.W0 AE /r
        inst("vfnmsub213ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xAE).r(), (_64b | compat) & avx512f),

        // VFNMSUB213PD - Fused negate-multiply-subtract double-precision (213 form)
        // EVEX.512.66.0F38.W1 AE /r
        inst("vfnmsub213pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xAE).r(), (_64b | compat) & avx512f),

        // VFNMSUB132PS - Fused negate-multiply-subtract single-precision (132 form)
        // EVEX.512.66.0F38.W0 9E /r
        inst("vfnmsub132ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x9E).r(), (_64b | compat) & avx512f),

        // VFNMSUB132PD - Fused negate-multiply-subtract double-precision (132 form)
        // EVEX.512.66.0F38.W1 9E /r
        inst("vfnmsub132pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x9E).r(), (_64b | compat) & avx512f),

        // VFNMSUB231PS - Fused negate-multiply-subtract single-precision (231 form)
        // EVEX.512.66.0F38.W0 BE /r
        inst("vfnmsub231ps", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0xBE).r(), (_64b | compat) & avx512f),

        // VFNMSUB231PD - Fused negate-multiply-subtract double-precision (231 form)
        // EVEX.512.66.0F38.W1 BE /r
        inst("vfnmsub231pd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0xBE).r(), (_64b | compat) & avx512f),

        // =========================================
        // VNNI (Vector Neural Network Instructions)
        // These are dot-product accumulate operations for neural network inference
        // =========================================

        // VPDPBUSD - Multiply unsigned bytes by signed bytes and accumulate to dwords
        // dst[i] = dst[i] + dot4(src1[4i:4i+3] unsigned, src2[4i:4i+3] signed)
        // EVEX.512.66.0F38.W0 50 /r
        inst("vpdpbusd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x50).r(), (_64b | compat) & avx512vnni),

        // VPDPBUSDS - Same as VPDPBUSD but with signed saturation
        // EVEX.512.66.0F38.W0 51 /r
        inst("vpdpbusds", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x51).r(), (_64b | compat) & avx512vnni),

        // VPDPWSSD - Multiply signed words and accumulate to dwords
        // dst[i] = dst[i] + dot2(src1[2i:2i+1], src2[2i:2i+1])
        // EVEX.512.66.0F38.W0 52 /r
        inst("vpdpwssd", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x52).r(), (_64b | compat) & avx512vnni),

        // VPDPWSSDS - Same as VPDPWSSD but with signed saturation
        // EVEX.512.66.0F38.W0 53 /r
        inst("vpdpwssds", fmt("Z_fma", [rw(zmm1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x53).r(), (_64b | compat) & avx512vnni),

        // =========================================
        // Mask Register Operations (K-register)
        // Note: 66 prefix = byte, no prefix = word, W1 = dword/qword
        // =========================================

        // KANDB - Bitwise AND of mask registers (byte)
        // VEX.L1.66.0F.W0 41 /r
        inst("kandb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x41).r(), (_64b | compat) & avx512dq),

        // KANDW - Bitwise AND of mask registers (word)
        // VEX.L1.0F.W0 41 /r
        inst("kandw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x41).r(), (_64b | compat) & avx512f),

        // KANDNB - Bitwise AND NOT of mask registers (byte)
        // VEX.L1.66.0F.W0 42 /r
        inst("kandnb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x42).r(), (_64b | compat) & avx512dq),

        // KANDNW - Bitwise AND NOT of mask registers (word)
        // VEX.L1.0F.W0 42 /r
        inst("kandnw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x42).r(), (_64b | compat) & avx512f),

        // KNOTB - Bitwise NOT of mask register (byte)
        // VEX.L0.66.0F.W0 44 /r
        inst("knotb", fmt("Ku", [w(k1), r(k2)]), vex(LZ)._66()._0f().w0().op(0x44).r(), (_64b | compat) & avx512dq),

        // KNOTW - Bitwise NOT of mask register (word)
        // VEX.L0.0F.W0 44 /r
        inst("knotw", fmt("Ku", [w(k1), r(k2)]), vex(LZ)._0f().w0().op(0x44).r(), (_64b | compat) & avx512f),

        // KORB - Bitwise OR of mask registers (byte)
        // VEX.L1.66.0F.W0 45 /r
        inst("korb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x45).r(), (_64b | compat) & avx512dq),

        // KORW - Bitwise OR of mask registers (word)
        // VEX.L1.0F.W0 45 /r
        inst("korw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x45).r(), (_64b | compat) & avx512f),

        // KXORB - Bitwise XOR of mask registers (byte)
        // VEX.L1.66.0F.W0 47 /r
        inst("kxorb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x47).r(), (_64b | compat) & avx512dq),

        // KXORW - Bitwise XOR of mask registers (word)
        // VEX.L1.0F.W0 47 /r
        inst("kxorw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x47).r(), (_64b | compat) & avx512f),

        // KXNORB - Bitwise XNOR of mask registers (byte)
        // VEX.L1.66.0F.W0 46 /r
        inst("kxnorb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x46).r(), (_64b | compat) & avx512dq),

        // KXNORW - Bitwise XNOR of mask registers (word)
        // VEX.L1.0F.W0 46 /r
        inst("kxnorw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x46).r(), (_64b | compat) & avx512f),

        // KORTESTB - OR mask registers and set flags (byte)
        // VEX.L0.66.0F.W0 98 /r
        inst("kortestb", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._66()._0f().w0().op(0x98).r(), (_64b | compat) & avx512dq),

        // KORTESTW - OR mask registers and set flags (word)
        // VEX.L0.0F.W0 98 /r
        inst("kortestw", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._0f().w0().op(0x98).r(), (_64b | compat) & avx512f),

        // =========================================
        // Compare-to-Mask Operations
        // =========================================

        // VPCMPD - Compare packed signed dwords to mask
        // EVEX.512.66.0F3A.W0 1F /r ib
        inst("vpcmpd", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x1F).r().ib(), (_64b | compat) & avx512f),

        // VPCMPQ - Compare packed signed qwords to mask
        // EVEX.512.66.0F3A.W1 1F /r ib
        inst("vpcmpq", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x1F).r().ib(), (_64b | compat) & avx512f),

        // VPCMPUD - Compare packed unsigned dwords to mask
        // EVEX.512.66.0F3A.W0 1E /r ib
        inst("vpcmpud", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x1E).r().ib(), (_64b | compat) & avx512f),

        // VPCMPUQ - Compare packed unsigned qwords to mask
        // EVEX.512.66.0F3A.W1 1E /r ib
        inst("vpcmpuq", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x1E).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Load/Store Operations
        // =========================================

        // VMOVDQU32 - Move unaligned packed dwords (load)
        // EVEX.512.F3.0F.W0 6F /r
        inst("vmovdqu32", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w0().op(0x6F).r(), (_64b | compat) & avx512f),

        // VMOVDQU32 - Move unaligned packed dwords (store)
        // EVEX.512.F3.0F.W0 7F /r
        inst("vmovdqu32", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._f3()._0f().w0().op(0x7F).r(), (_64b | compat) & avx512f),

        // VMOVDQU64 - Move unaligned packed qwords (load)
        // EVEX.512.F3.0F.W1 6F /r
        inst("vmovdqu64", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w1().op(0x6F).r(), (_64b | compat) & avx512f),

        // VMOVDQU64 - Move unaligned packed qwords (store)
        // EVEX.512.F3.0F.W1 7F /r
        inst("vmovdqu64", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._f3()._0f().w1().op(0x7F).r(), (_64b | compat) & avx512f),

        // Masked load/store variants (for conditional memory operations)

        // VMOVDQU32 - Masked load dwords (zeroing mode)
        // EVEX.512.F3.0F.W0 6F /r with k-mask
        inst("vmovdqu32", fmt("Zl_km", [w(zmm1), r(k1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w0().op(0x6F).r().zero_mask(), (_64b | compat) & avx512f),

        // VMOVDQU64 - Masked load qwords (zeroing mode)
        // EVEX.512.F3.0F.W1 6F /r with k-mask
        inst("vmovdqu64", fmt("Zl_km", [w(zmm1), r(k1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w1().op(0x6F).r().zero_mask(), (_64b | compat) & avx512f),

        // VMOVDQU32 - Masked store dwords
        // EVEX.512.F3.0F.W0 7F /r with k-mask
        // Note: Stores use merge_mask (z=0) because there's no destination register to zero
        inst("vmovdqu32", fmt("Zs_km", [r(k1), w(m512), r(zmm1)]), evex(L512, Full)._f3()._0f().w0().op(0x7F).r().merge_mask(), (_64b | compat) & avx512f),

        // VMOVDQU64 - Masked store qwords
        // EVEX.512.F3.0F.W1 7F /r with k-mask
        // Note: Stores use merge_mask (z=0) because there's no destination register to zero
        inst("vmovdqu64", fmt("Zs_km", [r(k1), w(m512), r(zmm1)]), evex(L512, Full)._f3()._0f().w1().op(0x7F).r().merge_mask(), (_64b | compat) & avx512f),

        // VMOVDQU32 - Move unaligned packed dwords 256-bit (load)
        // EVEX.256.F3.0F.W0 6F /r
        inst("vmovdqu32", fmt("Yl", [w(ymm1), r(ymm_m256)]), evex(L256, Full)._f3()._0f().w0().op(0x6F).r(), (_64b | compat) & avx512f & avx512vl),

        // VMOVDQU32 - Move unaligned packed dwords 256-bit (store)
        // EVEX.256.F3.0F.W0 7F /r
        inst("vmovdqu32", fmt("Ys", [w(ymm_m256), r(ymm1)]), evex(L256, Full)._f3()._0f().w0().op(0x7F).r(), (_64b | compat) & avx512f & avx512vl),

        // VMOVDQU64 - Move unaligned packed qwords 256-bit (load)
        // EVEX.256.F3.0F.W1 6F /r
        inst("vmovdqu64", fmt("Yl", [w(ymm1), r(ymm_m256)]), evex(L256, Full)._f3()._0f().w1().op(0x6F).r(), (_64b | compat) & avx512f & avx512vl),

        // VMOVDQU64 - Move unaligned packed qwords 256-bit (store)
        // EVEX.256.F3.0F.W1 7F /r
        inst("vmovdqu64", fmt("Ys", [w(ymm_m256), r(ymm1)]), evex(L256, Full)._f3()._0f().w1().op(0x7F).r(), (_64b | compat) & avx512f & avx512vl),

        // VMOVDQU8 - Move unaligned packed bytes (load)
        // EVEX.512.F2.0F.W0 6F /r
        inst("vmovdqu8", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f2()._0f().w0().op(0x6F).r(), (_64b | compat) & avx512bw),

        // VMOVDQU8 - Move unaligned packed bytes (store)
        // EVEX.512.F2.0F.W0 7F /r
        inst("vmovdqu8", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._f2()._0f().w0().op(0x7F).r(), (_64b | compat) & avx512bw),

        // VMOVDQU16 - Move unaligned packed words (load)
        // EVEX.512.F2.0F.W1 6F /r
        inst("vmovdqu16", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f2()._0f().w1().op(0x6F).r(), (_64b | compat) & avx512bw),

        // VMOVDQU16 - Move unaligned packed words (store)
        // EVEX.512.F2.0F.W1 7F /r
        inst("vmovdqu16", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._f2()._0f().w1().op(0x7F).r(), (_64b | compat) & avx512bw),

        // VMOVDQA32 - Move aligned packed dwords (load)
        // EVEX.512.66.0F.W0 6F /r
        inst("vmovdqa32", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0x6F).r(), (_64b | compat) & avx512f),

        // VMOVDQA32 - Move aligned packed dwords (store)
        // EVEX.512.66.0F.W0 7F /r
        inst("vmovdqa32", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._66()._0f().w0().op(0x7F).r(), (_64b | compat) & avx512f),

        // VMOVDQA64 - Move aligned packed qwords (load)
        // EVEX.512.66.0F.W1 6F /r
        inst("vmovdqa64", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x6F).r(), (_64b | compat) & avx512f),

        // VMOVDQA64 - Move aligned packed qwords (store)
        // EVEX.512.66.0F.W1 7F /r
        inst("vmovdqa64", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._66()._0f().w1().op(0x7F).r(), (_64b | compat) & avx512f),

        // VMOVAPS - Move aligned packed single-precision (load)
        // EVEX.512.0F.W0 28 /r
        inst("vmovaps", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x28).r(), (_64b | compat) & avx512f),

        // VMOVAPS - Move aligned packed single-precision (store)
        // EVEX.512.0F.W0 29 /r
        inst("vmovaps", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._0f().w0().op(0x29).r(), (_64b | compat) & avx512f),

        // VMOVAPD - Move aligned packed double-precision (load)
        // EVEX.512.66.0F.W1 28 /r
        inst("vmovapd", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x28).r(), (_64b | compat) & avx512f),

        // VMOVAPD - Move aligned packed double-precision (store)
        // EVEX.512.66.0F.W1 29 /r
        inst("vmovapd", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._66()._0f().w1().op(0x29).r(), (_64b | compat) & avx512f),

        // VMOVUPS - Move unaligned packed single-precision (load)
        // EVEX.512.0F.W0 10 /r
        inst("vmovups", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x10).r(), (_64b | compat) & avx512f),

        // VMOVUPS - Move unaligned packed single-precision (store)
        // EVEX.512.0F.W0 11 /r
        inst("vmovups", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._0f().w0().op(0x11).r(), (_64b | compat) & avx512f),

        // VMOVUPD - Move unaligned packed double-precision (load)
        // EVEX.512.66.0F.W1 10 /r
        inst("vmovupd", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x10).r(), (_64b | compat) & avx512f),

        // VMOVUPD - Move unaligned packed double-precision (store)
        // EVEX.512.66.0F.W1 11 /r
        inst("vmovupd", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Full)._66()._0f().w1().op(0x11).r(), (_64b | compat) & avx512f),

        // =========================================
        // Type Conversion Operations
        // =========================================

        // VCVTDQ2PS - Convert packed i32 to packed f32
        // EVEX.512.0F.W0 5B /r
        inst("vcvtdq2ps", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._0f().w0().op(0x5B).r(), (_64b | compat) & avx512f),

        // VCVTPS2DQ - Convert packed f32 to packed i32 (round)
        // EVEX.512.66.0F.W0 5B /r
        inst("vcvtps2dq", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0x5B).r(), (_64b | compat) & avx512f),

        // VCVTTPS2DQ - Convert packed f32 to packed i32 (truncate)
        // EVEX.512.F3.0F.W0 5B /r
        inst("vcvttps2dq", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w0().op(0x5B).r(), (_64b | compat) & avx512f),

        // VCVTQQ2PD - Convert packed i64 to packed f64
        // EVEX.512.F3.0F.W1 E6 /r
        inst("vcvtqq2pd", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._f3()._0f().w1().op(0xE6).r(), (_64b | compat) & avx512dq),

        // VCVTPD2QQ - Convert packed f64 to packed i64 (round)
        // EVEX.512.66.0F.W1 7B /r
        inst("vcvtpd2qq", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x7B).r(), (_64b | compat) & avx512dq),

        // VCVTTPD2QQ - Convert packed f64 to packed i64 (truncate)
        // EVEX.512.66.0F.W1 7A /r
        inst("vcvttpd2qq", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x7A).r(), (_64b | compat) & avx512dq),

        // VCVTPS2PD - Convert packed f32 to packed f64 (8 floats -> 8 doubles = 256-bit source)
        // EVEX.512.0F.W0 5A /r (tuple type: Half since source is half-size)
        inst("vcvtps2pd", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Half)._0f().w0().op(0x5A).r(), (_64b | compat) & avx512f),

        // VCVTPD2PS - Convert packed f64 to packed f32 (8 doubles -> 8 floats = zmm -> ymm)
        // EVEX.512.66.0F.W1 5A /r
        inst("vcvtpd2ps", fmt("yZ", [w(ymm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0x5A).r(), (_64b | compat) & avx512f),

        // VCVTDQ2PD - Convert packed i32 to packed f64 (8 ints -> 8 doubles = 256-bit source)
        // EVEX.512.F3.0F.W0 E6 /r (tuple type: Half since source is half-size)
        inst("vcvtdq2pd", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Half)._f3()._0f().w0().op(0xE6).r(), (_64b | compat) & avx512f),

        // VCVTPD2DQ - Convert packed f64 to packed i32 (8 doubles -> 8 ints = zmm -> ymm)
        // EVEX.512.F2.0F.W1 E6 /r
        inst("vcvtpd2dq", fmt("yZ", [w(ymm1), r(zmm_m512)]), evex(L512, Full)._f2()._0f().w1().op(0xE6).r(), (_64b | compat) & avx512f),

        // VCVTTPD2DQ - Convert packed f64 to packed i32 with truncation (8 doubles -> 8 ints = zmm -> ymm)
        // EVEX.512.66.0F.W1 E6 /r
        inst("vcvttpd2dq", fmt("yZ", [w(ymm1), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xE6).r(), (_64b | compat) & avx512f),

        // Zero-extend operations
        // VPMOVZXBD - Zero-extend bytes to dwords
        // EVEX.512.66.0F38.W0 31 /r
        inst("vpmovzxbd", fmt("Zu", [w(zmm1), r(xmm_m128)]), evex(L512, Full)._66()._0f38().w0().op(0x31).r(), (_64b | compat) & avx512f),

        // VPMOVZXBQ - Zero-extend bytes to qwords
        // EVEX.512.66.0F38.W0 32 /r
        inst("vpmovzxbq", fmt("Zu", [w(zmm1), r(xmm_m64)]), evex(L512, Full)._66()._0f38().w0().op(0x32).r(), (_64b | compat) & avx512f),

        // VPMOVZXWD - Zero-extend words to dwords (16 words -> 16 dwords = 256-bit source)
        // EVEX.512.66.0F38.W0 33 /r
        inst("vpmovzxwd", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Full)._66()._0f38().w0().op(0x33).r(), (_64b | compat) & avx512f),

        // VPMOVZXWQ - Zero-extend words to qwords
        // EVEX.512.66.0F38.W0 34 /r
        inst("vpmovzxwq", fmt("Zu", [w(zmm1), r(xmm_m128)]), evex(L512, Full)._66()._0f38().w0().op(0x34).r(), (_64b | compat) & avx512f),

        // VPMOVZXDQ - Zero-extend dwords to qwords (8 dwords -> 8 qwords = 256-bit source)
        // EVEX.512.66.0F38.W0 35 /r
        inst("vpmovzxdq", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Full)._66()._0f38().w0().op(0x35).r(), (_64b | compat) & avx512f),

        // Sign-extend operations
        // VPMOVSXBD - Sign-extend bytes to dwords
        // EVEX.512.66.0F38.W0 21 /r
        inst("vpmovsxbd", fmt("Zu", [w(zmm1), r(xmm_m128)]), evex(L512, Full)._66()._0f38().w0().op(0x21).r(), (_64b | compat) & avx512f),

        // VPMOVSXBQ - Sign-extend bytes to qwords
        // EVEX.512.66.0F38.W0 22 /r
        inst("vpmovsxbq", fmt("Zu", [w(zmm1), r(xmm_m64)]), evex(L512, Full)._66()._0f38().w0().op(0x22).r(), (_64b | compat) & avx512f),

        // VPMOVSXWD - Sign-extend words to dwords (16 words -> 16 dwords = 256-bit source)
        // EVEX.512.66.0F38.W0 23 /r
        inst("vpmovsxwd", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Full)._66()._0f38().w0().op(0x23).r(), (_64b | compat) & avx512f),

        // VPMOVSXWQ - Sign-extend words to qwords
        // EVEX.512.66.0F38.W0 24 /r
        inst("vpmovsxwq", fmt("Zu", [w(zmm1), r(xmm_m128)]), evex(L512, Full)._66()._0f38().w0().op(0x24).r(), (_64b | compat) & avx512f),

        // VPMOVSXDQ - Sign-extend dwords to qwords (8 dwords -> 8 qwords = 256-bit source)
        // EVEX.512.66.0F38.W0 25 /r
        inst("vpmovsxdq", fmt("Zy", [w(zmm1), r(ymm_m256)]), evex(L512, Full)._66()._0f38().w0().op(0x25).r(), (_64b | compat) & avx512f),

        // Truncation operations (down-convert)
        // These take a zmm register and produce a smaller xmm/ymm result

        // Note: Truncation ops have unusual encoding where source (zmm) goes in reg field
        // and dest (xmm/ymm) goes in r/m field. These need special handling.
        // For now, keeping them out of the DSL and using manual emission.

        // =========================================
        // Compress/Expand Operations
        // =========================================

        // VPCOMPRESSD - Store sparse packed dwords
        // EVEX.512.66.0F38.W0 8B /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src - uses .mr() for swapped encoding
        inst("vpcompressd", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x8B).mr(), (_64b | compat) & avx512f),

        // VPCOMPRESSQ - Store sparse packed qwords
        // EVEX.512.66.0F38.W1 8B /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src - uses .mr() for swapped encoding
        inst("vpcompressq", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x8B).mr(), (_64b | compat) & avx512f),

        // VPEXPANDD - Load sparse packed dwords
        // EVEX.512.66.0F38.W0 89 /r
        inst("vpexpandd", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x89).r(), (_64b | compat) & avx512f),

        // VPEXPANDQ - Load sparse packed qwords
        // EVEX.512.66.0F38.W1 89 /r
        inst("vpexpandq", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x89).r(), (_64b | compat) & avx512f),

        // VCOMPRESSPS - Store sparse packed single-precision floats
        // EVEX.512.66.0F38.W0 8A /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src - uses .mr() for swapped encoding
        inst("vcompressps", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x8A).mr(), (_64b | compat) & avx512f),

        // VCOMPRESSPD - Store sparse packed double-precision floats
        // EVEX.512.66.0F38.W1 8A /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src - uses .mr() for swapped encoding
        inst("vcompresspd", fmt("Zs", [w(zmm_m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x8A).mr(), (_64b | compat) & avx512f),

        // VEXPANDPS - Load sparse packed single-precision floats
        // EVEX.512.66.0F38.W0 88 /r
        inst("vexpandps", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x88).r(), (_64b | compat) & avx512f),

        // VEXPANDPD - Load sparse packed double-precision floats
        // EVEX.512.66.0F38.W1 88 /r
        inst("vexpandpd", fmt("Zl", [w(zmm1), r(zmm_m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x88).r(), (_64b | compat) & avx512f),

        // Register-to-register compress/expand with masking
        // These use the mask to control which elements participate in compress/expand

        // VPCOMPRESSD register form with mask - compress src elements where mask=1 to consecutive dst positions
        // EVEX.512.66.0F38.W0 8B /r  (same opcode, but reg-to-reg form)
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src - uses .mr() for swapped encoding
        inst("vpcompressd", fmt("Z_km_c", [w(zmm1), r(k1), r(zmm2)]), evex(L512, Full)._66()._0f38().w0().op(0x8B).mr().zero_mask(), (_64b | compat) & avx512f),
        // VPCOMPRESSQ register form with mask
        inst("vpcompressq", fmt("Z_km_c", [w(zmm1), r(k1), r(zmm2)]), evex(L512, Full)._66()._0f38().w1().op(0x8B).mr().zero_mask(), (_64b | compat) & avx512f),

        // VPEXPANDD register form with mask - expand consecutive src elements to dst positions where mask=1
        // EVEX.512.66.0F38.W0 89 /r - uses normal .r() encoding (dst in reg, src in r/m)
        inst("vpexpandd", fmt("Z_km_e", [w(zmm1), r(k1), r(zmm2)]), evex(L512, Full)._66()._0f38().w0().op(0x89).r().zero_mask(), (_64b | compat) & avx512f),
        // VPEXPANDQ register form with mask
        inst("vpexpandq", fmt("Z_km_e", [w(zmm1), r(k1), r(zmm2)]), evex(L512, Full)._66()._0f38().w1().op(0x89).r().zero_mask(), (_64b | compat) & avx512f),

        // Memory compress/expand with masking
        // VPCOMPRESSD memory form with mask - store compressed dwords to memory
        // Note: Stores use merge_mask (z=0) because there's no destination register to zero
        inst("vpcompressd", fmt("Zs_km_c", [r(k1), w(m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x8B).r().merge_mask(), (_64b | compat) & avx512f),
        // VPCOMPRESSQ memory form with mask - store compressed qwords to memory
        // Note: Stores use merge_mask (z=0) because there's no destination register to zero
        inst("vpcompressq", fmt("Zs_km_c", [r(k1), w(m512), r(zmm1)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x8B).r().merge_mask(), (_64b | compat) & avx512f),
        // VPEXPANDD memory form with mask - load and expand dwords from memory
        inst("vpexpandd", fmt("Zl_km_e", [w(zmm1), r(k1), r(m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w0().op(0x89).r().zero_mask(), (_64b | compat) & avx512f),
        // VPEXPANDQ memory form with mask - load and expand qwords from memory
        inst("vpexpandq", fmt("Zl_km_e", [w(zmm1), r(k1), r(m512)]), evex(L512, Tuple1Scalar)._66()._0f38().w1().op(0x89).r().zero_mask(), (_64b | compat) & avx512f),

        // =========================================
        // Gather/Scatter Operations
        // Note: Removed - these require VSIB (vector index) addressing mode
        // which uses a ZMM register as the index, not supported yet
        // =========================================

        // =========================================
        // KMOV Operations (mask register moves)
        // =========================================

        // KMOVW - Move mask register (k to k)
        // VEX.L0.0F.W0 90 /r
        inst("kmovw", fmt("Kk", [w(k1), r(k2)]), vex(LZ)._0f().w0().op(0x90).r(), (_64b | compat) & avx512f),

        // KMOVB - Move mask register (k to k)
        // VEX.L0.66.0F.W0 90 /r
        inst("kmovb", fmt("Kk", [w(k1), r(k2)]), vex(LZ)._66()._0f().w0().op(0x90).r(), (_64b | compat) & avx512dq),

        // KMOVD - Move mask register (k to k)
        // VEX.L0.66.0F.W1 90 /r
        inst("kmovd", fmt("Kk", [w(k1), r(k2)]), vex(LZ)._66()._0f().w1().op(0x90).r(), (_64b | compat) & avx512bw),

        // KMOVQ - Move mask register (k to k)
        // VEX.L0.0F.W1 90 /r
        inst("kmovq", fmt("Kk", [w(k1), r(k2)]), vex(LZ)._0f().w1().op(0x90).r(), (_64b | compat) & avx512bw),

        // KMOV - Move mask register to GPR variants
        // These are essential for extracting mask bits for popcnt operations

        // KMOVW r32, k - Move 16-bit mask to GPR (zero-extended)
        // VEX.L0.0F.W0 93 /r
        inst("kmovw", fmt("Rk", [w(r32), r(k1)]), vex(LZ)._0f().w0().op(0x93).r(), (_64b | compat) & avx512f),

        // KMOVB r32, k - Move 8-bit mask to GPR (zero-extended)
        // VEX.L0.66.0F.W0 93 /r
        inst("kmovb", fmt("Rk", [w(r32), r(k1)]), vex(LZ)._66()._0f().w0().op(0x93).r(), (_64b | compat) & avx512dq),

        // KMOVD r32, k - Move 32-bit mask to GPR
        // VEX.L0.F2.0F.W0 93 /r
        inst("kmovd", fmt("Rk", [w(r32), r(k1)]), vex(LZ)._f2()._0f().w0().op(0x93).r(), (_64b | compat) & avx512bw),

        // KMOVQ r64, k - Move 64-bit mask to GPR
        // VEX.L0.F2.0F.W1 93 /r
        inst("kmovq", fmt("Rk64", [w(r64), r(k1)]), vex(LZ)._f2()._0f().w1().op(0x93).r(), (_64b | compat) & avx512bw),

        // =========================================
        // Mask Shift Operations
        // =========================================

        // KSHIFTLW - Shift mask left
        // VEX.L0.66.0F3A.W1 32 /r ib
        inst("kshiftlw", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w1().op(0x32).r().ib(), (_64b | compat) & avx512f),

        // KSHIFTRW - Shift mask right
        // VEX.L0.66.0F3A.W1 30 /r ib
        inst("kshiftrw", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w1().op(0x30).r().ib(), (_64b | compat) & avx512f),

        // KSHIFTLB - Shift mask left (byte)
        // VEX.L0.66.0F3A.W0 32 /r ib
        inst("kshiftlb", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w0().op(0x32).r().ib(), (_64b | compat) & avx512dq),

        // KSHIFTRB - Shift mask right (byte)
        // VEX.L0.66.0F3A.W0 30 /r ib
        inst("kshiftrb", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w0().op(0x30).r().ib(), (_64b | compat) & avx512dq),

        // KSHIFTLD - Shift mask left (dword)
        // VEX.L0.66.0F3A.W0 33 /r ib
        inst("kshiftld", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w0().op(0x33).r().ib(), (_64b | compat) & avx512bw),

        // KSHIFTRD - Shift mask right (dword)
        // VEX.L0.66.0F3A.W0 31 /r ib
        inst("kshiftrd", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w0().op(0x31).r().ib(), (_64b | compat) & avx512bw),

        // KSHIFTLQ - Shift mask left (qword)
        // VEX.L0.66.0F3A.W1 33 /r ib
        inst("kshiftlq", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w1().op(0x33).r().ib(), (_64b | compat) & avx512bw),

        // KSHIFTRQ - Shift mask right (qword)
        // VEX.L0.66.0F3A.W1 31 /r ib
        inst("kshiftrq", fmt("Ki", [w(k1), r(k2), r(imm8)]), vex(LZ)._66()._0f3a().w1().op(0x31).r().ib(), (_64b | compat) & avx512bw),

        // =========================================
        // Mask Unpack Operations
        // =========================================

        // KUNPCKBW - Unpack and interleave byte masks
        // VEX.L1.66.0F.W0 4B /r
        inst("kunpckbw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x4B).r(), (_64b | compat) & avx512f),

        // KUNPCKWD - Unpack and interleave word masks
        // VEX.L1.0F.W0 4B /r
        inst("kunpckwd", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x4B).r(), (_64b | compat) & avx512bw),

        // KUNPCKDQ - Unpack and interleave dword masks
        // VEX.L1.0F.W1 4B /r
        inst("kunpckdq", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w1().op(0x4B).r(), (_64b | compat) & avx512bw),

        // =========================================
        // Mask Add Operations
        // =========================================

        // KADDB - Add byte masks
        // VEX.L1.66.0F.W0 4A /r
        inst("kaddb", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w0().op(0x4A).r(), (_64b | compat) & avx512dq),

        // KADDW - Add word masks
        // VEX.L1.0F.W0 4A /r
        inst("kaddw", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w0().op(0x4A).r(), (_64b | compat) & avx512dq),

        // KADDD - Add dword masks
        // VEX.L1.66.0F.W1 4A /r
        inst("kaddd", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._66()._0f().w1().op(0x4A).r(), (_64b | compat) & avx512bw),

        // KADDQ - Add qword masks
        // VEX.L1.0F.W1 4A /r
        inst("kaddq", fmt("K", [w(k1), r(k2), r(k3)]), vex(L256)._0f().w1().op(0x4A).r(), (_64b | compat) & avx512bw),

        // =========================================
        // Mask Test Operations
        // =========================================

        // KTESTB - Test byte masks
        // VEX.L0.66.0F.W0 99 /r
        inst("ktestb", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._66()._0f().w0().op(0x99).r(), (_64b | compat) & avx512dq),

        // KTESTW - Test word masks
        // VEX.L0.0F.W0 99 /r
        inst("ktestw", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._0f().w0().op(0x99).r(), (_64b | compat) & avx512dq),

        // KTESTD - Test dword masks
        // VEX.L0.66.0F.W1 99 /r
        inst("ktestd", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._66()._0f().w1().op(0x99).r(), (_64b | compat) & avx512bw),

        // KTESTQ - Test qword masks
        // VEX.L0.0F.W1 99 /r
        inst("ktestq", fmt("Kt", [r(k1), r(k2)]), vex(LZ)._0f().w1().op(0x99).r(), (_64b | compat) & avx512bw),

        // =========================================
        // FP Compare with Immediate
        // =========================================

        // VCMPPS - Compare packed single-precision floats
        // EVEX.512.0F.W0 C2 /r ib
        inst("vcmpps", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._0f().w0().op(0xC2).r().ib(), (_64b | compat) & avx512f),

        // VCMPPD - Compare packed double-precision floats
        // EVEX.512.66.0F.W1 C2 /r ib
        inst("vcmppd", fmt("ZC", [w(k1), r(zmm2), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f().w1().op(0xC2).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Extract/Insert Lane Operations
        // =========================================

        // VEXTRACTI32X4 - Extract 128-bit from 512-bit
        // EVEX.512.66.0F3A.W0 39 /r ib
        inst("vextracti32x4", fmt("Ze", [w(xmm_m128), r(zmm1), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x39).r().ib(), (_64b | compat) & avx512f),

        // VEXTRACTI64X2 - Extract 128-bit from 512-bit (qword)
        // EVEX.512.66.0F3A.W1 39 /r ib
        inst("vextracti64x2", fmt("Ze", [w(xmm_m128), r(zmm1), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x39).r().ib(), (_64b | compat) & avx512dq),

        // VEXTRACTI32X8 - Extract 256-bit from 512-bit
        // EVEX.512.66.0F3A.W0 3B /r ib
        inst("vextracti32x8", fmt("Ze8", [w(ymm_m256), r(zmm1), r(imm8)]), evex(L512, Tuple8)._66()._0f3a().w0().op(0x3B).r().ib(), (_64b | compat) & avx512dq),

        // VEXTRACTI64X4 - Extract 256-bit from 512-bit (qword)
        // EVEX.512.66.0F3A.W1 3B /r ib
        inst("vextracti64x4", fmt("Ze4", [w(ymm_m256), r(zmm1), r(imm8)]), evex(L512, Tuple4)._66()._0f3a().w1().op(0x3B).r().ib(), (_64b | compat) & avx512f),

        // VINSERTI32X4 - Insert 128-bit into 512-bit
        // EVEX.512.66.0F3A.W0 38 /r ib
        inst("vinserti32x4", fmt("Zi", [w(zmm1), r(zmm2), r(xmm_m128), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x38).r().ib(), (_64b | compat) & avx512f),

        // VINSERTI64X2 - Insert 128-bit into 512-bit (qword)
        // EVEX.512.66.0F3A.W1 38 /r ib
        inst("vinserti64x2", fmt("Zi", [w(zmm1), r(zmm2), r(xmm_m128), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x38).r().ib(), (_64b | compat) & avx512dq),

        // VINSERTI32X8 - Insert 256-bit into 512-bit
        // EVEX.512.66.0F3A.W0 3A /r ib
        inst("vinserti32x8", fmt("Zi8", [w(zmm1), r(zmm2), r(ymm_m256), r(imm8)]), evex(L512, Tuple8)._66()._0f3a().w0().op(0x3A).r().ib(), (_64b | compat) & avx512dq),

        // VINSERTI64X4 - Insert 256-bit into 512-bit (qword)
        // EVEX.512.66.0F3A.W1 3A /r ib
        inst("vinserti64x4", fmt("Zi4", [w(zmm1), r(zmm2), r(ymm_m256), r(imm8)]), evex(L512, Tuple4)._66()._0f3a().w1().op(0x3A).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // FP Extract/Insert
        // =========================================

        // VEXTRACTF32X4 - Extract 128-bit FP from 512-bit
        // EVEX.512.66.0F3A.W0 19 /r ib
        inst("vextractf32x4", fmt("Ze", [w(xmm_m128), r(zmm1), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x19).r().ib(), (_64b | compat) & avx512f),

        // VEXTRACTF64X2 - Extract 128-bit FP from 512-bit (qword)
        // EVEX.512.66.0F3A.W1 19 /r ib
        inst("vextractf64x2", fmt("Ze", [w(xmm_m128), r(zmm1), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x19).r().ib(), (_64b | compat) & avx512dq),

        // VEXTRACTF32X8 - Extract 256-bit FP from 512-bit
        // EVEX.512.66.0F3A.W0 1B /r ib
        inst("vextractf32x8", fmt("Ze8", [w(ymm_m256), r(zmm1), r(imm8)]), evex(L512, Tuple8)._66()._0f3a().w0().op(0x1B).r().ib(), (_64b | compat) & avx512dq),

        // VEXTRACTF64X4 - Extract 256-bit FP from 512-bit (qword)
        // EVEX.512.66.0F3A.W1 1B /r ib
        inst("vextractf64x4", fmt("Ze4", [w(ymm_m256), r(zmm1), r(imm8)]), evex(L512, Tuple4)._66()._0f3a().w1().op(0x1B).r().ib(), (_64b | compat) & avx512f),

        // VINSERTF32X4 - Insert 128-bit FP into 512-bit
        // EVEX.512.66.0F3A.W0 18 /r ib
        inst("vinsertf32x4", fmt("Zi", [w(zmm1), r(zmm2), r(xmm_m128), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x18).r().ib(), (_64b | compat) & avx512f),

        // VINSERTF64X2 - Insert 128-bit FP into 512-bit (qword)
        // EVEX.512.66.0F3A.W1 18 /r ib
        inst("vinsertf64x2", fmt("Zi", [w(zmm1), r(zmm2), r(xmm_m128), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x18).r().ib(), (_64b | compat) & avx512dq),

        // VINSERTF32X8 - Insert 256-bit FP into 512-bit
        // EVEX.512.66.0F3A.W0 1A /r ib
        inst("vinsertf32x8", fmt("Zi8", [w(zmm1), r(zmm2), r(ymm_m256), r(imm8)]), evex(L512, Tuple8)._66()._0f3a().w0().op(0x1A).r().ib(), (_64b | compat) & avx512dq),

        // VINSERTF64X4 - Insert 256-bit FP into 512-bit (qword)
        // EVEX.512.66.0F3A.W1 1A /r ib
        inst("vinsertf64x4", fmt("Zi4", [w(zmm1), r(zmm2), r(ymm_m256), r(imm8)]), evex(L512, Tuple4)._66()._0f3a().w1().op(0x1A).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // FP Special Operations
        // =========================================

        // VRCP14PS - Reciprocal approximation (single)
        // EVEX.512.66.0F38.W0 4C /r
        inst("vrcp14ps", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x4C).r(), (_64b | compat) & avx512f),

        // VRCP14PD - Reciprocal approximation (double)
        // EVEX.512.66.0F38.W1 4C /r
        inst("vrcp14pd", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x4C).r(), (_64b | compat) & avx512f),

        // VRSQRT14PS - Reciprocal square root approximation (single)
        // EVEX.512.66.0F38.W0 4E /r
        inst("vrsqrt14ps", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x4E).r(), (_64b | compat) & avx512f),

        // VRSQRT14PD - Reciprocal square root approximation (double)
        // EVEX.512.66.0F38.W1 4E /r
        inst("vrsqrt14pd", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x4E).r(), (_64b | compat) & avx512f),

        // VGETEXPPS - Get exponent (single)
        // EVEX.512.66.0F38.W0 42 /r
        inst("vgetexpps", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x42).r(), (_64b | compat) & avx512f),

        // VGETEXPPD - Get exponent (double)
        // EVEX.512.66.0F38.W1 42 /r
        inst("vgetexppd", fmt("Zu", [w(zmm1), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x42).r(), (_64b | compat) & avx512f),

        // VGETMANTPS - Get mantissa (single)
        // EVEX.512.66.0F3A.W0 26 /r ib
        inst("vgetmantps", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x26).r().ib(), (_64b | compat) & avx512f),

        // VGETMANTPD - Get mantissa (double)
        // EVEX.512.66.0F3A.W1 26 /r ib
        inst("vgetmantpd", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x26).r().ib(), (_64b | compat) & avx512f),

        // VRNDSCALEPS - Round to int with scale (single)
        // EVEX.512.66.0F3A.W0 08 /r ib
        inst("vrndscaleps", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w0().op(0x08).r().ib(), (_64b | compat) & avx512f),

        // VRNDSCALEPD - Round to int with scale (double)
        // EVEX.512.66.0F3A.W1 09 /r ib
        inst("vrndscalepd", fmt("Z_i", [w(zmm1), r(zmm_m512), r(imm8)]), evex(L512, Full)._66()._0f3a().w1().op(0x09).r().ib(), (_64b | compat) & avx512f),

        // =========================================
        // Mask to Vector / Vector to Mask
        // =========================================

        // VPMOVM2D - Convert mask to dword vector (all 1s or all 0s per element)
        // EVEX.512.F3.0F38.W0 38 /r
        inst("vpmovm2d", fmt("Zk", [w(zmm1), r(k1)]), evex(L512, Full)._f3()._0f38().w0().op(0x38).r(), (_64b | compat) & avx512dq),

        // VPMOVM2Q - Convert mask to qword vector
        // EVEX.512.F3.0F38.W1 38 /r
        inst("vpmovm2q", fmt("Zk", [w(zmm1), r(k1)]), evex(L512, Full)._f3()._0f38().w1().op(0x38).r(), (_64b | compat) & avx512dq),

        // VPMOVM2B - Convert mask to byte vector
        // EVEX.512.F3.0F38.W0 28 /r
        inst("vpmovm2b", fmt("Zk", [w(zmm1), r(k1)]), evex(L512, Full)._f3()._0f38().w0().op(0x28).r(), (_64b | compat) & avx512bw),

        // VPMOVM2W - Convert mask to word vector
        // EVEX.512.F3.0F38.W1 28 /r
        inst("vpmovm2w", fmt("Zk", [w(zmm1), r(k1)]), evex(L512, Full)._f3()._0f38().w1().op(0x28).r(), (_64b | compat) & avx512bw),

        // VPMOVD2M - Convert dword vector to mask (test sign bit)
        // EVEX.512.F3.0F38.W0 39 /r
        inst("vpmovd2m", fmt("kZ", [w(k1), r(zmm1)]), evex(L512, Full)._f3()._0f38().w0().op(0x39).r(), (_64b | compat) & avx512dq),

        // VPMOVQ2M - Convert qword vector to mask
        // EVEX.512.F3.0F38.W1 39 /r
        inst("vpmovq2m", fmt("kZ", [w(k1), r(zmm1)]), evex(L512, Full)._f3()._0f38().w1().op(0x39).r(), (_64b | compat) & avx512dq),

        // VPMOVB2M - Convert byte vector to mask
        // EVEX.512.F3.0F38.W0 29 /r
        inst("vpmovb2m", fmt("kZ", [w(k1), r(zmm1)]), evex(L512, Full)._f3()._0f38().w0().op(0x29).r(), (_64b | compat) & avx512bw),

        // VPMOVW2M - Convert word vector to mask
        // EVEX.512.F3.0F38.W1 29 /r
        inst("vpmovw2m", fmt("kZ", [w(k1), r(zmm1)]), evex(L512, Full)._f3()._0f38().w1().op(0x29).r(), (_64b | compat) & avx512bw),

        // =========================================
        // VP2INTERSECT Operations
        // Note: Removed - these require AVX512VP2INTERSECT feature
        // and capstone may not support decoding them
        // =========================================

        // =========================================
        // Broadcast from GPR
        // =========================================

        // Note: VPBROADCASTD/Q from GPR variants removed - code gen doesn't support GPR with EVEX encoding

        // =========================================
        // Masked Arithmetic Operations (merge-masking)
        // These use {k} merge-masking where unselected elements retain their values
        // Format: dest{k} = op(src1, src2) where k selects which elements are written
        // =========================================

        // VPADDD with merge-mask - Packed 32-bit integer add with mask
        // EVEX.512.66.0F.W0 FE /r {k}
        inst("vpaddd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xFE).r().merge_mask(), (_64b | compat) & avx512f),

        // VPADDQ with merge-mask - Packed 64-bit integer add with mask
        // EVEX.512.66.0F.W1 D4 /r {k}
        inst("vpaddq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xD4).r().merge_mask(), (_64b | compat) & avx512f),

        // VPSUBD with merge-mask - Packed 32-bit integer subtract with mask
        // EVEX.512.66.0F.W0 FA /r {k}
        inst("vpsubd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xFA).r().merge_mask(), (_64b | compat) & avx512f),

        // VPSUBQ with merge-mask - Packed 64-bit integer subtract with mask
        // EVEX.512.66.0F.W1 FB /r {k}
        inst("vpsubq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xFB).r().merge_mask(), (_64b | compat) & avx512f),

        // VPANDD with merge-mask - Packed bitwise AND (32-bit) with mask
        // EVEX.512.66.0F.W0 DB /r {k}
        inst("vpandd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xDB).r().merge_mask(), (_64b | compat) & avx512f),

        // VPANDQ with merge-mask - Packed bitwise AND (64-bit) with mask
        // EVEX.512.66.0F.W1 DB /r {k}
        inst("vpandq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xDB).r().merge_mask(), (_64b | compat) & avx512f),

        // VPORD with merge-mask - Packed bitwise OR (32-bit) with mask
        // EVEX.512.66.0F.W0 EB /r {k}
        inst("vpord", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xEB).r().merge_mask(), (_64b | compat) & avx512f),

        // VPORQ with merge-mask - Packed bitwise OR (64-bit) with mask
        // EVEX.512.66.0F.W1 EB /r {k}
        inst("vporq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xEB).r().merge_mask(), (_64b | compat) & avx512f),

        // VPXORD with merge-mask - Packed bitwise XOR (32-bit) with mask
        // EVEX.512.66.0F.W0 EF /r {k}
        inst("vpxord", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xEF).r().merge_mask(), (_64b | compat) & avx512f),

        // VPXORQ with merge-mask - Packed bitwise XOR (64-bit) with mask
        // EVEX.512.66.0F.W1 EF /r {k}
        inst("vpxorq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xEF).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMULLD with merge-mask - Packed 32-bit integer multiply with mask
        // EVEX.512.66.0F38.W0 40 /r {k}
        inst("vpmulld", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x40).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMULLQ with merge-mask - Packed 64-bit integer multiply with mask
        // EVEX.512.66.0F38.W1 40 /r {k}
        inst("vpmullq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x40).r().merge_mask(), (_64b | compat) & avx512dq),

        // VPANDND with merge-mask - Packed bitwise AND NOT (32-bit) with mask
        // EVEX.512.66.0F.W0 DF /r {k}
        inst("vpandnd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w0().op(0xDF).r().merge_mask(), (_64b | compat) & avx512f),

        // VPANDNQ with merge-mask - Packed bitwise AND NOT (64-bit) with mask
        // EVEX.512.66.0F.W1 DF /r {k}
        inst("vpandnq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f().w1().op(0xDF).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMINSD with merge-mask - Packed signed 32-bit minimum with mask
        // EVEX.512.66.0F38.W0 39 /r {k}
        inst("vpminsd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x39).r().merge_mask(), (_64b | compat) & avx512f),
        // VPMINSQ with merge-mask
        inst("vpminsq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x39).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMINUD with merge-mask - Packed unsigned 32-bit minimum with mask
        // EVEX.512.66.0F38.W0 3B /r {k}
        inst("vpminud", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3B).r().merge_mask(), (_64b | compat) & avx512f),
        // VPMINUQ with merge-mask
        inst("vpminuq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3B).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMAXSD with merge-mask - Packed signed 32-bit maximum with mask
        // EVEX.512.66.0F38.W0 3D /r {k}
        inst("vpmaxsd", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3D).r().merge_mask(), (_64b | compat) & avx512f),
        // VPMAXSQ with merge-mask
        inst("vpmaxsq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3D).r().merge_mask(), (_64b | compat) & avx512f),

        // VPMAXUD with merge-mask - Packed unsigned 32-bit maximum with mask
        // EVEX.512.66.0F38.W0 3F /r {k}
        inst("vpmaxud", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x3F).r().merge_mask(), (_64b | compat) & avx512f),
        // VPMAXUQ with merge-mask
        inst("vpmaxuq", fmt("Z_km", [rw(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x3F).r().merge_mask(), (_64b | compat) & avx512f),

        // =========================================
        // Blend Operations (select based on mask)
        // =========================================
        //
        // VPBLENDMD/Q: Conditional blend using k-register mask
        // dst[i] = mask[i] ? src2[i] : src1[i]
        // Note: src1 provides the "else" value (when mask=0)
        //       src2 provides the "then" value (when mask=1)

        // VPBLENDMD - Blend 32-bit integers based on mask
        // EVEX.512.66.0F38.W0 64 /r {k}
        inst("vpblendmd", fmt("Z_km", [w(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w0().op(0x64).r().merge_mask(), (_64b | compat) & avx512f),

        // VPBLENDMQ - Blend 64-bit integers based on mask
        // EVEX.512.66.0F38.W1 64 /r {k}
        inst("vpblendmq", fmt("Z_km", [w(zmm1), r(k1), r(zmm2), r(zmm_m512)]), evex(L512, Full)._66()._0f38().w1().op(0x64).r().merge_mask(), (_64b | compat) & avx512f),

        // =========================================
        // Truncation Operations (no saturation)
        // =========================================
        //
        // These pack/truncate larger elements into smaller ones.
        // The output is smaller than the input (zmm → xmm or ymm).

        // VPMOVDB - Truncate dwords (32-bit) to bytes (8-bit)
        // zmm source → xmm destination (16 dwords → 16 bytes)
        // EVEX.512.F3.0F38.W0 31 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        // Uses .mr() because destination goes in r/m, source in reg (opposite of normal)
        inst("vpmovdb", fmt("xZ", [w(xmm1), r(zmm1)]), evex(L512, QuarterMem)._f3()._0f38().w0().op(0x31).mr(), (_64b | compat) & avx512f),

        // VPMOVDW - Truncate dwords (32-bit) to words (16-bit)
        // zmm source → ymm destination (16 dwords → 16 words)
        // EVEX.512.F3.0F38.W0 33 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovdw", fmt("yZ", [w(ymm1), r(zmm1)]), evex(L512, HalfMem)._f3()._0f38().w0().op(0x33).mr(), (_64b | compat) & avx512f),

        // VPMOVQB - Truncate qwords (64-bit) to bytes (8-bit)
        // zmm source → xmm destination (8 qwords → 8 bytes, only low 64 bits used)
        // EVEX.512.F3.0F38.W0 32 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovqb", fmt("xZ", [w(xmm1), r(zmm1)]), evex(L512, EigthMem)._f3()._0f38().w0().op(0x32).mr(), (_64b | compat) & avx512f),

        // VPMOVQD - Truncate qwords (64-bit) to dwords (32-bit)
        // zmm source → ymm destination (8 qwords → 8 dwords)
        // EVEX.512.F3.0F38.W0 35 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovqd", fmt("yZ", [w(ymm1), r(zmm1)]), evex(L512, HalfMem)._f3()._0f38().w0().op(0x35).mr(), (_64b | compat) & avx512f),

        // VPMOVQW - Truncate qwords (64-bit) to words (16-bit)
        // zmm source → xmm destination (8 qwords → 8 words, only low 128 bits used)
        // EVEX.512.F3.0F38.W0 34 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovqw", fmt("xZ", [w(xmm1), r(zmm1)]), evex(L512, QuarterMem)._f3()._0f38().w0().op(0x34).mr(), (_64b | compat) & avx512f),

        // VPMOVWB - Truncate words (16-bit) to bytes (8-bit)
        // zmm source → ymm destination (32 words → 32 bytes)
        // EVEX.512.F3.0F38.W0 30 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovwb", fmt("yZ", [w(ymm1), r(zmm1)]), evex(L512, HalfMem)._f3()._0f38().w0().op(0x30).mr(), (_64b | compat) & avx512bw),

        // =========================================
        // Truncation with Signed Saturation
        // =========================================

        // VPMOVSDB - Truncate dwords to bytes with signed saturation
        // EVEX.512.F3.0F38.W0 21 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovsdb", fmt("xZ", [w(xmm1), r(zmm1)]), evex(L512, QuarterMem)._f3()._0f38().w0().op(0x21).mr(), (_64b | compat) & avx512f),

        // VPMOVSQD - Truncate qwords to dwords with signed saturation
        // EVEX.512.F3.0F38.W0 25 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovsqd", fmt("yZ", [w(ymm1), r(zmm1)]), evex(L512, HalfMem)._f3()._0f38().w0().op(0x25).mr(), (_64b | compat) & avx512f),

        // =========================================
        // Truncation with Unsigned Saturation
        // =========================================

        // VPMOVUSDB - Truncate dwords to bytes with unsigned saturation
        // EVEX.512.F3.0F38.W0 11 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovusdb", fmt("xZ", [w(xmm1), r(zmm1)]), evex(L512, QuarterMem)._f3()._0f38().w0().op(0x11).mr(), (_64b | compat) & avx512f),

        // VPMOVUSQD - Truncate qwords to dwords with unsigned saturation
        // EVEX.512.F3.0F38.W0 15 /r
        // Op/En A: ModRM:r/m (w) = dest, ModRM:reg (r) = src
        inst("vpmovusqd", fmt("yZ", [w(ymm1), r(zmm1)]), evex(L512, HalfMem)._f3()._0f38().w0().op(0x15).mr(), (_64b | compat) & avx512f),
    ]
}
