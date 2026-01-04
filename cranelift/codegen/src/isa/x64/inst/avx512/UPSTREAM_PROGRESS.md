# AVX-512 Upstream Readiness Progress

## Status: IN PROGRESS

Last Updated: 2025-12-29

---

## Critical Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| C1 | Missing ZMM16-31 registers | DONE | Fully implemented! 32 registers now available |
| C2 | Vendor-specific branding | DONE | Removed Turin/Zen 5/AMD EPYC from avx512.isle + tests |
| C3 | Missing CLIF filetests | DONE | Added avx512.clif with 17 compile tests |
| C4 | Platform gating missing | DONE | Added #![cfg(target_arch)] to all 3 test files |
| C5 | Operand corrections | DONE | Fixed VCVTPD2PS/VCVTPD2DQ/VCVTTPD2DQ to use zmm_m512 |

## High Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| H1 | Missing unsigned comparisons | DONE | Added VPCMPUD/VPCMPUQ support for I32X16/I64X8 |
| H2 | Missing AVX-512DQ guard | DONE | Added has_avx512dq to vextracti64x2/vinserti64x2 rules |
| H3 | Rule formatting | DONE | Fixed 246 rules (rule 17/18 spacing) |
| H4 | VCVTPS2PD tuple type | DONE | Fixed to Half, also fixed VCVTDQ2PD |
| H5 | Move benchmarks | DONE | Already in correct location (cranelift/jit/tests/) |
| H6 | Missing edge case tests | DONE | Added 16 edge case tests for boundary values, special FP values |

## Medium Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| M1 | Run cargo fmt | DONE | Formatting fixed |
| M2 | Add #[must_use] | DONE | Added to Kmask new/enc/to_string |
| M3 | Document priority scheme | DONE | Already in avx512.isle header |
| M4 | K-register allocation | DEFER | Complex - future work |
| M5 | Missing instructions | DONE | Added VMOVAPS/VMOVAPD/VMOVUPS/VMOVUPD (512-bit) |
| M6 | Remove println in tests | DONE | Removed redundant PASS/PASSED messages |

## Low Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| L1 | Format name consolidation | DONE | Added comprehensive docs to avx512.rs header |
| L2 | VL variants | DEFER | 128/256-bit EVEX - future work |
| L3 | Gather/scatter support | DEFER | Requires VSIB addressing mode - not in DSL |

---

## Test Results

| Test Suite | Status | Count |
|------------|--------|-------|
| cranelift-assembler-x64 | ✅ PASS | 16 tests |
| cranelift-codegen | ✅ PASS | 218 tests |
| avx512_sql_ops | ✅ PASS | 22 tests |
| avx512_e2e | ✅ PASS | 119 tests |
| avx512_perf | ✅ PASS | 8 tests |

---

## Change Log

### 2025-12-28
- Created tracking document
- C1: **FULLY IMPLEMENTED ZMM16-31 register support!**
  - Added XMM16-31 encoding constants to xmm.rs
  - Added xmm16()-xmm31() register constructors to regs.rs
  - Added `enable_simd32` shared setting to conditionally expose registers
  - Updated abi.rs to add xmm16-31 to allocatable pool when enabled
  - This doubles the available SIMD registers for AVX-512 code!
- C2: Removed vendor branding (Turin/Zen5/AMD EPYC) from avx512.isle and tests
- C4: Added platform gating (#![cfg(target_arch)]) to all 3 test files
- C5: Fixed VCVTPD2PS/VCVTPD2DQ/VCVTTPD2DQ to use zmm_m512 for memory operands
- H2: Added has_avx512dq guard to vextracti64x2/vinserti64x2 ISLE rules
- H3: Fixed 246 ISLE rules with proper spacing (rule 17 vs rule 17)
- H4: Fixed VCVTPS2PD and VCVTDQ2PD tuple type from Full to Half
- M1: Ran cargo fmt
- M2: Added #[must_use] to Kmask methods
- Fixed fuzz test vcmpps/vcmpngeps/vpcmp mnemonic comparison issues
- All tests passing (265+ total)

### 2025-12-28 (cont'd)
- Fixed OnceLock static naming in abi.rs (PINNED_SIMD32_ENV, PINNED_ENV, SIMD32_ENV, DEFAULT_ENV)
- C3: Added CLIF filetests at cranelift/filetests/filetests/isa/x64/avx512.clif
  - 17 functions testing: iadd, isub, imul, band, bor, bxor, fadd, fsub, fmul, fdiv, sqrt, ineg
  - Covers i64x8, i32x16, f64x8, f32x16 types

### 2025-12-29
- H1: Added unsigned integer comparisons (VPCMPUD/VPCMPUQ)
  - Added x64_512_vpcmpud/vpcmpuq wrappers in inst.isle
  - Added specific helpers: vpcmpultd, vpcmpuled, vpcmpugtd, vpcmpuged (32-bit)
  - Added specific helpers: vpcmpultq, vpcmpuleq, vpcmpugtq, vpcmpugeq (64-bit)
  - Added lowering rules for UnsignedGreaterThan, UnsignedLessThan, etc.
- H5: Confirmed benchmarks already in correct location (cranelift/jit/tests/)
- M5: Added floating-point move instructions
  - VMOVAPS/VMOVAPD (aligned, 512-bit)
  - VMOVUPS/VMOVUPD (unaligned, 512-bit)
- M6: Cleaned up test output
  - Removed 109 redundant "PASSED" println statements from avx512_e2e.rs
  - Removed "PASS" statements from avx512_sql_ops.rs
  - Kept informational "Skipping" and benchmark output messages
- H6: Added comprehensive edge case tests to avx512_e2e.rs
  - Integer boundary tests: MIN/MAX values, wraparound behavior
  - Float special values: NaN, Infinity, -0.0, subnormals
  - Division edge cases: div by zero, 0/0, Inf/Inf
  - sqrt edge cases: sqrt(0), sqrt(-1), sqrt(Inf)
  - Bitwise pattern tests: all-ones, all-zeros, x&x, x^x
  - Negation overflow: -MIN = MIN for signed integers
  - Signed/unsigned comparison boundary tests
  - Float comparison with NaN (unordered)
  - Lane pattern tests (alternating, first/last only)
- L1: Added format naming documentation to avx512.rs
  - Documented primary prefixes: Z (512-bit), Y (256-bit), K (k-mask), x/y (cross-width)
  - Documented suffixes: _unary, l/s (load/store), _km (masked), _fma, _i (immediate)
  - Documented specialized patterns: Z_ternlog, Z_perm2, ZC (compare), etc.
- L3: Reviewed gather/scatter status
  - VSIB addressing mode (vector index) not yet supported in DSL
  - Instructions like VPGATHERD/VPSCATTERD need this first
  - Appropriately deferred until DSL supports VSIB

### 2025-12-29 (Quality Fixes)
- Fixed assembler test flakiness caused by incorrect tuple types
  - Implemented `Tuple4` tuple type for 256-bit extract/insert instructions
  - Fixed 8 instructions using wrong tuple type (`Full` → `Tuple4`/`Tuple8`):
    - vextractf64x4, vextracti64x4, vinsertf64x4, vinserti64x4 → Tuple4
    - vextractf32x8, vextracti32x8, vinsertf32x8, vinserti32x8 → Tuple8
  - Fixed compress/expand instructions to use `Tuple1Scalar` (12 instruction forms)
- Changed `unimplemented!()` to `unreachable!()` in emit.rs FP special ops
- Removed remaining vendor branding from defs.rs and emit.rs
- Added comprehensive emit.rs documentation (module overview, K-register encoding, MergeMode)
- Enhanced K-register allocation documentation in inst.isle
- Created README.md for avx512 module with architecture overview

---

## AVX-512 Feature Matrix

### Supported Vector Types

| Type | Lanes | Element | Width | Status |
|------|-------|---------|-------|--------|
| I8X64 | 64 | i8 | 512-bit | ✅ Full |
| I16X32 | 32 | i16 | 512-bit | ✅ Full |
| I32X16 | 16 | i32 | 512-bit | ✅ Full |
| I64X8 | 8 | i64 | 512-bit | ✅ Full |
| F32X16 | 16 | f32 | 512-bit | ✅ Full |
| F64X8 | 8 | f64 | 512-bit | ✅ Full |

### Instruction Coverage

| Category | Instructions | Status |
|----------|-------------|--------|
| Integer ALU | VPADD/SUB/MULL (B/W/D/Q) | ✅ Complete |
| Integer Compare | VPCMP (signed/unsigned) | ✅ Complete |
| Integer Min/Max | VPMIN/MAX (signed/unsigned) | ✅ Complete |
| Bitwise | VPAND/OR/XOR/ANDN | ✅ Complete |
| Shifts | VPSLL/SRL/SRA (D/Q) | ✅ Complete |
| FP ALU | VADD/SUB/MUL/DIV (PS/PD) | ✅ Complete |
| FP Compare | VCMPPS/VCMPPD | ✅ Complete |
| FP Min/Max | VMINPS/VMAXPS/etc | ✅ Complete |
| FP Special | VSQRT, VRCP14, VRSQRT14 | ✅ Complete |
| FMA | VFMADD/SUB 132/213/231 | ✅ Complete |
| Conversions | VCVT (int↔fp, precision) | ✅ Complete |
| Shuffles | VPSHUFB, VPERMD, etc | ✅ Complete |
| Insert/Extract | VEXTRACTI/VINSERTI | ✅ Complete |
| K-mask ALU | KAND/OR/XOR/NOT | ✅ Complete |
| Compress/Expand | VPCOMPRESS/VPEXPAND | ✅ Complete |
| Conflict Detection | VPCONFLICT | ✅ Complete |
| Gather/Scatter | VPGATHER/VPSCATTER | ⏸️ Deferred |
| VL variants | 128/256-bit EVEX | ⏸️ Deferred |

### Required CPU Features

| Feature | Purpose | Required For |
|---------|---------|--------------|
| AVX-512F | Foundation | All 512-bit ops |
| AVX-512BW | Byte/Word | I8X64, I16X32 ops |
| AVX-512DQ | DQ extensions | VPMULLQ, VEXTRACTI64X2 |
| AVX-512VL | VL variants | Future 128/256-bit work |
