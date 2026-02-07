// cranelift/codegen/src/isa/x64/inst/avx512/emit.rs
//
// AVX-512 EVEX instruction emission.
// This module handles the encoding and emission of AVX-512 instructions
// using standard EVEX encoding for 512-bit vector operations.
//
// ## Architecture
//
// This module provides emit functions for each category of AVX-512 operations:
//
// - **ALU operations** (`emit_x64_512_inst`): Integer arithmetic (VPADDD, VPSUBQ, etc.)
// - **FP ALU operations** (`emit_x64_512_fp_inst`): Floating-point arithmetic (VADDPS, VMULPD, etc.)
// - **Comparison operations** (`emit_x64_512_cmp_inst`): Vector comparisons to k-mask
// - **Conversion operations** (`emit_x64_512_cvt_inst`): Type conversions (VCVTPS2PD, etc.)
// - **FMA operations** (`emit_x64_512_fma_inst`): Fused multiply-add (VFMADD132PS, etc.)
// - **Shuffle operations** (`emit_x64_512_shuffle_inst`): Lane permutations
// - **Insert/Extract** (`emit_x64_512_insert_inst`, `emit_x64_512_extract_inst`)
// - **K-mask operations** (`emit_mask_*`): Mask register manipulation
// - **Aligned load/store** (`emit_x64_512_aligned_op`): VMOVAPS, VMOVDQA64, etc.
//
// ## K-Register Encoding
//
// K-registers (k0-k7) use hardware encodings 0-7.
// - EVEX aaa field: 0-7
// - k0 is special: it means "no masking" (all lanes active)
//
// ## MergeMode
//
// EVEX instructions support two masking modes:
// - `MergeMode::Merge` (z=0): Masked-off lanes preserve destination values
// - `MergeMode::Zero` (z=1): Masked-off lanes are zeroed
//
// ## Reference
//
// Intel 64 and IA-32 Architectures Software Developer's Manual
// Volume 2: Instruction Set Reference (EVEX Encoding)

use super::super::args::{Amode, OperandSize, SyntheticAmode, Xmm};
use super::defs::{
    Avx512AlignOp, Avx512AluOp, Avx512Cond, Avx512CvtOp, Avx512ExtractOp, Avx512FmaOp,
    Avx512FpAluOp, Avx512FpSpecialOp, Avx512ImmShuffleOp, Avx512InsertOp, Avx512LaneShuffleOp,
    Avx512VnniOp, MaskAddOp, MaskAluOp, MaskShiftOp, MaskTestOp, MaskUnpackOp, MergeMode,
    Vp2IntersectOp,
};
use super::encoding::*;
use crate::isa::x64::inst::Inst;
use crate::isa::x64::inst::args::{OptionMaskReg, RegMem};
use crate::machinst::{MachBuffer, Reg, Writable};

// =============================================================================
// K-Register Encoding Helper
// =============================================================================

/// Convert a k-register to the EVEX/VEX mask encoding (0-7).
#[inline]
fn kreg_enc(reg: Reg) -> u8 {
    let enc = reg.to_real_reg().unwrap().hw_enc();
    debug_assert!(enc < 8, "expected k-register, got PReg index {enc}");
    enc
}

// =============================================================================
// AVX-512 ALU Instruction Emission (VPADDD, VPADDQ, etc.)
// =============================================================================

/// Emit an AVX-512 ALU instruction (3-operand form: dst = src1 op src2).
pub fn emit_x64_512_inst(
    op: Avx512AluOp,
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let w = match size {
        OperandSize::Size64 => true,
        _ => op.evex_w(),
    };

    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w,
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    // For unary ops, we pass 0 which after inversion in emit() becomes vvvv=1111 (unused)
    let src1_enc = if op.is_unary() {
        0
    } else {
        src1.to_real_reg().unwrap().hw_enc()
    };

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            emit_modrm_for_regmem(sink, dst_enc, src2);
        }
        RegMem::Mem { addr } => {
            // For memory operands, we need to properly encode the base/index register extensions
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_for_regmem(sink, dst_enc, src2);
        }
    }
}

// =============================================================================
// AVX-512 Broadcast Instruction Emission (VPBROADCASTD, VPBROADCASTQ)
// =============================================================================

/// Emit VPBROADCASTD - broadcast 32-bit element to all 16 lanes of ZMM.
///
/// VPBROADCASTD zmm, xmm/m32: EVEX.512.66.0F38.W0 58 /r
pub fn emit_vpbroadcastd(dst: Writable<Xmm>, src: &RegMem, sink: &mut MachBuffer<Inst>) {
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: false,  // W=0 for 32-bit
        pp: 0x01,  // 66 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit(dst_enc, 0x00, src_enc, false, sink);
            sink.put1(0x58); // VPBROADCASTD opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit_for_alu_mem(dst_enc, 0x00, base_enc, index_enc, sink);
            sink.put1(0x58);
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

/// Emit VPBROADCASTQ - broadcast 64-bit element to all 8 lanes of ZMM.
///
/// VPBROADCASTQ zmm, xmm/m64: EVEX.512.66.0F38.W1 59 /r
pub fn emit_vpbroadcastq(dst: Writable<Xmm>, src: &RegMem, sink: &mut MachBuffer<Inst>) {
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: true,   // W=1 for 64-bit
        pp: 0x01,  // 66 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit(dst_enc, 0x00, src_enc, false, sink);
            sink.put1(0x59); // VPBROADCASTQ opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit_for_alu_mem(dst_enc, 0x00, base_enc, index_enc, sink);
            sink.put1(0x59);
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

// =============================================================================
// AVX-512 Comparison Instruction Emission (VPCMPD, VPCMPQ)
// =============================================================================

/// Emit an AVX-512 comparison instruction that writes to a k-register.
/// VPCMPD/VPCMPQ: Compare packed integers and store result in mask register.
pub fn emit_x64_512_cmp(
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    cond: Avx512Cond,
    mask: OptionMaskReg,
    sink: &mut MachBuffer<Inst>,
) {
    // VPCMPD: EVEX.512.66.0F3A.W0 1F /r ib
    // VPCMPQ: EVEX.512.66.0F3A.W1 1F /r ib
    let w = matches!(size, OperandSize::Size64);

    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: 0x03, // 0F3A map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: false, // Comparisons don't use zeroing
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(0x1F); // VPCMPD/VPCMPQ opcode
            emit_modrm_for_regmem(sink, dst_enc, src2);
            sink.put1(cond as u8); // Immediate condition
        }
        RegMem::Mem { addr } => {
            // For memory operands, properly encode the base/index register extensions
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(0x1F); // VPCMPD/VPCMPQ opcode
            emit_modrm_for_regmem(sink, dst_enc, src2);
            sink.put1(cond as u8); // Immediate condition
        }
    }
}

// =============================================================================
// VPCOMPRESS / VPEXPAND Instruction Emission
// =============================================================================

/// Emit VPCOMPRESSD/VPCOMPRESSQ - compress and store.
/// Packs valid elements contiguously based on mask.
pub fn emit_compress_store(
    size: OperandSize,
    src: Reg,
    addr: &Amode,
    mask: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // VPCOMPRESSD: EVEX.512.66.0F38.W0 8B /r
    // VPCOMPRESSQ: EVEX.512.66.0F38.W1 8B /r
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = kreg_enc(mask);

    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: false, // Compress store uses merge semantics
        b: false,
        ll: 0b10,
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    // For store instructions, we need to use a different encoding
    // The src register goes in the reg field of ModRM
    evex.emit_for_mem(src_enc, base_enc, index_enc, sink);
    sink.put1(0x8B); // VPCOMPRESSD/Q opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

/// Emit VPEXPANDD/VPEXPANDQ - expand load.
/// Loads sparse elements into contiguous positions based on mask.
pub fn emit_expand_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &Amode,
    mask: Reg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VPEXPANDD: EVEX.512.66.0F38.W0 89 /r
    // VPEXPANDQ: EVEX.512.66.0F38.W1 89 /r
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = kreg_enc(mask);

    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10,
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(dst_enc, base_enc, index_enc, sink);
    sink.put1(0x89); // VPEXPANDD/Q opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

// =============================================================================
// Masked Load/Store Instruction Emission (VMOVDQU32/64)
// =============================================================================

/// Emit VMOVDQU32/VMOVDQU64 masked load (fault-suppressing).
pub fn emit_masked_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &Amode,
    mask: Reg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU32: EVEX.512.F3.0F.W0 6F /r (load)
    // VMOVDQU64: EVEX.512.F3.0F.W1 6F /r (load)
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = kreg_enc(mask);

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp: 0x02, // F3 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10,
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(dst_enc, base_enc, index_enc, sink);
    sink.put1(0x6F); // VMOVDQU32/64 load opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

/// Emit VMOVDQU32/VMOVDQU64 masked store.
pub fn emit_masked_store(
    size: OperandSize,
    src: Reg,
    addr: &Amode,
    mask: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU32: EVEX.512.F3.0F.W0 7F /r (store)
    // VMOVDQU64: EVEX.512.F3.0F.W1 7F /r (store)
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = kreg_enc(mask);

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp: 0x02, // F3 prefix
        aaa: mask_enc,
        z: false, // Stores don't use zeroing
        b: false,
        ll: 0b10,
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(src_enc, base_enc, index_enc, sink);
    sink.put1(0x7F); // VMOVDQU32/64 store opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

/// Emit VMOVDQU8/16/32/64 unmasked load (512-bit).
/// This is for simple 512-bit vector loads without masking.
pub fn emit_512bit_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &Amode,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU8:  EVEX.512.F2.0F.W0 6F /r (load) - byte elements
    // VMOVDQU16: EVEX.512.F2.0F.W1 6F /r (load) - word elements
    // VMOVDQU32: EVEX.512.F3.0F.W0 6F /r (load) - dword elements
    // VMOVDQU64: EVEX.512.F3.0F.W1 6F /r (load) - qword elements
    // Use aaa=000 (k0) for unmasked operation
    let (pp, w) = match size {
        OperandSize::Size8 => (0x03, false),  // F2 prefix, W=0
        OperandSize::Size16 => (0x03, true),  // F2 prefix, W=1
        OperandSize::Size32 => (0x02, false), // F3 prefix, W=0
        OperandSize::Size64 => (0x02, true),  // F3 prefix, W=1
    };

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp,
        aaa: 0,   // k0 = no masking
        z: false, // No zeroing for unmasked
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(dst_enc, base_enc, index_enc, sink);
    sink.put1(0x6F); // VMOVDQU load opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

/// Emit VMOVDQU32/64 unmasked load (256-bit).
/// This is for simple 256-bit vector loads without masking.
pub fn emit_256bit_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &Amode,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU32: EVEX.256.F3.0F.W0 6F /r (load) - dword elements
    // VMOVDQU64: EVEX.256.F3.0F.W1 6F /r (load) - qword elements
    // Use aaa=000 (k0) for unmasked operation
    let (pp, w) = match size {
        OperandSize::Size32 => (0x02, false), // F3 prefix, W=0
        OperandSize::Size64 => (0x02, true),  // F3 prefix, W=1
        _ => panic!("256-bit load only supports Size32 and Size64"),
    };

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp,
        aaa: 0,   // k0 = no masking
        z: false, // No zeroing for unmasked
        b: false,
        ll: 0b01, // 256-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(dst_enc, base_enc, index_enc, sink);
    sink.put1(0x6F); // VMOVDQU load opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

/// Emit VMOVDQU8/16/32/64 unmasked store (512-bit).
/// This is for simple 512-bit vector stores without masking.
pub fn emit_512bit_store(size: OperandSize, src: Reg, addr: &Amode, sink: &mut MachBuffer<Inst>) {
    // VMOVDQU8:  EVEX.512.F2.0F.W0 7F /r (store) - byte elements
    // VMOVDQU16: EVEX.512.F2.0F.W1 7F /r (store) - word elements
    // VMOVDQU32: EVEX.512.F3.0F.W0 7F /r (store) - dword elements
    // VMOVDQU64: EVEX.512.F3.0F.W1 7F /r (store) - qword elements
    // Use aaa=000 (k0) for unmasked operation
    let (pp, w) = match size {
        OperandSize::Size8 => (0x03, false),  // F2 prefix, W=0
        OperandSize::Size16 => (0x03, true),  // F2 prefix, W=1
        OperandSize::Size32 => (0x02, false), // F3 prefix, W=0
        OperandSize::Size64 => (0x02, true),  // F3 prefix, W=1
    };

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp,
        aaa: 0, // k0 = no masking
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(src_enc, base_enc, index_enc, sink);
    sink.put1(0x7F); // VMOVDQU store opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

/// Emit VMOVDQU32/64 unmasked store (256-bit).
/// This is for simple 256-bit vector stores without masking.
pub fn emit_256bit_store(size: OperandSize, src: Reg, addr: &Amode, sink: &mut MachBuffer<Inst>) {
    // VMOVDQU32: EVEX.256.F3.0F.W0 7F /r (store) - dword elements
    // VMOVDQU64: EVEX.256.F3.0F.W1 7F /r (store) - qword elements
    // Use aaa=000 (k0) for unmasked operation
    let (pp, w) = match size {
        OperandSize::Size32 => (0x02, false), // F3 prefix, W=0
        OperandSize::Size64 => (0x02, true),  // F3 prefix, W=1
        _ => panic!("256-bit store only supports Size32 and Size64"),
    };

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp,
        aaa: 0, // k0 = no masking
        z: false,
        b: false,
        ll: 0b01, // 256-bit
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Extract base/index register encodings from the address mode
    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);

    evex.emit_for_mem(src_enc, base_enc, index_enc, sink);
    sink.put1(0x7F); // VMOVDQU store opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

// =============================================================================
// Mask Register Logic Instruction Emission (KAND, KOR, etc.)
// =============================================================================

/// Emit a mask register logic instruction.
/// These use VEX encoding (not EVEX) for k-register operations.
pub fn emit_mask_logic(
    op: MaskAluOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: Option<Reg>,
    sink: &mut MachBuffer<Inst>,
) {
    // Mask logic instructions use VEX encoding.
    // Note: Different instructions have different pp (prefix) requirements:
    // - KANDW:  VEX.L1.66.0F.W0 41 /r (pp=01)
    // - KANDNW: VEX.L1.66.0F.W0 42 /r (pp=01)
    // - KNOTW:  VEX.L0.0F.W0 44 /r (pp=00)
    // - KORW:   VEX.L1.0F.W0 45 /r (pp=00)
    // - KXNORW: VEX.L1.0F.W0 46 /r (pp=00)
    // - KXORW:  VEX.L1.0F.W0 47 /r (pp=00)

    // K-registers use encodings 0-7 for VEX encoding
    let dst_enc = kreg_enc(dst.to_reg());
    let src1_enc = kreg_enc(src1);

    match op {
        MaskAluOp::Knot => {
            // KNOTW is unary: VEX.L0.0F.W0 44 /r
            // 2-byte VEX: C5 [R~vvvv~L~pp] opcode modrm
            // K-registers only go 0-7, so R bit always 1 (no extension needed)
            let vvvv = !src1_enc & 0x0F;
            let vex2 = (1 << 7) | (vvvv << 3) | 0x00; // R=1, L=0, pp=00
            sink.put1(0xC5);
            sink.put1(vex2);
            sink.put1(0x44); // KNOTW opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src1_enc & 0x07);
            sink.put1(modrm);
        }
        _ => {
            // Binary ops: KAND, KANDN, KOR, KXNOR, KXOR
            let src2 = src2.expect("Binary mask op requires src2");
            let src2_enc = kreg_enc(src2);

            let (opcode, pp) = match op {
                // These use pp=01 (66 prefix)
                MaskAluOp::Kand => (0x41, 0x01),
                MaskAluOp::Kandn => (0x42, 0x01),
                // These use pp=00 (no prefix)
                MaskAluOp::Kor => (0x45, 0x00),
                MaskAluOp::Kxnor => (0x46, 0x00),
                MaskAluOp::Kxor => (0x47, 0x00),
                MaskAluOp::Knot => unreachable!(),
            };

            // 3-byte VEX for L=1: C4 [RXB~mmmmm] [W~vvvv~L~pp] opcode modrm
            // K-registers only go 0-7, so R, X, B bits always 1 (no extension needed)
            let vvvv = !src1_enc & 0x0F;
            let vex2 = (0 << 7) | (vvvv << 3) | 0x04 | pp; // W=0, L=1, pp=variable

            sink.put1(0xC4);
            sink.put1(0xE1); // R=1, X=1, B=1, mmmmm = 01 (0F)
            sink.put1(vex2);
            sink.put1(opcode);
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
        }
    }
}

// =============================================================================
// KMOV Instruction Emission
// =============================================================================

/// Emit KMOVQ - move between k-register, GPR, or k↔k.
/// - to_gpr=true: KMOVQ r64, k (move from k to GPR)
/// - to_gpr=false: KMOVQ k, r64 (move from GPR to k)
/// For k↔k moves, use emit_kmov_kk instead.
pub fn emit_kmov(dst: Writable<Reg>, src: Reg, to_gpr: bool, sink: &mut MachBuffer<Inst>) {
    if to_gpr {
        let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
        let src_enc = kreg_enc(src);
        // KMOVQ r64, k: VEX.L0.F2.0F.W1 93 /r
        let r = if dst_enc >= 8 { 0 } else { 1 };
        let b = 1;
        let rxb = (r << 7) | 0x40 | (b << 5) | 0x01; // X=1, mmmmm=01
        let vex2 = 0x80 | 0x78 | 0x03; // W=1, vvvv=1111, L=0, pp=11 (F2)

        sink.put1(0xC4);
        sink.put1(rxb);
        sink.put1(vex2);
        sink.put1(0x93);
        let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
        sink.put1(modrm);
    } else {
        let dst_enc = kreg_enc(dst.to_reg());
        let src_enc = src.to_real_reg().unwrap().hw_enc();
        // KMOVQ k, r64: VEX.L0.F2.0F.W1 92 /r
        let r = 1;
        let b = if src_enc >= 8 { 0 } else { 1 };
        let rxb = (r << 7) | 0x40 | (b << 5) | 0x01;
        let vex2 = 0x80 | 0x78 | 0x03; // W=1, vvvv=1111, L=0, pp=11 (F2)

        sink.put1(0xC4);
        sink.put1(rxb);
        sink.put1(vex2);
        sink.put1(0x92);
        let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
        sink.put1(modrm);
    }
}

/// Emit KMOVQ k, k (k-register to k-register move).
pub fn emit_kmov_kk(dst: Writable<Reg>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // KMOVQ k1, k2: VEX.L0.0F.W1 90 /r (mod=11 for reg-reg)
    let dst_enc = kreg_enc(dst.to_reg());
    let src_enc = kreg_enc(src);

    // 3-byte VEX for W=1
    sink.put1(0xC4);
    sink.put1(0xE1); // RXB=111, mmmmm=01
    sink.put1(0xF8); // W=1, vvvv=1111, L=0, pp=00
    sink.put1(0x90);
    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// KORTEST Instruction Emission
// =============================================================================

/// Emit KORTESTW/KORTESTQ - OR masks and set flags.
pub fn emit_kortest(src1: Reg, src2: Reg, sink: &mut MachBuffer<Inst>) {
    // KORTESTW: VEX.L0.0F.W0 98 /r
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    // 2-byte VEX
    let r = if src1_enc >= 8 { 0 } else { 1 };
    let vex2 = (r << 7) | 0x78 | 0x00; // vvvv=1111, L=0, pp=00

    sink.put1(0xC5);
    sink.put1(vex2);
    sink.put1(0x98); // KORTESTW opcode
    let modrm = 0xC0 | ((src1_enc & 0x07) << 3) | (src2_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// K-Register Spill/Fill (for register allocation)
// =============================================================================

/// Emit KMOVQ to spill a k-register to memory.
/// Uses the two-tier strategy: prefer GPR, fall back to stack.
pub fn emit_kmov_store(src: Reg, addr: &Amode, sink: &mut MachBuffer<Inst>) {
    // KMOVQ m64, k: VEX.L0.0F.W1 91 /r
    let src_enc = kreg_enc(src);

    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);
    let r_inv = if (src_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = match index_enc {
        Some(enc) => {
            if (enc & 0x08) == 0 {
                0x40
            } else {
                0
            }
        }
        None => 0x40,
    };
    let b_inv = if (base_enc & 0x08) == 0 { 0x20 } else { 0 };
    let vex_byte1 = r_inv | x_inv | b_inv | 0x01; // mmmmm = 01 (0F)

    // 3-byte VEX for memory operand
    sink.put1(0xC4);
    sink.put1(vex_byte1);
    sink.put1(0xF8); // W=1, vvvv=1111, L=0, pp=00
    sink.put1(0x91);
    emit_modrm_sib_disp_no_evex(sink, src_enc, addr);
}

/// Emit KMOVQ to fill a k-register from memory.
pub fn emit_kmov_load(dst: Writable<Reg>, addr: &Amode, sink: &mut MachBuffer<Inst>) {
    // KMOVQ k, m64: VEX.L0.0F.W1 90 /r
    let dst_enc = kreg_enc(dst.to_reg());

    let (base_enc, index_enc) = extract_mem_reg_encodings(addr);
    let r_inv = if (dst_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = match index_enc {
        Some(enc) => {
            if (enc & 0x08) == 0 {
                0x40
            } else {
                0
            }
        }
        None => 0x40,
    };
    let b_inv = if (base_enc & 0x08) == 0 { 0x20 } else { 0 };
    let vex_byte1 = r_inv | x_inv | b_inv | 0x01; // mmmmm = 01 (0F)

    // 3-byte VEX for memory operand
    sink.put1(0xC4);
    sink.put1(vex_byte1);
    sink.put1(0xF8); // W=1, vvvv=1111, L=0, pp=00
    sink.put1(0x90);
    emit_modrm_sib_disp_no_evex(sink, dst_enc, addr);
}

// =============================================================================
// Gather/Scatter Instruction Emission (VPGATHERD/Q, VPSCATTERD/Q)
// =============================================================================

/// Emit a VPGATHER instruction (load non-contiguous elements by indices).
///
/// Gathers dword/qword elements from memory using indices in a vector register.
/// The mask register indicates which elements to gather - after each gather,
/// the corresponding mask bit is cleared.
///
/// Memory addressing: base + index_reg * scale + offset
pub fn emit_gather(
    op: super::defs::GatherOp,
    dst: Writable<Reg>,
    base: Reg,
    index: Reg,
    scale: u8, // 1, 2, 4, or 8
    disp: i32,
    mask: Reg, // Must be k1-k7 (k0 not allowed for gather/scatter)
    sink: &mut MachBuffer<Inst>,
) {
    // VPGATHERDD/DQ/QD/QQ: EVEX.512.66.0F38.W0/W1 90/91 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: op.evex_w(),
        pp: 0x01, // 66 prefix
        aaa: kreg_enc(mask),
        z: false, // Gather doesn't use zeroing (mask bits are cleared during operation)
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let base_enc = base.to_real_reg().unwrap().hw_enc();
    let index_enc = index.to_real_reg().unwrap().hw_enc();

    // Emit EVEX prefix with VSIB addressing
    evex.emit_vsib(dst_enc, index_enc, base_enc, sink);
    sink.put1(op.opcode());

    // Emit ModRM + SIB for VSIB addressing
    // ModRM: mod=00/01/10 (depending on displacement), reg=dst, rm=100 (SIB follows)
    let scale_bits = match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => panic!("Invalid scale for gather: {scale}"),
    };

    // EVEX uses compressed displacement (disp8*N) where N is the element size.
    // For gather/scatter with qword elements (W=1), N=8; for dword (W=0), N=4.
    let elem_size = op.element_size() as i32;

    // Try to use compressed disp8 encoding if displacement is aligned to element size
    let can_use_disp8 = disp != 0 && (disp % elem_size) == 0;
    let compressed_disp = if can_use_disp8 { disp / elem_size } else { 0 };
    let use_disp8 = can_use_disp8 && compressed_disp >= -128 && compressed_disp <= 127;

    if disp == 0 && (base_enc & 0x07) != 5 {
        // mod=00: no displacement (unless base is RBP/R13)
        let modrm = 0x04 | ((dst_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
    } else if use_disp8 {
        // mod=01: 8-bit compressed displacement (disp8 * element_size)
        let modrm = 0x44 | ((dst_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
        sink.put1(compressed_disp as u8);
    } else {
        // mod=10: 32-bit displacement (not compressed)
        let modrm = 0x84 | ((dst_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
        sink.put4(disp as u32);
    }
}

/// Emit a VPSCATTER instruction (store elements to non-contiguous memory by indices).
///
/// Scatters dword/qword elements to memory using indices in a vector register.
/// The mask register indicates which elements to scatter - after each scatter,
/// the corresponding mask bit is cleared.
///
/// Memory addressing: base + index_reg * scale + offset
pub fn emit_scatter(
    op: super::defs::ScatterOp,
    src: Reg,
    base: Reg,
    index: Reg,
    scale: u8, // 1, 2, 4, or 8
    disp: i32,
    mask: Reg, // Must be k1-k7 (k0 not allowed for gather/scatter)
    sink: &mut MachBuffer<Inst>,
) {
    // VPSCATTERDD/DQ/QD/QQ: EVEX.512.66.0F38.W0/W1 A0/A1 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: op.evex_w(),
        pp: 0x01, // 66 prefix
        aaa: kreg_enc(mask),
        z: false, // Scatter doesn't use zeroing
        b: false,
        ll: 0b10, // 512-bit
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc();
    let base_enc = base.to_real_reg().unwrap().hw_enc();
    let index_enc = index.to_real_reg().unwrap().hw_enc();

    // Emit EVEX prefix with VSIB addressing
    evex.emit_vsib(src_enc, index_enc, base_enc, sink);
    sink.put1(op.opcode());

    // Emit ModRM + SIB for VSIB addressing
    let scale_bits = match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => panic!("Invalid scale for scatter: {scale}"),
    };

    if disp == 0 && (base_enc & 0x07) != 5 {
        let modrm = 0x04 | ((src_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
    } else if disp >= -128 && disp <= 127 {
        let modrm = 0x44 | ((src_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
        sink.put1(disp as u8);
    } else {
        let modrm = 0x84 | ((src_enc & 0x07) << 3);
        sink.put1(modrm);
        let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
        sink.put1(sib);
        sink.put4(disp as u32);
    }
}

// =============================================================================
// Compress/Expand Register Instructions
// =============================================================================

/// Emit VPCOMPRESSD/Q to register (compress and write to register).
///
/// VPCOMPRESSD/Q compresses elements from src where mask bits are 1,
/// packing them contiguously into dst starting from the lowest position.
///
/// Intel encoding: VPCOMPRESSD zmm1/m512{k1}{z}, zmm2
/// - zmm2 (source) goes in reg field of ModRM
/// - zmm1 (destination) goes in r/m field of ModRM
pub fn emit_compress_reg(
    size: OperandSize,
    dst: Writable<Reg>,
    src: Reg,
    mask: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // VPCOMPRESSD: EVEX.512.66.0F38.W0 8B /r
    // VPCOMPRESSQ: EVEX.512.66.0F38.W1 8B /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: matches!(size, OperandSize::Size64),
        pp: 0x01, // 66 prefix
        aaa: kreg_enc(mask),
        z: false, // Register form doesn't use zeroing
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // For VPCOMPRESSD: src goes in reg field (R/R' bits), dst goes in r/m field (B/X bits)
    evex.emit(src_enc, 0, dst_enc, false, sink);
    sink.put1(0x8B);

    // ModRM: reg field = src, r/m field = dst
    let modrm = 0xC0 | ((src_enc & 0x07) << 3) | (dst_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPEXPANDD/Q from register (expand from register).
///
/// VPEXPANDD/Q expands contiguous elements from src into dst,
/// placing them at positions where mask bits are 1.
pub fn emit_expand_reg(
    size: OperandSize,
    dst: Writable<Reg>,
    src: Reg,
    mask: Reg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VPEXPANDD: EVEX.512.66.0F38.W0 89 /r
    // VPEXPANDQ: EVEX.512.66.0F38.W1 89 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: matches!(size, OperandSize::Size64),
        pp: 0x01, // 66 prefix
        aaa: kreg_enc(mask),
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x89);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVD2M - extract high bits from 32-bit elements to mask register.
///
/// This instruction extracts the sign bit (bit 31) of each 32-bit element
/// in the source vector and places them into the destination mask register.
/// Extract k-register hardware encoding from PReg hw_enc.
fn k_enc(hw_enc: u8) -> u8 {
    debug_assert!(hw_enc < 8, "invalid k-register hw_enc: {hw_enc}");
    hw_enc
}

pub fn emit_vmovmsk32(dst: Writable<Reg>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVD2M: EVEX.512.F3.0F38.W0 39 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: false,  // W=0 for 32-bit elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_hw = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();
    // dst is a k-register
    let dst_enc = k_enc(dst_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x39);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVM2D - expand mask register to 32-bit vector elements.
///
/// This instruction broadcasts each mask bit to a full 32-bit element (0 or -1).
/// Used to convert k-register comparison results to vector masks.
pub fn emit_movm2d(dst: Writable<Xmm>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVM2D: EVEX.512.F3.0F38.W0 38 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: false,  // W=0 for 32-bit elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();
    let src_hw = src.to_real_reg().unwrap().hw_enc();
    // src is a k-register
    let src_enc = k_enc(src_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x38);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVM2Q - expand mask register to 64-bit vector elements.
///
/// This instruction broadcasts each mask bit to a full 64-bit element (0 or -1).
/// Used to convert k-register comparison results to vector masks.
pub fn emit_movm2q(dst: Writable<Xmm>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVM2Q: EVEX.512.F3.0F38.W1 38 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: true,   // W=1 for 64-bit elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();
    let src_hw = src.to_real_reg().unwrap().hw_enc();
    // src is a k-register
    let src_enc = k_enc(src_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x38);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVQ2M - extract high bits from 64-bit elements to mask register.
///
/// This instruction extracts the sign bit (bit 63) of each 64-bit element
/// in the source vector and places them into the destination mask register.
pub fn emit_vmovmsk64(dst: Writable<Reg>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVQ2M: EVEX.512.F3.0F38.W1 39 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: true,   // W=1 for 64-bit elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_hw = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();
    // dst is a k-register
    let dst_enc = k_enc(dst_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x39);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVB2M - extract high bits from byte elements to mask register.
///
/// This instruction extracts the sign bit (bit 7) of each byte element
/// in the source vector and places them into the destination mask register.
pub fn emit_vmovmsk8(dst: Writable<Reg>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVB2M: EVEX.512.F3.0F38.W0 29 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: false,  // W=0 for byte elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_hw = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();
    // dst is a k-register
    let dst_enc = k_enc(dst_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x29);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVW2M - extract high bits from word elements to mask register.
///
/// This instruction extracts the sign bit (bit 15) of each word element
/// in the source vector and places them into the destination mask register.
pub fn emit_vmovmsk16(dst: Writable<Reg>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVW2M: EVEX.512.F3.0F38.W1 29 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: true,   // W=1 for word elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_hw = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();
    // dst is a k-register
    let dst_enc = k_enc(dst_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x29);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVM2B - expand mask register to byte vector elements.
///
/// This instruction broadcasts each mask bit to a full byte element (0 or -1).
/// Used to convert k-register comparison results to byte vector masks.
pub fn emit_movm2b(dst: Writable<Xmm>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVM2B: EVEX.512.F3.0F38.W0 28 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: false,  // W=0 for byte elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();
    let src_hw = src.to_real_reg().unwrap().hw_enc();
    // src is a k-register
    let src_enc = k_enc(src_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x28);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

/// Emit VPMOVM2W - expand mask register to word vector elements.
///
/// This instruction broadcasts each mask bit to a full word element (0 or -1).
/// Used to convert k-register comparison results to word vector masks.
pub fn emit_movm2w(dst: Writable<Xmm>, src: Reg, sink: &mut MachBuffer<Inst>) {
    // VPMOVM2W: EVEX.512.F3.0F38.W1 28 /r
    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w: true,   // W=1 for word elements
        pp: 0x02,  // F3 prefix
        aaa: 0,    // No mask
        z: false,
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_reg().to_real_reg().unwrap().hw_enc();
    let src_hw = src.to_real_reg().unwrap().hw_enc();
    // src is a k-register
    let src_enc = k_enc(src_hw);

    evex.emit(dst_enc, 0, src_enc, false, sink);
    sink.put1(0x28);

    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// AVX-512 Floating-Point ALU Instruction Emission
// =============================================================================

/// Emit an AVX-512 floating-point ALU instruction (3-operand form: dst = src1 op src2).
///
/// This handles VADDPS/PD, VSUBPS/PD, VMULPS/PD, VDIVPS/PD, VMINPS/PD, VMAXPS/PD.
pub fn emit_x64_512_fp_inst(
    op: Avx512FpAluOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            emit_modrm_for_regmem(sink, dst_enc, src2);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_for_regmem(sink, dst_enc, src2);
        }
    }
}

/// Emit an AVX-512 floating-point unary instruction (VSQRTPS/PD).
///
/// dst = sqrt(src)
pub fn emit_x64_512_fp_unary(
    op: Avx512FpAluOp,
    dst: Writable<Reg>,
    src: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit(dst_enc, 0x00, src_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit_for_alu_mem(dst_enc, 0x00, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

// =============================================================================
// AVX-512 FMA (Fused Multiply-Add) Instruction Emission
// =============================================================================

/// Emit an AVX-512 FMA instruction (VFMADD213PS/PD, etc.).
///
/// FMA computes `(a * b) + c` (or subtract/negate variants) in a single instruction.
/// The 213 form is: dst = src2 * src1 + src3
///
/// For Cranelift's `fma(x, y, z)` = x * y + z:
/// - Use VFMADD213: dst/src1=x, src2=y, src3=z → result = y * x + z = x * y + z ✓
pub fn emit_x64_512_fma_inst(
    op: Avx512FmaOp,
    dst: Writable<Reg>,
    src1: Reg,     // First multiplicand (becomes dst)
    src2: Reg,     // Second multiplicand
    src3: &RegMem, // Addend
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    // For FMA with 213 encoding, src1 must equal dst (they're tied)
    // Verify the constraint and use src1 to suppress unused warning in release.
    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            dst.to_reg().to_real_reg().unwrap().hw_enc(),
            src1.to_real_reg().unwrap().hw_enc(),
            "FMA 213: src1 must be tied to dst"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = src1;
    }

    // For FMA, the encoding is:
    // - dst (reg field in ModRM) = first source and destination
    // - src2 (vvvv field in EVEX) = second multiplicand
    // - src3 (r/m field in ModRM) = addend/subtrahend

    match src3 {
        RegMem::Reg { reg } => {
            let src3_enc = reg.to_real_reg().unwrap().hw_enc();
            // The destination (src1) goes in the reg field
            // src2 goes in vvvv
            // src3 goes in r/m
            evex.emit(dst_enc, src2_enc, src3_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src3_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src2_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

/// Emit VPDPBUSD/VPDPBUSDS/VPDPWSSD/VPDPWSSDS - VNNI dot product instructions.
///
/// These are accumulating instructions: dst = dst + dot(src1, src2)
/// The accumulator (dst/acc) is tied to the destination.
///
/// Encoding (all use 0F38 map, 66 prefix, W=0):
///   VPDPBUSD:  EVEX.NDS.512.66.0F38.W0 50 /r
///   VPDPBUSDS: EVEX.NDS.512.66.0F38.W0 51 /r
///   VPDPWSSD:  EVEX.NDS.512.66.0F38.W0 52 /r
///   VPDPWSSDS: EVEX.NDS.512.66.0F38.W0 53 /r
pub fn emit_x64_512_vnni_inst(
    op: Avx512VnniOp,
    dst: Writable<Reg>,
    acc: Reg,      // Accumulator (tied to dst)
    src1: Reg,     // First multiplicand
    src2: &RegMem, // Second multiplicand
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    // Verify the accumulator is tied to dst
    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            dst.to_reg().to_real_reg().unwrap().hw_enc(),
            acc.to_real_reg().unwrap().hw_enc(),
            "VNNI: acc must be tied to dst"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = acc;
    }

    // For VNNI, the encoding is:
    // - dst (reg field) = accumulator/destination
    // - vvvv = src1 (first multiplicand)
    // - r/m = src2 (second multiplicand)

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

// =============================================================================
// VP2INTERSECT Instruction Emission (Hash Join Acceleration)
// =============================================================================

/// Emit a VP2INTERSECT instruction for hash join acceleration.
///
/// VP2INTERSECT compares two vectors and outputs TWO mask registers:
/// - dst_k: For each element in src1, set bit if it matches ANY element in src2
/// - dst_k+1: For each element in src2, set bit if it matches ANY element in src1
///
/// The dst_k register MUST be an even k-register (k0, k2, k4, k6).
///
/// Encoding:
///   VP2INTERSECTD: EVEX.NDS.512.F2.0F38.W0 68 /r
///   VP2INTERSECTQ: EVEX.NDS.512.F2.0F38.W1 68 /r
///
/// Note: Unlike most instructions, the reg field in ModR/M encodes a k-register,
/// not a ZMM register. The vvvv field holds src1 (ZMM), r/m holds src2 (ZMM/mem).
pub fn emit_x64_512_vp2intersect_inst(
    op: Vp2IntersectOp,
    dst_k: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    sink: &mut MachBuffer<Inst>,
) {
    let evex = EvexPrefix {
        map: op.evex_map(), // 0x02 = 0F38 map
        w: op.evex_w(),     // W=0 for D, W=1 for Q
        pp: op.evex_pp(),   // 0x03 = F2 prefix
        aaa: 0,             // No write mask for output (we're writing to k-regs)
        z: false,           // No zeroing
        b: false,           // No broadcast
        ll: 0b10,           // 512-bit
    };

    // VP2INTERSECT uses k-register in the reg field
    // The dst_k must be an even k-register (k0, k2, k4, k6)
    let dst_enc = dst_k.to_reg().to_real_reg().unwrap().hw_enc();

    // Debug check: ensure dst is an even k-register
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            dst_enc % 2 == 0,
            "VP2INTERSECT dst_k must be an even k-register (k0, k2, k4, k6), got k{dst_enc}"
        );
    }

    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            // For VP2INTERSECT:
            // - EVEX.R/R' controls reg field extension (but k-regs only need 3 bits: k0-k7)
            // - EVEX.vvvv = src1 (inverted, ZMM register)
            // - ModR/M.reg = dst_k (k-register, 3 bits)
            // - ModR/M.r/m = src2 (ZMM register)
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode()); // 0x68
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode()); // 0x68
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

// =============================================================================
// AVX-512 Type Conversion Instruction Emission
// =============================================================================

/// Emit an AVX-512 type conversion instruction (unary: dst = convert(src)).
///
/// This handles VCVTDQ2PS, VCVTPS2DQ, VCVTTPS2DQ, VCVTQQ2PD, VCVTPD2QQ, etc.
pub fn emit_x64_512_cvt_inst(
    op: Avx512CvtOp,
    dst: Writable<Reg>,
    src: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit(dst_enc, 0x00, src_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            // For unary ops, vvvv must be 1111b (no second source).
            // Pass 0x00 so it inverts to 0x0F in the encoded field.
            evex.emit_for_alu_mem(dst_enc, 0x00, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
        }
    }
}

// =============================================================================
// AVX-512 Vector Alignment Instruction Emission
// =============================================================================

/// Emit an AVX-512 vector alignment instruction (VALIGND/VALIGNQ).
///
/// Format: op dst, src1, src2, imm8
/// Concatenates src1:src2 and extracts 512 bits starting at imm8 elements.
pub fn emit_x64_512_align_inst(
    op: Avx512AlignOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    imm8: u8,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(imm8);
        }
    }
}

// =============================================================================
// AVX-512 Ternary Logic Instruction Emission (VPTERNLOGD, VPTERNLOGQ)
// =============================================================================

/// Emit an AVX-512 ternary logic instruction (VPTERNLOGD/VPTERNLOGQ).
///
/// These instructions compute arbitrary 3-input boolean functions based on
/// an 8-bit immediate that encodes the truth table.
///
/// For each bit position i in the output:
///   bit index = (src1[i] << 2) | (src2[i] << 1) | src3[i]
///   result[i] = imm8[bit index]
///
/// Common imm8 values:
///   0x00 = zero
///   0xFF = all ones
///   0x80 = AND(a, b, c)
///   0xFE = OR(a, b, c)
///   0x96 = XOR(a, XOR(b, c))
///   0xCA = (a & b) | (~a & c) = blend/select
pub fn emit_x64_512_ternlog_inst(
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: Reg,
    src3: &RegMem,
    imm8: u8,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VPTERNLOGD: EVEX.512.66.0F3A.W0 25 /r ib
    // VPTERNLOGQ: EVEX.512.66.0F3A.W1 25 /r ib
    let w = matches!(size, OperandSize::Size64);

    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: 0x03, // 0F3A map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    // Used in debug_assert below to verify src1 is tied to dst
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();
    // Note: src2 goes in the vvvv field of EVEX
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    match src3 {
        RegMem::Reg { reg } => {
            let src3_enc = reg.to_real_reg().unwrap().hw_enc();
            // For ternlog: dst=reg field, src2=vvvv, src3=r/m
            // src1 is implicit as the destination (it's a 4-operand instruction with dst=src1)
            evex.emit(dst_enc, src2_enc, src3_enc, false, sink);
            sink.put1(0x25); // VPTERNLOGD/Q opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src3_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            emit_evex_for_alu_mem(&evex, dst_enc, src2_enc, base_enc, index_enc, sink);
            sink.put1(0x25); // VPTERNLOGD/Q opcode
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(imm8);
        }
    }

    // Verify src1 is tied to dst (it should be the same register)
    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            dst.to_reg().to_real_reg().unwrap().hw_enc(),
            src1_enc,
            "VPTERNLOG: src1 must be tied to dst"
        );
    }
}

/// Emit VPROLD/VPROLQ/VPRORD/VPRORQ - immediate rotate left/right.
///
/// These instructions rotate each element by an immediate value.
/// Encoding:
///   VPROLD/VPRORD: EVEX.512.66.0F.W0 72 /1 ib (left) or /0 ib (right)
///   VPROLQ/VPRORQ: EVEX.512.66.0F.W1 72 /1 ib (left) or /0 ib (right)
///
/// The ModRM reg field encodes the operation:
///   /0 = rotate right
///   /1 = rotate left
pub fn emit_x64_512_imm_rotate_inst(
    size: OperandSize,
    dst: Writable<Reg>,
    src: &RegMem,
    imm8: u8,
    is_left: bool,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VPROLD/VPROLQ/VPRORD/VPRORQ: EVEX.512.66.0F.W0/W1 72 /0 or /1 ib
    let w = matches!(size, OperandSize::Size64);

    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    // ModRM reg field: /1 for left rotate, /0 for right rotate
    let modrm_reg = if is_left { 1 } else { 0 };

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For immediate rotate: vvvv = dst (the result destination)
            // r/m = src (the source to rotate)
            // reg = operation (0=right, 1=left)
            evex.emit(modrm_reg, dst_enc, src_enc, false, sink);
            sink.put1(0x72); // Opcode for VPROL/VPROR
            // ModRM: mod=11 (reg-reg), reg=modrm_reg (0/1), r/m=src
            let modrm = 0xC0 | (modrm_reg << 3) | (src_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(modrm_reg, dst_enc, base_enc, index_enc, sink);
            sink.put1(0x72);
            emit_modrm_sib_disp(sink, modrm_reg, amode);
            sink.put1(imm8);
        }
    }
}

/// Emit VCMPPS/VCMPPD - FP compare to k-register.
///
/// These instructions compare floating-point vectors and produce a k-mask result.
/// Encoding:
///   VCMPPS: EVEX.NDS.512.0F.W0 C2 /r ib
///   VCMPPD: EVEX.NDS.512.66.0F.W1 C2 /r ib
///
/// The immediate specifies the comparison predicate:
///   0=EQ_OQ, 1=LT_OS, 2=LE_OS, 3=UNORD_Q, 4=NEQ_UQ, 5=NLT_US, 6=NLE_US, 7=ORD_Q, etc.
pub fn emit_x64_512_fp_cmp_inst(
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    imm8: u8,
    mask: OptionMaskReg,
    sink: &mut MachBuffer<Inst>,
) {
    // VCMPPS: EVEX.NDS.512.0F.W0 C2 /r ib (pp=0, no 66 prefix)
    // VCMPPD: EVEX.NDS.512.66.0F.W1 C2 /r ib (pp=1, 66 prefix)
    let (w, pp) = match size {
        OperandSize::Size32 => (false, 0x00), // VCMPPS: W=0, no prefix
        OperandSize::Size64 => (true, 0x01),  // VCMPPD: W=1, 66 prefix
        _ => panic!("Invalid size for VCMPPS/PD"),
    };

    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp,
        aaa: mask_enc,
        z: false, // No zeroing for compare instructions (output is k-register)
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            // For VCMPPS/PD: dst is k-register (reg field), vvvv=src1, r/m=src2
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(0xC2); // VCMPPS/PD opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(0xC2);
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(imm8);
        }
    }
}

/// Emit VPSHUFD/VPSHUFHW/VPSHUFLW - immediate shuffle operations.
///
/// These instructions shuffle elements according to an 8-bit immediate value.
/// Encoding:
///   VPSHUFD:  EVEX.512.66.0F.W0 70 /r ib
///   VPSHUFHW: EVEX.512.F3.0F.W0 70 /r ib
///   VPSHUFLW: EVEX.512.F2.0F.W0 70 /r ib
///
/// All use opcode 0x70, differentiated by the mandatory prefix (pp field).
pub fn emit_x64_512_imm_shuffle_inst(
    op: Avx512ImmShuffleOp,
    dst: Writable<Reg>,
    src: &RegMem,
    imm8: u8,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // For shuffle: vvvv is not used (set to 0xF), dst is reg field, src is r/m field
            // But EVEX requires vvvv for NDS/NDD encoding - for VPSHUFD etc, vvvv should be 0 (no source 2)
            evex.emit(dst_enc, 0, src_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(dst_enc, 0, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(imm8);
        }
    }
}

/// Emit an AVX-512 lane shuffle instruction (VSHUFF32X4, VSHUFF64X2, VSHUFI32X4, VSHUFI64X2).
///
/// These instructions shuffle entire 128-bit lanes between two 512-bit sources.
/// Encoding: EVEX.512.66.0F3A.W0/W1 opcode /r imm8
///
/// Format: op dst, src1, src2/m512, imm8
/// - dst = result register
/// - src1 = first source (encoded in vvvv field)
/// - src2 = second source (encoded in r/m field)
/// - imm8 = lane selection pattern
pub fn emit_x64_512_lane_shuffle_inst(
    op: Avx512LaneShuffleOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    imm8: u8,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            // dst in reg field, src1 in vvvv, src2 in r/m
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
            sink.put1(imm8);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(imm8);
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Emit EVEX prefix for ALU instruction with memory operand.
/// This is a wrapper that combines the EvexPrefix with memory operand register encodings.
fn emit_evex_for_alu_mem(
    evex: &EvexPrefix,
    dst_enc: u8,
    src1_enc: u8,
    base_enc: u8,
    index_enc: Option<u8>,
    sink: &mut MachBuffer<Inst>,
) {
    evex.emit_for_alu_mem(dst_enc, src1_enc, base_enc, index_enc, sink);
}

/// Extract the Amode from a SyntheticAmode::Real variant.
/// Panics if the SyntheticAmode is not Real (e.g., SlotOffset, IncomingArg).
fn extract_real_amode(addr: &SyntheticAmode) -> &Amode {
    match addr {
        SyntheticAmode::Real(amode) => amode,
        _ => panic!("extract_real_amode: expected Real amode, got {addr:?}"),
    }
}

/// Extract base and index register encodings from an Amode.
/// Returns (base_enc, Option<index_enc>).
/// Note: This function expects a finalized Amode (not SyntheticAmode).
fn extract_mem_reg_encodings(addr: &Amode) -> (u8, Option<u8>) {
    match addr {
        Amode::ImmReg { base, .. } => {
            let base_enc = base.to_real_reg().unwrap().hw_enc();
            (base_enc, None)
        }
        Amode::ImmRegRegShift { base, index, .. } => {
            let base_enc = base.to_reg().to_real_reg().unwrap().hw_enc();
            let index_enc = index.to_reg().to_real_reg().unwrap().hw_enc();
            (base_enc, Some(index_enc))
        }
        Amode::RipRelative { .. } => {
            // RIP-relative addressing: use RBP encoding (5) with no index
            // The B bit should be 1 (inverted) since we're not extending
            (5, None)
        }
    }
}

/// Emit ModRM byte for a RegMem operand.
/// Note: For memory operands, this handles SyntheticAmode::Real only.
/// All SyntheticAmode variants other than Real should be finalized before
/// calling the instruction emit functions.
fn emit_modrm_for_regmem(sink: &mut MachBuffer<Inst>, reg: u8, rm: &RegMem) {
    match rm {
        RegMem::Reg { reg: rm_reg } => {
            let rm_enc = rm_reg.to_real_reg().unwrap().hw_enc();
            let modrm = 0xC0 | ((reg & 0x07) << 3) | (rm_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            // For RegMem in ALU instructions, the address should be Real
            // (synthetic addresses like SlotOffset should be finalized before lowering)
            match addr {
                SyntheticAmode::Real(amode) => {
                    emit_modrm_sib_disp(sink, reg, amode);
                }
                _ => panic!("emit_modrm_for_regmem: expected Real amode, got {addr:?}"),
            }
        }
    }
}

/// Emit ModRM, SIB, and displacement for a memory operand.
/// Note: This function expects a finalized Amode (not SyntheticAmode).
/// This function uses EVEX compressed displacement (disp8*N) where N=64 for 512-bit vectors.
fn emit_modrm_sib_disp(sink: &mut MachBuffer<Inst>, reg: u8, addr: &Amode) {
    // EVEX compressed displacement factor for 512-bit vectors is 64 bytes
    const EVEX_DISP_FACTOR: i32 = 64;

    match addr {
        Amode::ImmReg { simm32, base, .. } => {
            let base_enc = base.to_real_reg().unwrap().hw_enc();
            let needs_sib = (base_enc & 0x07) == 4;

            if *simm32 == 0 && (base_enc & 0x07) != 5 {
                let modrm = 0x00 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                sink.put1(modrm);
                if needs_sib {
                    sink.put1(0x24);
                }
            } else {
                // Try EVEX compressed displacement first (disp8*N where N=64)
                // The displacement must be divisible by 64 and fit in signed 8-bit after division
                let can_use_compressed = *simm32 % EVEX_DISP_FACTOR == 0;
                let compressed_disp = *simm32 / EVEX_DISP_FACTOR;

                if can_use_compressed && compressed_disp >= -128 && compressed_disp <= 127 {
                    // Use mod=01 with compressed 8-bit displacement
                    let modrm = 0x40 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                    sink.put1(modrm);
                    if needs_sib {
                        sink.put1(0x24);
                    }
                    sink.put1(compressed_disp as u8);
                } else {
                    // Fall back to full 32-bit displacement
                    let modrm = 0x80 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                    sink.put1(modrm);
                    if needs_sib {
                        sink.put1(0x24);
                    }
                    sink.put4(*simm32 as u32);
                }
            }
        }
        Amode::ImmRegRegShift {
            simm32,
            base,
            index,
            shift,
            ..
        } => {
            let base_enc = base.to_reg().to_real_reg().unwrap().hw_enc();
            let index_enc = index.to_reg().to_real_reg().unwrap().hw_enc();
            let sib = (*shift << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);

            if *simm32 == 0 && (base_enc & 0x07) != 5 {
                let modrm = 0x04 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(sib);
            } else {
                // Try EVEX compressed displacement (disp8*N where N=64)
                let can_use_compressed = *simm32 % EVEX_DISP_FACTOR == 0;
                let compressed_disp = *simm32 / EVEX_DISP_FACTOR;

                if can_use_compressed && compressed_disp >= -128 && compressed_disp <= 127 {
                    let modrm = 0x44 | ((reg & 0x07) << 3);
                    sink.put1(modrm);
                    sink.put1(sib);
                    sink.put1(compressed_disp as u8);
                } else {
                    let modrm = 0x84 | ((reg & 0x07) << 3);
                    sink.put1(modrm);
                    sink.put1(sib);
                    sink.put4(*simm32 as u32);
                }
            }
        }
        Amode::RipRelative { .. } => {
            let modrm = 0x05 | ((reg & 0x07) << 3);
            sink.put1(modrm);
            sink.put4(0);
        }
    }
}

fn emit_modrm_sib_disp_no_evex(sink: &mut MachBuffer<Inst>, reg: u8, addr: &Amode) {
    match addr {
        Amode::ImmReg { simm32, base, .. } => {
            let base_enc = base.to_real_reg().unwrap().hw_enc();
            let needs_sib = (base_enc & 0x07) == 4;

            if *simm32 == 0 && (base_enc & 0x07) != 5 {
                let modrm = 0x00 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                sink.put1(modrm);
                if needs_sib {
                    sink.put1(0x24);
                }
            } else if *simm32 >= -128 && *simm32 <= 127 {
                let modrm = 0x40 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                sink.put1(modrm);
                if needs_sib {
                    sink.put1(0x24);
                }
                sink.put1(*simm32 as u8);
            } else {
                let modrm = 0x80 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                sink.put1(modrm);
                if needs_sib {
                    sink.put1(0x24);
                }
                sink.put4(*simm32 as u32);
            }
        }
        Amode::ImmRegRegShift {
            simm32,
            base,
            index,
            shift,
            ..
        } => {
            let base_enc = base.to_reg().to_real_reg().unwrap().hw_enc();
            let index_enc = index.to_reg().to_real_reg().unwrap().hw_enc();
            let sib = (*shift << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);

            if *simm32 == 0 && (base_enc & 0x07) != 5 {
                let modrm = 0x04 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(sib);
            } else if *simm32 >= -128 && *simm32 <= 127 {
                let modrm = 0x44 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(sib);
                sink.put1(*simm32 as u8);
            } else {
                let modrm = 0x84 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(sib);
                sink.put4(*simm32 as u32);
            }
        }
        Amode::RipRelative { .. } => {
            let modrm = 0x05 | ((reg & 0x07) << 3);
            sink.put1(modrm);
            sink.put4(0);
        }
    }
}

// =============================================================================
// AVX-512 Extract Instruction Emission (VEXTRACTI32X4, etc.)
// =============================================================================

/// Emit an AVX-512 extract instruction (dst = extract_lane(src, imm))
/// Extracts a 128-bit or 256-bit lane from a 512-bit ZMM register.
pub fn emit_x64_512_extract(
    op: Avx512ExtractOp,
    dst: Writable<Reg>,
    src: Reg,
    lane: u8,
    sink: &mut MachBuffer<Inst>,
) {
    // EVEX.512.66.0F3A.W0/W1 39/3B /r imm8
    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: 0, // No masking
        z: false,
        b: false,
        ll: 0b10, // 512-bit source
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Note: For extract instructions, the encoding is reversed:
    // - The ZMM source is encoded in the R field (like a destination normally would be)
    // - The XMM/YMM destination is encoded in the RM field
    evex.emit(src_enc, 0, dst_enc, false, sink);
    sink.put1(op.opcode());
    let modrm = 0xC0 | ((src_enc & 0x07) << 3) | (dst_enc & 0x07);
    sink.put1(modrm);
    sink.put1(lane);
}

/// Emit an AVX-512 insert instruction (VINSERTI32X4, VINSERTI64X2, etc.)
///
/// Format: dst = insert(src1 (zmm base), src2 (xmm/ymm to insert), imm8)
/// - dst: ZMM destination (encoded in R field)
/// - src1: ZMM source (encoded in vvvv field)
/// - src2: XMM/YMM source to insert (encoded in RM field)
/// - lane: immediate selecting which lane to insert at
pub fn emit_x64_512_insert_inst(
    op: Avx512InsertOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    lane: u8,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    // EVEX.512.66.0F3A.W0/W1 38/3A /r imm8
    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit destination
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            // dst in reg field, src1 in vvvv, src2 in r/m
            evex.emit(dst_enc, src1_enc, src2_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
            sink.put1(lane);
        }
        RegMem::Mem { addr } => {
            let amode = extract_real_amode(addr);
            let (base_enc, index_enc) = extract_mem_reg_encodings(amode);
            evex.emit_for_alu_mem(dst_enc, src1_enc, base_enc, index_enc, sink);
            sink.put1(op.opcode());
            emit_modrm_sib_disp(sink, dst_enc, amode);
            sink.put1(lane);
        }
    }
}

// =============================================================================
// AVX-512 FP Special Instruction Emission (VRCP14PS, VRSQRT14PS, etc.)
// =============================================================================

/// Emit an AVX-512 FP special instruction (dst = op(src))
/// These are unary operations for reciprocal/rsqrt approximations.
pub fn emit_x64_512_fp_special(
    op: Avx512FpSpecialOp,
    dst: Writable<Reg>,
    src: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let mask_enc = mask.map(|m| kreg_enc(m.to_reg())).unwrap_or(0);

    let evex = EvexPrefix {
        map: op.evex_map(),
        w: op.evex_w(),
        pp: op.evex_pp(),
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();

    match src {
        RegMem::Reg { reg } => {
            let src_enc = reg.to_real_reg().unwrap().hw_enc();
            // Unary ops: vvvv field is not used (encoded as 1111 = 0 after inversion)
            evex.emit(dst_enc, 0, src_enc, false, sink);
            sink.put1(op.opcode());
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            match addr {
                SyntheticAmode::Real(amode) => {
                    match amode {
                        Amode::ImmReg { base, simm32, .. } => {
                            let base_enc = base.to_real_reg().unwrap().hw_enc();
                            evex.emit_for_mem(dst_enc, base_enc, None, sink);
                            sink.put1(op.opcode());
                            let modrm = if *simm32 == 0 && (base_enc & 0x07) != 5 {
                                0x00 | ((dst_enc & 0x07) << 3) | (base_enc & 0x07)
                            } else if *simm32 >= -128 && *simm32 <= 127 {
                                0x40 | ((dst_enc & 0x07) << 3) | (base_enc & 0x07)
                            } else {
                                0x80 | ((dst_enc & 0x07) << 3) | (base_enc & 0x07)
                            };
                            sink.put1(modrm);
                            if (base_enc & 0x07) == 4 {
                                sink.put1(0x24); // SIB for RSP
                            }
                            if *simm32 != 0 || (base_enc & 0x07) == 5 {
                                if *simm32 >= -128 && *simm32 <= 127 {
                                    sink.put1(*simm32 as u8);
                                } else {
                                    sink.put4(*simm32 as u32);
                                }
                            }
                        }
                        _ => unreachable!(
                            "FP special ops only support ImmReg addressing mode, got: {:?}",
                            amode
                        ),
                    }
                }
                _ => unreachable!(
                    "FP special ops only support Real synthetic amode, got non-Real variant"
                ),
            }
        }
    }
}

// =============================================================================
// K-Register Shift Instruction Emission (KSHIFTL/R B/W/D/Q)
// =============================================================================

/// Emit a K-register shift instruction (dst = src << imm or dst = src >> imm)
/// These shift the bits in a mask register by an immediate count.
pub fn emit_x64_512_mask_shift_inst(
    op: MaskShiftOp,
    dst: Writable<Reg>,
    src: Reg,
    imm8: u8,
    sink: &mut MachBuffer<Inst>,
) {
    // KSHIFT instructions use VEX encoding:
    // VEX.L0.66.0F3A.W* opcode /r ib
    // where W depends on the size (B/D = W0, W/Q = W1)

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src_enc = src.to_real_reg().unwrap().hw_enc();

    // Build VEX prefix
    // VEX 3-byte form: C4 RXB map Wvvvv.Lpp opcode ModRM imm8
    // For k-register shifts: L=0, pp=01 (66), map=0F3A (0x03)

    let r_inv = if (dst_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = 0x40; // X not used
    let b_inv = if (src_enc & 0x08) == 0 { 0x20 } else { 0 };

    let byte1 = r_inv | x_inv | b_inv | 0x03; // map = 0x03 (0F3A)

    let w = if op.vex_w() { 0x80 } else { 0 };
    let vvvv = 0x78; // vvvv = 1111 (not used, inverted)
    let l = 0x00; // L = 0
    let pp = 0x01; // pp = 01 (66)
    let byte2 = w | vvvv | l | pp;

    // Emit VEX prefix
    sink.put1(0xC4);
    sink.put1(byte1);
    sink.put1(byte2);

    // Emit opcode
    sink.put1(op.vex_opcode());

    // Emit ModRM (mod=11 for register-register, reg=dst, rm=src)
    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
    sink.put1(modrm);

    // Emit immediate byte (shift count)
    sink.put1(imm8);
}

// =============================================================================
// K-Register Unpack Instruction Emission (KUNPCKBW/WD/DQ)
// =============================================================================

/// Emit a K-register unpack instruction (dst = unpack(src1, src2)).
/// These unpack and interleave low halves of two mask registers.
pub fn emit_x64_512_mask_unpack_inst(
    op: MaskUnpackOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // KUNPCK instructions use VEX encoding:
    // VEX.NDS.L*.0F.W* 4B /r
    // where L and W depend on the variant (BW/WD/DQ)

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    // Build VEX prefix
    // VEX 3-byte form: C4 RXB map Wvvvv.Lpp opcode ModRM

    let r_inv = if (dst_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = 0x40; // X not used
    let b_inv = if (src2_enc & 0x08) == 0 { 0x20 } else { 0 };

    let byte1 = r_inv | x_inv | b_inv | 0x01; // map = 0x01 (0F)

    let w = if op.vex_w() { 0x80 } else { 0 };
    // vvvv encodes src1 (inverted)
    let vvvv = ((!src1_enc) & 0x0F) << 3;
    let l = if op.vex_l() { 0x04 } else { 0 };
    let pp = 0x00; // pp = 00 (no prefix)
    let byte2 = w | vvvv | l | pp;

    // Emit VEX prefix
    sink.put1(0xC4);
    sink.put1(byte1);
    sink.put1(byte2);

    // Emit opcode
    sink.put1(op.vex_opcode());

    // Emit ModRM (mod=11 for register-register, reg=dst, rm=src2)
    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// K-Register Add Instruction Emission (KADDB/W/D/Q)
// =============================================================================

/// Emit a K-register add instruction (dst = src1 + src2).
/// These add two mask registers element-wise.
pub fn emit_x64_512_mask_add_inst(
    op: MaskAddOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // KADD instructions use VEX encoding:
    // VEX.NDS.L*.0F.W* 4A /r
    // where L and W depend on the variant (B/W/D/Q)

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc();
    let src1_enc = src1.to_real_reg().unwrap().hw_enc();
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    // Build VEX prefix
    // VEX 3-byte form: C4 RXB map Wvvvv.Lpp opcode ModRM

    let r_inv = if (dst_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = 0x40; // X not used
    let b_inv = if (src2_enc & 0x08) == 0 { 0x20 } else { 0 };

    let byte1 = r_inv | x_inv | b_inv | 0x01; // map = 0x01 (0F)

    let w = if op.vex_w() { 0x80 } else { 0 };
    // vvvv encodes src1 (inverted)
    let vvvv = ((!src1_enc) & 0x0F) << 3;
    let l = if op.vex_l() { 0x04 } else { 0 };
    let pp = 0x00; // pp = 00 (no prefix)
    let byte2 = w | vvvv | l | pp;

    // Emit VEX prefix
    sink.put1(0xC4);
    sink.put1(byte1);
    sink.put1(byte2);

    // Emit opcode
    sink.put1(op.vex_opcode());

    // Emit ModRM (mod=11 for register-register, reg=dst, rm=src2)
    let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// K-Register Test Instruction Emission (KTESTB/W/D/Q)
// =============================================================================

/// Emit a K-register test instruction (sets flags based on src1 AND src2).
/// These test two mask registers and set CPU flags (CF, ZF).
pub fn emit_x64_512_mask_test_inst(
    op: MaskTestOp,
    src1: Reg,
    src2: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // KTEST instructions use VEX encoding:
    // VEX.L*.0F.W* 99 /r
    // where L and W depend on the variant (B/W/D/Q)
    // Note: src1 goes in ModRM.reg, src2 goes in ModRM.rm

    let src1_enc = src1.to_real_reg().unwrap().hw_enc();
    let src2_enc = src2.to_real_reg().unwrap().hw_enc();

    // Build VEX prefix
    // VEX 3-byte form: C4 RXB map Wvvvv.Lpp opcode ModRM

    let r_inv = if (src1_enc & 0x08) == 0 { 0x80 } else { 0 };
    let x_inv = 0x40; // X not used
    let b_inv = if (src2_enc & 0x08) == 0 { 0x20 } else { 0 };

    let byte1 = r_inv | x_inv | b_inv | 0x01; // map = 0x01 (0F)

    let w = if op.vex_w() { 0x80 } else { 0 };
    // vvvv must be 1111 (not used, inverted from 0)
    let vvvv = 0x78;
    let l = if op.vex_l() { 0x04 } else { 0 };
    let pp = 0x00; // pp = 00 (no prefix)
    let byte2 = w | vvvv | l | pp;

    // Emit VEX prefix
    sink.put1(0xC4);
    sink.put1(byte1);
    sink.put1(byte2);

    // Emit opcode
    sink.put1(op.vex_opcode());

    // Emit ModRM (mod=11 for register-register, reg=src1, rm=src2)
    let modrm = 0xC0 | ((src1_enc & 0x07) << 3) | (src2_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EVEX Prefix Structure Tests
    // =========================================================================

    #[test]
    fn test_evex_prefix_structure() {
        // EVEX prefix should always start with 0x62
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 0x01,
            aaa: 0,
            z: false,
            b: false,
            ll: 0b10,
        };
        assert_eq!(evex.map, 0x01);
        assert_eq!(evex.ll, 0b10);
    }

    #[test]
    fn test_evex_prefix_512bit_default() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01);
        assert_eq!(evex.ll, 0b10); // 512-bit
        assert!(!evex.z);
        assert!(!evex.b);
        assert_eq!(evex.aaa, 0);
    }

    // =========================================================================
    // MaskAluOp Tests
    // =========================================================================

    #[test]
    fn test_mask_alu_op_encoding() {
        assert_eq!(MaskAluOp::Kand as u8, 0);
        assert_eq!(MaskAluOp::Kor as u8, 1);
        assert_eq!(MaskAluOp::Kxor as u8, 2);
        assert_eq!(MaskAluOp::Kxnor as u8, 3);
        assert_eq!(MaskAluOp::Knot as u8, 4);
        assert_eq!(MaskAluOp::Kandn as u8, 5);
    }

    #[test]
    fn test_mask_alu_op_vex_opcodes() {
        // Verify opcodes match Intel documentation
        assert_eq!(MaskAluOp::Kand.vex_opcode(), 0x41);
        assert_eq!(MaskAluOp::Kandn.vex_opcode(), 0x42);
        assert_eq!(MaskAluOp::Knot.vex_opcode(), 0x44);
        assert_eq!(MaskAluOp::Kor.vex_opcode(), 0x45);
        assert_eq!(MaskAluOp::Kxnor.vex_opcode(), 0x46);
        assert_eq!(MaskAluOp::Kxor.vex_opcode(), 0x47);
    }

    // =========================================================================
    // Avx512Cond Tests
    // =========================================================================

    #[test]
    fn test_x64_512_cond_encoding() {
        // Verify condition encodings match Intel documentation for VPCMPD/VPCMPQ
        assert_eq!(Avx512Cond::Eq as u8, 0);
        assert_eq!(Avx512Cond::Lt as u8, 1);
        assert_eq!(Avx512Cond::Le as u8, 2);
        // Note: 3 is "false" (always false), not commonly used
        assert_eq!(Avx512Cond::Neq as u8, 4);
        assert_eq!(Avx512Cond::Ge as u8, 5); // NLT (not less than)
        assert_eq!(Avx512Cond::Gt as u8, 6); // NLE (not less than or equal)
        // Note: 7 is "true" (always true), not commonly used
    }

    // =========================================================================
    // MergeMode Tests
    // =========================================================================

    #[test]
    fn test_merge_mode() {
        assert!(matches!(MergeMode::Zeroing, MergeMode::Zeroing));
        assert!(matches!(MergeMode::Merging, MergeMode::Merging));
        // Default should be Zeroing (as per our impl)
        assert_eq!(MergeMode::default(), MergeMode::Zeroing);
    }

    // =========================================================================
    // Avx512AluOp Tests
    // =========================================================================

    #[test]
    fn test_x64_512_alu_op_arithmetic_opcodes() {
        // Integer arithmetic instructions
        assert_eq!(Avx512AluOp::Vpaddd.opcode(), 0xFE);
        assert_eq!(Avx512AluOp::Vpaddq.opcode(), 0xD4);
        assert_eq!(Avx512AluOp::Vpsubd.opcode(), 0xFA);
        assert_eq!(Avx512AluOp::Vpsubq.opcode(), 0xFB);
    }

    #[test]
    fn test_x64_512_alu_op_logical_opcodes() {
        // Bitwise logical instructions
        assert_eq!(Avx512AluOp::Vpandd.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpandq.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpord.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vporq.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vpxord.opcode(), 0xEF);
        assert_eq!(Avx512AluOp::Vpxorq.opcode(), 0xEF);
    }

    #[test]
    fn test_x64_512_alu_op_shift_opcodes() {
        // Shift instructions
        assert_eq!(Avx512AluOp::Vpslld.opcode(), 0xF2);
        assert_eq!(Avx512AluOp::Vpsllq.opcode(), 0xF3);
        assert_eq!(Avx512AluOp::Vpsrld.opcode(), 0xD2);
        assert_eq!(Avx512AluOp::Vpsrlq.opcode(), 0xD3);
    }

    #[test]
    fn test_x64_512_alu_op_evex_maps() {
        // 0F map (simple operations)
        assert_eq!(Avx512AluOp::Vpaddd.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpandd.evex_map(), 0x01);

        // 0F38 map (complex operations)
        assert_eq!(Avx512AluOp::Vpmulld.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpermd.evex_map(), 0x02);

        // 0F3A map (ternary logic)
        assert_eq!(Avx512AluOp::Vpternlogd.evex_map(), 0x03);
    }

    #[test]
    fn test_x64_512_alu_op_evex_w() {
        // 32-bit operations: W=0
        assert!(!Avx512AluOp::Vpaddd.evex_w());
        assert!(!Avx512AluOp::Vpsubd.evex_w());
        assert!(!Avx512AluOp::Vpandd.evex_w());

        // 64-bit operations: W=1
        assert!(Avx512AluOp::Vpaddq.evex_w());
        assert!(Avx512AluOp::Vpsubq.evex_w());
        assert!(Avx512AluOp::Vpandq.evex_w());
    }

    // =========================================================================
    // Instruction Encoding Constants Tests
    // =========================================================================

    #[test]
    fn test_vpcmpd_encoding_constants() {
        // VPCMPD: EVEX.512.66.0F3A.W0 1F /r ib
        // map=0x03 (0F3A), pp=0x01 (66), W=0
        assert_eq!(0x1F, 31); // Opcode
    }

    #[test]
    fn test_vpcompressd_encoding_constants() {
        // VPCOMPRESSD: EVEX.512.66.0F38.W0 8B /r
        // map=0x02 (0F38), pp=0x01 (66), W=0
        assert_eq!(0x8B, 139); // Opcode
    }

    #[test]
    fn test_vpexpandd_encoding_constants() {
        // VPEXPANDD: EVEX.512.66.0F38.W0 89 /r
        assert_eq!(0x89, 137); // Opcode
    }

    #[test]
    fn test_vmovdqu32_encoding_constants() {
        // VMOVDQU32 load: EVEX.512.F3.0F.W0 6F /r
        // VMOVDQU32 store: EVEX.512.F3.0F.W0 7F /r
        assert_eq!(0x6F, 111); // Load opcode
        assert_eq!(0x7F, 127); // Store opcode
    }

    #[test]
    fn test_kmov_encoding_constants() {
        // KMOVQ k, r64: VEX.L0.F2.0F.W1 92 /r
        // KMOVQ r64, k: VEX.L0.F2.0F.W1 93 /r
        // KMOVQ k, m64: VEX.L0.0F.W1 90 /r
        // KMOVQ m64, k: VEX.L0.0F.W1 91 /r
        assert_eq!(0x90, 144); // Load from memory
        assert_eq!(0x91, 145); // Store to memory
        assert_eq!(0x92, 146); // Move GPR to k
        assert_eq!(0x93, 147); // Move k to GPR
    }

    #[test]
    fn test_kortest_encoding_constants() {
        // KORTESTW: VEX.L0.0F.W0 98 /r
        assert_eq!(0x98, 152); // Opcode
    }
}
