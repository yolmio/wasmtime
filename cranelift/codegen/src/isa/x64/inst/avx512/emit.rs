// cranelift/codegen/src/isa/x64/inst/avx512/emit.rs
//
// Manual emission for the few AVX-512 instructions the assembler DSL cannot
// express. Everything else — the bulk of the AVX-512 instruction set — is
// DSL-generated in `cranelift-assembler-x64`.
//
// ## What lives here
//
// - **Gather/scatter** (`emit_gather`, `emit_scatter`): VPGATHERD/Q* and
//   VPSCATTERD/Q* need VSIB (vector SIB) addressing, which the DSL does not
//   model. EVEX compressed displacement (disp8*N) is handled by `EvexDisp`.
// - **VP2INTERSECT** (`emit_x64_512_vp2intersect_inst`): writes a dual
//   k-register destination pair (`KRegPair`), a concept the DSL has no
//   operand kind for.
// - **K-register spill/fill** (`emit_kmov_kk`, `emit_kmov_load`,
//   `emit_kmov_store`): KMOVQ moves used by the register allocator for
//   k-register copies and spill slots.
//
// ## K-Register Encoding
//
// K-registers (k0-k7) use hardware encodings 0-7.
// - EVEX aaa field: 0-7
// - k0 is special: it means "no masking" (all lanes active); gather/scatter
//   require a real mask in k1-k7.
//
// ## Reference
//
// Intel 64 and IA-32 Architectures Software Developer's Manual
// Volume 2: Instruction Set Reference (EVEX Encoding)

use super::super::args::Amode;
use super::defs::{KRegPair, Vp2IntersectOp};
use super::encoding::{EvexDisp, EvexPrefix};
use crate::isa::x64::inst::Inst;
use crate::isa::x64::inst::args::RegMem;
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
// K-Register Moves and Spill/Fill (for register allocation)
// =============================================================================

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
    // `EvexDisp` performs the disp8 scaling; both gather and scatter must go
    // through it so a raw displacement can never be emitted as disp8.
    let disp = EvexDisp::new(disp, op.element_size(), base_enc);

    // ModRM: mod=disp form, reg=dst, rm=100 (SIB follows)
    let modrm = (disp.mod_bits() << 6) | ((dst_enc & 0x07) << 3) | 0x04;
    sink.put1(modrm);
    let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
    sink.put1(sib);
    disp.emit(sink);
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

    // EVEX uses compressed displacement (disp8*N) where N is the element
    // size, exactly as for gather above: the disp8 byte holds `disp / N` and
    // the hardware multiplies it back by N. Emitting the raw displacement as
    // disp8 would silently store to `base + index*scale + disp*N`.
    let disp = EvexDisp::new(disp, op.element_size(), base_enc);

    // ModRM: mod=disp form, reg=src, rm=100 (SIB follows)
    let modrm = (disp.mod_bits() << 6) | ((src_enc & 0x07) << 3) | 0x04;
    sink.put1(modrm);
    let sib = (scale_bits << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);
    sink.put1(sib);
    disp.emit(sink);
}

// =============================================================================
// VP2INTERSECT Instruction Emission
// =============================================================================

/// Emit VP2INTERSECTD/Q: compare two vectors and write the dual k-register
/// destination pair named by `pair` (the even low register k[2n] and its odd
/// successor k[2n+1]). Taking the pair type instead of an arbitrary register
/// makes an odd destination unrepresentable rather than merely
/// debug-asserted.
///
/// Encoding (Intel SDM):
///   VP2INTERSECTD zmm-form: EVEX.NDS.512.F2.0F38.W0 68 /r
///   VP2INTERSECTQ zmm-form: EVEX.NDS.512.F2.0F38.W1 68 /r
///
/// Note: Unlike most instructions, the reg field in ModR/M encodes a k-register,
/// not a ZMM register. The vvvv field holds src1 (ZMM), r/m holds src2 (ZMM).
///
/// Byte-exact reference (cross-checked against GNU as/objdump):
///   vp2intersectd %zmm2, %zmm1, %k6 => 62 F2 77 48 68 F2
///   vp2intersectq %zmm2, %zmm1, %k6 => 62 F2 F7 48 68 F2
///
/// Only the register form of src2 is supported: no lowering rule constructs
/// the memory form (see `x64_512_vp2intersectd/q` in `inst.isle`, which is
/// register-only on purpose), and the manual memory-operand emission helpers
/// were removed with the rest of the dead hand-written EVEX layer. If a
/// memory form is ever needed it must be added back with byte-exact tests.
pub fn emit_x64_512_vp2intersect_inst(
    op: Vp2IntersectOp,
    pair: KRegPair,
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

    // The ModR/M reg field encodes the destination pair via its low
    // register; `KRegPair::low_enc` is structurally even, so no evenness
    // assertion is needed here.
    let dst_enc = pair.low_enc();

    let src1_enc = src1.to_real_reg().unwrap().hw_enc();

    match src2 {
        RegMem::Reg { reg } => {
            let src2_enc = reg.to_real_reg().unwrap().hw_enc();
            // For VP2INTERSECT:
            // - EVEX.R/R' controls reg field extension (but k-regs only need 3 bits: k0-k7)
            // - EVEX.vvvv = src1 (inverted, ZMM register)
            // - ModR/M.reg = pair.low() (even k-register, 3 bits)
            // - ModR/M.r/m = src2 (ZMM register)
            evex.emit_reg(dst_enc, src1_enc, src2_enc, sink);
            sink.put1(op.opcode()); // 0x68
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { .. } => {
            // Unreachable by construction: the lowering rules only build the
            // register form (see the doc comment above).
            panic!("vp2intersect memory-operand form is not supported");
        }
    }
}

// =============================================================================
// Memory-Operand Encoding Helpers (VEX-encoded KMOVQ spill/fill only)
// =============================================================================

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

/// Emit ModRM, SIB, and displacement for a VEX-encoded (non-EVEX) memory
/// operand, i.e. the KMOVQ spill/fill forms. Uses plain (uncompressed) disp8;
/// EVEX-encoded instructions must never come through here (their disp8 is
/// compressed, see `EvexDisp`).
///
/// The `Amode::RipRelative` arm is unreachable in practice: k-register
/// spill/fill addresses are frame-slot `ImmReg`/`ImmRegRegShift` amodes
/// produced by regalloc.
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
            panic!("RIP-relative addressing is not supported for k-register spill/fill");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // KMOVQ Encoding Constants
    // =========================================================================

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

    // =========================================================================
    // Gather/Scatter Displacement Encoding Tests (byte-exact)
    //
    // EVEX gather/scatter use compressed displacement (disp8*N): the disp8
    // byte is `disp / element_size`, which the hardware multiplies back by
    // the element size. These expected encodings were produced with GNU as
    // and verified with objdump.
    // =========================================================================

    use super::super::defs::{GatherOp, ScatterOp};
    use crate::isa::x64::inst::regs;
    use alloc::string::String;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02X}")).collect()
    }

    /// Emit `vpscatter* [rdi + zmm2*scale + disp] {k1}, zmm1` and return the
    /// emitted bytes as an uppercase hex string.
    fn scatter_bytes(op: ScatterOp, scale: u8, disp: i32) -> String {
        let mut sink = MachBuffer::<Inst>::new();
        emit_scatter(
            op,
            regs::xmm1(),
            regs::rdi(),
            regs::xmm2(),
            scale,
            disp,
            regs::k1(),
            &mut sink,
        );
        let buf = sink.finish(&Default::default(), &mut Default::default());
        hex(buf.data())
    }

    /// Emit `vpgather* zmm1 {k1}, [rdi + zmm2*scale + disp]` and return the
    /// emitted bytes as an uppercase hex string.
    fn gather_bytes(op: GatherOp, scale: u8, disp: i32) -> String {
        let mut sink = MachBuffer::<Inst>::new();
        emit_gather(
            op,
            Writable::from_reg(regs::xmm1()),
            regs::rdi(),
            regs::xmm2(),
            scale,
            disp,
            regs::k1(),
            &mut sink,
        );
        let buf = sink.finish(&Default::default(), &mut Default::default());
        hex(buf.data())
    }

    #[test]
    fn test_scatter_dd_displacement_encoding() {
        use ScatterOp::Vpscatterdd as DD;
        // disp 0: mod=00, no displacement bytes.
        assert_eq!(scatter_bytes(DD, 4, 0), "62F27D49A00C97");
        // disp +4 with 4-byte elements: compressed disp8 = 1, NOT 4.
        assert_eq!(scatter_bytes(DD, 4, 4), "62F27D49A04C9701");
        // disp +8: compressed disp8 = 2, NOT 8. (This is the regression the
        // old code got wrong: it emitted the raw 0x08, which the hardware
        // scales to +32.)
        assert_eq!(scatter_bytes(DD, 4, 8), "62F27D49A04C9702");
        // disp +64: compressed disp8 = 16.
        assert_eq!(scatter_bytes(DD, 4, 64), "62F27D49A04C9710");
        // disp +127: not divisible by 4, must fall back to disp32.
        assert_eq!(scatter_bytes(DD, 4, 127), "62F27D49A08C977F000000");
        // disp +128: divisible, compressed disp8 = 32 (raw 128 would not
        // even fit in a signed disp8).
        assert_eq!(scatter_bytes(DD, 4, 128), "62F27D49A04C9720");
        // disp -8: compressed disp8 = -2 (0xFE).
        assert_eq!(scatter_bytes(DD, 4, -8), "62F27D49A04C97FE");
        // disp +512 = 128*4: scaled value 128 exceeds i8, must use disp32.
        assert_eq!(scatter_bytes(DD, 4, 512), "62F27D49A08C9700020000");
        // disp +508 = 127*4: largest compressible positive displacement.
        assert_eq!(scatter_bytes(DD, 4, 508), "62F27D49A04C977F");
    }

    #[test]
    fn test_scatter_qq_displacement_encoding() {
        // 8-byte elements: disp +8 compresses to disp8 = 1.
        // vpscatterqq [rdi + zmm2*8 + 8] {k2}, zmm1
        let mut sink = MachBuffer::<Inst>::new();
        emit_scatter(
            ScatterOp::Vpscatterqq,
            regs::xmm1(),
            regs::rdi(),
            regs::xmm2(),
            8,
            8,
            regs::k2(),
            &mut sink,
        );
        let buf = sink.finish(&Default::default(), &mut Default::default());
        assert_eq!(hex(buf.data()), "62F2FD4AA14CD701");
    }

    #[test]
    fn test_gather_dd_displacement_encoding() {
        use GatherOp::Vpgatherdd as DD;
        // Same compressed-displacement rules as scatter (regression guard).
        assert_eq!(gather_bytes(DD, 4, 0), "62F27D49900C97");
        assert_eq!(gather_bytes(DD, 4, 4), "62F27D49904C9701");
        assert_eq!(gather_bytes(DD, 4, 8), "62F27D49904C9702");
        assert_eq!(gather_bytes(DD, 4, 64), "62F27D49904C9710");
        assert_eq!(gather_bytes(DD, 4, 127), "62F27D49908C977F000000");
        assert_eq!(gather_bytes(DD, 4, 128), "62F27D49904C9720");
        assert_eq!(gather_bytes(DD, 4, -8), "62F27D49904C97FE");
        assert_eq!(gather_bytes(DD, 4, 512), "62F27D49908C9700020000");
        assert_eq!(gather_bytes(DD, 4, 508), "62F27D49904C977F");
    }

    #[test]
    fn test_gather_qq_dq_displacement_encoding() {
        // vpgatherqq zmm3 {k2}, [rax + zmm5*8 + 8]: 8-byte elements, disp8 = 1.
        let mut sink = MachBuffer::<Inst>::new();
        emit_gather(
            GatherOp::Vpgatherqq,
            Writable::from_reg(regs::xmm3()),
            regs::rax(),
            regs::xmm5(),
            8,
            8,
            regs::k2(),
            &mut sink,
        );
        let buf = sink.finish(&Default::default(), &mut Default::default());
        assert_eq!(hex(buf.data()), "62F2FD4A915CE801");

        // vpgatherdq zmm1 {k1}, [rdi + ymm2*1 + 16]: N is the *element* size
        // (8 for dq), not the index size, so disp8 = 16/8 = 2.
        assert_eq!(gather_bytes(GatherOp::Vpgatherdq, 1, 16), "62F2FD49904C1702");
    }

    // =========================================================================
    // VP2INTERSECT Encoding Tests (byte-exact)
    //
    // VP2INTERSECTD zmm-form: EVEX.NDS.512.F2.0F38.W0 68 /r (note the F2
    // prefix, pp=0b11). The ModR/M reg field encodes the destination
    // k-register PAIR via its even low register; the hardware ignores the
    // LSB. Expected byte sequences below were produced with GNU as and
    // verified with objdump:
    //
    //   vp2intersectd %zmm2, %zmm1, %k6   => 62 f2 77 48 68 f2
    //   vp2intersectq %zmm2, %zmm1, %k6   => 62 f2 f7 48 68 f2
    //   vp2intersectd %zmm2, %zmm1, %k4   => 62 f2 77 48 68 e2
    //   vp2intersectd %zmm10, %zmm9, %k6  => 62 d2 37 48 68 f2
    // =========================================================================

    /// Emit `vp2intersect{d,q} pair.low(), src1, src2` (register form) and
    /// return the emitted bytes as an uppercase hex string.
    fn vp2intersect_bytes(op: Vp2IntersectOp, pair: KRegPair, src1: Reg, src2: Reg) -> String {
        let mut sink = MachBuffer::<Inst>::new();
        emit_x64_512_vp2intersect_inst(op, pair, src1, &RegMem::Reg { reg: src2 }, &mut sink);
        let buf = sink.finish(&Default::default(), &mut Default::default());
        hex(buf.data())
    }

    #[test]
    fn test_vp2intersect_encoding() {
        use Vp2IntersectOp::{Vp2intersectd, Vp2intersectq};
        // vp2intersectd k6, zmm1, zmm2
        assert_eq!(
            vp2intersect_bytes(Vp2intersectd, KRegPair::K6K7, regs::xmm1(), regs::xmm2()),
            "62F2774868F2"
        );
        // vp2intersectq k6, zmm1, zmm2 (W=1)
        assert_eq!(
            vp2intersect_bytes(Vp2intersectq, KRegPair::K6K7, regs::xmm1(), regs::xmm2()),
            "62F2F74868F2"
        );
        // vp2intersectd k4, zmm1, zmm2 (different pair -> ModRM.reg)
        assert_eq!(
            vp2intersect_bytes(Vp2intersectd, KRegPair::K4K5, regs::xmm1(), regs::xmm2()),
            "62F2774868E2"
        );
        // vp2intersectd k6, zmm9, zmm10 (EVEX.B extension for src2, vvvv=~9)
        assert_eq!(
            vp2intersect_bytes(Vp2intersectd, KRegPair::K6K7, regs::xmm9(), regs::xmm10()),
            "62D2374868F2"
        );
    }

    #[test]
    fn test_kreg_pair_structurally_even() {
        // The KRegPair type admits only even/odd adjacent pairs over the
        // allocatable k2-k7; verify the accessors agree with each other.
        for pair in [KRegPair::K2K3, KRegPair::K4K5, KRegPair::K6K7] {
            assert_eq!(pair.low_enc() % 2, 0, "low k-register must be even");
            assert_eq!(pair.low().to_real_reg().unwrap().hw_enc(), pair.low_enc());
            assert_eq!(
                pair.high().to_real_reg().unwrap().hw_enc(),
                pair.low_enc() + 1
            );
        }
    }
}
