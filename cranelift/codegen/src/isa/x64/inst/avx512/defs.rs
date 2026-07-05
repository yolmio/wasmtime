// cranelift/codegen/src/isa/x64/inst/avx512/defs.rs
//
// Type definitions for the few AVX-512 instructions that still have manual
// (hand-written) emission paths. Everything else — the bulk of the AVX-512
// instruction set — is DSL-generated in `cranelift-assembler-x64` and needs
// no definitions here.
//
// The manual layer consists of:
// - `GatherOp` / `ScatterOp`: VSIB gather/scatter (VPGATHER*/VPSCATTER*)
// - `Vp2IntersectOp` / `KRegPair`: VP2INTERSECTD/Q with its dual k-register
//   destination pair
// - `Avx512Cond`: comparison-immediate encoding shared with DSL VPCMP*
//   constructors via `avx512_cond_to_u8`

/// AVX-512 comparison conditions (for VPCMPD, VPCMPQ, etc.)
///
/// The discriminants are the architectural immediate-byte encodings; the
/// ISLE extern constructor `avx512_cond_to_u8` relies on `*cond as u8`
/// producing exactly these values.
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
            // - 0x91 for 64-bit indices (Q): VPGATHERQQ
            // The W bit (element size) is a separate encoding field.
            GatherOp::Vpgatherdd | GatherOp::Vpgatherdq => 0x90,
            GatherOp::Vpgatherqq => 0x91,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            GatherOp::Vpgatherdd => false, // W=0 for 32-bit elements
            GatherOp::Vpgatherdq | GatherOp::Vpgatherqq => true, // W=1 for 64-bit elements
        }
    }

    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u8 {
        match self {
            GatherOp::Vpgatherdd => 4,
            GatherOp::Vpgatherdq | GatherOp::Vpgatherqq => 8,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            GatherOp::Vpgatherdd => "vpgatherdd",
            GatherOp::Vpgatherdq => "vpgatherdq",
            GatherOp::Vpgatherqq => "vpgatherqq",
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
            // - 0xA1 for 64-bit indices (Q): VPSCATTERQQ
            // The W bit (element size) is a separate encoding field.
            ScatterOp::Vpscatterdd | ScatterOp::Vpscatterdq => 0xA0,
            ScatterOp::Vpscatterqq => 0xA1,
        }
    }

    /// Returns the EVEX.W bit for this operation.
    pub fn evex_w(&self) -> bool {
        match self {
            ScatterOp::Vpscatterdd => false, // W=0 for 32-bit elements
            ScatterOp::Vpscatterdq | ScatterOp::Vpscatterqq => true, // W=1 for 64-bit elements
        }
    }

    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u8 {
        match self {
            ScatterOp::Vpscatterdd => 4,
            ScatterOp::Vpscatterdq | ScatterOp::Vpscatterqq => 8,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            ScatterOp::Vpscatterdd => "vpscatterdd",
            ScatterOp::Vpscatterdq => "vpscatterdq",
            ScatterOp::Vpscatterqq => "vpscatterqq",
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

/// An even-aligned, adjacent pair of *allocatable* AVX-512 mask registers.
///
/// VP2INTERSECTD/Q writes TWO consecutive k-registers: the destination
/// encoded in ModRM.reg (whose LSB is ignored by hardware, i.e. it must be
/// even) and its odd successor. regalloc2 has no native register-pair
/// support, so instructions that need the pair pin both halves with fixed
/// constraints derived from this type.
///
/// Type-safety invariant: the only inhabitants are the three even/odd
/// adjacent pairs among the allocatable mask registers k2-k7 (k0 is the
/// hardwired "no mask" register and k1 is only ever used as the
/// value-independent source of the `kxnorw k1, k1, kN` all-ones mask
/// idiom, so gather/scatter masks allocate from k2-k7; neither k0 nor k1
/// is in the register allocator's `MachineEnv`).
/// An odd-based or non-adjacent pair is unrepresentable, so evenness never
/// needs to be `debug_assert!`ed at emission time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KRegPair {
    /// k2 (even, low) and k3 (high).
    K2K3,
    /// k4 (even, low) and k5 (high).
    K4K5,
    /// k6 (even, low) and k7 (high).
    K6K7,
}

impl KRegPair {
    /// The even (low) k-register of the pair, as a real `Reg` of
    /// `RegClass::Vector`.
    pub fn low(self) -> crate::machinst::Reg {
        use crate::isa::x64::inst::regs;
        match self {
            KRegPair::K2K3 => regs::k2(),
            KRegPair::K4K5 => regs::k4(),
            KRegPair::K6K7 => regs::k6(),
        }
    }

    /// The odd (high) k-register of the pair, as a real `Reg` of
    /// `RegClass::Vector`.
    pub fn high(self) -> crate::machinst::Reg {
        use crate::isa::x64::inst::regs;
        match self {
            KRegPair::K2K3 => regs::k3(),
            KRegPair::K4K5 => regs::k5(),
            KRegPair::K6K7 => regs::k7(),
        }
    }

    /// Hardware encoding of the even (low) k-register; structurally even by
    /// construction.
    pub fn low_enc(self) -> u8 {
        match self {
            KRegPair::K2K3 => 2,
            KRegPair::K4K5 => 4,
            KRegPair::K6K7 => 6,
        }
    }

    /// Human-readable name of the pair for pretty-printing.
    pub fn name(self) -> &'static str {
        match self {
            KRegPair::K2K3 => "%k2/%k3",
            KRegPair::K4K5 => "%k4/%k5",
            KRegPair::K6K7 => "%k6/%k7",
        }
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x64_512_cond_encoding() {
        // Verify condition encodings match Intel documentation for
        // VPCMPD/VPCMPQ; the ISLE extern constructor `avx512_cond_to_u8`
        // relies on these discriminants (`*cond as u8`).
        assert_eq!(Avx512Cond::Eq as u8, 0);
        assert_eq!(Avx512Cond::Lt as u8, 1);
        assert_eq!(Avx512Cond::Le as u8, 2);
        // Note: 3 is "false" (always false), not representable on purpose.
        assert_eq!(Avx512Cond::Neq as u8, 4);
        assert_eq!(Avx512Cond::Ge as u8, 5); // NLT (not less than)
        assert_eq!(Avx512Cond::Gt as u8, 6); // NLE (not less than or equal)
        // Note: 7 is "true" (always true), not representable on purpose.
    }
}
