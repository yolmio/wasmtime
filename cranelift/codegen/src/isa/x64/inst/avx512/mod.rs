// cranelift/codegen/src/isa/x64/inst/avx512/mod.rs
//
// Manual implementations for the few AVX-512 instructions the assembler DSL
// (`cranelift-assembler-x64`) cannot express. The bulk of the AVX-512
// instruction set is DSL-generated and does NOT live here.
//
// The hand-written layer consists of exactly:
//
// - VSIB gather/scatter (VPGATHERD/Q*, VPSCATTERD/Q*): the DSL has no VSIB
//   addressing mode.
// - VP2INTERSECTD/Q: writes a dual k-register destination pair (`KRegPair`),
//   which the DSL cannot model.
// - K-register spill/reload/copy (KMOVQ k,k / k,m64 / m64,k): used by the
//   register allocator.
//
// Modules:
//
// - `defs`: operand/opcode types for the instructions above (`GatherOp`,
//   `ScatterOp`, `Vp2IntersectOp`, `KRegPair`) plus `Avx512Cond`, whose
//   discriminants feed the DSL VPCMP* constructors via `avx512_cond_to_u8`
// - `emit`: the emission functions themselves
// - `encoding`: EVEX prefix emission (`EvexPrefix`) and compressed
//   displacement classification (`EvexDisp`) for the manual EVEX paths

pub mod defs;
pub mod emit;
pub mod encoding;

pub use defs::*;
