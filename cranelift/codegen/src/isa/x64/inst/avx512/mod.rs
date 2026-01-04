// cranelift/codegen/src/isa/x64/inst/avx512/mod.rs
//
// AVX-512: Manual implementations for AVX-512 instructions.
//
// Most AVX-512 instructions are now DSL-generated in assembler-x64.
// This module contains manual implementations for instructions that
// require special handling:
//
// - `defs`: Type definitions (opcodes, enums) for manual instructions
// - `emit`: EVEX emission for gather/scatter (VSIB), vp2intersect, k-register spill/fill
// - `encoding`: EVEX prefix encoding utilities
// - `regs`: K-register utilities

pub mod defs;
pub mod emit;
pub mod encoding;
pub mod regs;

pub use defs::*;
