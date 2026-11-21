//! The shader interface: entry point, interface leaves, uniforms,
//! samplers — built from the validated [`Sema`](glcc_frontend::sema::Sema).
//!
//! Entry points take and return interface structs whose leaves carry
//! `[[prism::semantic(NAME)]]`; inputs flatten to one `v`-register per
//! leaf and outputs to one `o`-slot per leaf, both in declaration
//! order. Uniforms get two layout tables computed from one leaf walk:
//! c-register packing for the ISA targets (a vector leaf takes a
//! whole c-register, scalars pack four per register, declaration order
//! decides, one global cursor across all uniforms) and std140 offsets
//! for SPIR-V's UBO form. Samplers get binding slots in declaration
//! order.

use std::format;
use std::string::{String, ToString};
use std::vec;

use glcc_frontend::lexer::Interner;
use glcc_frontend::parse::{ArrayBound, PrismStage, Ty};
use glcc_frontend::sema::{BindingKind, Sema, StructId};

use crate::ir::{RegTy, Scalar, Stage};

/// One flattened interface leaf: a scalar or vector with its hardware
/// binding. Inputs occupy one `v`-register each (the index in
/// [`Interface::inputs`]); outputs are written by [`crate::ir::Stmt::Exit`]
/// in [`Interface::outputs`] order.
#[derive(Debug, Clone)]
pub struct InterfaceLeaf {
    pub semantic: String,
    pub ty: RegTy,
}

/// A host-set uniform object with its two layout tables. `creg` is
/// the first c-register the object occupies; leaves carry each
/// readable piece's member path, register type, absolute c-register,
/// and first lane; std140 offsets are relative to the object's own
/// base (each uniform is its own UBO for SPIR-V).
#[derive(Debug, Clone)]
pub struct UniformVar {
    pub name: String,
    pub creg: u8,
    pub leaves: Vec<UniformLeaf>,
    pub std140_size: u32,
}

/// One readable piece of a uniform object.
#[derive(Debug, Clone)]
pub struct UniformLeaf {
    /// Member indices (and array element indices) from the object root.
    pub path: Vec<u32>,
    pub ty: RegTy,
    /// Absolute c-register.
    pub creg: u8,
    /// First lane of the register the leaf occupies (scalars pack
    /// four per register).
    pub lane: u8,
    /// std140 byte offset within the object.
    pub std140_offset: u32,
}

/// A host-bound combined texture+sampler.
#[derive(Debug, Clone)]
pub struct SamplerVar {
    pub name: String,
}

/// The interface of one shader program.
#[derive(Debug, Clone)]
pub struct Interface {
    pub stage: Stage,
    pub entry: String,
    pub inputs: Vec<InterfaceLeaf>,
    pub outputs: Vec<InterfaceLeaf>,
    pub uniforms: Vec<UniformVar>,
    pub samplers: Vec<SamplerVar>,
}

impl Interface {
    /// The uniform object a MIR global name refers to, if any.
    pub fn uniform_by_name(&self, name: &str) -> Option<u32> {
        self.uniforms.iter().position(|u| u.name == name).map(|i| i as u32)
    }

    /// The sampler a MIR global name refers to, if any.
    pub fn sampler_by_name(&self, name: &str) -> Option<u32> {
        self.samplers.iter().position(|s| s.name == name).map(|i| i as u32)
    }

    /// The uniform leaf at `path` within uniform `object`, for mapping
    /// field reads.
    pub fn uniform_leaf(&self, object: u32, path: &[u32]) -> Option<&UniformLeaf> {
        let uniform = &self.uniforms[object as usize];
        uniform.leaves.iter().find(|leaf| leaf.path == path)
    }

    /// The c-register a uniform leaf reads (its own, or the object's
    /// base plus the leaf offset — leaves store absolute registers, so
    /// the leaf's own).
    pub fn uniform_creg(&self, object: u32, path: &[u32]) -> Option<(u8, u8)> {
        let leaf = self.uniform_leaf(object, path)?;
        Some((leaf.creg, leaf.lane))
    }
}

/// Build the interface of the (single) entry point of a validated
/// prism translation unit. The checker guarantees: at least one
/// defined entry point with external linkage, interface structs whose
/// leaves all carry semantics, no pointers, and file-scope samplers.
pub fn build(sema: &Sema, interner: &Interner) -> Result<Interface, String> {
    let (name, stage, param_tys, ret_ty) = find_entry(sema, interner)?;
    let mut interface = Interface {
        stage: match stage {
            PrismStage::Vertex => Stage::Vertex,
            PrismStage::Fragment => Stage::Fragment,
        },
        entry: name,
        inputs: Vec::new(),
        outputs: Vec::new(),
        uniforms: Vec::new(),
        samplers: Vec::new(),
    };
    for param in &param_tys {
        flatten_interface(sema, interner, param, &mut interface.inputs)?;
    }
    flatten_interface(sema, interner, &ret_ty, &mut interface.outputs)?;
    collect_resources(sema, interner, &mut interface)?;
    pack_uniform_cregs(&mut interface.uniforms);
    Ok(interface)
}

fn find_entry(sema: &Sema, interner: &Interner) -> Result<(String, PrismStage, Vec<Ty>, Ty), String> {
    let mut found: Option<(String, PrismStage, Vec<Ty>, Ty)> = None;
    for (id, attrs) in &sema.prism_attrs {
        let Some(stage) = attrs.stage else { continue };
        let binding = &sema.decls[id.0 as usize];
        if binding.kind != BindingKind::Func {
            return Err("prism entry point attributes must be on functions".to_string());
        }
        let name = name_of(sema, interner, id.0);
        if found.is_some() {
            return Err(format!(
                "multiple prism entry points ('{}' and '{}'): one entry point per shader",
                found.as_ref().unwrap().0,
                name
            ));
        }
        let Ty::Func { ret, params, .. } = binding.ty.unqualified() else {
            return Err(format!("'{name}' is not a function"));
        };
        found = Some((
            name,
            stage,
            params.iter().map(|p| p.unqualified().clone()).collect(),
            ret.unqualified().clone(),
        ));
    }
    found.ok_or_else(|| "no prism entry point: a shader needs [[prism::vertex]] or [[prism::fragment]]".to_string())
}

fn name_of(sema: &Sema, interner: &Interner, decl: u32) -> String {
    sema.decls[decl as usize]
        .name
        .map(|n| interner.resolve(n).unwrap_or("?").to_string())
        .unwrap_or_else(|| "?".to_string())
}

/// Flatten an interface struct into its semantic leaves. `void` (a
/// no-input parameter list or a bare `return`) contributes nothing;
/// anything else must be an interface struct — the checker enforced
/// this, so a violation here is a bug, not a diagnostic.
fn flatten_interface(sema: &Sema, interner: &Interner, ty: &Ty, out: &mut Vec<InterfaceLeaf>) -> Result<(), String> {
    match ty.unqualified() {
        Ty::Void => Ok(()),
        Ty::Struct(id) => flatten_struct(sema, interner, *id, out),
        _ => Err("interface types must be structs".to_string()),
    }
}

fn flatten_struct(sema: &Sema, interner: &Interner, id: StructId, out: &mut Vec<InterfaceLeaf>) -> Result<(), String> {
    let def = &sema.structs[id.0 as usize];
    for member in &def.members {
        match member.ty.unqualified() {
            Ty::Struct(inner) if member.name.is_none() => flatten_struct(sema, interner, *inner, out)?,
            _ => {
                let semantic = member
                    .prism
                    .semantic
                    .map(|s| interner.resolve(s).unwrap_or("?").to_string())
                    .ok_or_else(|| "interface member without a semantic".to_string())?;
                out.push(InterfaceLeaf {
                    semantic,
                    ty: interface_reg_ty(&member.ty)?,
                });
            }
        }
    }
    Ok(())
}

/// Register type of an interface leaf (scalar or 2/3/4-lane vector of
/// float or ≤32-bit integer).
fn interface_reg_ty(ty: &Ty) -> Result<RegTy, String> {
    if let Ty::Vector { elem, len } = ty.unqualified() {
        if *len < 2 || *len > 4 {
            return Err("interface vectors are 2/3/4 lanes".to_string());
        }
        return Ok(RegTy::Vector {
            elem: leaf_scalar(elem)?,
            lanes: *len as u8,
        });
    }
    Ok(RegTy::Scalar(leaf_scalar(ty)?))
}

/// Scalar kind of an interface leaf.
fn leaf_scalar(ty: &Ty) -> Result<Scalar, String> {
    Ok(match ty.unqualified() {
        Ty::Float => Scalar::F32,
        Ty::Bool | Ty::Char | Ty::SignedChar => Scalar::Int { bits: 32 },
        Ty::UnsignedChar => Scalar::Int { bits: 32 },
        Ty::Short => Scalar::Int { bits: 32 },
        Ty::UnsignedShort => Scalar::Int { bits: 32 },
        Ty::Int | Ty::UnsignedInt => Scalar::i32(),
        Ty::BitInt { width, .. } if *width <= 32 => Scalar::i32(),
        _ => return Err("unsupported interface leaf type".to_string()),
    })
}

/// File-scope objects: uniforms (`[[prism::uniform]]`) and samplers
/// (`sampler2D`), in declaration order.
fn collect_resources(sema: &Sema, interner: &Interner, interface: &mut Interface) -> Result<(), String> {
    for (index, binding) in sema.decls.iter().enumerate() {
        let attrs = match sema.prism_attrs.get(&glcc_frontend::sema::DeclId(index as u32)) {
            Some(attrs) => *attrs,
            None => continue,
        };
        let _id = index;
        if attrs.stage.is_some() {
            continue;
        }
        if !attrs.uniform {
            continue;
        }
        if binding.kind != BindingKind::Var {
            return Err("[[prism::uniform]] must mark a file-scope object".to_string());
        }
        let name = name_of(sema, interner, index as u32);
        let ty = binding.ty.unqualified().clone();
        let mut leaves = Vec::new();
        let mut std140_offset = 0u32;
        collect_uniform_leaves(sema, &ty, &mut Vec::new(), &mut std140_offset, &mut leaves)?;
        if leaves.is_empty() {
            return Err(format!("uniform '{name}' has no readable leaves"));
        }
        interface.uniforms.push(UniformVar {
            name,
            creg: 0,
            std140_size: round16(std140_offset),
            leaves,
        });
    }
    // Samplers live only at file scope (the checker rejects block
    // scopes), where a bare declaration is a tentative definition;
    // sampler-typed parameters of prototypes have neither flag.
    for decl in sema.decls.iter() {
        if decl.kind != BindingKind::Var || !(decl.tentative || decl.defined) {
            continue;
        }
        if !sema.is_prism_sampler_ty(&decl.ty, interner) {
            continue;
        }
        let name = decl
            .name
            .map(|n| interner.resolve(n).unwrap_or("?").to_string())
            .unwrap_or_else(|| "?".to_string());
        interface.samplers.push(SamplerVar { name });
    }
    Ok(())
}

/// Flatten a uniform object into readable leaves (scalars, vectors)
/// with member paths and std140 offsets, advancing `offset` past the
/// whole object. c-register packing happens afterwards over the flat
/// leaf list.
fn collect_uniform_leaves(sema: &Sema, ty: &Ty, path: &mut Vec<u32>, offset: &mut u32, out: &mut Vec<UniformLeaf>) -> Result<(), String> {
    match ty.unqualified() {
        Ty::Struct(id) => {
            let def = &sema.structs[id.0 as usize];
            for (index, member) in def.members.iter().enumerate() {
                *offset = (*offset).next_multiple_of(16);
                path.push(index as u32);
                collect_uniform_leaves(sema, &member.ty, path, offset, out)?;
                path.pop();
            }
            *offset = round16(*offset);
            Ok(())
        }
        Ty::Array { elem, bound, .. } => {
            let len = match bound {
                Some(ArrayBound::Fixed(n)) => *n,
                _ => return Err("uniform arrays need a constant bound".to_string()),
            };
            let stride = round16(std140_size(elem));
            for index in 0..len {
                path.push(index as u32);
                let saved = *offset;
                collect_uniform_leaves(sema, elem, path, offset, out)?;
                *offset = saved + stride;
                path.pop();
            }
            Ok(())
        }
        ty => {
            *offset = (*offset).next_multiple_of(std140_align(ty));
            out.push(UniformLeaf {
                path: path.clone(),
                ty: interface_reg_ty(ty)?,
                creg: 0,
                lane: 0,
                std140_offset: *offset,
            });
            *offset += std140_size(ty);
            Ok(())
        }
    }
}

/// std140 alignment of a leaf type.
fn std140_align(ty: &Ty) -> u32 {
    match ty.unqualified() {
        Ty::Vector { elem, len } => match (**elem).unqualified() {
            Ty::Float | Ty::Bool | Ty::Int | Ty::UnsignedInt if *len == 2 => 8,
            _ => 16,
        },
        _ => 4,
    }
}

/// Assign absolute c-registers to every uniform leaf, one global
/// cursor in declaration order: a vector leaf takes a whole register,
/// scalars pack four per register.
fn pack_uniform_cregs(uniforms: &mut [UniformVar]) {
    let mut next_reg = 0u8;
    let mut scalar_reg = 0u8;
    let mut scalar_slot = 4u8;
    for uniform in uniforms.iter_mut() {
        let mut first_reg = None;
        for leaf in &mut uniform.leaves {
            let (reg, lane) = if leaf.ty.lanes() > 1 {
                let reg = next_reg;
                next_reg += 1;
                scalar_slot = 4;
                (reg, 0)
            } else {
                if scalar_slot == 4 {
                    scalar_reg = next_reg;
                    next_reg += 1;
                    scalar_slot = 0;
                }
                let slot = scalar_slot;
                scalar_slot += 1;
                (scalar_reg, slot)
            };
            first_reg.get_or_insert(reg);
            leaf.creg = reg;
            leaf.lane = lane;
        }
        uniform.creg = first_reg.unwrap_or(0);
    }
}

/// std140 size/alignment of a type (GLSL rules: vec3 aligns to 16 and
/// strides 16 in arrays, arrays round their element stride to 16,
/// structs align to 16 and round their size).
pub fn std140_size(ty: &Ty) -> u32 {
    match ty.unqualified() {
        Ty::Vector { elem, len } => match (**elem).unqualified() {
            Ty::Float | Ty::Bool | Ty::Int | Ty::UnsignedInt => match len {
                2 => 8,
                3 => 12,
                _ => 16,
            },
            _ => 16,
        },
        Ty::Array { elem, bound, .. } => {
            let stride = round16(std140_size(elem));
            let count = match bound {
                Some(ArrayBound::Fixed(n)) => *n,
                _ => 1,
            };
            stride * count.max(1) as u32
        }
        _ => 4,
    }
}

fn round16(size: u32) -> u32 {
    size.div_ceil(16) * 16
}
