//! Fern types to MIR types, with layouts.
//!
//! A named data value, a closure, and a tuple are all heap cells behind
//! an erased pointer, so recursion, sharing, and escaping cost nothing to
//! model. A cell's byte layout is derived from the source type where a
//! cell is built or taken apart (constructor calls, patterns, capture
//! access, tuple fields); the unit type is a one-byte value so it fits
//! anywhere a value is expected.

use std::collections::HashMap;

use glcc_mir as mir;

use crate::defs::Defs;
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

pub(crate) struct Types<'d> {
    defs: &'d Defs,
    cache: HashMap<String, mir::Ty>,
    layouts: HashMap<String, DataLayout>,
    tuple_layouts: HashMap<String, TupleLayout>,
}

/// The byte layout of one data type instance's cells: an `i32` tag at
/// offset zero, then one payload struct per constructor.
#[derive(Debug, Clone)]
pub(crate) struct DataLayout {
    pub cell_size: u64,
    /// Where each constructor's payload struct starts in the cell.
    pub payload_offsets: Vec<u64>,
    /// Per constructor, the offset of each field in its payload.
    pub field_offsets: Vec<Vec<u64>>,
    /// Per constructor, each field's MIR type.
    pub field_tys: Vec<Vec<mir::Ty>>,
}

/// The byte layout of a tuple cell: the fields, packed from offset zero.
#[derive(Debug, Clone)]
pub(crate) struct TupleLayout {
    pub cell_size: u64,
    pub offsets: Vec<u64>,
    pub field_tys: Vec<mir::Ty>,
}

/// The byte layout of a closure's environment: the code pointer takes
/// the first slot, captures follow in capture order.
#[derive(Debug, Clone)]
pub(crate) struct EnvLayout {
    pub size: u64,
    pub offsets: Vec<u64>,
}

impl<'d> Types<'d> {
    pub(crate) fn new(defs: &'d Defs) -> Types<'d> {
        Types {
            defs,
            cache: HashMap::new(),
            layouts: HashMap::new(),
            tuple_layouts: HashMap::new(),
        }
    }

    pub(crate) fn get(&mut self, ty: &Ty, span: Span) -> KResult<mir::Ty> {
        let ty = normalize(ty);
        match &ty {
            Ty::Named { name, .. } => self.named(&ty, name, span),
            Ty::Tuple(vs) if vs.is_empty() => Ok(mir::Ty::int(8)),
            Ty::Tuple(_) => Ok(self.memo(&ty, mir::Ty::ptr(mir::Ty::Void))),
            Ty::Arrow(..) => Ok(mir::Ty::ptr(mir::Ty::Void)),
            Ty::App(..) => Err(Diag::new(
                format!("`{ty}` reached lowering with an unresolved head"),
                span,
            )),
            Ty::Param(p) => Err(Diag::new(
                format!("type parameter `{p}` reached lowering uninstantiated"),
                span,
            )),
            Ty::Var(v) => Err(Diag::new(
                format!("inference variable `?{v}` reached lowering"),
                span,
            )),
        }
    }

    fn named(&mut self, orig: &Ty, name: &str, span: Span) -> KResult<mir::Ty> {
        match name {
            // LP64: Int is a signed 64-bit integer.
            "Int" => return Ok(mir::Ty::int(64)),
            "Float" => return Ok(mir::Ty::Float(mir::FloatKind::F64)),
            "Bool" => return Ok(mir::Ty::Bool),
            "Unit" => return Ok(mir::Ty::int(8)),
            _ => {}
        }
        if self.defs.data_by_name(name).is_none() {
            return Err(Diag::new(format!("cannot find type `{name}`"), span));
        }
        if !orig.is_concrete() {
            return Err(Diag::new(
                format!("`{orig}` reached lowering uninstantiated"),
                span,
            ));
        }
        Ok(mir::Ty::ptr(mir::Ty::Void))
    }

    fn memo(&mut self, ty: &Ty, out: mir::Ty) -> mir::Ty {
        self.cache.insert(ty.mangle(), out.clone());
        out
    }

    /// The cell layout of the data type instance `ty` (a concrete
    /// `Named`). Cached per mangled type, so every site agrees.
    pub(crate) fn data_layout(&mut self, ty: &Ty, span: Span) -> KResult<DataLayout> {
        let ty = normalize(ty);
        let Ty::Named { name, args } = &ty else {
            return Err(Diag::new(format!("`{ty}` is not a data type"), span));
        };
        let key = ty.mangle();
        if let Some(l) = self.layouts.get(&key) {
            return Ok(l.clone());
        }
        let data_id = self
            .defs
            .data_by_name(name)
            .ok_or_else(|| Diag::new(format!("cannot find type `{name}`"), span))?;
        let ctor_args: Vec<Vec<Ty>> = {
            let data = &self.defs.data[data_id.index()];
            let subst: HashMap<String, Ty> = data
                .params
                .iter()
                .map(|(n, _)| n.clone())
                .zip(args.iter().cloned())
                .collect();
            data.ctors
                .iter()
                .map(|c| c.args.iter().map(|a| a.substitute(&subst)).collect())
                .collect()
        };
        let mut field_tys = Vec::with_capacity(ctor_args.len());
        for args in &ctor_args {
            let mut tys = Vec::with_capacity(args.len());
            for a in args {
                tys.push(self.get(a, span)?);
            }
            field_tys.push(tys);
        }
        let mut cell_fields = Vec::with_capacity(field_tys.len() + 1);
        cell_fields.push(mir::Ty::int(32));
        for tys in &field_tys {
            cell_fields.push(struct_ty(tys.clone()));
        }
        let cell = struct_ty(cell_fields);
        let cell_offsets = struct_offsets(&cell, span)?;
        let payload_offsets: Vec<u64> = (0..field_tys.len()).map(|i| cell_offsets[i + 1]).collect();
        let mut field_offsets = Vec::with_capacity(field_tys.len());
        for tys in &field_tys {
            field_offsets.push(struct_offsets(&struct_ty(tys.clone()), span)?);
        }
        let layout = DataLayout {
            cell_size: cell.size_bytes(),
            payload_offsets,
            field_offsets,
            field_tys,
        };
        self.layouts.insert(key, layout.clone());
        Ok(layout)
    }

    /// The cell layout of the tuple type `ty`. Cached per mangled type,
    /// so every site agrees.
    pub(crate) fn tuple_layout(&mut self, ty: &Ty, span: Span) -> KResult<TupleLayout> {
        let ty = normalize(ty);
        let Ty::Tuple(vs) = &ty else {
            return Err(Diag::new(format!("`{ty}` is not a tuple type"), span));
        };
        let key = ty.mangle();
        if let Some(l) = self.tuple_layouts.get(&key) {
            return Ok(l.clone());
        }
        let mut field_tys = Vec::with_capacity(vs.len());
        for v in vs {
            field_tys.push(self.get(v, span)?);
        }
        let cell = struct_ty(field_tys.clone());
        let offsets = struct_offsets(&cell, span)?;
        let layout = TupleLayout {
            cell_size: cell.size_bytes(),
            offsets,
            field_tys,
        };
        self.tuple_layouts.insert(key, layout.clone());
        Ok(layout)
    }

    /// The environment layout for a closure with the given capture
    /// types: one code-pointer slot, then the captures.
    pub(crate) fn env_layout(&mut self, captures: &[Ty], span: Span) -> KResult<EnvLayout> {
        let mut fields = Vec::with_capacity(captures.len() + 1);
        fields.push(mir::Ty::ptr(mir::Ty::Void));
        for c in captures {
            fields.push(self.get(c, span)?);
        }
        let cell = struct_ty(fields);
        let offsets = struct_offsets(&cell, span)?;
        Ok(EnvLayout {
            size: cell.size_bytes(),
            offsets: offsets[1..].to_vec(),
        })
    }
}

/// Flatten `App` spines onto their head so lowering only ever sees
/// saturated `Named` types.
pub(crate) fn normalize(ty: &Ty) -> Ty {
    match ty {
        Ty::App(h, a) => {
            let head = normalize(h);
            let arg = normalize(a);
            match head {
                Ty::Named { name, mut args } => {
                    args.push(arg);
                    Ty::Named { name, args }
                }
                other => Ty::app(other, arg),
            }
        }
        Ty::Arrow(s, d) => Ty::arrow(normalize(s), normalize(d)),
        Ty::Tuple(vs) => Ty::Tuple(vs.iter().map(normalize).collect()),
        Ty::Named { name, args } => Ty::Named {
            name: name.clone(),
            args: args.iter().map(normalize).collect(),
        },
        other => other.clone(),
    }
}

pub(crate) fn struct_offsets(ty: &mir::Ty, span: Span) -> KResult<Vec<u64>> {
    match ty {
        mir::Ty::Struct { offsets, .. } => Ok(offsets.clone()),
        other => Err(Diag::new(
            format!("cannot select a field of `{other:?}`"),
            span,
        )),
    }
}

/// A struct layout with natural alignment and padding. Empty aggregates
/// keep one byte so storage never has zero size (the backend rejects
/// zero-sized slots).
pub(crate) fn struct_ty(fields: Vec<mir::Ty>) -> mir::Ty {
    let offsets = mir::dense_field_offsets(&fields);
    let align = fields
        .iter()
        .map(mir::ty::ty_align)
        .max()
        .unwrap_or(1)
        .max(1);
    let end = fields
        .iter()
        .zip(&offsets)
        .map(|(f, off)| off + f.size_bytes())
        .max()
        .unwrap_or(0);
    mir::Ty::Struct {
        fields,
        offsets,
        size: align_up(end, align).max(1),
    }
}

fn align_up(n: u64, align: u64) -> u64 {
    (n + align - 1) / align * align
}
