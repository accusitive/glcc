//! Struct/union layout and type sizes (`sizeof`).
//!
//! All queries read `Sema::target`, so sizes follow the target being
//! compiled for. Anything unknown — incomplete types, `[]` bounds,
//! variable-length bounds, opaque builtins, recursive types — yields
//! `None` rather than a guess.
//!
//! Bit-fields follow the x86-64 GCC model, verified field-by-field
//! against GCC 14 (`offsetof` where observable, sizes and alignments
//! throughout): a nonzero width packs into the remaining bits of the
//! aligned type-sized block containing the bit position (crossing into
//! the next block when it does not fit, whatever the neighboring
//! fields' types); `:0` aligns the position to its type without
//! contributing size or alignment; unnamed nonzero widths consume
//! space but contribute no alignment; named bit-fields contribute
//! their type's alignment. A full-width bit-field behaves like a
//! regular member. Multi-byte-wide values never straddle oddly: the
//! rules above hold across `short`/`int` type mixing and zero-width
//! edge cases.

use alloc::vec::Vec;
use core::cell::RefCell;

use crate::{
    check::CheckedExpr,
    eval::{Value, eval_const},
    lexer::Interner,
    parse::{ArrayBound, Ty},
    sema::{Member, Sema, StructDef, StructId},
};

use string_interner::symbol::SymbolU32;

/// Layout of one named member: the byte offset of its storage (for a
/// bit-field, of the storage unit holding its first bit) and, for
/// bit-fields, the bit index of that first bit within the byte at
/// `offset` (from the low bit on little-endian targets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemberLayout {
    pub name: SymbolU32,
    pub offset: u64,
    pub bit: Option<u32>,
}

/// Layout of a struct or union: total size, required alignment, one
/// entry per named member in declaration order, and every member
/// slot's byte offset for designator walks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    pub size: u64,
    pub align: u64,
    pub members: Vec<MemberLayout>,
    /// Byte offset of every member in declaration order (parallel to
    /// the definition's `members`, including anonymous members and
    /// unnamed bit-fields, which get no `members` entry of their own).
    /// Used to walk designator paths that cross anonymous members.
    pub member_offsets: Vec<u64>,
}

/// RAII guard for the layout recursion stack: recursive types (invalid
/// C, rejected by the parser, but still constructible below it) must
/// yield `None`, not hang.
struct VisitGuard<'a> {
    visiting: &'a RefCell<Vec<StructId>>,
    id: StructId,
}

impl<'a> VisitGuard<'a> {
    fn enter(sema: &'a Sema, id: StructId) -> Option<Self> {
        let mut visiting = sema.visiting.borrow_mut();
        if visiting.contains(&id) {
            return None;
        }
        visiting.push(id);
        drop(visiting);
        Some(Self { visiting: &sema.visiting, id })
    }
}

impl Drop for VisitGuard<'_> {
    fn drop(&mut self) {
        let mut visiting = self.visiting.borrow_mut();
        debug_assert_eq!(visiting.last(), Some(&self.id));
        visiting.pop();
    }
}

fn align_up(pos: u64, align: u64) -> Option<u64> {
    debug_assert!(align > 0);
    let extra = (align - pos % align) % align;
    pos.checked_add(extra)
}

/// Bit-position variant of `align_up` (the struct cursor counts bits,
/// which overflows `u64` for objects near the address-space limit).
fn align_up_bits(pos: u128, align: u128) -> Option<u128> {
    debug_assert!(align > 0);
    let extra = (align - pos % align) % align;
    pos.checked_add(extra)
}

impl Sema {
    /// Whether a value of this type can exist: primitives (opaque
    /// builtins included) and pointers always can; structs, unions,
    /// and enums need completed definitions; arrays need complete
    /// elements. `void` and function types never can. Qualification is
    /// transparent throughout.
    pub fn is_complete(&self, ty: &Ty) -> bool {
        match ty.unqualified() {
            Ty::Void | Ty::Func { .. } => false,
            Ty::Struct(id) => self.structs.get(id.0 as usize).is_some_and(|d| d.complete),
            Ty::Enum(id) => self.enums.get(id.0 as usize).is_some_and(|d| d.complete),
            Ty::Array { elem, .. } => self.is_complete(elem),
            _ => true,
        }
    }

    /// Size in bytes, or `None` when unknowable (see the module docs).
    /// Qualification never changes the size.
    pub fn type_size(&self, ty: &Ty, interner: &Interner) -> Option<u64> {
        let t = &self.target;
        match ty.unqualified() {
            // `unqualified` strips every layer, so this never matches;
            // the arm exists only for exhaustiveness.
            Ty::Qualified { .. } => unreachable!("unqualified type is never qualified"),
            Ty::Void | Ty::Func { .. } => None,
            Ty::Bool => Some(t.bool_size),
            Ty::Char | Ty::SignedChar | Ty::UnsignedChar => Some(t.char_size),
            Ty::Short | Ty::UnsignedShort => Some(t.short_size),
            Ty::Int | Ty::UnsignedInt => Some(t.int_size),
            Ty::Long | Ty::UnsignedLong => Some(t.long_size),
            Ty::LongLong | Ty::UnsignedLongLong => Some(t.long_long_size),
            Ty::Int128 | Ty::UnsignedInt128 => Some(t.int128_size),
            Ty::Float => Some(t.float_size),
            Ty::Double => Some(t.double_size),
            Ty::LongDouble => Some(t.long_double_size),
            Ty::Complex(inner) => self.type_size(inner, interner)?.checked_mul(2),
            Ty::Pointer { .. } => Some(t.pointer_size),
            Ty::Enum(id) => {
                let repr = self.enums.get(id.0 as usize).filter(|d| d.complete).map(|d| d.repr.clone())?;
                self.type_size(&repr, interner)
            }
            Ty::Struct(id) => Some(self.struct_layout(*id, interner)?.size),
            Ty::Array { elem, .. } => {
                let elem = self.type_size(elem, interner)?;
                let n = self.array_count(ty)?;
                elem.checked_mul(n)
            }
            Ty::BuiltinVaList => Some(t.va_list_size),
            Ty::Float128 => Some(16),
        }
    }

    /// Required alignment in bytes, or `None` when unknowable. Arrays
    /// (even `[]`) align like their elements. Qualification never
    /// changes the alignment.
    pub fn type_align(&self, ty: &Ty, interner: &Interner) -> Option<u64> {
        let t = &self.target;
        match ty.unqualified() {
            // `unqualified` strips every layer, so this never matches;
            // the arm exists only for exhaustiveness.
            Ty::Qualified { .. } => unreachable!("unqualified type is never qualified"),
            Ty::Void | Ty::Func { .. } => None,
            Ty::Bool => Some(t.bool_align),
            Ty::Char | Ty::SignedChar | Ty::UnsignedChar => Some(t.char_align),
            Ty::Short | Ty::UnsignedShort => Some(t.short_align),
            Ty::Int | Ty::UnsignedInt => Some(t.int_align),
            Ty::Long | Ty::UnsignedLong => Some(t.long_align),
            Ty::LongLong | Ty::UnsignedLongLong => Some(t.long_long_align),
            Ty::Int128 | Ty::UnsignedInt128 => Some(t.int128_align),
            Ty::Float => Some(t.float_align),
            Ty::Double => Some(t.double_align),
            Ty::LongDouble => Some(t.long_double_align),
            Ty::Complex(inner) => self.type_align(inner, interner),
            Ty::Pointer { .. } => Some(t.pointer_align),
            Ty::Enum(id) => {
                let repr = self.enums.get(id.0 as usize).filter(|d| d.complete).map(|d| d.repr.clone())?;
                self.type_align(&repr, interner)
            }
            Ty::Struct(id) => Some(self.struct_layout(*id, interner)?.align),
            Ty::Array { elem, .. } => self.type_align(elem, interner),
            Ty::BuiltinVaList => Some(t.va_list_align),
            Ty::Float128 => Some(16),
        }
    }

    /// Element count of an array type. `[]`, `[*]`, and VLA bounds
    /// yield `None` (the last is a variable-length array, whose size is
    /// a runtime value, not a layout fact).
    fn array_count(&self, ty: &Ty) -> Option<u64> {
        let Ty::Array {
            bound: Some(ArrayBound::Fixed(n)),
            ..
        } = ty
        else {
            return None;
        };
        Some(*n)
    }

    /// Folded bit-field width in bits, or `None` for a non-constant
    /// width. Parse-time validation rejects negative, over-wide, and
    /// named zero widths, so layout treats those as unknowable too.
    pub fn bit_width(&self, width: &CheckedExpr, type_bits: u64, interner: &Interner) -> Option<u64> {
        match eval_const(width, self, interner)? {
            Value::Int(n) if n >= 0 && (n as u64) <= type_bits => u64::try_from(n).ok(),
            _ => None,
        }
    }

    /// Full layout of a struct or union, or `None` for incomplete,
    /// recursive, or otherwise unlayable definitions. Successful
    /// results memoize in the definition (layouts are pure in it);
    /// failures never cache, so recursive types keep yielding `None`.
    pub fn struct_layout(&self, id: StructId, interner: &Interner) -> Option<StructLayout> {
        if let Some(cached) = self.structs.get(id.0 as usize).and_then(|d| d.layout_cache.borrow().clone()) {
            return Some(cached);
        }
        let layout = self.compute_layout(id, interner)?;
        self.structs[id.0 as usize].layout_cache.replace(Some(layout.clone()));
        Some(layout)
    }

    /// Uncached layout computation; see `struct_layout`.
    fn compute_layout(&self, id: StructId, interner: &Interner) -> Option<StructLayout> {
        let def = self.structs.get(id.0 as usize).filter(|d| d.complete)?;
        let _guard = VisitGuard::enter(self, id)?;
        if def.is_union {
            return self.union_layout(def, interner);
        }
        // Bit position from the struct start; byte offsets derive from
        // it, so bit-fields and members share one cursor. The cursor is
        // `u128`: an object can be as large as 2^63 bytes (GCC's cap),
        // whose bit count overflows `u64`.
        let mut pos: u128 = 0;
        let mut struct_align: u64 = 1;
        let mut members = Vec::new();
        let mut member_offsets = Vec::new();
        for member in &def.members {
            // Flexible array members (`T name[]`, no bound) contribute
            // no size but align like their element (C99 §6.7.2.1p16:
            // `sizeof` ignores them); bound-less arrays elsewhere in
            // the list get the same tolerant treatment.
            let (size, natural) = match &member.ty {
                Ty::Array { elem, bound, .. } if bound.is_none() => (0, self.type_align(elem, interner)?),
                _ => (self.type_size(&member.ty, interner)?, self.type_align(&member.ty, interner)?),
            };
            let align = Self::member_align(def.packed, member, natural);
            match &member.bitfield {
                None => {
                    let at = align_up(u64::try_from(pos.div_ceil(8)).ok()?, align)?;
                    // Anonymous members reserve their nested type's
                    // full extent (alignment and tail padding included)
                    // but contribute no entry of their own.
                    if let Some(name) = member.name {
                        members.push(MemberLayout { name, offset: at, bit: None });
                    }
                    member_offsets.push(at);
                    pos = u128::from(at).checked_mul(8)?.checked_add(u128::from(size).checked_mul(8)?)?;
                    struct_align = struct_align.max(align);
                }
                Some(width) => {
                    let type_bits = size.checked_mul(8)?;
                    let w = self.bit_width(width, type_bits, interner)?;
                    if w == 0 {
                        // `:0` (unnamed — named zero widths never parse):
                        // align the position, contribute nothing else.
                        pos = align_up_bits(pos, u128::from(type_bits))?;
                        member_offsets.push(u64::try_from(pos.div_ceil(8)).ok()?);
                        continue;
                    }
                    // The aligned type-sized block holding the position;
                    // the field packs into its remainder, else the next
                    // block — whatever the neighbors' types.
                    let type_bits = u128::from(type_bits);
                    let w = u128::from(w);
                    let mut block = pos / type_bits * type_bits;
                    if type_bits - (pos - block) < w {
                        block = block.checked_add(type_bits)?;
                    }
                    let at = block.max(pos);
                    // The field's storage unit is the type-sized block
                    // containing it: accesses read-modify-write the
                    // whole unit, so the unit must never overhang the
                    // struct. A field starting mid-block shares the
                    // block (and thus the unit) with its predecessors.
                    let unit = at / type_bits * type_bits;
                    // Only named widths get an entry; unnamed ones just
                    // consume space (and, unlike named ones, no
                    // alignment). Every slot records its unit offset
                    // for designator walks.
                    member_offsets.push(u64::try_from(unit / 8).ok()?);
                    if let Some(name) = member.name {
                        let bit = if self.target.little_endian {
                            (at - unit) as u32
                        } else {
                            // Big-endian numbers bits from the unit's
                            // high end; only the layout shape is
                            // modeled, not exact BE bit order.
                            ((type_bits - (at - unit) - w) % type_bits) as u32
                        };
                        members.push(MemberLayout {
                            name,
                            offset: u64::try_from(unit / 8).ok()?,
                            bit: Some(bit),
                        });
                        struct_align = struct_align.max(align);
                    }
                    pos = at.checked_add(w)?;
                }
            }
        }
        if let Some(n) = def.align_override {
            struct_align = struct_align.max(n);
        }
        Some(StructLayout {
            size: align_up(u64::try_from(pos.div_ceil(8)).ok()?, struct_align)?,
            align: struct_align,
            members,
            member_offsets,
        })
    }

    /// Effective alignment of a member: the natural type alignment,
    /// dropped to 1 by `packed` (on the struct or the member), then
    /// raised by a member `aligned(n)` (a minimum, like GCC).
    fn member_align(def_packed: bool, member: &Member, natural: u64) -> u64 {
        let mut align = if def_packed || member.packed { 1 } else { natural };
        if let Some(n) = member.align_override {
            align = align.max(n);
        }
        align
    }

    /// Union layout: every member starts at bit zero; the size covers
    /// the largest member and the alignment is the largest member
    /// alignment (named bit-fields count, `:0` and unnamed widths do
    /// not). `packed`/`aligned` apply as in structs.
    fn union_layout(&self, def: &StructDef, interner: &Interner) -> Option<StructLayout> {
        let mut size: u64 = 0;
        let mut align: u64 = 1;
        let mut members = Vec::new();
        let mut member_offsets = Vec::new();
        for member in &def.members {
            // Flexible array members contribute no size here either
            // (see the struct arm).
            let (msize, natural) = match &member.ty {
                Ty::Array { elem, bound, .. } if bound.is_none() => (0, self.type_align(elem, interner)?),
                _ => (self.type_size(&member.ty, interner)?, self.type_align(&member.ty, interner)?),
            };
            let malign = Self::member_align(def.packed, member, natural);
            // Every union member starts at zero, entry or not.
            member_offsets.push(0);
            match &member.bitfield {
                None => {
                    size = size.max(msize);
                    align = align.max(malign);
                    if let Some(name) = member.name {
                        members.push(MemberLayout { name, offset: 0, bit: None });
                    }
                }
                Some(width) => {
                    let w = self.bit_width(width, msize.checked_mul(8)?, interner)?;
                    if w == 0 {
                        continue;
                    }
                    size = size.max(w.div_ceil(8));
                    if let Some(name) = member.name {
                        align = align.max(malign);
                        members.push(MemberLayout { name, offset: 0, bit: Some(0) });
                    }
                }
            }
        }
        if let Some(n) = def.align_override {
            align = align.max(n);
        }
        Some(StructLayout {
            size: align_up(size, align)?,
            align,
            members,
            member_offsets,
        })
    }
}
