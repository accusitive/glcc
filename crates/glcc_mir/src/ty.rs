use smallvec::SmallVec;

/// Scalar float kinds. Integer sign lives on the operation, not the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatKind {
    F32,
    F64,
    /// 80-bit extended carrier (x86-64 `long double`), held as 16 bytes.
    F80,
    /// IEEE-754 quadruple (aarch64/riscv64 `long double`, `__float128`).
    F128,
}

/// Width of one limb in a [`Ty::WideInt`]. Also the largest integer the
/// backend lowers natively (with [`MAX_SCALAR_INT_BITS`]).
pub const LIMB_BITS: u32 = 64;

/// Integers wider than this stay [`Ty::Int`]: Cranelift handles them as
/// scalar `i128` values. Anything wider is legalized to [`Ty::WideInt`].
pub const MAX_SCALAR_INT_BITS: u32 = 128;

/// Middle IR type. Scalars mirror Cranelift 1:1; aggregates are structural
/// so optimizations see through them instead of opaque byte blobs.
///
/// Structs carry their C layout: `offsets` is the byte offset of each
/// field and `size` the padded total, both from the frontend layout.
/// Field indices stay 1:1 with members (no explicit pad fields), so
/// every `Field(i)` step keeps meaning member `i` while sizes, strides,
/// and offsets agree with C.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Void,
    Bool,
    Int {
        bits: u32,
    },
    /// An integer wider than [`MAX_SCALAR_INT_BITS`], after
    /// legalization. It is an aggregate: its in-memory and SSA
    /// representation is a little-endian sequence of [`LIMB_BITS`]-bit
    /// limbs, so every arithmetic operation has been rewritten to limb
    /// operations by the legalization pass. `bits` keeps the integer's
    /// storage width (a multiple of 8), so sizes and pointer arithmetic
    /// match the pre-legalization [`Ty::Int`] exactly.
    WideInt {
        bits: u32,
    },
    Float(FloatKind),
    /// Code or data address. `pointee` is structural, not a runtime check.
    /// `bits` is the pointer's own width (MSVC `__ptr32`/`__ptr64`); on
    /// every supported target the ordinary width is 64.
    Ptr {
        pointee: Box<Ty>,
        bits: u32,
    },
    /// C `va_list`: opaque at MIR level, sized per target at lowering.
    /// Only addressable, never loaded by value; manipulated through
    /// the `Va*` instructions.
    BuiltinVaList {
        size: u64,
        align: u64,
    },
    Array {
        elem: Box<Ty>,
        len: u64,
    },
    Struct {
        fields: Vec<Ty>,
        offsets: Vec<u64>,
        size: u64,
    },
}

impl Ty {
    pub fn int(bits: u32) -> Self {
        Ty::Int { bits }
    }

    pub fn ptr(pointee: Ty) -> Self {
        Ty::Ptr {
            pointee: Box::new(pointee),
            bits: 64,
        }
    }

    pub fn ptr_bits(pointee: Ty, bits: u32) -> Self {
        Ty::Ptr { pointee: Box::new(pointee), bits }
    }

    /// Width in bits of a pointer type, `None` for anything else.
    pub fn ptr_width(&self) -> Option<u32> {
        match self {
            Ty::Ptr { bits, .. } => Some(*bits),
            _ => None,
        }
    }

    pub fn is_void(&self) -> bool {
        matches!(self, Ty::Void)
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(self, Ty::Array { .. } | Ty::Struct { .. } | Ty::WideInt { .. })
    }

    pub fn is_wide_int(&self) -> bool {
        matches!(self, Ty::WideInt { .. })
    }

    /// Number of [`LIMB_BITS`]-sized limbs in a wide integer.
    pub fn limb_count(&self) -> u32 {
        match self {
            Ty::WideInt { bits } => (*bits).div_ceil(LIMB_BITS),
            _ => panic!("limb count of a non-wide integer"),
        }
    }

    /// Type of limb `index`: a full-width limb everywhere but the last
    /// one, which holds the remaining `bits % LIMB_BITS` bits (or a full
    /// limb when the width divides evenly).
    pub fn limb_ty(&self, index: u32) -> Ty {
        let Ty::WideInt { bits } = self else {
            panic!("limb type of a non-wide integer");
        };
        let full = index + 1 < self.limb_count();
        Ty::Int {
            bits: if full { LIMB_BITS } else { bits - index * LIMB_BITS },
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, Ty::Bool | Ty::Int { .. } | Ty::Float(_) | Ty::Ptr { .. })
    }

    /// Byte size under LP64 (signed `char`), matching the frontend target
    /// assumptions. Structs pack without padding here; lowering inserts
    /// alignment padding when computing field offsets.
    pub fn size_bytes(&self) -> u64 {
        match self {
            Ty::Void => 0,
            Ty::Bool => 1,
            Ty::Int { bits } => ((*bits).max(8) / 8) as u64,
            Ty::WideInt { bits } => ((*bits).max(8) / 8) as u64,
            Ty::Float(k) => match k {
                FloatKind::F32 => 4,
                FloatKind::F64 => 8,
                FloatKind::F80 | FloatKind::F128 => 16,
            },
            Ty::Ptr { bits, .. } => (*bits as u64).max(8) / 8,
            Ty::BuiltinVaList { size, .. } => *size,
            Ty::Array { elem, len } => elem.size_bytes() * len,
            Ty::Struct { size, .. } => *size,
        }
    }
}

/// Function signature: parameter types plus return, with C variadics noted
/// so the inliner refuses them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub variadic: bool,
}

impl Signature {
    pub fn new(params: Vec<Ty>, ret: Ty) -> Self {
        Signature { params, ret, variadic: false }
    }

    /// One step of a GEP through this type: field selection into structs,
    /// constant or value indexing into arrays (bounds are unchecked here,
    /// verified at lowering). A constant or dynamic step over a pointed-
    /// to scalar is pointer arithmetic over a single element.
    pub fn gep_step(ty: &Ty, index: &GepIndex) -> Option<Ty> {
        match (ty, index) {
            (Ty::Struct { fields, .. }, GepIndex::Field(i)) => fields.get(*i as usize).cloned(),
            (Ty::Array { elem, .. }, GepIndex::Index(_) | GepIndex::ConstIndex(_)) => Some(elem.as_ref().clone()),
            (Ty::Int { .. } | Ty::Float(_) | Ty::Bool | Ty::Ptr { .. }, GepIndex::ConstIndex(_)) => Some(ty.clone()),
            (Ty::Int { .. } | Ty::Float(_) | Ty::Bool | Ty::Ptr { .. }, GepIndex::Index(_)) => Some(ty.clone()),
            (Ty::WideInt { .. }, GepIndex::ConstIndex(_) | GepIndex::Index(_)) => Some(ty.clone()),
            _ => None,
        }
    }
}

/// One GEP navigation step over an aggregate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GepIndex {
    Field(u32),
    Index(super::ir::Value),
    ConstIndex(u64),
}

/// Byte offset of each field under its C layout, stored on the type
/// by lowering from the frontend layout.
pub fn struct_field_offsets(ty: &Ty) -> Vec<u64> {
    match ty {
        Ty::Struct { offsets, .. } => offsets.clone(),
        _ => panic!("field offsets of non-struct"),
    }
}

/// Dense field layout from member types alone: per-field natural
/// alignment, no trailing padding. Only for contexts with no frontend
/// layout (demos, tests, and the incomplete-struct fallback in type
/// lowering); production paths use the stored C layout above.
pub fn dense_field_offsets(fields: &[Ty]) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut off = 0u64;
    for f in fields {
        let align = ty_align(f).max(1);
        off = off.next_multiple_of(align);
        offsets.push(off);
        off += f.size_bytes();
    }
    offsets
}

/// Natural alignment mirror used for field layout: primitives by
/// width (capped at 16), pointers at 8, aggregates by their strictest
/// member. Shared with lowering so structural sizes agree.
pub fn ty_align(ty: &Ty) -> u64 {
    match ty {
        Ty::Void | Ty::Bool => 1,
        Ty::Int { bits } => ((*bits).max(8) / 8).min(16) as u64,
        // `_BitInt` storage only aligns to 8 past the first 8-byte unit,
        // even when the object spans several of them (clang does the
        // same), so a wide value never demands 16-byte alignment.
        Ty::WideInt { bits } => ((*bits).max(8) / 8).min(8) as u64,
        Ty::Float(k) => match k {
            FloatKind::F32 => 4,
            FloatKind::F64 => 8,
            // Quad and x87 extended both align to 16 bytes.
            FloatKind::F128 | FloatKind::F80 => 16,
        },
        Ty::Ptr { bits, .. } => ((*bits as u64).max(8) / 8).min(16),
        Ty::BuiltinVaList { align, .. } => *align,
        Ty::Array { elem, .. } => ty_align(elem),
        Ty::Struct { fields, .. } => fields.iter().map(ty_align).max().unwrap_or(1),
    }
}

/// Flat operand list for display and use-counting.
pub fn gep_index_operands(idx: &[GepIndex]) -> SmallVec<[super::ir::Value; 4]> {
    idx.iter()
        .filter_map(|i| match i {
            GepIndex::Index(v) => Some(*v),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_size_and_offsets() {
        let point = Ty::Struct {
            fields: vec![Ty::int(32), Ty::int(32)],
            offsets: vec![0, 4],
            size: 8,
        };
        assert!(point.is_aggregate());
        assert_eq!(point.size_bytes(), 8);
        assert_eq!(struct_field_offsets(&point), vec![0, 4]);
        assert_eq!(dense_field_offsets(&[Ty::int(8), Ty::int(32)]), vec![0, 4]);
    }

    #[test]
    fn gep_steps_through_aggregates() {
        let row = Ty::Array {
            elem: Box::new(Ty::int(32)),
            len: 4,
        };
        let elem = Signature::gep_step(&row, &GepIndex::ConstIndex(2)).unwrap();
        assert_eq!(elem, Ty::int(32));
        let s = Ty::Struct {
            fields: vec![Ty::int(32), row],
            offsets: vec![0, 4],
            size: 20,
        };
        let arr = Signature::gep_step(&s, &GepIndex::Field(1)).unwrap();
        assert!(arr.is_aggregate());
        assert!(Signature::gep_step(&s, &GepIndex::Field(9)).is_none());
    }
}
