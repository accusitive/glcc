use glcc_frontend as fe;
use glcc_mir as mir;

/// Lower a checked frontend type to a MIR type. Qualifiers vanish;
/// signedness vanishes (it lives on operations, not types).
///
/// Complex (`_Complex`, GNU `__complex__`) erases to a two-field
/// struct of the component type: real part at offset zero, imaginary
/// part right after, sized at twice the component. The frontend
/// alignment of a complex type is its component's alignment, and the
/// size is already a multiple of it, so the structural layout agrees
/// with C exactly. All complex operations desugar during lowering to
/// component-wise MIR over those fields, and the aggregate calling
/// convention classifies the pair the way clang classifies C complex
/// (`float _Complex` as one F64 carrier, `double _Complex` as two F64
/// pieces, integer complex as integer pieces, `long double _Complex`
/// in memory) — verified both directions against clang, including
/// real libm calls like `cexp`.
///
/// One known divergence, inherited from the legacy backend:
/// `long double _Complex` *returns* stay memory-based (hidden
/// out-pointer) while GCC returns them in `st(0)`/`st(1)`, which
/// Cranelift cannot express — so calls into foreign code returning
/// `long double _Complex` do not interoperate (arguments do, as does
/// everything else).
///
/// Loudly unfinished: VLAs (needs dynamic slots). Unions and
/// bit-field structs lower as byte blobs (see `lower_union` and
/// `lower_bitfield_struct`); `packed` and `aligned` members lower
/// structurally, with offsets and sizes taken from the frontend
/// layout, so member access stays byte-exact. Only alignment
/// *metadata* is missing from MIR types, so stack slots and by-value
/// classification use natural alignment (globals carry the real one).
pub fn lower_ty(ty: &fe::parse::Ty, sema: &fe::sema::Sema, interner: &fe::lexer::Interner) -> mir::Ty {
    match ty.unqualified() {
        fe::parse::Ty::Void => mir::Ty::Void,
        fe::parse::Ty::Bool => mir::Ty::Bool,
        fe::parse::Ty::Char | fe::parse::Ty::SignedChar | fe::parse::Ty::UnsignedChar => mir::Ty::int(8),
        fe::parse::Ty::Short | fe::parse::Ty::UnsignedShort => mir::Ty::int(16),
        fe::parse::Ty::Int | fe::parse::Ty::UnsignedInt => mir::Ty::int(32),
        fe::parse::Ty::Long | fe::parse::Ty::UnsignedLong => mir::Ty::int((sema.target.long_size * 8) as u32),
        fe::parse::Ty::LongLong | fe::parse::Ty::UnsignedLongLong => mir::Ty::int(64),
        fe::parse::Ty::Int128 | fe::parse::Ty::UnsignedInt128 => mir::Ty::int(128),
        fe::parse::Ty::BitInt { width, .. } => {
            let bits = u32::try_from(u64::from(*width).div_ceil(8).next_power_of_two().min(8) * 8).expect("bit-int storage width");
            mir::Ty::int(bits)
        }
        fe::parse::Ty::Float => mir::Ty::Float(mir::FloatKind::F32),
        fe::parse::Ty::Double => mir::Ty::Float(mir::FloatKind::F64),
        fe::parse::Ty::LongDouble => match sema.target.long_double {
            fe::target::LongDoubleKind::F80 => mir::Ty::Float(mir::FloatKind::F80),
            fe::target::LongDoubleKind::F128 => mir::Ty::Float(mir::FloatKind::F128),
            fe::target::LongDoubleKind::F64 => mir::Ty::Float(mir::FloatKind::F64),
        },
        fe::parse::Ty::Float128 => mir::Ty::Float(mir::FloatKind::F128),
        fe::parse::Ty::Complex(inner) => {
            let comp = lower_ty(inner, sema, interner);
            let size = sema.type_size(inner, interner).expect("complex component has a size");
            mir::Ty::Struct {
                fields: vec![comp.clone(), comp],
                offsets: vec![0, size],
                size: size * 2,
            }
        }
        fe::parse::Ty::Pointer { pointee, size, .. } => {
            // Struct pointees erase to `void`: navigating into one
            // re-derives its layout from the frontend per hop
            // (`place` uses the checker's byte offsets), so keeping
            // the definition here buys nothing and diverges on
            // self-referential structs (`struct S { struct S *next; }`,
            // `FILE`, mutual recursion).
            let bits = size.bits((sema.target.pointer_size * 8) as u32);
            if matches!(pointee.unqualified(), fe::parse::Ty::Struct(_)) {
                mir::Ty::ptr_bits(mir::Ty::Void, bits)
            } else {
                mir::Ty::ptr_bits(lower_ty(pointee, sema, interner), bits)
            }
        }
        fe::parse::Ty::BuiltinVaList => mir::Ty::BuiltinVaList {
            size: sema.target.va_list_size,
            align: sema.target.va_list_align,
        },
        fe::parse::Ty::Func { .. } => mir::Ty::ptr(mir::Ty::Void),
        fe::parse::Ty::Array { elem, bound, .. } => match bound {
            // Unsized and starred arrays only occur decayed as pointers;
            // a bare one here names no storage.
            None => mir::Ty::ptr(lower_ty(elem, sema, interner)),
            Some(fe::parse::ArrayBound::Fixed(n)) => mir::Ty::Array {
                elem: Box::new(lower_ty(elem, sema, interner)),
                len: *n,
            },
            Some(fe::parse::ArrayBound::Star) => mir::Ty::ptr(lower_ty(elem, sema, interner)),
            // A VLA only survives as a decayed pointer or a pointee;
            // objects allocate dynamically (see `lower_local`).
            Some(fe::parse::ArrayBound::Vla(_)) => mir::Ty::ptr(lower_ty(elem, sema, interner)),
        },
        fe::parse::Ty::Struct(id) => lower_struct(*id, sema, interner),
        fe::parse::Ty::Enum(id) => {
            let repr = sema.enums[id.0 as usize].repr.clone();
            lower_ty(&repr, sema, interner)
        }
        fe::parse::Ty::Qualified { .. } => {
            unreachable!("unqualified() never returns Qualified")
        }
    }
}

/// Reject a struct embedding itself by value (through further
/// structs or fixed arrays): invalid C whose structural expansion
/// would otherwise recurse forever. Pointer and function edges cut
/// the walk, matching what `lower_ty` expands.
fn check_acyclic(id: fe::sema::StructId, sema: &fe::sema::Sema, stack: &mut Vec<fe::sema::StructId>) {
    let def = &sema.structs[id.0 as usize];
    for member in &def.members {
        check_ty_acyclic(&member.ty, sema, stack);
    }
}

fn check_ty_acyclic(ty: &fe::parse::Ty, sema: &fe::sema::Sema, stack: &mut Vec<fe::sema::StructId>) {
    match ty.unqualified() {
        fe::parse::Ty::Struct(id) => {
            if stack.contains(id) {
                panic!("struct definition embeds itself by value");
            }
            stack.push(*id);
            check_acyclic(*id, sema, stack);
            stack.pop();
        }
        fe::parse::Ty::Array { elem, .. } => check_ty_acyclic(elem, sema, stack),
        // Complex components are real scalars and cannot carry a
        // cycle, but the walk stays total by descending anyway.
        fe::parse::Ty::Complex(inner) => check_ty_acyclic(inner, sema, stack),
        _ => {}
    }
}

fn lower_struct(id: fe::sema::StructId, sema: &fe::sema::Sema, interner: &fe::lexer::Interner) -> mir::Ty {
    // By-value self-embedding (`struct S { struct S v; }`, invalid C the
    // parser rejects) would diverge the structural expansion below, so
    // reject it loudly up front. Only value edges recurse: pointers (the
    // legal self-reference) and function types (erased to `void*`) cannot
    // carry a cycle into MIR.
    check_acyclic(id, sema, &mut vec![id]);
    let def = &sema.structs[id.0 as usize];
    if def.is_union {
        return lower_union(id, def, sema, interner);
    }
    if def.members.iter().any(|m| m.bitfield.is_some()) {
        return lower_bitfield_struct(id, def, sema, interner);
    }
    let mut fields = Vec::with_capacity(def.members.len());
    for member in &def.members {
        fields.push(lower_ty(&member.ty, sema, interner));
    }
    // The C layout rides along: offsets and padded size from the
    // frontend, so sizes, strides, and field steps agree with C. No
    // frontend layout (incomplete use) falls back to dense packing,
    // matching the previous behavior exactly.
    match sema.struct_layout(id, interner) {
        Some(layout) => mir::Ty::Struct {
            fields,
            offsets: layout.member_offsets,
            size: layout.size,
        },
        None => {
            let offsets = mir::dense_field_offsets(&fields);
            let size = fields.iter().map(|f| f.size_bytes()).sum();
            mir::Ty::Struct { fields, offsets, size }
        }
    }
}

/// A struct carrying bit-fields as coalesced storage units: each
/// maximal same-start run with bit-fields emits one unit covering
/// its extent (next start, else struct size) as the smallest integer
/// type. Extents cover hidden unnamed widths by construction, and
/// mixed declared types share honestly; padding inside the extent is
/// indeterminate and harmless. Runs without bit-fields emit
/// member-wise as usual. Field indices shift by construction, so all
/// member access on such structs resolves through byte offsets (never
/// symbolic field steps).
fn lower_bitfield_struct(id: fe::sema::StructId, def: &fe::sema::StructDef, sema: &fe::sema::Sema, interner: &fe::lexer::Interner) -> mir::Ty {
    let layout = sema.struct_layout(id, interner).expect("layout of bitfield struct");
    let mut fields = Vec::new();
    let mut offsets = Vec::new();
    let mut i = 0;
    while i < def.members.len() {
        let start = layout.member_offsets[i];
        let mut j = i + 1;
        while j < def.members.len() && layout.member_offsets[j] == start {
            j += 1;
        }
        if !def.members[i..j].iter().any(|m| m.bitfield.is_some()) {
            for (k, m) in def.members[i..j].iter().enumerate() {
                // Member-wise runs keep one field per member at the
                // member's own offset.
                offsets.push(layout.member_offsets[i + k]);
                fields.push(lower_ty(&m.ty, sema, interner));
            }
        } else {
            let end = if j < def.members.len() { layout.member_offsets[j] } else { layout.size };
            if end > start {
                offsets.push(start);
                fields.push(unit_covering(end - start));
            }
        }
        i = j;
    }
    mir::Ty::Struct {
        fields,
        offsets,
        size: layout.size,
    }
}

/// Smallest integer type covering `bytes` (units never exceed 16).
fn unit_covering(bytes: u64) -> mir::Ty {
    match bytes {
        0 => panic!("empty storage unit"),
        1 => mir::Ty::int(8),
        2 => mir::Ty::int(16),
        3..=4 => mir::Ty::int(32),
        5..=8 => mir::Ty::int(64),
        _ => mir::Ty::int(128),
    }
}

/// A union as a byte blob of its exact C size: members overlay at
/// offset zero, so member access resolves through byte offsets (never
/// symbolic field steps) and the value only needs to be the right
/// number of bytes. The size comes from the frontend layout when it
/// exists — `packed`/`aligned` members break the "strictest member
/// alignment accompanies a member at least as large" coincidence the
/// natural computation relied on — and is recomputed structurally
/// otherwise. Classification follows wherever an eightbyte has
/// integer coverage: x86-64 merges overlay votes with integer priority
/// (confirmed against clang and the SysV-ABI writeup below), AAPCS64
/// takes X registers for any non-homogeneous union, and RISC-V takes
/// GPRs unconditionally. Divergences, documented rather than fixed
/// here: any eightbyte with floating-point but no integer coverage
/// crosses in SIMD registers on x86-64/AAPCS64 (we send integer
/// registers) — all-FP unions, plus mixed unions like
/// `union{float f[3]; short s[3]}` whose upper eightbyte is float-only.
/// 16-byte-aligned unions additionally miss the ARM/RISC-V
/// even-register skip.
fn lower_union(id: fe::sema::StructId, def: &fe::sema::StructDef, sema: &fe::sema::Sema, interner: &fe::lexer::Interner) -> mir::Ty {
    let len = match sema.struct_layout(id, interner) {
        Some(layout) => layout.size,
        None => {
            let mut size = 0u64;
            let mut align = 1u64;
            for member in &def.members {
                // Bit-fields count at their declared unit size: units
                // are what overlay, and padding analysis agrees with
                // the C size by the argument above.
                let f = lower_ty(&member.ty, sema, interner);
                size = size.max(f.size_bytes());
                align = align.max(mir::ty::ty_align(&f));
            }
            size.next_multiple_of(align.max(1))
        }
    };
    mir::Ty::Array {
        elem: Box::new(mir::Ty::int(8)),
        len,
    }
}

/// Whether a frontend type is a complex type (`_Complex`, GNU
/// `__complex__`). Qualification is transparent.
pub fn is_complex(ty: &fe::parse::Ty) -> bool {
    matches!(ty.unqualified(), fe::parse::Ty::Complex(_))
}

/// The real component type of a complex frontend type.
pub fn complex_component(ty: &fe::parse::Ty) -> fe::parse::Ty {
    match ty.unqualified() {
        fe::parse::Ty::Complex(inner) => inner.as_ref().clone(),
        _ => panic!("complex component of a non-complex type"),
    }
}

/// Whether an integer frontend type is signed. Plain `char` follows the
/// compilation target, mirroring the checker's rule.
pub fn is_signed(ty: &fe::parse::Ty, sema: &fe::sema::Sema) -> bool {
    match ty.unqualified() {
        fe::parse::Ty::Bool
        | fe::parse::Ty::UnsignedChar
        | fe::parse::Ty::UnsignedShort
        | fe::parse::Ty::UnsignedInt
        | fe::parse::Ty::UnsignedLong
        | fe::parse::Ty::UnsignedLongLong
        | fe::parse::Ty::UnsignedInt128 => false,
        fe::parse::Ty::BitInt { signed, .. } => *signed,
        fe::parse::Ty::Char => sema.target.char_signed,
        // An enum is signed exactly when its representation is.
        fe::parse::Ty::Enum(id) => {
            let repr = sema.enums[id.0 as usize].repr.clone();
            is_signed(&repr, sema)
        }
        _ => true,
    }
}

/// Width in bits of an integer MIR type.
pub fn int_bits(ty: &mir::Ty) -> u32 {
    match ty {
        mir::Ty::Bool => 1,
        mir::Ty::Int { bits } => *bits,
        _ => panic!("not an integer MIR type"),
    }
}
