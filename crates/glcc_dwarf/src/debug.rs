//! Lowered debug records and the DWARF writer.
//!
//! [`LoweredDebug`] is what `glcc_lower_cl` hands over once every Clif
//! function and data object exists: interned source locations, one
//! [`FuncDebug`] per defined function (machine size, source map, frame-
//! relative variable homes), and module-level [`GlobalDebug`] records.
//! [`emit_with_debug`] turns them, plus the MIR module's C-free file
//! and type tables, into `.debug_*` sections appended to a
//! `cranelift-object` product. DWARF specifics (gimli, section
//! relocations, line programs) live only in this module.

use std::collections::HashMap;
use std::path::Path;

use cranelift_module::{DataId, FuncId};
use gimli::write::{
    Address, AttributeValue, DwarfUnit, EndianVec, Expression, FileId, LineProgram, LineString, RelocateWriter, Relocation, RelocationTarget, Sections, Unit, UnitEntryId,
};
use gimli::{Encoding, Format, LineEncoding, SectionId};
use glcc_mir as mir;
use glcc_mir::debug::{BaseEncoding, DebugType, DebugTypeId, Location, MemberLocation, Qualifier};
use object::write::{Object, SectionKind, SymbolId};
use object::{RelocationEncoding, RelocationKind};

/// One machine-code range of a function and the source location index
/// it maps to. Sorted by `start`; offsets are relative to the function
/// symbol, so they survive PIE and ASLR.
#[derive(Debug, Clone, Copy)]
pub struct SrcLocRow {
    pub start: u32,
    pub end: u32,
    /// Index into `LoweredDebug::locations`; 0 means "no location".
    pub loc: u32,
}

/// One defined function's debug record.
#[derive(Debug, Clone)]
pub struct FuncDebug {
    pub func_id: FuncId,
    pub name: String,
    pub decl: Option<Location>,
    pub size: u32,
    pub srclocs: Vec<SrcLocRow>,
    /// Whether the function has external linkage (`DW_AT_external`).
    pub external: bool,
    /// Parameters in declaration order (`DW_TAG_formal_parameter`).
    pub params: Vec<VarDebug>,
    /// Body-level variables (whole-subprogram range).
    pub vars: Vec<VarDebug>,
    /// Nested lexical blocks; PC ranges are resolved at emission from
    /// the function's source map.
    pub scopes: Vec<ScopeDebug>,
}

/// One variable or parameter with a resolved home.
#[derive(Debug, Clone)]
pub struct VarDebug {
    pub name: Option<String>,
    pub decl: Option<Location>,
    pub debug_ty: Option<DebugTypeId>,
    pub loc: VarLoc,
}

/// Where a variable lives: a CFA-relative frame offset for automatics
/// homed in explicit stack slots, or a data symbol for globals and
/// block-scope statics.
#[derive(Debug, Clone, Copy)]
pub enum VarLoc {
    /// Offset from the CFA (`DW_OP_fbreg`).
    Fbreg(i64),
    /// Link-time address (`DW_OP_addr` against the data symbol).
    Static(DataId),
}

/// One lexical block before PC-range resolution.
#[derive(Debug, Clone)]
pub struct ScopeDebug {
    pub start: Location,
    pub end: Location,
    pub vars: Vec<VarDebug>,
    pub children: Vec<ScopeDebug>,
}

/// One module-level object.
#[derive(Debug, Clone)]
pub struct GlobalDebug {
    pub name: Option<String>,
    pub decl: Option<Location>,
    pub debug_ty: Option<DebugTypeId>,
    /// Link-time home; `None` for an `extern` declaration with no
    /// storage in this module, emitted as `DW_AT_declaration` without
    /// a location.
    pub data: Option<DataId>,
    pub external: bool,
}

/// Everything lowered during one module's code generation.
#[derive(Debug, Default)]
pub struct LoweredDebug {
    /// Location table; index 0 is the "no location" slot.
    locations: Vec<Option<Location>>,
    loc_index: HashMap<Location, u32>,
    pub funcs: Vec<FuncDebug>,
    pub globals: Vec<GlobalDebug>,
}

impl LoweredDebug {
    pub fn new() -> Self {
        Self {
            locations: vec![None],
            ..Default::default()
        }
    }

    /// Intern one source location, `None` becoming index 0 (the
    /// `SourceLoc::default()` Cranelift uses for "no location").
    pub fn intern_location(&mut self, loc: Option<Location>) -> u32 {
        let Some(loc) = loc else { return 0 };
        if let Some(&index) = self.loc_index.get(&loc) {
            return index;
        }
        let index = self.locations.len() as u32;
        self.locations.push(Some(loc));
        self.loc_index.insert(loc, index);
        index
    }

    /// The location behind one interned index, if any.
    pub fn location(&self, index: u32) -> Option<Location> {
        self.locations.get(index as usize).copied().flatten()
    }
}

#[derive(Debug, Clone)]
struct DebugSection {
    writer: EndianVec<gimli::RunTimeEndian>,
    relocations: Vec<Relocation>,
}

impl DebugSection {
    fn new(endian: gimli::RunTimeEndian) -> Self {
        Self {
            writer: EndianVec::new(endian),
            relocations: Vec::new(),
        }
    }
}

impl RelocateWriter for DebugSection {
    type Writer = EndianVec<gimli::RunTimeEndian>;

    fn writer(&self) -> &Self::Writer {
        &self.writer
    }

    fn writer_mut(&mut self) -> &mut Self::Writer {
        &mut self.writer
    }

    fn relocate(&mut self, relocation: Relocation) {
        self.relocations.push(relocation);
    }
}

/// Wrap a DIE in a qualifier DIE (`const`/`volatile`/`restrict`).
fn wrap_qual(unit: &mut Unit, parent: UnitEntryId, qual: Qualifier, inner: UnitEntryId) -> UnitEntryId {
    use gimli::constants::*;
    let tag = match qual {
        Qualifier::Const => DW_TAG_const_type,
        Qualifier::Volatile => DW_TAG_volatile_type,
        Qualifier::Restrict => DW_TAG_restrict_type,
    };
    let id = unit.add(parent, tag);
    unit.get_mut(id).set(DW_AT_type, AttributeValue::UnitRef(inner));
    id
}

/// Emit one base-type DIE.
fn base_die(unit: &mut Unit, parent: UnitEntryId, name: &str, encoding: gimli::constants::DwAte, size: u64) -> UnitEntryId {
    use gimli::constants::*;
    let id = unit.add(parent, DW_TAG_base_type);
    let entry = unit.get_mut(id);
    entry.set(DW_AT_name, AttributeValue::String(name.as_bytes().to_vec()));
    entry.set(DW_AT_encoding, AttributeValue::Encoding(encoding));
    entry.set(DW_AT_byte_size, AttributeValue::Udata(size));
    id
}

/// DWARF base encoding of one MIR scalar encoding.
fn ate(encoding: BaseEncoding) -> gimli::constants::DwAte {
    use gimli::constants::*;
    match encoding {
        BaseEncoding::Boolean => DW_ATE_boolean,
        BaseEncoding::Signed => DW_ATE_signed,
        BaseEncoding::Unsigned => DW_ATE_unsigned,
        BaseEncoding::SignedChar => DW_ATE_signed_char,
        BaseEncoding::UnsignedChar => DW_ATE_unsigned_char,
        BaseEncoding::Float => DW_ATE_float,
        BaseEncoding::ComplexFloat => DW_ATE_complex_float,
    }
}

/// Emit the DIE for one MIR type-table entry, memoized in `cache` by
/// table index (the table is already structurally deduplicated). Struct
/// and enum DIEs insert themselves before their members so pointers
/// back to the same type terminate.
fn ty_die(unit: &mut Unit, root: UnitEntryId, cache: &mut [Option<UnitEntryId>], types: &[DebugType], id: DebugTypeId) -> Option<UnitEntryId> {
    use gimli::constants::*;
    let index = id as usize;
    if let Some(cached) = cache.get(index).copied().flatten() {
        return Some(cached);
    }
    let ty = types.get(index)?;
    let die = match ty {
        DebugType::Base { name, encoding, size } => base_die(unit, root, name, ate(*encoding), *size),
        DebugType::Pointer { pointee } => {
            // LP64: every pointer is 8 bytes.
            let pointee_id = pointee.and_then(|p| ty_die(unit, root, cache, types, p));
            let id = unit.add(root, DW_TAG_pointer_type);
            let entry = unit.get_mut(id);
            entry.set(DW_AT_byte_size, AttributeValue::Udata(8));
            if let Some(pid) = pointee_id {
                entry.set(DW_AT_type, AttributeValue::UnitRef(pid));
            }
            id
        }
        DebugType::Qualified { qualifier, inner } => {
            let inner_id = ty_die(unit, root, cache, types, *inner)?;
            wrap_qual(unit, root, *qualifier, inner_id)
        }
        DebugType::Array { elem, count } => {
            let elem_id = ty_die(unit, root, cache, types, *elem)?;
            let id = unit.add(root, DW_TAG_array_type);
            unit.get_mut(id).set(DW_AT_type, AttributeValue::UnitRef(elem_id));
            // Constant bounds become a subrange child (`upper_bound`
            // is `count - 1` in DWARF); VLAs and `[]` stay boundless.
            if let Some(count) = count {
                let sub = unit.add(id, DW_TAG_subrange_type);
                let entry = unit.get_mut(sub);
                entry.set(DW_AT_type, AttributeValue::UnitRef(elem_id));
                entry.set(DW_AT_upper_bound, AttributeValue::Udata(count - 1));
            }
            id
        }
        DebugType::Struct {
            name,
            is_union,
            size,
            declaration,
            members,
        } => {
            let tag = if *is_union { DW_TAG_union_type } else { DW_TAG_structure_type };
            let id = unit.add(root, tag);
            cache[index] = Some(id);
            if let Some(name) = name {
                unit.get_mut(id).set(DW_AT_name, AttributeValue::String(name.as_bytes().to_vec()));
            }
            if *declaration {
                unit.get_mut(id).set(DW_AT_declaration, AttributeValue::Flag(true));
                return Some(id);
            }
            unit.get_mut(id).set(DW_AT_byte_size, AttributeValue::Udata(*size));
            for member in members {
                let mid = unit.add(id, DW_TAG_member);
                if let Some(name) = &member.name {
                    unit.get_mut(mid).set(DW_AT_name, AttributeValue::String(name.as_bytes().to_vec()));
                }
                if let Some(mty) = ty_die(unit, root, cache, types, member.ty) {
                    unit.get_mut(mid).set(DW_AT_type, AttributeValue::UnitRef(mty));
                }
                // Bit-fields locate by struct-relative bit offset alone
                // (like gcc); plain members by byte offset.
                let entry = unit.get_mut(mid);
                match member.location {
                    MemberLocation::Bit { offset, size } => {
                        entry.set(DW_AT_bit_size, AttributeValue::Udata(size));
                        entry.set(DW_AT_data_bit_offset, AttributeValue::Udata(offset));
                    }
                    MemberLocation::Byte(offset) => {
                        entry.set(DW_AT_data_member_location, AttributeValue::Udata(offset));
                    }
                }
            }
            return Some(id);
        }
        DebugType::Enum {
            name,
            repr,
            size,
            signed,
            declaration,
            enumerators,
        } => {
            let id = unit.add(root, DW_TAG_enumeration_type);
            cache[index] = Some(id);
            if let Some(name) = name {
                unit.get_mut(id).set(DW_AT_name, AttributeValue::String(name.as_bytes().to_vec()));
            }
            if *declaration {
                unit.get_mut(id).set(DW_AT_declaration, AttributeValue::Flag(true));
                return Some(id);
            }
            let repr_id = ty_die(unit, root, cache, types, *repr);
            {
                let entry = unit.get_mut(id);
                entry.set(DW_AT_byte_size, AttributeValue::Udata(*size));
                if let Some(rid) = repr_id {
                    entry.set(DW_AT_type, AttributeValue::UnitRef(rid));
                }
            }
            for e in enumerators {
                let ed = unit.add(id, DW_TAG_enumerator);
                let entry = unit.get_mut(ed);
                entry.set(DW_AT_name, AttributeValue::String(e.name.as_bytes().to_vec()));
                if *signed {
                    entry.set(DW_AT_const_value, AttributeValue::Sdata(e.value as i64));
                } else {
                    entry.set(DW_AT_const_value, AttributeValue::Udata(e.value as u64));
                }
            }
            return Some(id);
        }
        DebugType::Func { ret, params, variadic } => {
            let id = unit.add(root, DW_TAG_subroutine_type);
            if let Some(rid) = ret.and_then(|r| ty_die(unit, root, cache, types, r)) {
                unit.get_mut(id).set(DW_AT_type, AttributeValue::UnitRef(rid));
            }
            for param in params {
                let pid = unit.add(id, DW_TAG_formal_parameter);
                if let Some(pty) = ty_die(unit, root, cache, types, *param) {
                    unit.get_mut(pid).set(DW_AT_type, AttributeValue::UnitRef(pty));
                }
            }
            if *variadic {
                unit.add(id, DW_TAG_unspecified_parameters);
            }
            id
        }
    };
    cache[index] = Some(die);
    Some(die)
}

/// Split a file path into `(dir, file)` for the line table. Absolute
/// paths keep their real directory; bare names land in `comp_dir`.
fn split_dir_file<'a>(path: &'a str, comp_dir: &'a str) -> (&'a str, &'a str) {
    let p = Path::new(path);
    match (p.parent(), p.file_name()) {
        (Some(dir), Some(_)) if !dir.as_os_str().is_empty() => {
            let dir_s = dir.to_str().unwrap_or(comp_dir);
            let file_s = p.file_name().and_then(|f| f.to_str()).unwrap_or(path);
            if dir_s.is_empty() { (comp_dir, file_s) } else { (dir_s, file_s) }
        }
        _ => (comp_dir, path),
    }
}

fn comp_dir_string() -> String {
    std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| ".".to_string())
}

/// Register one module file table entry in the line program.
fn intern_file(file_ids: &mut [Option<FileId>], line_prog: &mut LineProgram, files: &[mir::File], comp_dir: &str, index: u32) -> Option<FileId> {
    let slot = file_ids.get_mut(index as usize)?;
    if let Some(id) = *slot {
        return Some(id);
    }
    let path = files.get(index as usize)?.path.as_str();
    if path.is_empty() {
        return None;
    }
    let (dir, file) = split_dir_file(path, comp_dir);
    if file.is_empty() {
        return None;
    }
    // The working directory is index 0; reuse it for bare names so the
    // table stays small and `llvm-dwarfdump --verify` is happy.
    let dir_id = if dir == comp_dir {
        line_prog.default_directory()
    } else {
        line_prog.add_directory(LineString::String(dir.as_bytes().to_vec()))
    };
    let id = line_prog.add_file(LineString::String(file.as_bytes().to_vec()), dir_id, None);
    *slot = Some(id);
    Some(id)
}

/// Location expression for one variable home: `DW_OP_fbreg` into its
/// slot or `DW_OP_addr` of its data symbol.
fn location_expr(loc: VarLoc, data_sym: &dyn Fn(DataId) -> Option<usize>) -> Option<Expression> {
    let mut expr = Expression::new();
    match loc {
        VarLoc::Fbreg(off) => {
            expr.op_fbreg(off);
            Some(expr)
        }
        VarLoc::Static(did) => {
            let idx = data_sym(did)?;
            expr.op_addr(Address::Symbol { symbol: idx, addend: 0 });
            Some(expr)
        }
    }
}

struct VarDie<'a> {
    name: Option<&'a str>,
    decl: Option<Location>,
    debug_ty: Option<DebugTypeId>,
    /// `None` for a declaration with no storage here (`extern`): the
    /// DIE is emitted without a location.
    loc: Option<VarLoc>,
    external: bool,
}

/// Emit one `DW_TAG_variable` / `DW_TAG_formal_parameter` DIE.
fn emit_var(
    unit: &mut Unit,
    parent: UnitEntryId,
    root: UnitEntryId,
    tag: gimli::constants::DwTag,
    var: VarDie<'_>,
    file_ids: &[Option<FileId>],
    types: &[DebugType],
    type_cache: &mut [Option<UnitEntryId>],
    data_sym: &dyn Fn(DataId) -> Option<usize>,
) {
    use gimli::constants::*;
    let location = match var.loc {
        Some(loc) => match location_expr(loc, data_sym) {
            Some(expr) => Some(expr),
            None => return,
        },
        None => None,
    };
    let die = unit.add(parent, tag);
    if let Some(ty_id) = var.debug_ty.and_then(|id| ty_die(unit, root, type_cache, types, id)) {
        unit.get_mut(die).set(DW_AT_type, AttributeValue::UnitRef(ty_id));
    }
    let entry = unit.get_mut(die);
    if let Some(name) = var.name {
        entry.set(DW_AT_name, AttributeValue::String(name.as_bytes().to_vec()));
    }
    if let Some(decl) = var.decl {
        if let Some(Some(file)) = file_ids.get(decl.file as usize) {
            entry.set(DW_AT_decl_file, AttributeValue::FileIndex(Some(*file)));
        }
        entry.set(DW_AT_decl_line, AttributeValue::Udata(u64::from(decl.line)));
    }
    match location {
        Some(expr) => entry.set(DW_AT_location, AttributeValue::Exprloc(expr)),
        None => entry.set(DW_AT_declaration, AttributeValue::Flag(true)),
    }
    if var.external {
        entry.set(DW_AT_external, AttributeValue::Flag(true));
    }
}

/// Source map rows of one function, sorted, with locations resolved.
fn resolve_rows(debug: &LoweredDebug, func: &FuncDebug) -> Vec<(u32, u32, Option<Location>)> {
    let mut rows: Vec<(u32, u32, Option<Location>)> = func.srclocs.iter().map(|row| (row.start, row.end, debug.location(row.loc))).collect();
    rows.sort_by_key(|row| row.0);
    rows
}

/// PC range of one scope: the extent of rows whose (start-only)
/// locations lie within the scope's location range. `None` when the
/// scope generated no code.
fn scope_range(scope: &ScopeDebug, rows: &[(u32, u32, Option<Location>)]) -> Option<(u32, u32)> {
    let mut start = u32::MAX;
    let mut end = 0u32;
    for (s, e, loc) in rows {
        let Some(loc) = loc else { continue };
        if loc.file != scope.start.file {
            continue;
        }
        if (loc.line, loc.column) >= (scope.start.line, scope.start.column) && (loc.line, loc.column) <= (scope.end.line, scope.end.column) {
            start = start.min(*s);
            end = end.max(*e);
        }
    }
    if end > start { Some((start, end)) } else { None }
}

struct ScopeCtx<'a> {
    rows: &'a [(u32, u32, Option<Location>)],
    file_ids: &'a [Option<FileId>],
    types: &'a [DebugType],
    data_sym: &'a dyn Fn(DataId) -> Option<usize>,
    sym_idx: usize,
}

/// Emit one scope level's children under `parent`: ranged scopes become
/// lexical-block DIEs, scopes that generated no code hoist their
/// variables and ranged descendants into `parent`.
fn emit_scopes(unit: &mut Unit, parent: UnitEntryId, root: UnitEntryId, type_cache: &mut [Option<UnitEntryId>], ctx: &ScopeCtx<'_>, scopes: &[ScopeDebug]) {
    use gimli::constants::*;
    for scope in scopes {
        match scope_range(scope, ctx.rows) {
            Some((start, end)) => {
                let die = unit.add(parent, DW_TAG_lexical_block);
                {
                    let entry = unit.get_mut(die);
                    entry.set(
                        DW_AT_low_pc,
                        AttributeValue::Address(Address::Symbol {
                            symbol: ctx.sym_idx,
                            addend: i64::from(start),
                        }),
                    );
                    entry.set(DW_AT_high_pc, AttributeValue::Udata(u64::from(end - start)));
                }
                for var in &scope.vars {
                    emit_var(unit, die, root, DW_TAG_variable, var_die(var, false), ctx.file_ids, ctx.types, type_cache, ctx.data_sym);
                }
                emit_scopes(unit, die, root, type_cache, ctx, &scope.children);
            }
            None => {
                for var in &scope.vars {
                    emit_var(unit, parent, root, DW_TAG_variable, var_die(var, false), ctx.file_ids, ctx.types, type_cache, ctx.data_sym);
                }
                emit_scopes(unit, parent, root, type_cache, ctx, &scope.children);
            }
        }
    }
}

fn var_die(var: &VarDebug, external: bool) -> VarDie<'_> {
    VarDie {
        name: var.name.as_deref(),
        decl: var.decl,
        debug_ty: var.debug_ty,
        loc: Some(var.loc),
        external,
    }
}

/// Emit the object with `.debug_*` sections appended: line table, a
/// compile unit with subprograms (parameters, scoped variables),
/// file-scope variables, and the deduplicated type tree. Unwind
/// (`.eh_frame`) is already handled by `cranelift-object`.
pub fn emit_with_debug(product: cranelift_object::ObjectProduct, debug: &LoweredDebug, module: &mir::Module) -> Vec<u8> {
    let cranelift_object::ObjectProduct {
        mut object,
        functions,
        data_objects,
    } = product;
    if debug.funcs.is_empty() && debug.globals.is_empty() {
        return object.write().unwrap();
    }

    // gimli symbol indices: functions first (line rows and subprogram
    // addresses use these positions), then data symbols appended below.
    let mut symbols: Vec<SymbolId> = Vec::with_capacity(debug.funcs.len());
    for f in &debug.funcs {
        let sym = functions[f.func_id].expect("defined function without a symbol").0;
        symbols.push(sym);
    }
    // gimli index for one data symbol, appending on first use.
    let mut data_idx: Vec<(DataId, usize)> = Vec::new();
    let intern_data = |did: DataId, symbols: &mut Vec<SymbolId>, data_idx: &mut Vec<(DataId, usize)>| -> Option<usize> {
        if let Some((_, idx)) = data_idx.iter().find(|(d, _)| *d == did) {
            return Some(*idx);
        }
        let sym = data_objects[did]?.0;
        let idx = symbols.len();
        symbols.push(sym);
        data_idx.push((did, idx));
        Some(idx)
    };
    // Pre-resolve every referenced data symbol (globals plus static
    // homes) so later lookups never mutate the symbol table.
    fn resolve_scope_dids(scope: &ScopeDebug, out: &mut Vec<DataId>) {
        for var in &scope.vars {
            if let VarLoc::Static(did) = var.loc {
                out.push(did);
            }
        }
        for child in &scope.children {
            resolve_scope_dids(child, out);
        }
    }
    let mut referenced: Vec<DataId> = Vec::new();
    for global in &debug.globals {
        if let Some(did) = global.data {
            referenced.push(did);
        }
    }
    for func in &debug.funcs {
        for var in func.params.iter().chain(func.vars.iter()) {
            if let VarLoc::Static(did) = var.loc {
                referenced.push(did);
            }
        }
        for block in &func.scopes {
            let mut dids = Vec::new();
            resolve_scope_dids(block, &mut dids);
            referenced.extend(dids);
        }
    }
    for did in referenced {
        intern_data(did, &mut symbols, &mut data_idx);
    }

    let endian = gimli::RunTimeEndian::Little;
    let encoding = Encoding {
        format: Format::Dwarf32,
        version: 4,
        address_size: 8,
    };
    let comp_dir = comp_dir_string();

    // File table: every file the module interned (declarations,
    // locations, and type references all resolve through it).
    let mut line_prog = LineProgram::new(
        encoding,
        LineEncoding::default(),
        LineString::String(comp_dir.clone().into_bytes()),
        None,
        LineString::String(b"glcc".to_vec()),
        None,
    );
    let mut file_ids: Vec<Option<FileId>> = vec![None; module.files.len()];
    for index in 0..module.files.len() as u32 {
        intern_file(&mut file_ids, &mut line_prog, &module.files, &comp_dir, index);
    }

    // Line rows, one sequence per function. Offsets are relative to the
    // function symbol, so everything survives PIE and ASLR.
    for (sym_idx, func) in debug.funcs.iter().enumerate() {
        line_prog.begin_sequence(Some(Address::Symbol { symbol: sym_idx, addend: 0 }));
        let mut rows: Vec<(u32, FileId, u64, u64)> = Vec::new();
        for (start, _end, loc) in resolve_rows(debug, func) {
            let Some(loc) = loc else { continue };
            let Some(file) = intern_file(&mut file_ids, &mut line_prog, &module.files, &comp_dir, loc.file) else {
                continue;
            };
            rows.push((start, file, u64::from(loc.line), u64::from(loc.column)));
        }
        rows.sort_by_key(|r| r.0);
        // Prologue skipping: gdb/lldb guess `prologue_end` as the second
        // line row, so emit an initial row at the function start whose
        // `is_statement` differs from the body rows. Prologue bytes that
        // carry no srcloc stay covered by this row.
        if let Some((_, file, line, col)) = rows.first().copied() {
            line_prog.row().address_offset = 0;
            line_prog.row().file = file;
            line_prog.row().line = line;
            line_prog.row().column = col;
            line_prog.row().is_statement = false;
            line_prog.generate_row();
        }
        for (start, file, line, col) in rows {
            line_prog.row().address_offset = u64::from(start);
            line_prog.row().file = file;
            line_prog.row().line = line;
            line_prog.row().column = col;
            line_prog.row().is_statement = true;
            line_prog.generate_row();
        }
        line_prog.end_sequence(u64::from(func.size));
    }

    // DIE tree: compile unit, file-scope variables, one subprogram per
    // function (parameters, body variables, lexical blocks), and the
    // deduplicated type tree. The line program moves into the unit
    // after emission (file ids are just indices until then).
    let mut dwarf = DwarfUnit::new(encoding);
    let root = dwarf.unit.root();
    let cu_name = debug
        .funcs
        .first()
        .and_then(|f| f.decl)
        .and_then(|loc| module.files.get(loc.file as usize))
        .or_else(|| debug.globals.first().and_then(|g| g.decl).and_then(|loc| module.files.get(loc.file as usize)))
        .map(|f| f.path.clone())
        .unwrap_or_else(|| "glcc".to_string());
    {
        let root_die = dwarf.unit.get_mut(root);
        root_die.set(gimli::constants::DW_AT_name, AttributeValue::String(cu_name.as_bytes().to_vec()));
        root_die.set(gimli::constants::DW_AT_comp_dir, AttributeValue::String(comp_dir.as_bytes().to_vec()));
        root_die.set(gimli::constants::DW_AT_producer, AttributeValue::String(b"glcc".to_vec()));
        root_die.set(gimli::constants::DW_AT_language, AttributeValue::Language(gimli::constants::DW_LANG_C));
    }
    // Shared type cache for the whole unit: identical table entries
    // reuse one DIE no matter which variable references them.
    let mut type_cache: Vec<Option<UnitEntryId>> = vec![None; module.debug_types.len()];
    // Data-symbol resolver for `DW_OP_addr` locations. Symbols were
    // pre-resolved above, so this never mutates.
    let data_sym = |did: DataId| data_idx.iter().find(|(d, _)| *d == did).map(|(_, i)| *i);
    for global in &debug.globals {
        emit_var(
            &mut dwarf.unit,
            root,
            root,
            gimli::constants::DW_TAG_variable,
            VarDie {
                name: global.name.as_deref(),
                decl: global.decl,
                debug_ty: global.debug_ty,
                loc: global.data.map(VarLoc::Static),
                external: global.external,
            },
            &file_ids,
            &module.debug_types,
            &mut type_cache,
            &data_sym,
        );
    }
    for (sym_idx, func) in debug.funcs.iter().enumerate() {
        let die = dwarf.unit.add(root, gimli::constants::DW_TAG_subprogram);
        let entry = dwarf.unit.get_mut(die);
        entry.set(gimli::constants::DW_AT_name, AttributeValue::String(func.name.as_bytes().to_vec()));
        if let Some(decl) = func.decl {
            if let Some(Some(file)) = file_ids.get(decl.file as usize) {
                entry.set(gimli::constants::DW_AT_decl_file, AttributeValue::FileIndex(Some(*file)));
            }
            entry.set(gimli::constants::DW_AT_decl_line, AttributeValue::Udata(u64::from(decl.line)));
        }
        entry.set(gimli::constants::DW_AT_low_pc, AttributeValue::Address(Address::Symbol { symbol: sym_idx, addend: 0 }));
        entry.set(gimli::constants::DW_AT_high_pc, AttributeValue::Udata(u64::from(func.size)));
        let mut frame_base = Expression::new();
        frame_base.op(gimli::constants::DW_OP_call_frame_cfa);
        entry.set(gimli::constants::DW_AT_frame_base, AttributeValue::Exprloc(frame_base));
        if func.external {
            entry.set(gimli::constants::DW_AT_external, AttributeValue::Flag(true));
        }
        // Parameters in order, then body-level variables, then nested
        // lexical blocks.
        for param in &func.params {
            emit_var(
                &mut dwarf.unit,
                die,
                root,
                gimli::constants::DW_TAG_formal_parameter,
                var_die(param, false),
                &file_ids,
                &module.debug_types,
                &mut type_cache,
                &data_sym,
            );
        }
        for var in &func.vars {
            emit_var(
                &mut dwarf.unit,
                die,
                root,
                gimli::constants::DW_TAG_variable,
                var_die(var, false),
                &file_ids,
                &module.debug_types,
                &mut type_cache,
                &data_sym,
            );
        }
        let rows = resolve_rows(debug, func);
        let ctx = ScopeCtx {
            rows: &rows,
            file_ids: &file_ids,
            types: &module.debug_types,
            data_sym: &data_sym,
            sym_idx,
        };
        emit_scopes(&mut dwarf.unit, die, root, &mut type_cache, &ctx, &func.scopes);
    }

    // The line program moves in last; file ids were plain indices until now.
    dwarf.unit.line_program = line_prog;

    let mut sections = Sections::new(DebugSection::new(endian));
    dwarf.write(&mut sections).expect("DWARF emission failed");

    append_debug_sections(&mut object, sections, &symbols);
    object.write().unwrap()
}

fn append_debug_sections(object: &mut Object, sections: Sections<DebugSection>, symbols: &[SymbolId]) {
    let Sections {
        debug_abbrev,
        debug_info,
        debug_line,
        debug_line_str,
        debug_ranges,
        debug_rnglists,
        debug_loc,
        debug_loclists,
        debug_str,
        debug_frame,
        eh_frame: _,
        ..
    } = sections;
    // (gimli section, ELF name, contents). `.eh_frame` is owned by
    // cranelift-object already; never duplicate it here.
    let parts: Vec<(SectionId, &'static str, DebugSection)> = vec![
        (SectionId::DebugAbbrev, ".debug_abbrev", debug_abbrev.0),
        (SectionId::DebugInfo, ".debug_info", debug_info.0),
        (SectionId::DebugLine, ".debug_line", debug_line.0),
        (SectionId::DebugLineStr, ".debug_line_str", debug_line_str.0),
        (SectionId::DebugRanges, ".debug_ranges", debug_ranges.0),
        (SectionId::DebugRngLists, ".debug_rnglists", debug_rnglists.0),
        (SectionId::DebugLoc, ".debug_loc", debug_loc.0),
        (SectionId::DebugLocLists, ".debug_loclists", debug_loclists.0),
        (SectionId::DebugStr, ".debug_str", debug_str.0),
        (SectionId::DebugFrame, ".debug_frame", debug_frame.0),
    ];
    // Create object sections for non-empty data first, so section
    // relocations always have a target symbol to point at.
    let mut map: HashMap<SectionId, object::write::SectionId> = HashMap::new();
    for (id, name, part) in parts.iter() {
        if part.writer.slice().is_empty() {
            continue;
        }
        let obj_id = object.add_section(Vec::new(), name.as_bytes().to_vec(), SectionKind::Debug);
        object.append_section_data(obj_id, part.writer.slice(), 1);
        map.insert(*id, obj_id);
    }
    // Second pass: relocations. Symbol targets are function symbols;
    // section targets are the debug sections created above.
    for (id, _name, part) in &parts {
        if part.relocations.is_empty() {
            continue;
        }
        let Some(&obj_id) = map.get(id) else {
            continue;
        };
        for reloc in &part.relocations {
            // References between debug sections (abbrev offsets, string
            // offsets, line-program offsets) are section-relative;
            // symbol references (function addresses) are absolute.
            //
            // COFF encodes those section-relative references as
            // `SECREL`. Wider absolute relocations would both encode the
            // wrong value and make MSVC's linker reject the object
            // (LNK2017, "ADDR32 relocation ... invalid"). ELF keeps
            // absolute references to section symbols: non-allocated
            // sections link at address zero, so the value lands on the
            // offset — and the ELF writer has no `SECREL` equivalent.
            let coff = matches!(object.format(), object::BinaryFormat::Coff);
            let (target, kind) = match reloc.target {
                RelocationTarget::Symbol(idx) => match symbols.get(idx) {
                    Some(&sym) => (sym, RelocationKind::Absolute),
                    None => continue,
                },
                RelocationTarget::Section(target_id) => {
                    // Lazily create empty targets (e.g. an empty
                    // `.debug_line_str` under DWARF-4): the relocation
                    // still needs a section symbol.
                    let sym = if let Some(&obj_id) = map.get(&target_id) {
                        object.section_symbol(obj_id)
                    } else {
                        let name = target_id.name();
                        let obj_id = object.add_section(Vec::new(), name.as_bytes().to_vec(), SectionKind::Debug);
                        map.insert(target_id, obj_id);
                        object.section_symbol(obj_id)
                    };
                    let kind = if coff { RelocationKind::SectionOffset } else { RelocationKind::Absolute };
                    (sym, kind)
                }
            };
            let flags = object::RelocationFlags::Generic {
                kind,
                encoding: RelocationEncoding::Generic,
                size: reloc.size * 8,
            };
            object
                .add_relocation(
                    obj_id,
                    object::write::Relocation {
                        offset: reloc.offset as u64,
                        symbol: target,
                        addend: reloc.addend,
                        flags,
                    },
                )
                .expect("debug relocation failed");
        }
    }
}
