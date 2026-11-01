//! MIR to portable Lua 5.1 source.
//!
//! Every MIR function becomes one anonymous Lua function installed in the
//! `FN` table. Functions are not declared as chunk-level locals, so a
//! translation unit is not capped by Lua 5.1's 200-local limit. The
//! control-flow graph is executed by a small state machine (`pc` plus an
//! if/elseif chain), which sidesteps Lua 5.1's lack of `goto`. SSA values
//! live in a per-call table, so functions with more values than Lua's
//! 200-local limit still work.
//!
//! Memory is a sparse byte table; pointers are ordinary Lua numbers. MIR
//! aggregates already lower to addresses, which is exactly the model used
//! here. Scalars are Lua numbers, with explicit width normalization for
//! integer operations.

use std::collections::HashMap;
use std::fmt::Write as _;

use glcc_mir as mir;
use mir::ty::GepIndex;
use mir::{Block, Const, DataReloc, Def, FloatKind, Func, Function, Inst, InstKind, Module, Terminator, Ty, Value};

use crate::runtime::PRELUDE;

const DATA_BASE: u64 = 0x1000;
const ARG_RESERVE: u64 = 0x20000;
const MIN_STACK: u64 = 1 << 24;

fn align_up(value: u64, align: u64) -> u64 {
    value.next_multiple_of(align.max(1))
}

fn unsupported(what: impl std::fmt::Display) -> ! {
    panic!("glcc_lower_lua does not lower {what} yet")
}

fn scalar_size(ty: &Ty) -> u64 {
    match ty {
        Ty::Bool => 1,
        Ty::Int { bits } => (*bits).max(8).div_ceil(8) as u64,
        Ty::Ptr { bits, .. } => (*bits).max(8).div_ceil(8) as u64,
        Ty::Float(FloatKind::F32) => 4,
        Ty::Float(FloatKind::F64) => 8,
        Ty::Float(k) => unsupported(format!("{k:?} memory values")),
        other => unsupported(format!("{other:?} as a scalar")),
    }
}

fn int_bits(ty: &Ty) -> u32 {
    match ty {
        Ty::Bool => 1,
        Ty::Int { bits } => *bits,
        Ty::Ptr { bits, .. } => *bits,
        other => unsupported(format!("{other:?} as an integer")),
    }
}

/// Static addresses for globals and aggregate constants.
struct Layout {
    global_addrs: Vec<u64>,
    blobs: Vec<HashMap<Value, u64>>,
    argv: u64,
    heap: u64,
    stack: u64,
}

impl Layout {
    fn new(module: &Module) -> Layout {
        let mut cursor = DATA_BASE;
        let mut global_addrs = Vec::with_capacity(module.globals.len());
        for global in module.globals.iter() {
            cursor = align_up(cursor, mir::ty::ty_align(&global.ty));
            global_addrs.push(cursor);
            cursor += global.ty.size_bytes().max(1);
        }

        let mut blobs = Vec::with_capacity(module.funcs.len());
        for func in module.funcs.iter() {
            let mut map = HashMap::new();
            for (value, data) in func.values.iter_enumerated() {
                let Def::Const(c) = &data.def else { continue };
                let wanted = match c {
                    Const::Bytes(_) => true,
                    Const::Zero | Const::Undef => data.ty.is_aggregate(),
                    _ => false,
                };
                if !wanted {
                    continue;
                }
                let size = match c {
                    Const::Bytes(bytes) => data.ty.size_bytes().max(bytes.len() as u64).max(1),
                    _ => data.ty.size_bytes().max(1),
                };
                cursor = align_up(cursor, mir::ty::ty_align(&data.ty));
                map.insert(value, cursor);
                cursor += size;
            }
            blobs.push(map);
        }

        let argv = align_up(cursor, 16);
        let heap = align_up(argv + ARG_RESERVE, 16);
        let stack = MIN_STACK.max(align_up(heap + (1 << 20), 16));
        Layout {
            global_addrs,
            blobs,
            argv,
            heap,
            stack,
        }
    }
}

/// Blocks reachable from the entry block, indexed by block.
fn reachable_blocks(func: &Function) -> Vec<bool> {
    let mut reachable = vec![false; func.blocks.len()];
    let mut work = vec![func.entry];
    reachable[func.entry.index()] = true;
    while let Some(block) = work.pop() {
        let Some(term) = &func.blocks[block].term else { continue };
        for succ in term.successors() {
            if !reachable[succ.index()] {
                reachable[succ.index()] = true;
                work.push(succ);
            }
        }
    }
    reachable
}

/// Per-function stack-frame slots for allocas and aggregate temporaries.
struct Frame {
    size: u64,
    slots: HashMap<u32, u64>,
}

impl Frame {
    fn build(func: &Function, reachable: &[bool]) -> Frame {
        let mut cursor = 0u64;
        let mut slots = HashMap::new();
        for (block_index, block) in func.blocks.iter_enumerated() {
            if !reachable[block_index.index()] {
                continue;
            }
            for &inst in &block.insts {
                let data = &func.insts[inst];
                let (size, align) = match &data.kind {
                    InstKind::Alloca { pointee, count } => {
                        let size = pointee.size_bytes().saturating_mul(*count).max(1);
                        (size, mir::ty::ty_align(pointee).max(1))
                    }
                    InstKind::Load { .. }
                    | InstKind::InsertValue { .. }
                    | InstKind::VaArg { .. }
                    | InstKind::Call { .. }
                    | InstKind::CallIndirect { .. }
                    | InstKind::CallExtern { .. } => {
                        let Some(result) = data.result else { continue };
                        let ty = &func.values[result].ty;
                        if !ty.is_aggregate() {
                            continue;
                        }
                        (ty.size_bytes().max(1), mir::ty::ty_align(ty).max(1))
                    }
                    _ => continue,
                };
                cursor = align_up(cursor, align);
                slots.insert(inst.index() as u32, cursor);
                cursor += size;
            }
        }
        Frame {
            size: align_up(cursor, 16),
            slots,
        }
    }
}

fn lua_quote(bytes: &[u8]) -> String {
    let mut out = String::from("\"");
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            0x20..=0x7e => out.push(b as char),
            _ => {
                let _ = write!(out, "\\{b:03}");
            }
        }
    }
    out.push('"');
    out
}

fn float_literal(value: f64) -> String {
    if value.is_nan() {
        "(0/0)".to_string()
    } else if value == f64::INFINITY {
        "(1/0)".to_string()
    } else if value == f64::NEG_INFINITY {
        "(-1/0)".to_string()
    } else if value == 0.0 {
        if value.is_sign_negative() { "(-0.0)".to_string() } else { "0.0".to_string() }
    } else {
        format!("{value:?}")
    }
}

fn const_float(c: &Const) -> f64 {
    match c {
        Const::Float { kind: FloatKind::F32, bits } => f32::from_bits(*bits as u32) as f64,
        Const::Float { kind: FloatKind::F64, bits } => f64::from_bits(*bits as u64),
        Const::Float { kind, .. } => unsupported(format!("{kind:?} constants")),
        _ => panic!("not a float constant"),
    }
}

fn int_bytes(bits: u32, value: i128) -> Vec<u8> {
    let n = bits.max(8).div_ceil(8) as usize;
    let raw = value as u128;
    (0..n).map(|i| (raw >> (i * 8)) as u8).collect()
}

fn float_bytes(c: &Const) -> Vec<u8> {
    match c {
        Const::Float { kind: FloatKind::F32, bits } => (*bits as u32).to_le_bytes().to_vec(),
        Const::Float { kind: FloatKind::F64, bits } => (*bits as u64).to_le_bytes().to_vec(),
        Const::Float { kind, .. } => unsupported(format!("{kind:?} constants")),
        _ => panic!("not a float constant"),
    }
}

fn field_of(ty: &Ty, index: u32) -> (u64, Ty) {
    match ty {
        Ty::Struct { fields, offsets, .. } => {
            let ty = fields.get(index as usize).unwrap_or_else(|| panic!("struct field {index} out of range")).clone();
            let off = offsets.get(index as usize).copied().unwrap_or_else(|| mir::ty::dense_field_offsets(fields)[index as usize]);
            (off, ty)
        }
        Ty::Array { elem, .. } => (elem.size_bytes() * index as u64, (**elem).clone()),
        other => unsupported(format!("field {index} of {other:?}")),
    }
}

fn element_of(ty: &Ty) -> Ty {
    match ty {
        Ty::Array { elem, .. } => (**elem).clone(),
        Ty::Ptr { pointee, .. } => (**pointee).clone(),
        other => other.clone(),
    }
}

pub fn lower_module(module: &Module) -> String {
    let layout = Layout::new(module);
    let mut out = String::new();
    out.push_str("-- Generated by glcc_lower_lua (MIR -> Lua 5.1).\n");
    out.push_str(PRELUDE);
    let _ = writeln!(out, "\nHEAP = {}\nSP = {}\nARGV = {}", layout.heap, layout.stack, layout.argv);
    emit_data(module, &layout, &mut out);

    let extern_base = module.funcs.len();
    for (index, ext) in module.externs.iter_enumerated() {
        let _ = writeln!(out, "FN[{}] = __extern({})", extern_base + index.index(), lua_quote(ext.name.as_bytes()));
    }

    for (index, func) in module.funcs.iter_enumerated() {
        let reachable = reachable_blocks(func);
        let frame = Frame::build(func, &reachable);
        let mut emitter = FnEmitter::new(module, index, func, &layout, &frame, reachable);
        emitter.emit();
        out.push_str(&emitter.out);
        out.push('\n');
    }

    for init in &module.init_funcs {
        let _ = writeln!(out, "FN[{}]()", init.index());
    }
    for global in module.globals.iter() {
        if global.section.as_deref() != Some(".init_array") {
            continue;
        }
        for (_, reloc) in &global.relocs {
            match reloc {
                DataReloc::Func { target } => {
                    let _ = writeln!(out, "FN[{}]()", target.index());
                }
                DataReloc::FuncExtern { target } => {
                    let _ = writeln!(out, "FN[{}]()", module.funcs.len() + target.index());
                }
                DataReloc::Global { .. } => {}
            }
        }
    }

    if let Some(main) = module.funcs.iter().position(|f| f.name == "main") {
        let params = module.funcs[mir::Func::from_usize(main)].sig.params.len();
        let _ = writeln!(out, "\ndo\n  local __rc");
        match params {
            0 => {
                let _ = writeln!(out, "  __rc = FN[{main}]()");
            }
            1 => {
                let _ = writeln!(out, "  local __argc = __argv()");
                let _ = writeln!(out, "  __rc = FN[{main}](__argc)");
            }
            _ => {
                let _ = writeln!(out, "  local __argc, __argvp = __argv()");
                let mut args = String::from("__argc, __argvp");
                for _ in 2..params {
                    args.push_str(", 0");
                }
                let _ = writeln!(out, "  __rc = FN[{main}]({args})");
            }
        }
        let _ = writeln!(out, "  if __rc ~= nil and __rc ~= 0 then os.exit(__rc) end\nend");
    }

    out
}

fn stdio_id(name: &str) -> Option<u64> {
    match name {
        "stdin" | "__stdinp" => Some(0),
        "stdout" | "__stdoutp" => Some(1),
        "stderr" | "__stderrp" => Some(2),
        _ => None,
    }
}

fn emit_data(module: &Module, layout: &Layout, out: &mut String) {
    for (index, global) in module.globals.iter_enumerated() {
        let addr = layout.global_addrs[index.index()];
        let Some(init) = &global.init else { continue };
        match init {
            Const::Undef | Const::Null | Const::Zero => {}
            Const::Int { bits, value } => {
                let bytes = int_bytes(*bits, *value);
                let _ = writeln!(out, "__putbytes({addr}, {})", lua_quote(&bytes));
            }
            Const::Float { .. } => {
                let bytes = float_bytes(init);
                let _ = writeln!(out, "__putbytes({addr}, {})", lua_quote(&bytes));
            }
            Const::Bytes(bytes) => {
                let _ = writeln!(out, "__putbytes({addr}, {})", lua_quote(bytes));
            }
        }
        for (offset, reloc) in &global.relocs {
            let value = reloc_value(module, layout, reloc);
            let _ = writeln!(out, "__wr({}, 8, {value})", addr + offset);
        }
        for (offset, reloc) in &global.relocs32 {
            let value = reloc_value(module, layout, reloc);
            let _ = writeln!(out, "__wr({}, 4, {value})", addr + offset);
        }
    }

    for (index, global) in module.globals.iter_enumerated() {
        if !global.external {
            continue;
        }
        let Some(id) = stdio_id(&global.name) else { continue };
        if !matches!(&global.ty, Ty::Ptr { .. }) {
            continue;
        }
        let addr = layout.global_addrs[index.index()];
        let size = global.ty.size_bytes().max(1);
        let _ = writeln!(out, "__wr({addr}, {size}, {id})");
    }

    for (func_index, func) in module.funcs.iter_enumerated() {
        let blobs = &layout.blobs[func_index.index()];
        for (value, addr) in blobs {
            let Some(Const::Bytes(bytes)) = func.const_value(*value) else { continue };
            let _ = writeln!(out, "__putbytes({addr}, {})", lua_quote(bytes));
        }
    }
}

fn reloc_value(module: &Module, layout: &Layout, reloc: &DataReloc) -> u64 {
    match reloc {
        DataReloc::Global { target, addend } => layout.global_addrs[target.index()].wrapping_add_signed(*addend),
        DataReloc::Func { target } => target.index() as u64,
        DataReloc::FuncExtern { target } => (module.funcs.len() + target.index()) as u64,
    }
}

struct FnEmitter<'a> {
    module: &'a Module,
    index: Func,
    func: &'a Function,
    layout: &'a Layout,
    frame: &'a Frame,
    reachable: Vec<bool>,
    out: String,
    indent: usize,
}

impl<'a> FnEmitter<'a> {
    fn new(module: &'a Module, index: Func, func: &'a Function, layout: &'a Layout, frame: &'a Frame, reachable: Vec<bool>) -> FnEmitter<'a> {
        FnEmitter {
            module,
            index,
            func,
            layout,
            frame,
            reachable,
            out: String::new(),
            indent: 0,
        }
    }

    fn line(&mut self, text: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(text.as_ref());
        self.out.push('\n');
    }

    fn ty(&self, value: Value) -> &Ty {
        &self.func.values[value].ty
    }

    fn expr(&self, value: Value) -> String {
        match &self.func.values[value].def {
            Def::Const(c) => self.const_expr(value, c),
            Def::Param(..) | Def::Inst(..) => format!("V[{}]", value.index()),
        }
    }

    fn const_expr(&self, value: Value, c: &Const) -> String {
        match c {
            Const::Int { value: v, .. } => format!("{v}"),
            Const::Float { .. } => float_literal(const_float(c)),
            Const::Null => "0".to_string(),
            Const::Undef => {
                if self.ty(value).is_aggregate() {
                    format!("{}", self.blob(value))
                } else {
                    "0".to_string()
                }
            }
            Const::Zero => {
                if self.ty(value).is_aggregate() {
                    format!("{}", self.blob(value))
                } else {
                    "0".to_string()
                }
            }
            Const::Bytes(_) => format!("{}", self.blob(value)),
        }
    }

    fn blob(&self, value: Value) -> u64 {
        *self.layout.blobs[self.index.index()]
            .get(&value)
            .unwrap_or_else(|| panic!("missing blob for %{}", value.index()))
    }

    fn slot(&self, inst: Inst) -> u64 {
        *self
            .frame
            .slots
            .get(&(inst.index() as u32))
            .unwrap_or_else(|| panic!("missing frame slot for inst {}", inst.index()))
    }

    fn addr_with_base(&self, offset: u64) -> String {
        if offset == 0 { "base".to_string() } else { format!("(base + {offset})") }
    }

    fn set(&mut self, value: Value, expr: impl AsRef<str>) {
        self.line(format!("V[{}] = {}", value.index(), expr.as_ref()));
    }

    fn has_va_start(&self) -> bool {
        self.func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .any(|&inst| matches!(&self.func.insts[inst].kind, InstKind::VaStart { .. }))
    }

    fn emit(&mut self) {
        let index = self.index.index();
        let entry = self.func.entry;
        let sret = self.func.sig.ret.is_aggregate();
        let extra = usize::from(sret);
        let mut params: Vec<String> = (0..self.func.sig.params.len()).map(|i| format!("__a{i}")).collect();
        if sret {
            params.push("__sret".to_string());
        }
        if self.func.sig.variadic {
            params.push("...".to_string());
        }
        let signature = format!("FN[{index}] = function({})", params.join(", "));

        self.line(signature);
        self.indent += 1;
        self.line("local base = SP");
        self.line(format!("SP = SP + {}", self.frame.size));
        self.line("local V = {}");
        self.line(format!("local pc = {}", entry.index()));
        if self.func.sig.variadic {
            self.line("local __vatab = { ... }");
            self.line("local __van = select('#', ...)");
        } else if self.has_va_start() {
            self.line("local __vatab = {}");
            self.line("local __van = 0");
        }
        if sret {
            self.line("local SRET = __sret");
        }
        for (i, &param) in self.func.blocks[entry].params.iter().enumerate() {
            if i < self.func.sig.params.len() + extra {
                self.line(format!("V[{}] = __a{}", param.index(), i));
            }
        }
        self.line("while true do");
        self.indent += 1;

        let mut first = true;
        for (block_index, block) in self.func.blocks.iter_enumerated() {
            if !self.reachable[block_index.index()] {
                continue;
            }
            if first {
                self.line(format!("if pc == {} then", block_index.index()));
                first = false;
            } else {
                self.line(format!("elseif pc == {} then", block_index.index()));
            }
            self.indent += 1;
            self.line(format!("-- b{}", block_index.index()));
            for &inst in &block.insts {
                self.emit_inst(inst);
            }
            match &block.term {
                Some(term) => self.emit_term(block_index, term),
                None => {
                    self.line("SP = base");
                    self.line("return");
                }
            }
            self.indent -= 1;
        }
        if first {
            self.line("if false then");
            self.indent += 1;
            self.indent -= 1;
        } else {
            self.line("else");
            self.indent += 1;
            self.line("SP = base");
            self.line("return");
            self.indent -= 1;
        }
        self.line("end");
        self.indent -= 1;
        self.line("end");
        self.indent -= 1;
        self.line("end");
    }

    fn emit_inst(&mut self, inst: Inst) {
        let data = &self.func.insts[inst];
        let result = data.result;
        let result_ty = result.map(|v| self.func.values[v].ty.clone());

        match &data.kind {
            InstKind::Add(a, b) => self.binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) + ({y})")),
            InstKind::Sub(a, b) => self.binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) - ({y})")),
            InstKind::Mul(a, b) => self.binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) * ({y})")),
            InstKind::FAdd(a, b) => self.float_binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) + ({y})")),
            InstKind::FSub(a, b) => self.float_binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) - ({y})")),
            InstKind::FMul(a, b) => self.float_binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) * ({y})")),
            InstKind::FDiv(a, b) => self.float_binary(result, result_ty.as_ref(), *a, *b, |x, y| format!("({x}) / ({y})")),
            InstKind::Udiv(a, b) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let (x, y) = (self.expr(*a), self.expr(*b));
                self.set(result.unwrap(), format!("__u(math.floor(__u({x}, {bits}) / __u({y}, {bits})), {bits})"));
            }
            InstKind::Sdiv(a, b) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let (x, y) = (self.expr(*a), self.expr(*b));
                self.set(result.unwrap(), format!("__u(__trunc(__s({x}, {bits}) / __s({y}, {bits})), {bits})"));
            }
            InstKind::Urem(a, b) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let (x, y) = (self.expr(*a), self.expr(*b));
                self.set(result.unwrap(), format!("__u(__u({x}, {bits}) % __u({y}, {bits}), {bits})"));
            }
            InstKind::Srem(a, b) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let (x, y) = (self.expr(*a), self.expr(*b));
                self.set(
                    result.unwrap(),
                    format!("__u(__s({x}, {bits}) - __trunc(__s({x}, {bits}) / __s({y}, {bits})) * __s({y}, {bits}), {bits})"),
                );
            }
            InstKind::And(a, b) => self.bit_binary(result, result_ty.as_ref(), *a, *b, "__band"),
            InstKind::Or(a, b) => self.bit_binary(result, result_ty.as_ref(), *a, *b, "__bor"),
            InstKind::Xor(a, b) => self.bit_binary(result, result_ty.as_ref(), *a, *b, "__bxor"),
            InstKind::Shl(a, b) => self.shift(result, result_ty.as_ref(), *a, *b, "__shl"),
            InstKind::LShr(a, b) => self.shift(result, result_ty.as_ref(), *a, *b, "__lshr"),
            InstKind::AShr(a, b) => self.shift(result, result_ty.as_ref(), *a, *b, "__ashr"),
            InstKind::Neg(a) => {
                let ty = result_ty.as_ref().unwrap();
                let x = self.expr(*a);
                if matches!(ty, Ty::Float(_)) {
                    self.set(result.unwrap(), format!("-({x})"));
                } else {
                    let bits = int_bits(ty);
                    self.set(result.unwrap(), format!("__u(-({x}), {bits})"));
                }
            }
            InstKind::Not(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__bnot({x}, {bits})"));
            }
            InstKind::FNeg(a) => {
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("-({x})"));
            }
            InstKind::Icmp { pred, lhs, rhs } => {
                let bits = int_bits(self.ty(*lhs));
                let (a, b) = (self.expr(*lhs), self.expr(*rhs));
                let cmp = match pred {
                    mir::IcmpPred::Eq => format!("__u({a}, {bits}) == __u({b}, {bits})"),
                    mir::IcmpPred::Ne => format!("__u({a}, {bits}) ~= __u({b}, {bits})"),
                    mir::IcmpPred::Ult => format!("__u({a}, {bits}) < __u({b}, {bits})"),
                    mir::IcmpPred::Ule => format!("__u({a}, {bits}) <= __u({b}, {bits})"),
                    mir::IcmpPred::Ugt => format!("__u({a}, {bits}) > __u({b}, {bits})"),
                    mir::IcmpPred::Uge => format!("__u({a}, {bits}) >= __u({b}, {bits})"),
                    mir::IcmpPred::Slt => format!("__s({a}, {bits}) < __s({b}, {bits})"),
                    mir::IcmpPred::Sle => format!("__s({a}, {bits}) <= __s({b}, {bits})"),
                    mir::IcmpPred::Sgt => format!("__s({a}, {bits}) > __s({b}, {bits})"),
                    mir::IcmpPred::Sge => format!("__s({a}, {bits}) >= __s({b}, {bits})"),
                };
                self.set(result.unwrap(), format!("({cmp}) and 1 or 0"));
            }
            InstKind::Fcmp { pred, lhs, rhs } => {
                let (a, b) = (self.expr(*lhs), self.expr(*rhs));
                let cmp = match pred {
                    mir::FcmpPred::Eq => format!("({a}) == ({b})"),
                    mir::FcmpPred::Ne => format!("({a}) ~= ({b})"),
                    mir::FcmpPred::Lt => format!("({a}) < ({b})"),
                    mir::FcmpPred::Le => format!("({a}) <= ({b})"),
                    mir::FcmpPred::Gt => format!("({a}) > ({b})"),
                    mir::FcmpPred::Ge => format!("({a}) >= ({b})"),
                    mir::FcmpPred::Uno => format!("({a}) ~= ({a}) or ({b}) ~= ({b})"),
                    mir::FcmpPred::Ord => format!("not (({a}) ~= ({a}) or ({b}) ~= ({b}))"),
                };
                self.set(result.unwrap(), format!("({cmp}) and 1 or 0"));
            }
            InstKind::Trunc(a) | InstKind::Zext(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__u({x}, {bits})"));
            }
            InstKind::Sext(a) => {
                let src = int_bits(self.ty(*a));
                int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__s({x}, {src})"));
            }
            InstKind::Bitcast(a) => {
                let src = self.ty(*a).clone();
                let dst = result_ty.clone().unwrap();
                let x = self.expr(*a);
                let expr = match (&src, &dst) {
                    (Ty::Float(FloatKind::F32), Ty::Int { .. } | Ty::Ptr { .. } | Ty::Bool) => format!("__f32frombits(__u({x}, 32))"),
                    (Ty::Float(FloatKind::F64), Ty::Int { .. } | Ty::Ptr { .. } | Ty::Bool) => format!("__i2f64({x})"),
                    (Ty::Int { .. } | Ty::Ptr { .. } | Ty::Bool, Ty::Float(FloatKind::F32)) => format!("__u(__f32bits({x}), 32)"),
                    (Ty::Int { .. } | Ty::Ptr { .. } | Ty::Bool, Ty::Float(FloatKind::F64)) => {
                        format!("(__u(__f64lo({x}), 32) + __u(__f64hi({x}), 32) * 4294967296)")
                    }
                    _ => x,
                };
                self.set(result.unwrap(), expr);
            }
            InstKind::SItoF(a) | InstKind::UItoF(a) => {
                let src = self.ty(*a);
                let bits = int_bits(src);
                let signed = matches!(&data.kind, InstKind::SItoF(_));
                let x = self.expr(*a);
                let val = if signed { format!("__s({x}, {bits})") } else { format!("__u({x}, {bits})") };
                let val = match result_ty.as_ref().unwrap() {
                    Ty::Float(FloatKind::F32) => format!("__f32({val})"),
                    Ty::Float(FloatKind::F64) => val,
                    other => unsupported(format!("conversion to {other:?}")),
                };
                self.set(result.unwrap(), val);
            }
            InstKind::SFtoI(a) | InstKind::UFtoI(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__u(__trunc({x}), {bits})"));
            }
            InstKind::Fptrunc(a) => {
                let x = self.expr(*a);
                match result_ty.as_ref().unwrap() {
                    Ty::Float(FloatKind::F32) => self.set(result.unwrap(), format!("__f32({x})")),
                    Ty::Float(FloatKind::F64) => self.set(result.unwrap(), x),
                    other => unsupported(format!("fptrunc to {other:?}")),
                }
            }
            InstKind::Fpext(a) => {
                let x = self.expr(*a);
                match result_ty.as_ref().unwrap() {
                    Ty::Float(FloatKind::F32) => self.set(result.unwrap(), format!("__f32({x})")),
                    Ty::Float(FloatKind::F64) => self.set(result.unwrap(), x),
                    other => unsupported(format!("fpext to {other:?}")),
                }
            }
            InstKind::PtrToInt(a) | InstKind::IntToPtr(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__u({x}, {bits})"));
            }
            InstKind::Select { cond, then_val, else_val } => {
                let (c, t, f) = (self.expr(*cond), self.expr(*then_val), self.expr(*else_val));
                self.set(result.unwrap(), format!("(({c}) ~= 0) and ({t}) or ({f})"));
            }
            InstKind::Alloca { .. } => {
                let off = self.slot(inst);
                let addr = self.addr_with_base(off);
                self.set(result.unwrap(), addr);
            }
            InstKind::Load { addr, .. } => {
                let ty = result_ty.clone().unwrap();
                let a = self.expr(*addr);
                if ty.is_aggregate() {
                    let slot = self.addr_with_base(self.slot(inst));
                    self.line(format!("__copy({slot}, {a}, {})", ty.size_bytes()));
                    self.set(result.unwrap(), slot);
                } else {
                    let load = self.load_expr(&ty, &a);
                    self.set(result.unwrap(), load);
                }
            }
            InstKind::Store { addr, val, .. } => {
                let ty = self.ty(*val).clone();
                let a = self.expr(*addr);
                let v = self.expr(*val);
                if ty.is_aggregate() {
                    self.line(format!("__copy({a}, {v}, {})", ty.size_bytes()));
                } else {
                    let store = self.store_stmt(&ty, &a, &v);
                    self.line(store);
                }
            }
            InstKind::GetElementPtr { base, indices } => {
                let addr = self.gep(*base, indices);
                self.set(result.unwrap(), addr);
            }
            InstKind::ExtractValue { agg, index } => {
                let agg_ty = self.ty(*agg).clone();
                let (off, elem) = field_of(&agg_ty, *index);
                let base = self.expr(*agg);
                let addr = if off == 0 { base } else { format!("({base} + {off})") };
                if elem.is_aggregate() {
                    self.set(result.unwrap(), addr);
                } else {
                    let load = self.load_expr(&elem, &addr);
                    self.set(result.unwrap(), load);
                }
            }
            InstKind::InsertValue { agg, elem, index } => {
                let agg_ty = self.ty(*agg).clone();
                let (off, elem_ty) = field_of(&agg_ty, *index);
                let base = self.expr(*agg);
                let value = self.expr(*elem);
                let slot = self.addr_with_base(self.slot(inst));
                let undef = matches!(self.func.const_value(*agg), Some(Const::Undef | Const::Zero));
                if undef {
                    self.line(format!("__fill({slot}, 0, {})", agg_ty.size_bytes()));
                } else {
                    self.line(format!("__copy({slot}, {base}, {})", agg_ty.size_bytes()));
                }
                let dst = if off == 0 { slot.clone() } else { format!("({slot} + {off})") };
                if elem_ty.is_aggregate() {
                    self.line(format!("__copy({dst}, {value}, {})", elem_ty.size_bytes()));
                } else {
                    let store = self.store_stmt(&elem_ty, &dst, &value);
                    self.line(store);
                }
                self.set(result.unwrap(), slot);
            }
            InstKind::Memcpy { dst, src, len } => {
                let (d, s) = (self.expr(*dst), self.expr(*src));
                self.line(format!("__copy({d}, {s}, {len})"));
            }
            InstKind::Memset { dst, byte, len } => {
                let d = self.expr(*dst);
                self.line(format!("__fill({d}, {byte}, {len})"));
            }
            InstKind::FuncAddr { func } => self.set(result.unwrap(), func.index().to_string()),
            InstKind::ExternAddr { target } => {
                self.set(result.unwrap(), (self.module.funcs.len() + target.index()).to_string());
            }
            InstKind::GlobalAddr { global } => {
                self.set(result.unwrap(), self.layout.global_addrs[global.index()].to_string());
            }
            InstKind::Call { callee, args } => self.emit_call(inst, result, result_ty.as_ref(), format!("FN[{}]", callee.index()), args),
            InstKind::CallIndirect { callee, args, .. } => {
                let target = format!("FN[{}]", self.expr(*callee));
                self.emit_call(inst, result, result_ty.as_ref(), target, args);
            }
            InstKind::CallExtern { target, args } => {
                let id = self.module.funcs.len() + target.index();
                self.emit_call(inst, result, result_ty.as_ref(), format!("FN[{id}]"), args);
            }
            InstKind::AtomicLoad { addr } => {
                let ty = result_ty.clone().unwrap();
                let a = self.expr(*addr);
                let load = self.load_expr(&ty, &a);
                self.set(result.unwrap(), load);
            }
            InstKind::AtomicStore { addr, val } => {
                let ty = self.ty(*val).clone();
                let (a, v) = (self.expr(*addr), self.expr(*val));
                if ty.is_aggregate() {
                    self.line(format!("__copy({a}, {v}, {})", ty.size_bytes()));
                } else {
                    let store = self.store_stmt(&ty, &a, &v);
                    self.line(store);
                }
            }
            InstKind::AtomicRmw { op, addr, val } => {
                let ty = self.ty(*val).clone();
                let bits = int_bits(&ty);
                let (a, v) = (self.expr(*addr), self.expr(*val));
                self.line(format!("local __old = __rd({a}, {})", scalar_size(&ty)));
                let update = match op {
                    mir::AtomicRmwOp::Add => format!("__u(__old + ({v}), {bits})"),
                    mir::AtomicRmwOp::Sub => format!("__u(__old - ({v}), {bits})"),
                    mir::AtomicRmwOp::And => format!("__band(__old, {v}, {bits})"),
                    mir::AtomicRmwOp::Nand => format!("__bnot(__band(__old, {v}, {bits}), {bits})"),
                    mir::AtomicRmwOp::Or => format!("__bor(__old, {v}, {bits})"),
                    mir::AtomicRmwOp::Xor => format!("__bxor(__old, {v}, {bits})"),
                    mir::AtomicRmwOp::Xchg => format!("__u({v}, {bits})"),
                    mir::AtomicRmwOp::Umin => format!("__u((__u(__old, {bits}) < __u({v}, {bits})) and __old or {v}, {bits})"),
                    mir::AtomicRmwOp::Umax => format!("__u((__u(__old, {bits}) > __u({v}, {bits})) and __old or {v}, {bits})"),
                    mir::AtomicRmwOp::Smin => format!("__u((__s(__old, {bits}) < __s({v}, {bits})) and __old or {v}, {bits})"),
                    mir::AtomicRmwOp::Smax => format!("__u((__s(__old, {bits}) > __s({v}, {bits})) and __old or {v}, {bits})"),
                };
                self.line(format!("__wr({a}, {}, {update})", scalar_size(&ty)));
                self.set(result.unwrap(), "__old");
            }
            InstKind::AtomicCas { addr, expected, replacement } => {
                let ty = self.ty(*expected).clone();
                let bits = int_bits(&ty);
                let (a, e, r) = (self.expr(*addr), self.expr(*expected), self.expr(*replacement));
                self.line(format!("local __old = __rd({a}, {})", scalar_size(&ty)));
                self.line(format!("if __u(__old, {bits}) == __u({e}, {bits}) then __wr({a}, {}, {r}) end", scalar_size(&ty)));
                self.set(result.unwrap(), "__old");
            }
            InstKind::AtomicFence => {}
            InstKind::Ceil(a) => {
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("math.ceil({x})"));
            }
            InstKind::Floor(a) => {
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("math.floor({x})"));
            }
            InstKind::Sqrt(a) => {
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("math.sqrt({x})"));
            }
            InstKind::Clz(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__clz({x}, {bits})"));
            }
            InstKind::Ctz(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__ctz({x}, {bits})"));
            }
            InstKind::Popcnt(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__popcnt({x}, {bits})"));
            }
            InstKind::Bswap(a) => {
                let bits = int_bits(result_ty.as_ref().unwrap());
                let x = self.expr(*a);
                self.set(result.unwrap(), format!("__bswap({x}, {bits})"));
            }
            InstKind::GetFramePointer => self.set(result.unwrap(), "base"),
            InstKind::VaStart { ap } => {
                let ap = self.expr(*ap);
                self.line(format!("__va_begin({ap}, __vatab, __van)"));
            }
            InstKind::VaArg { ap } => {
                let ap = self.expr(*ap);
                let ty = result_ty.as_ref().unwrap();
                let value = format!("__va_arg({ap})");
                if ty.is_aggregate() {
                    let slot = self.addr_with_base(self.slot(inst));
                    let tmp = format!("__vav{}", inst.index());
                    self.line(format!("local {tmp} = {value}"));
                    self.line(format!("__copy({slot}, {tmp}, {})", ty.size_bytes()));
                    self.set(result.unwrap(), slot);
                } else {
                    let value = match ty {
                        Ty::Float(FloatKind::F32) => format!("__f32({value})"),
                        Ty::Float(FloatKind::F64) => value,
                        other => {
                            let bits = int_bits(other);
                            format!("__u({value}, {bits})")
                        }
                    };
                    self.set(result.unwrap(), value);
                }
            }
            InstKind::VaEnd { .. } => {}
            InstKind::VaCopy { dst, src } => {
                let dst = self.expr(*dst);
                let src = self.expr(*src);
                self.line(format!("__copy({dst}, {src}, 8)"));
            }
            InstKind::Await { .. } => unsupported("await (the coroutine pass should have removed it)"),
            InstKind::InlineAsm { .. } => unsupported("inline assembly"),
        }
    }

    fn binary(&mut self, result: Option<Value>, ty: Option<&Ty>, a: Value, b: Value, f: impl Fn(String, String) -> String) {
        let bits = int_bits(ty.unwrap());
        let (a, b) = (self.expr(a), self.expr(b));
        self.set(result.unwrap(), format!("__u({}, {bits})", f(a, b)));
    }

    fn float_binary(&mut self, result: Option<Value>, ty: Option<&Ty>, a: Value, b: Value, f: impl Fn(String, String) -> String) {
        let (a, b) = (self.expr(a), self.expr(b));
        let expr = f(a, b);
        let expr = match ty.unwrap() {
            Ty::Float(FloatKind::F32) => format!("__f32({expr})"),
            Ty::Float(FloatKind::F64) => expr,
            other => unsupported(format!("float arithmetic on {other:?}")),
        };
        self.set(result.unwrap(), expr);
    }

    fn bit_binary(&mut self, result: Option<Value>, ty: Option<&Ty>, a: Value, b: Value, op: &str) {
        let bits = int_bits(ty.unwrap());
        let (a, b) = (self.expr(a), self.expr(b));
        self.set(result.unwrap(), format!("{op}({a}, {b}, {bits})"));
    }

    fn shift(&mut self, result: Option<Value>, ty: Option<&Ty>, a: Value, b: Value, op: &str) {
        let bits = int_bits(ty.unwrap());
        let (a, b) = (self.expr(a), self.expr(b));
        self.set(result.unwrap(), format!("{op}({a}, __u({b}, {bits}), {bits})"));
    }

    fn load_expr(&self, ty: &Ty, addr: &str) -> String {
        match ty {
            Ty::Float(FloatKind::F32) => format!("__rf32({addr})"),
            Ty::Float(FloatKind::F64) => format!("__rf64({addr})"),
            _ => format!("__rd({addr}, {})", scalar_size(ty)),
        }
    }

    fn store_stmt(&self, ty: &Ty, addr: &str, value: &str) -> String {
        match ty {
            Ty::Float(FloatKind::F32) => format!("__wf32({addr}, {value})"),
            Ty::Float(FloatKind::F64) => format!("__wf64({addr}, {value})"),
            _ => format!("__wr({addr}, {}, {value})", scalar_size(ty)),
        }
    }

    fn gep(&self, base: Value, indices: &[GepIndex]) -> String {
        let Ty::Ptr { pointee, .. } = self.ty(base) else {
            panic!("gep base is not a pointer");
        };
        let mut cur = (**pointee).clone();
        let mut offset = String::new();
        for index in indices {
            let (step, next) = match index {
                GepIndex::Field(field) => {
                    let (off, next) = field_of(&cur, *field);
                    (off.to_string(), next)
                }
                GepIndex::ConstIndex(i) => {
                    let elem = element_of(&cur);
                    let size = elem.size_bytes().max(1);
                    (format!("{}", i * size), elem)
                }
                GepIndex::Index(v) => {
                    let elem = element_of(&cur);
                    let size = elem.size_bytes().max(1);
                    (format!("({}) * {size}", self.expr(*v)), elem)
                }
            };
            if !offset.is_empty() {
                offset.push_str(" + ");
            }
            offset.push_str(&step);
            cur = next;
        }
        let base = self.expr(base);
        if offset.is_empty() || offset == "0" { base } else { format!("({base} + {offset})") }
    }

    fn emit_call(&mut self, inst: Inst, result: Option<Value>, ret: Option<&Ty>, callee: String, args: &[Value]) {
        let mut argv: Vec<String> = args.iter().map(|&a| self.expr(a)).collect();
        let aggregate = ret.is_some_and(Ty::is_aggregate);
        let slot = if aggregate {
            let slot = self.addr_with_base(self.slot(inst));
            argv.push(slot.clone());
            Some(slot)
        } else {
            None
        };
        let call = format!("{callee}({})", argv.join(", "));
        match (result, slot) {
            (Some(result), Some(slot)) => {
                self.line(call);
                self.set(result, slot);
            }
            (Some(result), None) => self.set(result, call),
            (None, _) => self.line(call),
        }
    }

    fn assign_args(&mut self, dest: Block, args: &[Value]) {
        if args.is_empty() {
            return;
        }
        let params = &self.func.blocks[dest].params;
        for (i, &arg) in args.iter().enumerate() {
            if i >= params.len() {
                break;
            }
            let value = self.expr(arg);
            self.line(format!("local __p{i} = {value}"));
        }
        for (i, &param) in params.iter().enumerate().take(args.len()) {
            self.line(format!("V[{}] = __p{i}", param.index()));
        }
    }

    fn emit_term(&mut self, block: Block, term: &Terminator) {
        match term {
            Terminator::Br { dest, args } => {
                self.assign_args(*dest, args);
                self.line(format!("pc = {}", dest.index()));
            }
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
                then_args,
                else_args,
            } => {
                let cond = self.expr(*cond);
                self.line(format!("if ({cond}) ~= 0 then"));
                self.indent += 1;
                self.assign_args(*then_block, then_args);
                self.line(format!("pc = {}", then_block.index()));
                self.indent -= 1;
                self.line("else");
                self.indent += 1;
                self.assign_args(*else_block, else_args);
                self.line(format!("pc = {}", else_block.index()));
                self.indent -= 1;
                self.line("end");
            }
            Terminator::Switch { scrutinee, cases, default } => {
                let bits = int_bits(self.ty(*scrutinee));
                let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
                let scrutinee = self.expr(*scrutinee);
                self.line(format!("local __sw = __u({scrutinee}, {bits})"));
                for (i, (case, dest)) in cases.iter().enumerate() {
                    let case = (*case as u128) & mask;
                    if i == 0 {
                        self.line(format!("if __sw == {case} then"));
                    } else {
                        self.line(format!("elseif __sw == {case} then"));
                    }
                    self.indent += 1;
                    self.line(format!("pc = {}", dest.index()));
                    self.indent -= 1;
                }
                self.line("else");
                self.indent += 1;
                self.line(format!("pc = {}", default.index()));
                self.indent -= 1;
                self.line("end");
            }
            Terminator::Return { val } => {
                if self.func.sig.ret.is_aggregate() {
                    let value = self.expr(val.unwrap_or_else(|| panic!("aggregate return without a value")));
                    self.line(format!("__copy(SRET, {value}, {})", self.func.sig.ret.size_bytes()));
                    self.line("SP = base");
                    self.line("return");
                } else if let Some(value) = val {
                    let value = self.expr(*value);
                    self.line("SP = base");
                    self.line(format!("return {value}"));
                } else {
                    self.line("SP = base");
                    self.line("return");
                }
                let _ = block;
            }
            Terminator::Unreachable => {
                self.line("SP = base");
                self.line("return");
            }
        }
    }
}
