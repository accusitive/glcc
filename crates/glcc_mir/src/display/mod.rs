use std::fmt::{self, Write};

use crate::debug::Location;
use crate::ir::{Function, InstKind, Module, Terminator};
use crate::ty::Ty;

fn ty_str(ty: &Ty) -> String {
    match ty {
        Ty::Void => "void".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Int { bits } => format!("i{bits}"),
        Ty::WideInt { bits } => format!("i{bits}"),
        Ty::Float(k) => format!("{k:?}").to_lowercase(),
        Ty::Ptr { pointee, bits } => {
            if *bits == 64 {
                format!("ptr({})", ty_str(pointee))
            } else {
                format!("ptr{bits}({})", ty_str(pointee))
            }
        }
        Ty::BuiltinVaList { .. } => "va_list".to_string(),
        Ty::Vector { elem, lanes } => format!("v{}x{}", lanes, ty_str(elem)),
        Ty::Array { elem, len } => format!("[{} x {}]", ty_str(elem), len),
        Ty::Struct { fields, .. } => {
            let mut s = String::from("struct{");
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&ty_str(f));
            }
            s.push('}');
            s
        }
    }
}

fn val_name(func: &Function, v: crate::ir::Value) -> String {
    match &func.values[v].def {
        crate::ir::Def::Const(c) => match c {
            crate::ir::Const::Int { bits, value } => format!("i{bits}({value})"),
            crate::ir::Const::Float { kind, bits } => format!("{kind:?}({bits:#x})"),
            crate::ir::Const::Null => "null".to_string(),
            crate::ir::Const::Undef => "undef".to_string(),
            crate::ir::Const::Zero => "zeroinit".to_string(),
            crate::ir::Const::Bytes(bytes) => format!("bytes({})", bytes.len()),
        },
        _ => format!("%{}", v.index()),
    }
}

pub(crate) fn inst_str(func: &Function, kind: &InstKind) -> String {
    match kind {
        InstKind::Add(a, b) => format!("add {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Sub(a, b) => format!("sub {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Mul(a, b) => format!("mul {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Udiv(a, b) => format!("udiv {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Sdiv(a, b) => format!("sdiv {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Urem(a, b) => format!("urem {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Srem(a, b) => format!("srem {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::And(a, b) => format!("and {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Or(a, b) => format!("or {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Xor(a, b) => format!("xor {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Shl(a, b) => format!("shl {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::LShr(a, b) => format!("lshr {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::AShr(a, b) => format!("ashr {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::Neg(a) => format!("neg {}", val_name(func, *a)),
        InstKind::Not(a) => format!("not {}", val_name(func, *a)),
        InstKind::FAdd(a, b) => format!("fadd {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::FSub(a, b) => format!("fsub {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::FMul(a, b) => format!("fmul {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::FDiv(a, b) => format!("fdiv {} {}", val_name(func, *a), val_name(func, *b)),
        InstKind::FNeg(a) => format!("fneg {}", val_name(func, *a)),
        InstKind::Icmp { pred, lhs, rhs } => format!("icmp.{pred:?} {} {}", val_name(func, *lhs), val_name(func, *rhs)),
        InstKind::Fcmp { pred, lhs, rhs } => format!("fcmp.{pred:?} {} {}", val_name(func, *lhs), val_name(func, *rhs)),
        InstKind::Trunc(a) => format!("trunc {}", val_name(func, *a)),
        InstKind::Zext(a) => format!("zext {}", val_name(func, *a)),
        InstKind::Sext(a) => format!("sext {}", val_name(func, *a)),
        InstKind::Bitcast(a) => format!("bitcast {}", val_name(func, *a)),
        InstKind::SItoF(a) => format!("sitof {}", val_name(func, *a)),
        InstKind::UItoF(a) => format!("uitof {}", val_name(func, *a)),
        InstKind::SFtoI(a) => format!("sftoi {}", val_name(func, *a)),
        InstKind::UFtoI(a) => format!("uftoi {}", val_name(func, *a)),
        InstKind::Fptrunc(a) => format!("fptrunc {}", val_name(func, *a)),
        InstKind::Fpext(a) => format!("fpext {}", val_name(func, *a)),
        InstKind::PtrToInt(a) => format!("ptrtoint {}", val_name(func, *a)),
        InstKind::IntToPtr(a) => format!("inttoptr {}", val_name(func, *a)),
        InstKind::Select { cond, then_val, else_val } => format!("select {} {} {}", val_name(func, *cond), val_name(func, *then_val), val_name(func, *else_val)),
        InstKind::Alloca { pointee, count } => {
            format!("alloca {} x{count}", ty_str(pointee))
        }
        InstKind::Load { addr, flags } => format!("load{} {}", if flags.volatile { ".vol" } else { "" }, val_name(func, *addr)),
        InstKind::Store { addr, val, flags } => {
            format!("store{} {} {}", if flags.volatile { ".vol" } else { "" }, val_name(func, *addr), val_name(func, *val))
        }
        InstKind::Call { callee, args } => {
            let list: Vec<_> = args.iter().map(|a| val_name(func, *a)).collect();
            format!("call f{}({})", callee.index(), list.join(", "))
        }
        InstKind::CallExtern { target, args } => {
            let list: Vec<_> = args.iter().map(|a| val_name(func, *a)).collect();
            format!("call.ext e{}({})", target.index(), list.join(", "))
        }
        InstKind::CallIndirect { callee, args, .. } => {
            let list: Vec<_> = args.iter().map(|a| val_name(func, *a)).collect();
            format!("call.ind {}({})", val_name(func, *callee), list.join(", "))
        }
        InstKind::Await { fut } => format!("await {}", val_name(func, *fut)),
        InstKind::AtomicLoad { addr } => format!("atomic.load {}", val_name(func, *addr)),
        InstKind::AtomicStore { addr, val } => format!("atomic.store {} {}", val_name(func, *addr), val_name(func, *val)),
        InstKind::AtomicRmw { op, addr, val } => format!("atomic.rmw.{op:?} {} {}", val_name(func, *addr), val_name(func, *val)),
        InstKind::AtomicCas { addr, expected, replacement } => format!("atomic.cas {} {} {}", val_name(func, *addr), val_name(func, *expected), val_name(func, *replacement)),
        InstKind::AtomicFence => "atomic.fence".to_string(),
        InstKind::InlineAsm { block } => {
            let list: Vec<_> = block
                .operands
                .iter()
                .map(|operand| format!("{}:{}", operand.constraint, val_name(func, operand.value)))
                .collect();
            format!("inline.asm [{}]", list.join(", "))
        }
        InstKind::GetFramePointer => "get.frame.pointer".to_string(),
        InstKind::GetElementPtr { base, indices } => {
            let steps: Vec<_> = indices
                .iter()
                .map(|index| match index {
                    crate::ty::GepIndex::Field(field) => format!(".{field}"),
                    crate::ty::GepIndex::Index(v) => val_name(func, *v),
                    crate::ty::GepIndex::ConstIndex(i) => format!("{i}"),
                })
                .collect();
            format!("gep {} [{}]", val_name(func, *base), steps.join(", "))
        }
        InstKind::ExtractValue { agg, index } => {
            format!("extract {} .{index}", val_name(func, *agg))
        }
        InstKind::InsertValue { agg, elem, index } => format!("insert {} {} .{index}", val_name(func, *agg), val_name(func, *elem)),
        InstKind::Splat(a) => format!("splat {}", val_name(func, *a)),
        InstKind::ExtractLane { vec, lane } => format!("extractlane {} {}", val_name(func, *vec), lane),
        InstKind::GlobalAddr { global } => format!("global g{}", global.index()),
        InstKind::Ceil(a) => format!("ceil {}", val_name(func, *a)),
        InstKind::Floor(a) => format!("floor {}", val_name(func, *a)),
        InstKind::Sqrt(a) => format!("sqrt {}", val_name(func, *a)),
        InstKind::Clz(a) => format!("clz {}", val_name(func, *a)),
        InstKind::Ctz(a) => format!("ctz {}", val_name(func, *a)),
        InstKind::Popcnt(a) => format!("popcnt {}", val_name(func, *a)),
        InstKind::Bswap(a) => format!("bswap {}", val_name(func, *a)),
        InstKind::FuncAddr { func } => format!("funcaddr f{}", func.index()),
        InstKind::ExternAddr { target } => format!("externaddr e{}", target.index()),
        InstKind::VaStart { ap } => format!("vastart {}", val_name(func, *ap)),
        InstKind::VaArg { ap } => format!("vaarg {}", val_name(func, *ap)),
        InstKind::VaEnd { ap } => format!("vaend {}", val_name(func, *ap)),
        InstKind::VaCopy { dst, src } => format!("vacopy {} {}", val_name(func, *dst), val_name(func, *src)),
        InstKind::Memset { dst, byte, len } => format!("memset {} {byte} len{len}", val_name(func, *dst)),
        InstKind::Memcpy { dst, src, len } => format!("memcpy {} {} len{len}", val_name(func, *dst), val_name(func, *src)),
        InstKind::EhTypeId { typeinfo } => format!("eh.typeid g{}", typeinfo.index()),
    }
}

/// A target block with its branch arguments: `b3` or `b3(%a, %b)`.
fn block_str(func: &Function, block: crate::ir::Block, args: &[crate::ir::Value]) -> String {
    if args.is_empty() {
        return format!("b{}", block.index());
    }
    let list: Vec<_> = args.iter().map(|a| val_name(func, *a)).collect();
    format!("b{}({})", block.index(), list.join(", "))
}

fn term_str(func: &Function, term: &Terminator) -> String {
    match term {
        Terminator::Br { dest, args } => format!("br {}", block_str(func, *dest, args)),
        Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args,
            else_args,
        } => format!(
            "brif {} {} {}",
            val_name(func, *cond),
            block_str(func, *then_block, then_args),
            block_str(func, *else_block, else_args)
        ),
        Terminator::Switch { scrutinee, cases, default } => {
            let cases: Vec<_> = cases.iter().map(|(value, block)| format!("{value}: b{}", block.index())).collect();
            format!("switch {} [{}] default b{}", val_name(func, *scrutinee), cases.join(", "), default.index())
        }
        Terminator::Return { val } => match val {
            Some(v) => format!("ret {}", val_name(func, *v)),
            None => "ret void".to_string(),
        },
        Terminator::Invoke { callee, args, sig, normal, unwind, .. } => {
            let list: Vec<_> = args.iter().map(|a| val_name(func, *a)).collect();
            let callee_str = match callee {
                crate::ir::InvokeCallee::Func(f) => format!("f{}", f.index()),
                crate::ir::InvokeCallee::Extern(e) => format!("e{}", e.index()),
                crate::ir::InvokeCallee::Indirect(v) => val_name(func, *v),
            };
            let kind = match callee {
                crate::ir::InvokeCallee::Func(_) => "invoke",
                crate::ir::InvokeCallee::Extern(_) => "invoke.ext",
                crate::ir::InvokeCallee::Indirect(_) => "invoke.ind",
            };
            format!(
                "{kind} {callee_str} sig={} ({}) to b{} unwind b{}",
                sig_str(sig),
                list.join(", "),
                normal.index(),
                unwind.index()
            )
        }
        Terminator::Resume { exn, sel } => format!("resume {} {}", val_name(func, *exn), val_name(func, *sel)),
        Terminator::Unreachable => "unreachable".to_string(),
    }
}

/// Signature in the parser's spelling: `sig=(i32, ptr(void)) -> i32`.
fn sig_str(sig: &crate::ty::Signature) -> String {
    let params: Vec<_> = sig.params.iter().map(ty_str).collect();
    format!("({}) -> {}", params.join(", "), ty_str(&sig.ret))
}

/// One source location as `path:line:column`, resolved through the
/// module's file table.
fn location_str(module: &Module, loc: Location) -> String {
    format!("{}:{}:{}", module.files[loc.file as usize].path, loc.line, loc.column)
}

/// Provenance suffix for the values a line mentions: for every operand
/// with a recorded [`crate::ir::ValueOrigin`], `val <- pass: text`
/// (plus `@ file:line:column` when the replaced definition was
/// located), joined with `, ` and preceded by `  ; `. Empty when no
/// mentioned value has an origin, so compact dumps stay unchanged.
fn provenance_suffix(func: &Function, module: &Module, values: &[crate::ir::Value]) -> String {
    let mut entries = Vec::new();
    for &v in values {
        let Some(origin) = func.origins.get(&v) else {
            continue;
        };
        let mut entry = format!("{} <- {}: {}", val_name(func, v), origin.pass, origin.text);
        if let Some(loc) = origin.loc {
            entry.push_str(&format!(" @ {}", location_str(module, loc)));
        }
        entries.push(entry);
    }
    if entries.is_empty() { String::new() } else { format!("  ; {}", entries.join(", ")) }
}

/// Render one function. With `None`, instructions print bare; with a
/// module, every located instruction carries its source position as a
/// `  ; path:line:column` suffix, and any line mentioning a
/// pass-materialized value carries its provenance after that.
fn dump_function_inner(out: &mut String, func: &Function, module: Option<&Module>) -> fmt::Result {
    let personality = func.personality.map(|e| format!(" personality=e{}", e.index())).unwrap_or_default();
    if func.inline {
        writeln!(out, "func @{} inline{personality}:", func.name)?;
    } else {
        writeln!(out, "func @{func_name}{personality}:", func_name = func.name)?;
    }
    for (id, block) in func.blocks.iter_enumerated() {
        let params: Vec<_> = block.params.iter().map(|p| format!("{}:{}", val_name(func, *p), ty_str(&func.values[*p].ty))).collect();
        let lpad = match &block.lpad {
            None => String::new(),
            Some(clauses) => {
                let list: Vec<_> = clauses
                    .iter()
                    .map(|c| match c {
                        crate::ir::LpadClause::Catch(g) => format!("catch g{}", g.index()),
                        crate::ir::LpadClause::CatchAll => "catchall".to_string(),
                        crate::ir::LpadClause::Cleanup => "cleanup".to_string(),
                    })
                    .collect();
                format!(" .lpad [{}]", list.join(", "))
            }
        };
        writeln!(out, "  b{}({}){lpad}:", id.index(), params.join(", "))?;
        for inst in &block.insts {
            let data = &func.insts[*inst];
            let suffix = match module {
                Some(module) => {
                    let loc = data.loc.map(|loc| format!("  ; {}", location_str(module, loc))).unwrap_or_default();
                    format!("{loc}{}", provenance_suffix(func, module, &data.kind.operands()))
                }
                None => String::new(),
            };
            match data.result {
                Some(r) => writeln!(out, "    %{}:{} = {}{suffix}", r.index(), ty_str(&func.values[r].ty), inst_str(func, &data.kind))?,
                None => writeln!(out, "    {}{suffix}", inst_str(func, &data.kind))?,
            }
        }
        match &block.term {
            Some(t) => {
                let suffix = match module {
                    Some(module) => provenance_suffix(func, module, &t.operands()),
                    None => String::new(),
                };
                writeln!(out, "    {}{suffix}", term_str(func, t))?;
            }
            None => writeln!(out, "    <unterminated>")?,
        }
    }
    Ok(())
}

/// Compact MIR for one function: instructions without source
/// positions.
pub fn dump_function(out: &mut String, func: &Function) -> fmt::Result {
    dump_function_inner(out, func, None)
}

/// [`dump_function`] with the source location of every located
/// instruction appended (`  ; path:line:column`): the same positions
/// DWARF line info is built from. `module` supplies the file table the
/// locations index.
pub fn dump_function_full(out: &mut String, func: &Function, module: &Module) -> fmt::Result {
    dump_function_inner(out, func, Some(module))
}

/// Compact text of every function in `module`, then one `.init @name`
/// line per registered constructor (same spelling the parser reads).
pub fn dump_module(module: &Module) -> String {
    let mut out = String::new();
    for func in module.funcs.iter() {
        dump_function(&mut out, func).unwrap();
    }
    dump_inits(module, &mut out);
    out
}

/// [`dump_module`] with per-instruction source locations: the positions
/// debug info is generated from, for inspecting debug-line quality.
/// Substantially larger than the compact dump.
pub fn dump_module_full(module: &Module) -> String {
    let mut out = String::new();
    for func in module.funcs.iter() {
        dump_function_full(&mut out, func, module).unwrap();
    }
    dump_inits(module, &mut out);
    out
}

fn dump_inits(module: &Module, out: &mut String) {
    for func in &module.init_funcs {
        writeln!(out, ".init @{}", module.funcs[*func].name).unwrap();
    }
}

#[cfg(test)]
mod tests;

