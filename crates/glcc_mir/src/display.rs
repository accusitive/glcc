use std::fmt::{self, Write};

use crate::debug::Location;
use crate::ir::{Function, InstKind, Module, Terminator};
use crate::ty::Ty;

fn ty_str(ty: &Ty) -> String {
    match ty {
        Ty::Void => "void".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Int { bits } => format!("i{bits}"),
        Ty::Float(k) => format!("{k:?}").to_lowercase(),
        Ty::Ptr(inner) => format!("ptr({})", ty_str(inner)),
        Ty::BuiltinVaList { .. } => "va_list".to_string(),
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

fn inst_str(func: &Function, kind: &InstKind) -> String {
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
        InstKind::Icmp { pred, lhs, rhs } => format!(
            "icmp.{pred:?} {} {}",
            val_name(func, *lhs),
            val_name(func, *rhs)
        ),
        InstKind::Fcmp { pred, lhs, rhs } => format!(
            "fcmp.{pred:?} {} {}",
            val_name(func, *lhs),
            val_name(func, *rhs)
        ),
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
        InstKind::Select {
            cond,
            then_val,
            else_val,
        } => format!(
            "select {} {} {}",
            val_name(func, *cond),
            val_name(func, *then_val),
            val_name(func, *else_val)
        ),
        InstKind::Alloca { pointee, count } => {
            format!("alloca {} x{count}", ty_str(pointee))
        }
        InstKind::Load { addr, flags } => format!(
            "load{} {}",
            if flags.volatile { ".vol" } else { "" },
            val_name(func, *addr)
        ),
        InstKind::Store { addr, val, flags } => {
            format!(
                "store{} {} {}",
                if flags.volatile { ".vol" } else { "" },
                val_name(func, *addr),
                val_name(func, *val)
            )
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
        InstKind::AtomicStore { addr, val } => format!(
            "atomic.store {} {}",
            val_name(func, *addr),
            val_name(func, *val)
        ),
        InstKind::AtomicRmw { op, addr, val } => format!(
            "atomic.rmw.{op:?} {} {}",
            val_name(func, *addr),
            val_name(func, *val)
        ),
        InstKind::AtomicCas {
            addr,
            expected,
            replacement,
        } => format!(
            "atomic.cas {} {} {}",
            val_name(func, *addr),
            val_name(func, *expected),
            val_name(func, *replacement)
        ),
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
        InstKind::InsertValue { agg, elem, index } => format!(
            "insert {} {} .{index}",
            val_name(func, *agg),
            val_name(func, *elem)
        ),
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
        InstKind::VaCopy { dst, src } => format!(
            "vacopy {} {}",
            val_name(func, *dst),
            val_name(func, *src)
        ),
        InstKind::Memset { dst, byte, len } => format!(
            "memset {} {byte} len{len}",
            val_name(func, *dst)
        ),
        InstKind::Memcpy { dst, src, len } => format!(
            "memcpy {} {} len{len}",
            val_name(func, *dst),
            val_name(func, *src)
        ),
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
        Terminator::Switch {
            scrutinee,
            cases,
            default,
        } => {
            let cases: Vec<_> = cases
                .iter()
                .map(|(value, block)| format!("{value}: b{}", block.index()))
                .collect();
            format!(
                "switch {} [{}] default b{}",
                val_name(func, *scrutinee),
                cases.join(", "),
                default.index()
            )
        }
        Terminator::Return { val } => match val {
            Some(v) => format!("ret {}", val_name(func, *v)),
            None => "ret void".to_string(),
        },
        Terminator::Unreachable => "unreachable".to_string(),
    }
}

/// One source location as `path:line:column`, resolved through the
/// module's file table.
fn location_str(module: &Module, loc: Location) -> String {
    format!(
        "{}:{}:{}",
        module.files[loc.file as usize].path, loc.line, loc.column
    )
}

/// Render one function. With `None`, instructions print bare; with a
/// module, every located instruction carries its source position as a
/// `  ; path:line:column` suffix.
fn dump_function_inner(out: &mut String, func: &Function, module: Option<&Module>) -> fmt::Result {
    if func.inline {
        writeln!(out, "func @{} inline:", func.name)?;
    } else {
        writeln!(out, "func @{}:", func.name)?;
    }
    for (id, block) in func.blocks.iter_enumerated() {
        let params: Vec<_> = block
            .params
            .iter()
            .map(|p| format!("{}:{}", val_name(func, *p), ty_str(&func.values[*p].ty)))
            .collect();
        writeln!(out, "  b{}({}):", id.index(), params.join(", "))?;
        for inst in &block.insts {
            let data = &func.insts[*inst];
            let loc = match (module, data.loc) {
                (Some(module), Some(loc)) => format!("  ; {}", location_str(module, loc)),
                _ => String::new(),
            };
            match data.result {
                Some(r) => writeln!(
                    out,
                    "    %{}:{} = {}{loc}",
                    r.index(),
                    ty_str(&func.values[r].ty),
                    inst_str(func, &data.kind)
                )?,
                None => writeln!(out, "    {}{loc}", inst_str(func, &data.kind))?,
            }
        }
        match &block.term {
            Some(t) => writeln!(out, "    {}", term_str(func, t))?,
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

/// Compact text of every function in `module`.
pub fn dump_module(module: &Module) -> String {
    let mut out = String::new();
    for func in module.funcs.iter() {
        dump_function(&mut out, func).unwrap();
    }
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::File;
    use crate::ir::{Block, Const, Def, Inst, InstData, Value, ValueData};
    use crate::ty::{GepIndex, Signature};

    fn param(func: &mut Function, block: Block, ty: Ty) -> Value {
        let index = func.blocks[block].params.len() as u32;
        let value = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(value);
        value
    }

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let inst = Inst::from_usize(func.insts.len());
        let value = func.values.push(ValueData {
            ty,
            def: Def::Inst(inst),
        });
        func.insts.push(InstData {
            block,
            kind,
            result: Some(value),
            loc: None,
        });
        func.blocks[block].insts.push(inst);
        value
    }

    #[test]
    fn full_dump_annotates_located_instructions() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut f, entry) = Function::new("t", sig);
        let located = emit(
            &mut f,
            entry,
            InstKind::Alloca {
                pointee: Ty::int(32),
                count: 1,
            },
            Ty::ptr(Ty::int(32)),
        );
        if let Def::Inst(inst) = &f.values[located].def {
            f.insts[*inst].loc = Some(Location {
                file: 0,
                line: 4,
                column: 7,
            });
        } else {
            panic!("emit produces an instruction");
        }
        let bare = emit(&mut f, entry, InstKind::GetFramePointer, Ty::ptr(Ty::Void));
        if let Def::Inst(inst) = &f.values[bare].def {
            assert_eq!(f.insts[*inst].loc, None);
        } else {
            panic!("emit produces an instruction");
        }
        f.blocks[entry].term = Some(Terminator::Return { val: None });

        let mut module = Module::default();
        module.files.push(File {
            path: "t.c".to_string(),
        });
        module.push_func(f);

        let plain = dump_module(&module);
        assert!(!plain.contains("t.c"), "{plain}");

        let full = dump_module_full(&module);
        let located_line = full
            .lines()
            .find(|l| l.contains("alloca"))
            .expect("alloca line");
        assert!(located_line.ends_with("; t.c:4:7"), "{located_line}");
        let bare_line = full
            .lines()
            .find(|l| l.contains("get.frame.pointer"))
            .expect("frame pointer line");
        assert!(!bare_line.contains(';'), "{bare_line}");
    }

    #[test]
    fn branches_print_targets_and_arguments() {
        let sig = Signature::new(vec![], Ty::Void);
        let (mut f, entry) = Function::new("t", sig);
        let scrut = param(&mut f, entry, Ty::int(32));
        let b1 = f.push_block(vec![]);
        let b2 = f.push_block(vec![]);
        let b3 = f.push_block(vec![]);
        let b4 = f.push_block(vec![Ty::ptr(Ty::int(32))]);
        let b5 = f.push_block(vec![]);
        f.blocks[entry].term = Some(Terminator::Switch {
            scrutinee: scrut,
            cases: vec![(0, b1), (1, b2)],
            default: b3,
        });
        let base = emit(
            &mut f,
            b1,
            InstKind::Alloca {
                pointee: Ty::int(32),
                count: 1,
            },
            Ty::ptr(Ty::int(32)),
        );
        let idx = f.push_const(Const::Int { bits: 64, value: 5 });
        let gep = emit(
            &mut f,
            b1,
            InstKind::GetElementPtr {
                base,
                indices: vec![
                    GepIndex::Field(1),
                    GepIndex::ConstIndex(3),
                    GepIndex::Index(idx),
                ]
                .into(),
            },
            Ty::ptr(Ty::int(32)),
        );
        f.blocks[b1].term = Some(Terminator::Br {
            dest: b4,
            args: [gep].into_iter().collect(),
        });
        f.blocks[b2].term = Some(Terminator::Br {
            dest: b4,
            args: [idx].into_iter().collect(),
        });
        let p = f.blocks[b4].params[0];
        let cond = f.push_const(Const::Int { bits: 1, value: 0 });
        f.blocks[b4].term = Some(Terminator::CondBr {
            cond,
            then_block: b5,
            else_block: b3,
            then_args: Default::default(),
            else_args: [p].into_iter().collect(),
        });
        f.blocks[b5].term = Some(Terminator::Return { val: None });
        f.blocks[b3].term = Some(Terminator::Return { val: None });

        let mut text = String::new();
        dump_function(&mut text, &f).unwrap();
        let gep_line = format!(
            "gep {} [.1, 3, {}]",
            val_name(&f, base),
            val_name(&f, idx)
        );
        assert!(text.contains(&gep_line), "{text}");
        assert!(
            text.contains(&format!("br b4({})", val_name(&f, gep))),
            "{text}"
        );
        assert!(
            text.contains(&format!("br b4({})", val_name(&f, idx))),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "switch {} [0: b1, 1: b2] default b3",
                val_name(&f, scrut)
            )),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "brif {} b5 b3({})",
                val_name(&f, cond),
                val_name(&f, p)
            )),
            "{text}"
        );
    }
}
