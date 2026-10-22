//! S-expression dump of checked translation units, for backend
//! debugging: every declaration, body, and expression with its type.
//! Nothing here affects checking; the golden test pins the format.

// Debug surface read only by tests; dead-code lints stay silenced.
#![allow(dead_code)]

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use string_interner::symbol::SymbolU32;

use crate::{
    check::{CheckedExpr, CheckedExprKind},
    lexer::Interner,
    parse::{AssignOp, BinaryOp, BlockItem, Designator, ForInit, FuncBody, Initializer, Stmt, TranslationUnit, Ty, UnaryOp},
    sema::{DeclId, Sema},
};

/// Dump a whole translation unit: one s-expression per external
/// declaration, bodies and initializers included.
pub fn dump_translation_unit(sema: &Sema, interner: &Interner, tu: &TranslationUnit) -> String {
    let mut out = Vec::new();
    for ed in &tu.external_declarations {
        out.push(match ed {
            crate::parse::ExternalDeclaration::Var(id) => dump_var(sema, interner, *id),
            crate::parse::ExternalDeclaration::Func(id) => dump_func(sema, interner, *id),
        });
    }
    out.join("\n")
}

fn name(interner: &Interner, sym: SymbolU32) -> String {
    interner.resolve(sym).unwrap_or("?").to_string()
}

fn dump_var(sema: &Sema, interner: &Interner, id: DeclId) -> String {
    let binding = &sema.decls[id.0 as usize];
    let var = match binding.name {
        Some(sym) => format!("{}#{}", name(interner, sym), id.0),
        None => format!("#{}", id.0),
    };
    match sema.inits.get(&id) {
        Some(init) => format!("(var {} :{} {})", var, dump_ty(&binding.ty, sema, interner), dump_init(init, sema, interner)),
        None => format!("(var {} :{})", var, dump_ty(&binding.ty, sema, interner)),
    }
}

fn dump_func(sema: &Sema, interner: &Interner, id: DeclId) -> String {
    let binding = &sema.decls[id.0 as usize];
    let func = match binding.name {
        Some(sym) => format!("{}#{}", name(interner, sym), id.0),
        None => format!("#{}", id.0),
    };
    match sema.bodies.get(&id) {
        Some(body) => format!("(func {} :{} {})", func, dump_ty(&binding.ty, sema, interner), dump_block(body, sema, interner)),
        None => format!("(func-decl {} :{})", func, dump_ty(&binding.ty, sema, interner)),
    }
}

fn dump_block(body: &FuncBody, sema: &Sema, interner: &Interner) -> String {
    let mut items = Vec::new();
    for param in &body.params {
        let binding = &sema.decls[param.0 as usize];
        let var = match binding.name {
            Some(sym) => format!("{}#{}", name(interner, sym), param.0),
            None => format!("#{}", param.0),
        };
        items.push(format!("(param {} :{})", var, dump_ty(&binding.ty, sema, interner)));
    }
    for item in &body.block.items {
        items.push(match item {
            BlockItem::Decl { ids, .. } => {
                let mut decls = Vec::new();
                for id in ids {
                    decls.push(dump_var(sema, interner, *id));
                }
                format!("(decl {})", decls.join(" "))
            }
            BlockItem::Stmt(stmt) => dump_stmt(stmt, sema, interner),
        });
    }
    format!("(block {})", items.join(" "))
}

fn dump_stmt(s: &Stmt, sema: &Sema, interner: &Interner) -> String {
    match s {
        Stmt::Compound(compound) => {
            let mut items = Vec::new();
            for item in &compound.items {
                items.push(match item {
                    BlockItem::Decl { ids, .. } => {
                        let mut decls = Vec::new();
                        for id in ids {
                            decls.push(dump_var(sema, interner, *id));
                        }
                        format!("(decl {})", decls.join(" "))
                    }
                    BlockItem::Stmt(stmt) => dump_stmt(stmt, sema, interner),
                });
            }
            format!("(block {})", items.join(" "))
        }
        Stmt::Decl { ids, .. } => {
            let decls = ids.iter().map(|id| dump_var(sema, interner, *id)).collect::<Vec<_>>();
            format!("(decl {})", decls.join(" "))
        }
        Stmt::Empty { .. } => "(empty)".to_string(),
        Stmt::Expr { expr, .. } => format!("(; {})", dump_ex(expr, sema, interner)),
        Stmt::Asm(asm) => {
            let mut out = format!("(asm {:?}", asm.template);
            for op in &asm.outputs {
                out.push_str(&format!(" (out {:?} {})", op.constraint, dump_ex(&op.expr, sema, interner)));
            }
            for op in &asm.inputs {
                out.push_str(&format!(" (in {:?} {})", op.constraint, dump_ex(&op.expr, sema, interner)));
            }
            for clobber in &asm.clobbers {
                out.push_str(&format!(" (clobber {:?})", clobber));
            }
            out.push(')');
            out
        }
        Stmt::Return { expr, .. } => match expr {
            Some(expr) => format!("(return {})", dump_ex(expr, sema, interner)),
            None => "(return)".to_string(),
        },
        Stmt::If { cond, then, els, .. } => match els {
            Some(els) => format!(
                "(if {} {} {})",
                dump_ex(cond, sema, interner),
                dump_stmt(then, sema, interner),
                dump_stmt(els, sema, interner)
            ),
            None => format!("(if {} {})", dump_ex(cond, sema, interner), dump_stmt(then, sema, interner)),
        },
        Stmt::While { cond, body, .. } => format!("(while {} {})", dump_ex(cond, sema, interner), dump_stmt(body, sema, interner)),
        Stmt::DoWhile { body, cond, .. } => format!("(do {} {})", dump_stmt(body, sema, interner), dump_ex(cond, sema, interner)),
        Stmt::For { init, cond, step, body, .. } => {
            let init = match init {
                ForInit::Decl(ids) => {
                    let mut decls = Vec::new();
                    for id in ids {
                        decls.push(dump_var(sema, interner, *id));
                    }
                    format!("(decl {})", decls.join(" "))
                }
                ForInit::Expr(expr) => dump_ex(expr, sema, interner),
                ForInit::Empty => "(empty)".to_string(),
            };
            let cond = cond.as_ref().map(|c| dump_ex(c, sema, interner)).unwrap_or_else(|| "(empty)".to_string());
            let step = step.as_ref().map(|s| dump_ex(s, sema, interner)).unwrap_or_else(|| "(empty)".to_string());
            format!("(for {} {} {} {})", init, cond, step, dump_stmt(body, sema, interner))
        }
        Stmt::Switch { cond, body, .. } => format!("(switch {} {})", dump_ex(cond, sema, interner), dump_stmt(body, sema, interner)),
        Stmt::Break { .. } => "(break)".to_string(),
        Stmt::Continue { .. } => "(continue)".to_string(),
        Stmt::Goto { label, .. } => format!("(goto {})", name(interner, *label)),
        Stmt::GotoComputed { expr, .. } => format!("(goto* {})", dump_ex(expr, sema, interner)),
        Stmt::Label { name: label, stmt, .. } => format!("(label {} {})", name(interner, *label), dump_stmt(stmt, sema, interner)),
        Stmt::Case { expr, hi, stmt, .. } => match expr {
            Some(expr) => match hi {
                Some(hi) => format!(
                    "(case {} ... {} {})",
                    dump_ex(expr, sema, interner),
                    dump_ex(hi, sema, interner),
                    dump_stmt(stmt, sema, interner)
                ),
                None => format!("(case {} {})", dump_ex(expr, sema, interner), dump_stmt(stmt, sema, interner)),
            },
            None => format!("(default {})", dump_stmt(stmt, sema, interner)),
        },
    }
}

fn dump_init(init: &Initializer, sema: &Sema, interner: &Interner) -> String {
    match init {
        Initializer::Expr(expr) => dump_ex(expr, sema, interner),
        Initializer::List { items, .. } => {
            let mut parts = Vec::new();
            for item in items {
                let mut designators = Vec::new();
                for designator in &item.designators {
                    designators.push(match designator {
                        Designator::Field { name: field, .. } => {
                            format!(".{}", name(interner, *field))
                        }
                        Designator::AnonField { index, .. } => {
                            format!(".[anon{index}]")
                        }
                        Designator::Index { index, .. } => {
                            format!("[{}]", dump_ex(index, sema, interner))
                        }
                    });
                }
                parts.push(format!("({}{})", designators.join(""), dump_init(&item.init, sema, interner)));
            }
            format!("(list {})", parts.join(" "))
        }
    }
}

fn dump_ex(e: &CheckedExpr, sema: &Sema, interner: &Interner) -> String {
    let t = dump_ty(&e.ty, sema, interner);
    match &e.kind {
        CheckedExprKind::Ident(id) => {
            let binding = &sema.decls[id.0 as usize];
            match binding.name {
                Some(sym) => format!("(ident {}#{} :{})", name(interner, sym), id.0, t),
                None => format!("(ident #{} :{})", id.0, t),
            }
        }
        CheckedExprKind::Int(lit) => format!("(int {} :{})", interner.resolve(lit.spelling).unwrap_or("?"), t),
        CheckedExprKind::Float(lit) => format!("(float {} :{})", interner.resolve(lit.spelling).unwrap_or("?"), t),
        CheckedExprKind::Char(lit) => format!("(char {} :{})", interner.resolve(lit.spelling).unwrap_or("?"), t),
        CheckedExprKind::Str(parts) => {
            let mut literals = Vec::new();
            for part in parts {
                // Spellings keep their quotes and prefix.
                literals.push(interner.resolve(part.spelling).unwrap_or("?").to_string());
            }
            format!("(str {} :{})", literals.join(" "), t)
        }
        CheckedExprKind::Unary(op, operand) => format!("({} {} :{})", unary_op(*op), dump_ex(operand, sema, interner), t),
        CheckedExprKind::Binary(op, lhs, rhs) => format!("({} {} {} :{})", binary_op(*op), dump_ex(lhs, sema, interner), dump_ex(rhs, sema, interner), t),
        CheckedExprKind::Assign(op, lhs, rhs) => format!("({}= {} {} :{})", assign_op(*op), dump_ex(lhs, sema, interner), dump_ex(rhs, sema, interner), t),
        CheckedExprKind::CompoundAssign { op, lhs, rhs, .. } => format!("({}= {} {} :{})", assign_op(*op), dump_ex(lhs, sema, interner), dump_ex(rhs, sema, interner), t),
        CheckedExprKind::Ternary { cond, then, els } => format!(
            "(?: {} {} {} :{})",
            dump_ex(cond, sema, interner),
            dump_ex(then, sema, interner),
            dump_ex(els, sema, interner),
            t
        ),
        CheckedExprKind::Elvis { cond, els } => format!("(?: {} {} :{})", dump_ex(cond, sema, interner), dump_ex(els, sema, interner), t),
        CheckedExprKind::Call { func, args } => {
            let mut parts = vec![dump_ex(func, sema, interner)];
            for arg in args {
                parts.push(dump_ex(arg, sema, interner));
            }
            format!("(call {} :{})", parts.join(" "), t)
        }
        CheckedExprKind::Await(inner) => {
            format!("(await {} :{})", dump_ex(inner, sema, interner), t)
        }
        CheckedExprKind::Atomic { op, args } => {
            let mut parts = Vec::new();
            for arg in args {
                parts.push(dump_ex(arg, sema, interner));
            }
            format!("(atomic-{:?} {} :{})", op, parts.join(" "), t)
        }
        CheckedExprKind::Overflow { op, lhs, rhs, res, res_ty } => format!(
            "({:?}-overflow {} {} {} :{})",
            op,
            dump_ex(lhs, sema, interner),
            dump_ex(rhs, sema, interner),
            dump_ex(res, sema, interner),
            dump_ty(res_ty, sema, interner),
        ),
        CheckedExprKind::Alloca { size } => {
            format!("(alloca {} :{})", dump_ex(size, sema, interner), t)
        }
        CheckedExprKind::FrameAddress { level, returns_address } => {
            format!("({} {} :{})", if *returns_address { "return-address" } else { "frame-address" }, level, t)
        }
        CheckedExprKind::Clz { ctz, arg, fallback } => {
            let mut parts = vec![dump_ex(arg, sema, interner)];
            if let Some(fb) = fallback {
                parts.push(dump_ex(fb, sema, interner));
            }
            format!("({} {} :{})", if *ctz { "ctz" } else { "clz" }, parts.join(" "), t)
        }
        CheckedExprKind::ConstantP(b) => format!("(constant-p {} :{})", i32::from(*b), t),
        CheckedExprKind::ClassifyType(code) => format!("(classify-type {} :{})", code, t),
        CheckedExprKind::FpClass { op, arg } => {
            let what = match op {
                crate::check::FpPredicate::IsNan => "isnan",
                crate::check::FpPredicate::IsInf => "isinf",
                crate::check::FpPredicate::IsInfSign => "isinf-sign",
                crate::check::FpPredicate::IsFinite => "isfinite",
                crate::check::FpPredicate::IsNormal => "isnormal",
                crate::check::FpPredicate::SignBit => "signbit",
            };
            format!("({what} {} :{})", dump_ex(arg, sema, interner), t)
        }
        CheckedExprKind::FpClassify { codes, arg } => {
            let mut parts = vec![dump_ex(arg, sema, interner)];
            parts.extend(codes.iter().map(|code| dump_ex(code, sema, interner)));
            format!("(fpclassify {} :{})", parts.join(" "), t)
        }
        CheckedExprKind::Subscript { base, index } => format!("(sub {} {} :{})", dump_ex(base, sema, interner), dump_ex(index, sema, interner), t),
        CheckedExprKind::Member { base, path, arrow, .. } => {
            let mut names = Vec::new();
            let mut id = match &base.ty {
                Ty::Pointer { pointee, .. } if *arrow => match pointee.unqualified() {
                    Ty::Struct(id) => *id,
                    _ => {
                        names.push("?".to_string());
                        return format!("(member {} {:?} :{})", dump_ex(base, sema, interner), path, t);
                    }
                },
                _ => match base.ty.unqualified() {
                    Ty::Struct(id) => *id,
                    _ => {
                        return format!("(member {} {:?} :{})", dump_ex(base, sema, interner), path, t);
                    }
                },
            };
            for &index in path {
                let def = &sema.structs[id.0 as usize];
                match def.members.get(index).and_then(|m| m.name) {
                    Some(sym) => names.push(name(interner, sym)),
                    None => names.push(format!("#{index}")),
                }
                let member = &def.members[index];
                if let Ty::Struct(next) = member.ty.unqualified() {
                    id = *next;
                }
            }
            format!("(member {} {} :{})", dump_ex(base, sema, interner), names.join("."), t)
        }
        CheckedExprKind::Cast { ty, expr } => format!("(cast {} {} :{})", dump_ty(ty, sema, interner), dump_ex(expr, sema, interner), t),
        CheckedExprKind::Generic { index, assocs, .. } => {
            let mut parts = Vec::new();
            for (ty, expr) in assocs {
                let label = match ty {
                    Some(ty) => dump_ty(ty, sema, interner),
                    None => "default".to_string(),
                };
                parts.push(format!("({} {})", label, dump_ex(expr, sema, interner)));
            }
            format!("(generic [{}] {} :{})", index, parts.join(" "), t)
        }
        CheckedExprKind::VaArg { ap, ty } => format!("(va-arg {} {} :{})", dump_ex(ap, sema, interner), dump_ty(ty, sema, interner), t),
        CheckedExprKind::StmtExpr { items } => {
            let mut parts = Vec::new();
            for item in items {
                parts.push(match item {
                    BlockItem::Decl { ids, .. } => {
                        let mut decls = Vec::new();
                        for id in ids {
                            decls.push(dump_var(sema, interner, *id));
                        }
                        format!("(decl {})", decls.join(" "))
                    }
                    BlockItem::Stmt(stmt) => dump_stmt(stmt, sema, interner),
                });
            }
            format!("(stmt-expr {} :{})", parts.join(" "), t)
        }
        CheckedExprKind::Offsetof { ty, offset } => {
            format!("(offsetof {} {} :{})", dump_ty(ty, sema, interner), offset, t)
        }
        CheckedExprKind::CompoundLiteral { ty, init, .. } => format!("(compound {} {} :{})", dump_ty(ty, sema, interner), dump_init(init, sema, interner), t),
        CheckedExprKind::Convert { expr, ty } => format!("(convert {} {} :{})", dump_ty(ty, sema, interner), dump_ex(expr, sema, interner), t),
        CheckedExprKind::SizeofExpr(expr) => {
            format!("(sizeof {} :{})", dump_ex(expr, sema, interner), t)
        }
        CheckedExprKind::SizeofTy(ty) => {
            format!("(sizeof {} :{})", dump_ty(ty, sema, interner), t)
        }
        CheckedExprKind::AlignofExpr(expr) => {
            format!("(alignof {} :{})", dump_ex(expr, sema, interner), t)
        }
        CheckedExprKind::AlignofTy(ty) => {
            format!("(alignof {} :{})", dump_ty(ty, sema, interner), t)
        }
        CheckedExprKind::LabelAddr(label) => format!("(labeladdr {} :{})", name(interner, *label), t),
    }
}

fn unary_op(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Pos => "pos",
        UnaryOp::Neg => "neg",
        UnaryOp::Not => "not",
        UnaryOp::BitNot => "bitnot",
        UnaryOp::Deref => "deref",
        UnaryOp::AddrOf => "addrof",
        UnaryOp::PreInc => "preinc",
        UnaryOp::PreDec => "predec",
        UnaryOp::PostInc => "postinc",
        UnaryOp::PostDec => "postdec",
        UnaryOp::Real => "real",
        UnaryOp::Imag => "imag",
    }
}

fn binary_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Rem => "%",
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::Eq => "==",
        BinaryOp::Ne => "!=",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitXor => "^",
        BinaryOp::BitOr => "|",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
        BinaryOp::Comma => ",",
    }
}

fn assign_op(op: AssignOp) -> &'static str {
    match op {
        AssignOp::Assign => "=",
        AssignOp::Add => "+",
        AssignOp::Sub => "-",
        AssignOp::Mul => "*",
        AssignOp::Div => "/",
        AssignOp::Rem => "%",
        AssignOp::Shl => "<<",
        AssignOp::Shr => ">>",
        AssignOp::BitAnd => "&",
        AssignOp::BitXor => "^",
        AssignOp::BitOr => "|",
    }
}

/// Short type rendering for dumps: `int`, `*const(char)`,
/// `struct S`, `int[3]`, `fn(int, ...) -> int`.
pub fn dump_ty(t: &Ty, sema: &Sema, interner: &Interner) -> String {
    match t {
        Ty::Void => "void".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Char => "char".to_string(),
        Ty::SignedChar => "signed char".to_string(),
        Ty::UnsignedChar => "unsigned char".to_string(),
        Ty::Short => "short".to_string(),
        Ty::UnsignedShort => "unsigned short".to_string(),
        Ty::Int => "int".to_string(),
        Ty::UnsignedInt => "unsigned int".to_string(),
        Ty::Long => "long".to_string(),
        Ty::UnsignedLong => "unsigned long".to_string(),
        Ty::LongLong => "long long".to_string(),
        Ty::UnsignedLongLong => "unsigned long long".to_string(),
        Ty::Int128 => "__int128".to_string(),
        Ty::UnsignedInt128 => "unsigned __int128".to_string(),
        Ty::Float => "float".to_string(),
        Ty::Double => "double".to_string(),
        Ty::LongDouble => "long double".to_string(),
        Ty::Complex(inner) => format!("{} _Complex", dump_ty(inner, sema, interner)),
        Ty::Pointer { qualifiers, size, pointee } => {
            let size = match size {
                crate::parse::PtrSize::Default => "",
                crate::parse::PtrSize::Ptr32 => " __ptr32",
                crate::parse::PtrSize::Ptr64 => " __ptr64",
            };
            format!("*{}{}{}", size, quals(qualifiers), dump_ty(pointee, sema, interner))
        }
        Ty::Func {
            ret,
            params,
            variadic,
            prototyped,
        } => {
            let mut parts: Vec<String> = params.iter().map(|p| dump_ty(p, sema, interner)).collect();
            if *variadic {
                parts.push("...".to_string());
            }
            if !prototyped {
                parts.push("?".to_string());
            }
            format!("fn({}) -> {}", parts.join(", "), dump_ty(ret, sema, interner))
        }
        Ty::Array { elem, quals: array_quals, bound } => {
            let bound = match bound {
                None => "".to_string(),
                Some(crate::parse::ArrayBound::Star) => "*".to_string(),
                Some(crate::parse::ArrayBound::Fixed(n)) => n.to_string(),
                Some(crate::parse::ArrayBound::Vla(_)) => "vla".to_string(),
            };
            format!("{}{}[{}]", quals(array_quals), dump_ty(elem, sema, interner), bound)
        }
        Ty::Struct(id) => match sema.structs.get(id.0 as usize).and_then(|d| d.tag) {
            Some(tag) => format!("struct {}", name(interner, tag)),
            None => format!("struct#{}", id.0),
        },
        Ty::Enum(id) => match sema.enums.get(id.0 as usize).and_then(|d| d.tag) {
            Some(tag) => format!("enum {}", name(interner, tag)),
            None => format!("enum#{}", id.0),
        },
        Ty::BuiltinVaList => "va_list".to_string(),
        Ty::Float128 => "__float128".to_string(),
        Ty::Qualified { quals: q, inner } => {
            format!("{}({})", quals(q).trim_end(), dump_ty(inner, sema, interner))
        }
    }
}

fn quals(quals: &[crate::parse::TypeQualifier]) -> String {
    let mut out = String::new();
    for q in quals {
        out.push_str(match q {
            crate::parse::TypeQualifier::Const => "const ",
            crate::parse::TypeQualifier::Restrict => "restrict ",
            crate::parse::TypeQualifier::Volatile => "volatile ",
        });
    }
    out
}
