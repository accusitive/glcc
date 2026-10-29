//! Expression AST and evaluator, modeled on nasm's `eval.c` semantics:
//! a value is a constant, or (section, offset) with a constant addend.

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    /// Symbol reference; `$` and `$$` are `Here`/`SectStart`.
    Sym(String),
    Here,
    SectStart,
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Shl(Box<Expr>, Box<Expr>),
    Shr(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    BitNot(Box<Expr>),
}

/// The fully-resolved shape of an expression value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RValue {
    Const(i64),
    /// Offset into a section of this module (a local or global label).
    SectVal { sect: usize, off: i64 },
    /// Reference to an external symbol plus addend; needs a relocation.
    ExternRef { sym: usize, off: i64 },
}

/// What symbol lookup feeds the evaluator.
pub trait SymLookup {
    /// None: symbol is not known at all. `Symbol::defined == false` covers
    /// forward references during early passes.
    fn lookup(&self, name: &str) -> Option<SymbolVal>;
}

#[derive(Clone, Copy, Debug)]
pub struct SymbolVal {
    /// Index of this symbol in the module's symbol table.
    pub index: usize,
    pub defined: bool,
    pub extern_: bool,
    /// Index into the module's section table; None for absolute symbols.
    pub sect: Option<usize>,
    pub value: i64,
}

pub struct EvalCtx<'a> {
    pub cur_sect: usize,
    pub cur_off: i64,
    pub syms: &'a dyn SymLookup,
    /// During early passes forward references are unknown; treat them as
    /// zero so sizes can be estimated, and let the pass loop re-encode
    /// once real values are known.
    pub optimistic: bool,
}

impl Expr {
    pub fn eval(&self, ctx: &EvalCtx) -> Result<RValue, String> {
        match self {
            Expr::Int(i) => Ok(RValue::Const(*i)),
            Expr::Here => Ok(RValue::SectVal { sect: ctx.cur_sect, off: ctx.cur_off }),
            Expr::SectStart => Ok(RValue::SectVal { sect: ctx.cur_sect, off: 0 }),
            Expr::Sym(name) => {
                let sym = ctx
                    .syms
                    .lookup(name)
                    .ok_or_else(|| format!("undefined symbol `{name}'"))?;
                if sym.extern_ {
                    Ok(RValue::ExternRef { sym: sym.index, off: sym.value })
                } else if sym.defined {
                    match sym.sect {
                        Some(sect) => Ok(RValue::SectVal { sect, off: sym.value }),
                        None => Ok(RValue::Const(sym.value)),
                    }
                } else if ctx.optimistic {
                    // Forward reference: fall back to the previous pass's
                    // value as an estimate; the pass loop re-encodes until
                    // estimates and real values agree.
                    match sym.sect {
                        Some(sect) => Ok(RValue::SectVal { sect, off: sym.value }),
                        None => Ok(RValue::Const(sym.value)),
                    }
                } else {
                    Err(format!("undefined symbol `{name}'"))
                }
            }
            Expr::Add(a, b) => {
                let (a, b) = (a.eval(ctx)?, b.eval(ctx)?);
                match (a, b) {
                    (RValue::Const(x), y) | (y, RValue::Const(x)) => Ok(add_const(y, x)),
                    _ => Err("cannot add two relocatable values".into()),
                }
            }
            Expr::Sub(a, b) => {
                let (a, b) = (a.eval(ctx)?, b.eval(ctx)?);
                match (a, b) {
                    (RValue::Const(x), RValue::Const(y)) => Ok(RValue::Const(x - y)),
                    (a, RValue::Const(y)) => Ok(add_const(a, -y)),
                    (
                        RValue::SectVal { sect: s1, off: o1 },
                        RValue::SectVal { sect: s2, off: o2 },
                    ) if s1 == s2 => Ok(RValue::Const(o1 - o2)),
                    (RValue::Const(x), RValue::SectVal { sect, off }) => {
                        Ok(RValue::SectVal { sect, off: x - off })
                    }
                    (RValue::Const(x), RValue::ExternRef { sym, off }) => {
                        Ok(RValue::ExternRef { sym, off: x - off })
                    }
                    _ => Err("cannot subtract these values".into()),
                }
            }
            Expr::Mul(a, b) => binary(ctx, a, b, |x, y| x * y),
            Expr::Div(a, b) => binary(ctx, a, b, |x, y| x / y),
            Expr::Shl(a, b) => binary(ctx, a, b, |x, y| x << y),
            Expr::Shr(a, b) => binary(ctx, a, b, |x, y| x >> y),
            Expr::And(a, b) => binary(ctx, a, b, |x, y| x & y),
            Expr::Or(a, b) => binary(ctx, a, b, |x, y| x | y),
            Expr::Xor(a, b) => binary(ctx, a, b, |x, y| x ^ y),
            Expr::Neg(a) => match a.eval(ctx)? {
                RValue::Const(x) => Ok(RValue::Const(-x)),
                RValue::SectVal { sect, off } => Ok(RValue::SectVal { sect, off: -off }),
                RValue::ExternRef { sym, off } => Ok(RValue::ExternRef { sym, off: -off }),
            },
            Expr::BitNot(a) => match a.eval(ctx)? {
                RValue::Const(x) => Ok(RValue::Const(!x)),
                _ => Err("cannot complement a relocatable value".into()),
            },
        }
    }
}

impl RValue {
    pub fn as_const(&self) -> Option<i64> {
        match self {
            RValue::Const(c) => Some(*c),
            _ => None,
        }
    }

    pub fn is_const(&self) -> bool {
        self.as_const().is_some()
    }
}

fn add_const(v: RValue, c: i64) -> RValue {
    match v {
        RValue::Const(x) => RValue::Const(x + c),
        RValue::SectVal { sect, off } => RValue::SectVal { sect, off: off + c },
        RValue::ExternRef { sym, off } => RValue::ExternRef { sym, off: off + c },
    }
}

fn binary(
    ctx: &EvalCtx,
    a: &Expr,
    b: &Expr,
    f: impl Fn(i64, i64) -> i64,
) -> Result<RValue, String> {
    match (a.eval(ctx)?, b.eval(ctx)?) {
        (RValue::Const(x), RValue::Const(y)) => Ok(RValue::Const(f(x, y))),
        _ => Err("operation needs constant operands".into()),
    }
}
