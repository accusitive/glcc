//! Global common-subexpression elimination.
//!
//! Walks the dominator tree with a scoped value-number table: an
//! expression is reused when an identical one was computed in a
//! dominating block, or earlier in the same block. Only instructions
//! whose result is entirely fixed by their operands participate —
//! memory reads, calls, allocas, atomics, and the `Va*` state machine
//! never merge, and neither do side-effecting ops.
//!
//! Commutative arithmetic and equality comparisons are keyed with
//! their operands canonically ordered, so `a + b` and `b + a` share a
//! value number.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::ir::{Block, Extern, FcmpPred, Func, Function, Global, IcmpPred, Inst, InstKind, Value};
use crate::pass::FunctionPass;
use crate::ty::{GepIndex, Ty};

use super::DomInfo;

/// Common-subexpression elimination over the dominator tree.
#[derive(Default)]
pub struct Cse;

impl FunctionPass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&mut self, func: &mut Function) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let reachable = func.reachable_blocks();
        let dom = DomInfo::compute(func, &reachable);
        let mut table = Table::default();
        // Rewrites are collected and applied in one sweep at the end:
        // rewriting per hit is O(function size) and makes CSE
        // quadratic in the largest functions.
        let mut redirect: std::collections::HashMap<Value, Value> = std::collections::HashMap::new();
        let mut changed = false;
        let mut stack = vec![Step::Enter(func.entry)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(block) => {
                    let mark = table.mark();
                    // LIFO: the children run before the scope closes.
                    stack.push(Step::Exit(mark));
                    let old = std::mem::take(&mut func.blocks[block].insts);
                    let mut kept = Vec::with_capacity(old.len());
                    for inst in old {
                        let Some(result) = func.insts[inst].result else {
                            kept.push(inst);
                            continue;
                        };
                        let Some(key) = key(func, inst, &redirect) else {
                            kept.push(inst);
                            continue;
                        };
                        match table.get(&key).copied() {
                            Some(canon) if canon != result => {
                                redirect.insert(result, canon);
                                changed = true;
                            }
                            _ => {
                                table.insert(key, result);
                                kept.push(inst);
                            }
                        }
                    }
                    func.blocks[block].insts = kept;
                    for child in dom.children[block.index()].iter().rev() {
                        stack.push(Step::Enter(*child));
                    }
                }
                Step::Exit(mark) => table.rollback(mark),
            }
        }
        if !redirect.is_empty() {
            func.remap_operands(&redirect);
        }
        changed
    }
}

enum Step {
    Enter(Block),
    Exit(usize),
}

/// Value-number table with an undo log, so a dominator-tree walk can
/// open a scope per block and drop it again in O(insertions).
#[derive(Default)]
struct Table {
    map: HashMap<Key, Value>,
    undo: Vec<(Key, Option<Value>)>,
}

impl Table {
    fn mark(&self) -> usize {
        self.undo.len()
    }

    fn get(&self, key: &Key) -> Option<&Value> {
        self.map.get(key)
    }

    fn insert(&mut self, key: Key, value: Value) {
        let prev = self.map.insert(key.clone(), value);
        self.undo.push((key, prev));
    }

    fn rollback(&mut self, mark: usize) {
        while self.undo.len() > mark {
            let (key, prev) = self.undo.pop().expect("undo entry");
            match prev {
                Some(value) => {
                    self.map.insert(key, value);
                }
                None => {
                    self.map.remove(&key);
                }
            }
        }
    }
}

/// Expression identity: the result type plus everything the result
/// depends on.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    ty: Ty,
    expr: ExprKey,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    Bin(BinOp, Value, Value),
    Un(UnOp, Value),
    Cmp(CmpOp, Value, Value),
    Select { cond: Value, then_val: Value, else_val: Value },
    Gep { base: Value, indices: SmallVec<[GepIndex; 4]> },
    ExtractValue { agg: Value, index: u32 },
    InsertValue { agg: Value, elem: Value, index: u32 },
    FuncAddr(Func),
    ExternAddr(Extern),
    GlobalAddr(Global),
    GetFramePointer,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Udiv,
    Sdiv,
    Urem,
    Srem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    FAdd,
    FSub,
    FMul,
    FDiv,
}

impl BinOp {
    fn commutative(self) -> bool {
        matches!(self, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor | BinOp::FAdd | BinOp::FMul)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum UnOp {
    Neg,
    Not,
    FNeg,
    Trunc,
    Zext,
    Sext,
    Bitcast,
    SItoF,
    UItoF,
    SFtoI,
    UFtoI,
    Fptrunc,
    Fpext,
    PtrToInt,
    IntToPtr,
    Ceil,
    Floor,
    Sqrt,
    Clz,
    Ctz,
    Popcnt,
    Bswap,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CmpOp {
    Int(IcmpPred),
    Float(FcmpPred),
}

/// The value number of `inst`'s result, when the result is fully
/// determined by the operands. Operands already CSE'd in this run are
/// resolved through `redirect`, so a later expression matches the same
/// key it would have matched if the rewrite had been applied eagerly.
fn key(func: &Function, inst: Inst, redirect: &std::collections::HashMap<Value, Value>) -> Option<Key> {
    let r = |v: Value| redirect.get(&v).copied().unwrap_or(v);
    let result = func.insts[inst].result?;
    let ty = func.values[result].ty.clone();
    let expr = match &func.insts[inst].kind {
        InstKind::Add(a, b) => bin(BinOp::Add, r(*a), r(*b)),
        InstKind::Sub(a, b) => bin(BinOp::Sub, r(*a), r(*b)),
        InstKind::Mul(a, b) => bin(BinOp::Mul, r(*a), r(*b)),
        InstKind::Udiv(a, b) => bin(BinOp::Udiv, r(*a), r(*b)),
        InstKind::Sdiv(a, b) => bin(BinOp::Sdiv, r(*a), r(*b)),
        InstKind::Urem(a, b) => bin(BinOp::Urem, r(*a), r(*b)),
        InstKind::Srem(a, b) => bin(BinOp::Srem, r(*a), r(*b)),
        InstKind::And(a, b) => bin(BinOp::And, r(*a), r(*b)),
        InstKind::Or(a, b) => bin(BinOp::Or, r(*a), r(*b)),
        InstKind::Xor(a, b) => bin(BinOp::Xor, r(*a), r(*b)),
        InstKind::Shl(a, b) => bin(BinOp::Shl, r(*a), r(*b)),
        InstKind::LShr(a, b) => bin(BinOp::LShr, r(*a), r(*b)),
        InstKind::AShr(a, b) => bin(BinOp::AShr, r(*a), r(*b)),
        InstKind::FAdd(a, b) => bin(BinOp::FAdd, r(*a), r(*b)),
        InstKind::FSub(a, b) => bin(BinOp::FSub, r(*a), r(*b)),
        InstKind::FMul(a, b) => bin(BinOp::FMul, r(*a), r(*b)),
        InstKind::FDiv(a, b) => bin(BinOp::FDiv, r(*a), r(*b)),
        InstKind::Neg(a) => ExprKey::Un(UnOp::Neg, r(*a)),
        InstKind::Not(a) => ExprKey::Un(UnOp::Not, r(*a)),
        InstKind::FNeg(a) => ExprKey::Un(UnOp::FNeg, r(*a)),
        InstKind::Trunc(a) => ExprKey::Un(UnOp::Trunc, r(*a)),
        InstKind::Zext(a) => ExprKey::Un(UnOp::Zext, r(*a)),
        InstKind::Sext(a) => ExprKey::Un(UnOp::Sext, r(*a)),
        InstKind::Bitcast(a) => ExprKey::Un(UnOp::Bitcast, r(*a)),
        InstKind::SItoF(a) => ExprKey::Un(UnOp::SItoF, r(*a)),
        InstKind::UItoF(a) => ExprKey::Un(UnOp::UItoF, r(*a)),
        InstKind::SFtoI(a) => ExprKey::Un(UnOp::SFtoI, r(*a)),
        InstKind::UFtoI(a) => ExprKey::Un(UnOp::UFtoI, r(*a)),
        InstKind::Fptrunc(a) => ExprKey::Un(UnOp::Fptrunc, r(*a)),
        InstKind::Fpext(a) => ExprKey::Un(UnOp::Fpext, r(*a)),
        InstKind::PtrToInt(a) => ExprKey::Un(UnOp::PtrToInt, r(*a)),
        InstKind::IntToPtr(a) => ExprKey::Un(UnOp::IntToPtr, r(*a)),
        InstKind::Ceil(a) => ExprKey::Un(UnOp::Ceil, r(*a)),
        InstKind::Floor(a) => ExprKey::Un(UnOp::Floor, r(*a)),
        InstKind::Sqrt(a) => ExprKey::Un(UnOp::Sqrt, r(*a)),
        InstKind::Clz(a) => ExprKey::Un(UnOp::Clz, r(*a)),
        InstKind::Ctz(a) => ExprKey::Un(UnOp::Ctz, r(*a)),
        InstKind::Popcnt(a) => ExprKey::Un(UnOp::Popcnt, r(*a)),
        InstKind::Bswap(a) => ExprKey::Un(UnOp::Bswap, r(*a)),
        InstKind::Icmp { pred, lhs, rhs } => {
            let (lhs, rhs) = if matches!(pred, IcmpPred::Eq | IcmpPred::Ne) {
                sorted(r(*lhs), r(*rhs))
            } else {
                (r(*lhs), r(*rhs))
            };
            ExprKey::Cmp(CmpOp::Int(*pred), lhs, rhs)
        }
        InstKind::Fcmp { pred, lhs, rhs } => {
            let (lhs, rhs) = if matches!(pred, FcmpPred::Eq | FcmpPred::Ne | FcmpPred::Uno | FcmpPred::Ord) {
                sorted(r(*lhs), r(*rhs))
            } else {
                (r(*lhs), r(*rhs))
            };
            ExprKey::Cmp(CmpOp::Float(*pred), lhs, rhs)
        }
        InstKind::Select { cond, then_val, else_val } => ExprKey::Select {
            cond: r(*cond),
            then_val: r(*then_val),
            else_val: r(*else_val),
        },
        InstKind::GetElementPtr { base, indices } => ExprKey::Gep {
            base: r(*base),
            indices: indices
                .iter()
                .map(|i| match i {
                    GepIndex::Index(v) => GepIndex::Index(r(*v)),
                    other => other.clone(),
                })
                .collect(),
        },
        InstKind::ExtractValue { agg, index } => ExprKey::ExtractValue { agg: r(*agg), index: *index },
        InstKind::InsertValue { agg, elem, index } => ExprKey::InsertValue {
            agg: r(*agg),
            elem: r(*elem),
            index: *index,
        },
        InstKind::FuncAddr { func: target } => ExprKey::FuncAddr(*target),
        InstKind::ExternAddr { target } => ExprKey::ExternAddr(*target),
        InstKind::GlobalAddr { global } => ExprKey::GlobalAddr(*global),
        InstKind::GetFramePointer => ExprKey::GetFramePointer,
        _ => return None,
    };
    Some(Key { ty, expr })
}

fn bin(op: BinOp, a: Value, b: Value) -> ExprKey {
    if op.commutative() {
        let (a, b) = sorted(a, b);
        ExprKey::Bin(op, a, b)
    } else {
        ExprKey::Bin(op, a, b)
    }
}

/// Canonical operand order for commutative operations.
fn sorted(a: Value, b: Value) -> (Value, Value) {
    if a.index() <= b.index() { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests;

