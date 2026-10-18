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
                        let Some(key) = key(func, inst) else {
                            kept.push(inst);
                            continue;
                        };
                        match table.get(&key).copied() {
                            Some(canon) if canon != result => {
                                func.replace_uses(result, canon);
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
/// determined by the operands.
fn key(func: &Function, inst: Inst) -> Option<Key> {
    let result = func.insts[inst].result?;
    let ty = func.values[result].ty.clone();
    let expr = match &func.insts[inst].kind {
        InstKind::Add(a, b) => bin(BinOp::Add, *a, *b),
        InstKind::Sub(a, b) => bin(BinOp::Sub, *a, *b),
        InstKind::Mul(a, b) => bin(BinOp::Mul, *a, *b),
        InstKind::Udiv(a, b) => bin(BinOp::Udiv, *a, *b),
        InstKind::Sdiv(a, b) => bin(BinOp::Sdiv, *a, *b),
        InstKind::Urem(a, b) => bin(BinOp::Urem, *a, *b),
        InstKind::Srem(a, b) => bin(BinOp::Srem, *a, *b),
        InstKind::And(a, b) => bin(BinOp::And, *a, *b),
        InstKind::Or(a, b) => bin(BinOp::Or, *a, *b),
        InstKind::Xor(a, b) => bin(BinOp::Xor, *a, *b),
        InstKind::Shl(a, b) => bin(BinOp::Shl, *a, *b),
        InstKind::LShr(a, b) => bin(BinOp::LShr, *a, *b),
        InstKind::AShr(a, b) => bin(BinOp::AShr, *a, *b),
        InstKind::FAdd(a, b) => bin(BinOp::FAdd, *a, *b),
        InstKind::FSub(a, b) => bin(BinOp::FSub, *a, *b),
        InstKind::FMul(a, b) => bin(BinOp::FMul, *a, *b),
        InstKind::FDiv(a, b) => bin(BinOp::FDiv, *a, *b),
        InstKind::Neg(a) => ExprKey::Un(UnOp::Neg, *a),
        InstKind::Not(a) => ExprKey::Un(UnOp::Not, *a),
        InstKind::FNeg(a) => ExprKey::Un(UnOp::FNeg, *a),
        InstKind::Trunc(a) => ExprKey::Un(UnOp::Trunc, *a),
        InstKind::Zext(a) => ExprKey::Un(UnOp::Zext, *a),
        InstKind::Sext(a) => ExprKey::Un(UnOp::Sext, *a),
        InstKind::Bitcast(a) => ExprKey::Un(UnOp::Bitcast, *a),
        InstKind::SItoF(a) => ExprKey::Un(UnOp::SItoF, *a),
        InstKind::UItoF(a) => ExprKey::Un(UnOp::UItoF, *a),
        InstKind::SFtoI(a) => ExprKey::Un(UnOp::SFtoI, *a),
        InstKind::UFtoI(a) => ExprKey::Un(UnOp::UFtoI, *a),
        InstKind::Fptrunc(a) => ExprKey::Un(UnOp::Fptrunc, *a),
        InstKind::Fpext(a) => ExprKey::Un(UnOp::Fpext, *a),
        InstKind::PtrToInt(a) => ExprKey::Un(UnOp::PtrToInt, *a),
        InstKind::IntToPtr(a) => ExprKey::Un(UnOp::IntToPtr, *a),
        InstKind::Ceil(a) => ExprKey::Un(UnOp::Ceil, *a),
        InstKind::Floor(a) => ExprKey::Un(UnOp::Floor, *a),
        InstKind::Sqrt(a) => ExprKey::Un(UnOp::Sqrt, *a),
        InstKind::Clz(a) => ExprKey::Un(UnOp::Clz, *a),
        InstKind::Ctz(a) => ExprKey::Un(UnOp::Ctz, *a),
        InstKind::Popcnt(a) => ExprKey::Un(UnOp::Popcnt, *a),
        InstKind::Bswap(a) => ExprKey::Un(UnOp::Bswap, *a),
        InstKind::Icmp { pred, lhs, rhs } => {
            let (lhs, rhs) = if matches!(pred, IcmpPred::Eq | IcmpPred::Ne) {
                sorted(*lhs, *rhs)
            } else {
                (*lhs, *rhs)
            };
            ExprKey::Cmp(CmpOp::Int(*pred), lhs, rhs)
        }
        InstKind::Fcmp { pred, lhs, rhs } => {
            let (lhs, rhs) = if matches!(pred, FcmpPred::Eq | FcmpPred::Ne | FcmpPred::Uno | FcmpPred::Ord) {
                sorted(*lhs, *rhs)
            } else {
                (*lhs, *rhs)
            };
            ExprKey::Cmp(CmpOp::Float(*pred), lhs, rhs)
        }
        InstKind::Select { cond, then_val, else_val } => ExprKey::Select {
            cond: *cond,
            then_val: *then_val,
            else_val: *else_val,
        },
        InstKind::GetElementPtr { base, indices } => ExprKey::Gep {
            base: *base,
            indices: indices.clone(),
        },
        InstKind::ExtractValue { agg, index } => ExprKey::ExtractValue { agg: *agg, index: *index },
        InstKind::InsertValue { agg, elem, index } => ExprKey::InsertValue {
            agg: *agg,
            elem: *elem,
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
    if a.index() <= b.index() {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Const, Def, InstData, Terminator, ValueData};
    use crate::ty::{FloatKind, Signature};
    use crate::verify::verify_func;

    fn param(func: &mut Function, block: Block, ty: Ty, index: u32) -> Value {
        let v = func.values.push(ValueData {
            ty,
            def: Def::Param(block, index),
        });
        func.blocks[block].params.push(v);
        v
    }

    fn emit(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let inst = func.insts.push(InstData {
            block,
            kind,
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(inst);
        func.blocks[block].insts.push(inst);
        result
    }

    fn iconst(func: &mut Function, bits: u32, value: i128) -> Value {
        func.push_const(Const::Int { bits, value })
    }

    fn returned(func: &Function) -> Value {
        for block in func.blocks.iter() {
            match &block.term {
                Some(Terminator::Return { val: Some(v) }) => return *v,
                _ => {}
            }
        }
        panic!("no value return in {func:?}");
    }

    fn run(func: &mut Function) -> bool {
        let changed = Cse.run(func);
        let errors = verify_func(func);
        assert!(errors.is_empty(), "verifier: {errors:?}");
        changed
    }

    fn block_insts(func: &Function, block: Block) -> Vec<Inst> {
        func.blocks[block].insts.clone()
    }

    fn int_func() -> (Function, Block, Value, Value) {
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        (func, entry, a, b)
    }

    #[test]
    fn merges_duplicate_in_one_block() {
        let (mut func, entry, a, b) = int_func();
        let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
        let second = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        // Both operands of the sum now name the first addition.
        let inst = crate::ir::Inst::from_usize(0);
        let _ = inst;
        let Some(inst) = def_inst(&func, sum) else {
            panic!("sum has no defining instruction");
        };
        match &func.insts[inst].kind {
            InstKind::Add(x, y) => {
                assert_eq!(*x, first);
                assert_eq!(*y, first);
            }
            other => panic!("expected add, got {other:?}"),
        }
        assert_eq!(block_insts(&func, entry).len(), 2, "duplicate must leave the block");
        assert!(!run(&mut func), "second run is a no-op");
    }

    #[test]
    fn merges_commutative_operands() {
        let (mut func, entry, a, b) = int_func();
        let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
        let second = emit(&mut func, entry, InstKind::Add(b, a), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(first, second), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        let Some(inst) = def_inst(&func, sum) else {
            panic!("sum has no defining instruction");
        };
        match &func.insts[inst].kind {
            InstKind::Add(x, y) => assert_eq!(*x, *y),
            other => panic!("expected add, got {other:?}"),
        }
    }

    #[test]
    fn keeps_ordered_comparisons_apart() {
        // `a < b` and `b < a` differ, so they must not merge.
        let (mut func, entry, a, b) = int_func();
        let lt = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Slt,
                lhs: a,
                rhs: b,
            },
            Ty::Bool,
        );
        let gt = emit(
            &mut func,
            entry,
            InstKind::Icmp {
                pred: IcmpPred::Slt,
                lhs: b,
                rhs: a,
            },
            Ty::Bool,
        );
        let sum = emit(&mut func, entry, InstKind::Xor(lt, gt), Ty::Bool);
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
    }

    #[test]
    fn merges_across_dominating_blocks() {
        let (mut func, entry, a, b) = int_func();
        let first = emit(&mut func, entry, InstKind::Add(a, b), Ty::int(32));
        let body = func.push_block(vec![]);
        func.blocks[entry].term = Some(Terminator::Br {
            dest: body,
            args: Default::default(),
        });
        let second = emit(&mut func, body, InstKind::Add(a, b), Ty::int(32));
        func.blocks[body].term = Some(Terminator::Return { val: Some(second) });
        assert!(run(&mut func));
        assert_eq!(returned(&func), first);
        assert!(block_insts(&func, body).is_empty());
    }

    #[test]
    fn keeps_sibling_branch_expressions_apart() {
        // Neither arm dominates the other, so both additions stay.
        let sig = Signature::new(vec![Ty::int(32), Ty::int(32), Ty::Bool], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::int(32), 0);
        let b = param(&mut func, entry, Ty::int(32), 1);
        let cond = param(&mut func, entry, Ty::Bool, 2);
        let then_block = func.push_block(vec![]);
        let else_block = func.push_block(vec![]);
        let merge = func.push_block(vec![Ty::int(32)]);
        func.blocks[entry].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: Default::default(),
            else_args: Default::default(),
        });
        let t1 = emit(&mut func, then_block, InstKind::Add(a, b), Ty::int(32));
        func.blocks[then_block].term = Some(Terminator::Br {
            dest: merge,
            args: [t1].into_iter().collect(),
        });
        let t2 = emit(&mut func, else_block, InstKind::Add(a, b), Ty::int(32));
        func.blocks[else_block].term = Some(Terminator::Br {
            dest: merge,
            args: [t2].into_iter().collect(),
        });
        let merged = func.blocks[merge].params[0];
        func.blocks[merge].term = Some(Terminator::Return { val: Some(merged) });
        assert!(!run(&mut func));
        assert_eq!(block_insts(&func, then_block).len(), 1);
        assert_eq!(block_insts(&func, else_block).len(), 1);
    }

    #[test]
    fn does_not_merge_memory_reads() {
        // A load sees whatever the last store left; only one alias
        // analysis could say the two loads agree.
        let sig = Signature::new(vec![Ty::ptr(Ty::int(32))], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let p = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        let l1 = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr: p,
                flags: crate::ir::MemFlags::plain(),
            },
            Ty::int(32),
        );
        let l2 = emit(
            &mut func,
            entry,
            InstKind::Load {
                addr: p,
                flags: crate::ir::MemFlags::plain(),
            },
            Ty::int(32),
        );
        let sum = emit(&mut func, entry, InstKind::Add(l1, l2), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
    }

    #[test]
    fn does_not_merge_va_args() {
        // Each `VaArg` advances the list; merging two would drop one
        // advance.
        let ap_ty = Ty::BuiltinVaList { size: 24, align: 8 };
        let sig = Signature::new(vec![Ty::ptr(ap_ty.clone())], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let ap = param(&mut func, entry, Ty::ptr(ap_ty), 0);
        let v1 = emit(&mut func, entry, InstKind::VaArg { ap }, Ty::int(32));
        let v2 = emit(&mut func, entry, InstKind::VaArg { ap }, Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(v1, v2), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
    }

    #[test]
    fn merges_identical_geps() {
        let sig = Signature::new(vec![Ty::ptr(Ty::int(32)), Ty::int(32)], Ty::int(32));
        let (mut func, entry) = Function::new("f", sig);
        let base = param(&mut func, entry, Ty::ptr(Ty::int(32)), 0);
        let index = param(&mut func, entry, Ty::int(32), 1);
        let indices: SmallVec<[GepIndex; 4]> = [GepIndex::Index(index)].into_iter().collect();
        let g1 = emit(&mut func, entry, InstKind::GetElementPtr { base, indices: indices.clone() }, Ty::ptr(Ty::int(32)));
        let g2 = emit(&mut func, entry, InstKind::GetElementPtr { base, indices }, Ty::ptr(Ty::int(32)));
        let p1 = emit(&mut func, entry, InstKind::PtrToInt(g1), Ty::int(64));
        let p2 = emit(&mut func, entry, InstKind::PtrToInt(g2), Ty::int(64));
        let diff = emit(&mut func, entry, InstKind::Sub(p1, p2), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(diff) });
        assert!(run(&mut func));
        // Both ptrtoint's now name the first GEP, and the subtraction
        // of a value with itself dies to inst-combine/DCE later.
        let Some(sub) = def_inst(&func, diff) else {
            panic!("difference has no defining instruction");
        };
        match &func.insts[sub].kind {
            InstKind::Sub(x, y) => {
                assert_eq!(*x, *y, "both sides must share the first GEP's address");
                let x = def_inst(&func, *x).expect("ptrtoint");
                match &func.insts[x].kind {
                    InstKind::PtrToInt(g) => assert_eq!(*g, g1),
                    other => panic!("expected ptrtoint, got {other:?}"),
                }
            }
            other => panic!("expected sub, got {other:?}"),
        }
    }

    #[test]
    fn merges_selects_and_unary_ops() {
        let (mut func, entry, a, b) = int_func();
        let cond = iconst(&mut func, 1, 1);
        let s1 = emit(&mut func, entry, InstKind::Select { cond, then_val: a, else_val: b }, Ty::int(32));
        let neg1 = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
        let neg2 = emit(&mut func, entry, InstKind::Neg(a), Ty::int(32));
        let s2 = emit(&mut func, entry, InstKind::Select { cond, then_val: a, else_val: b }, Ty::int(32));
        let sum1 = emit(&mut func, entry, InstKind::Add(s1, s2), Ty::int(32));
        let sum = emit(&mut func, entry, InstKind::Add(sum1, neg1), Ty::int(32));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        assert_eq!(block_insts(&func, entry).len(), 4, "one select, one neg, two adds");
        let _ = neg2;
    }

    #[test]
    fn distinguishes_result_types() {
        // `zext x` to i32 and to i64 are different expressions.
        let sig = Signature::new(vec![Ty::int(8)], Ty::int(64));
        let (mut func, entry) = Function::new("f", sig);
        let x = param(&mut func, entry, Ty::int(8), 0);
        let z32 = emit(&mut func, entry, InstKind::Zext(x), Ty::int(32));
        let z64 = emit(&mut func, entry, InstKind::Zext(x), Ty::int(64));
        let wide = emit(&mut func, entry, InstKind::Zext(z32), Ty::int(64));
        let sum = emit(&mut func, entry, InstKind::Add(wide, z64), Ty::int(64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(!run(&mut func));
    }

    #[test]
    fn merges_float_binops_commutatively() {
        let sig = Signature::new(vec![Ty::Float(FloatKind::F64), Ty::Float(FloatKind::F64)], Ty::Float(FloatKind::F64));
        let (mut func, entry) = Function::new("f", sig);
        let a = param(&mut func, entry, Ty::Float(FloatKind::F64), 0);
        let b = param(&mut func, entry, Ty::Float(FloatKind::F64), 1);
        let m1 = emit(&mut func, entry, InstKind::FMul(a, b), Ty::Float(FloatKind::F64));
        let m2 = emit(&mut func, entry, InstKind::FMul(b, a), Ty::Float(FloatKind::F64));
        let sum = emit(&mut func, entry, InstKind::FAdd(m1, m2), Ty::Float(FloatKind::F64));
        func.blocks[entry].term = Some(Terminator::Return { val: Some(sum) });
        assert!(run(&mut func));
        let Some(inst) = def_inst(&func, sum) else {
            panic!("sum has no defining instruction");
        };
        match &func.insts[inst].kind {
            InstKind::FAdd(x, y) => assert_eq!(*x, *y),
            other => panic!("expected fadd, got {other:?}"),
        }
    }

    fn def_inst(func: &Function, v: Value) -> Option<Inst> {
        match &func.values[v].def {
            Def::Inst(i) if i.index() < func.insts.len() => Some(*i),
            _ => None,
        }
    }
}
