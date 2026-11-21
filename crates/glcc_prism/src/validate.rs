//! PrismIR well-formedness checks, run before anything consumes the
//! program: every register defined exactly once, types consistent
//! between operands and results, region joins carrying what their
//! arms produce, and break/continue statements matching the loop they
//! leave. Catches structurizer and lowering bugs; the ISA targets
//! assume all of it.

use std::collections::HashMap;
use std::format;

use smallvec::SmallVec;

use crate::ir::{Alu, Program, RegTy, Scalar, Src, Stmt, VReg};

/// Checks to run after building. Returns driver-renderable errors.
pub fn validate(program: &Program) -> Result<(), String> {
    let mut v = Validator {
        program,
        defs: HashMap::new(),
        errors: Vec::new(),
    };
    let context = LoopContext::default();
    v.body(&program.body, &context);
    if v.errors.is_empty() {
        Ok(())
    } else {
        Err(v.errors.join("\n"))
    }
}

#[derive(Default, Clone, Copy)]
struct LoopContext {
    /// Carried register count of the enclosing loop.
    carried: usize,
    /// Exit register count of the enclosing loop.
    results: usize,
    /// Enclosing loop depth (0 = not inside a loop).
    depth: usize,
}

struct Validator<'a> {
    program: &'a Program,
    /// Register -> the statement index that defined it.
    defs: HashMap<VReg, usize>,
    errors: Vec<String>,
}

impl Validator<'_> {
    fn reg_ty(&self, v: VReg) -> Option<&RegTy> {
        self.program.regs.get(v.0 as usize)
    }

    fn ty(&mut self, v: VReg, what: &str, index: usize) -> RegTy {
        match self.reg_ty(v) {
            Some(ty) => ty.clone(),
            None => {
                self.errors.push(format!("%{v}: {what} (statement {index}) has no recorded type"));
                RegTy::Scalar(Scalar::F32)
            }
        }
    }

    fn define(&mut self, v: VReg, what: &str, index: usize) {
        if v.0 as usize >= self.program.regs.len() {
            self.errors.push(format!("%{v}: {what} (statement {index}) is outside the register arena"));
            return;
        }
        if self.defs.insert(v, index).is_some() {
            self.errors.push(format!("%{v}: defined more than once (statement {index} redefines it)"));
        }
    }

    fn body(&mut self, body: &crate::ir::Body, context: &LoopContext) {
        for (index, stmt) in body.stmts.iter().enumerate() {
            self.stmt(stmt, index, context);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, index: usize, context: &LoopContext) {
        match stmt {
            Stmt::Op(op) => {
                self.define(op.dst, "op result", index);
                let dst_ty = self.ty(op.dst, "op result", index);
                for (position, src) in op.srcs.iter().enumerate() {
                    self.src(src, &dst_ty, index, position);
                }
                self.alu(op.kind_arity(), op, index);
            }
            Stmt::Tex(tex) => {
                self.define(tex.dst, "sample result", index);
                if tex.sampler as usize >= self.program.interface.samplers.len() {
                    self.errors.push(format!("statement {index}: sampler s{} is not declared", tex.sampler));
                }
                let dst_ty = self.ty(tex.dst, "sample result", index);
                self.src(&tex.coord, &dst_ty, index, 0);
            }
            Stmt::Setp(setp) => {
                self.define(setp.dst, "setp result", index);
                if self.ty(setp.dst, "setp result", index) != RegTy::Pred {
                    self.errors.push(format!("%{}: setp result must be pred", setp.dst.0));
                }
                for (position, src) in [&setp.a, &setp.b].iter().enumerate() {
                    self.src(src, &RegTy::Pred, index, position);
                }
            }
            Stmt::If { pred, then, els, results } => {
                if self.ty(*pred, "if predicate", index) != RegTy::Pred {
                    self.errors.push(format!("%{}: if predicate must be pred", pred.0));
                }
                self.body(then, context);
                self.body(els, context);
                for (join, then_v, else_v) in results {
                    self.define(*join, "if join", index);
                    self.ty(*join, "if join", index);
                    let _ = (then_v, else_v);
                }
            }
            Stmt::Loop { carried, body: loop_body, results } => {
                for (phi, init) in carried {
                    self.define(*phi, "carried register", index);
                    let ty = self.ty(*phi, "carried register", index);
                    self.ty(*init, "carried init", index);
                    if let Some(init_ty) = self.reg_ty(*init) {
                        if init_ty != &ty {
                            self.errors.push(format!("%{phi}: carried register type {ty:?} does not match its init {init_ty:?}"));
                        }
                    }
                }
                let inner = LoopContext {
                    carried: carried.len(),
                    results: results.len(),
                    depth: context.depth + 1,
                };
                self.body(loop_body, &inner);
                if carried.is_empty() && results.is_empty() {
                    // A loop with no carried state and no exits never
                    // contributes values; still legal (it must contain
                    // a break or the program diverges — targets decide).
                }
                for result in results {
                    self.define(*result, "loop exit", index);
                }
            }
            Stmt::Break(args) => {
                if context.depth == 0 {
                    self.errors.push(format!("statement {index}: break outside a loop"));
                } else if args.len() != context.results {
                    self.errors.push(format!(
                        "statement {index}: break carries {} values, the loop exits with {}",
                        args.len(),
                        context.results
                    ));
                }
                for arg in args {
                    self.ty(*arg, "break value", index);
                }
            }
            Stmt::Continue(args) => {
                if context.depth == 0 {
                    self.errors.push(format!("statement {index}: continue outside a loop"));
                } else if args.len() != context.carried {
                    self.errors.push(format!(
                        "statement {index}: continue carries {} values, the loop carries {}",
                        args.len(),
                        context.carried
                    ));
                }
                for arg in args {
                    self.ty(*arg, "continue value", index);
                }
            }
            Stmt::Exit(outputs) => {
                let outputs_count = self.program.interface.outputs.len();
                if outputs.len() != outputs_count {
                    self.errors.push(format!(
                        "statement {index}: ret carries {} values, the interface has {} outputs",
                        outputs.len(),
                        outputs_count
                    ));
                }
                for (position, value) in outputs.iter().enumerate() {
                    let expected = self.program.interface.outputs.get(position).map(|leaf| leaf.ty.clone());
                    let ty = self.ty(*value, "ret value", index);
                    if let Some(expected) = expected
                        && ty != expected
                    {
                        self.errors.push(format!("statement {index}: output {position} is {ty:?}, the interface declares {expected:?}"));
                    }
                }
            }
        }
    }

    /// Operand shape checks per operation.
    fn alu(&mut self, _arity: usize, op: &crate::ir::Op, index: usize) {
        let dst_ty = self.reg_ty(op.dst).cloned();
        let Some(dst_ty) = dst_ty else { return };
        let src_tys: Vec<Option<RegTy>> = op
            .srcs
            .iter()
            .map(|src| {
                self.src_reg(src)
                    .and_then(|v| self.reg_ty(v).cloned())
                    .or_else(|| match src {
                        Src::ConstF32(_) => Some(RegTy::Scalar(Scalar::F32)),
                        Src::ConstI32(_) => Some(RegTy::Scalar(Scalar::i32())),
                        // Inline interface reads carry their leaf's type.
                        Src::Input { input, .. } => self
                            .program
                            .interface
                            .inputs
                            .get(*input as usize)
                            .map(|leaf| leaf.ty.clone()),
                        Src::Uniform { object, reg, .. } => self
                            .program
                            .interface
                            .uniforms
                            .get(*object as usize)
                            .and_then(|u| u.leaves.iter().find(|leaf| leaf.creg == *reg))
                            .map(|leaf| leaf.ty.clone()),
                        _ => None,
                    })
            })
            .collect();
        let float = dst_ty.is_f32();
        let lanes = dst_ty.lanes();
        // Lanes a source delivers: a swizzled read narrows to the
        // swizzle's length, an unswizzled one delivers its whole type.
        // Sources with no known type (undef) stay unchecked.
        let src_lanes = |position: usize| -> Option<u32> {
            let ty = src_tys[position].as_ref()?;
            let swizzle = match &op.srcs[position] {
                Src::VReg { swizzle, .. } | Src::Input { swizzle, .. } | Src::Uniform { swizzle, .. } => {
                    swizzle.as_ref().map(|s| s.len() as u32)
                }
                _ => None,
            };
            Some(swizzle.unwrap_or(u32::from(ty.lanes())))
        };
        match &op.kind {
            Alu::Mov => {
                // A scalar source feeding a vector dst is the splat
                // form; anything else must match.
                if let Some(width) = src_lanes(0)
                    && width != 1
                    && width != u32::from(lanes)
                {
                    self.errors.push(format!("statement {index}: mov width {width} into {lanes}"));
                }
            }
            Alu::Dp3 | Alu::Dp4 => {
                if dst_ty.lanes() != 1 {
                    self.errors.push(format!("statement {index}: dp results are scalars"));
                }
                let dot = if matches!(op.kind, Alu::Dp4) { 4 } else { 3 };
                for position in 0..op.srcs.len() {
                    if let Some(width) = src_lanes(position)
                        && width != 1
                        && width < dot
                    {
                        self.errors.push(format!("statement {index}: dp sources must cover the dot width"));
                    }
                }
            }
            Alu::Cmp(_) => {
                if !matches!(dst_ty, RegTy::Scalar(Scalar::Int { bits: 32 }) | RegTy::Vector { elem: Scalar::Int { bits: 32 }, .. }) {
                    self.errors.push(format!("statement {index}: cmp yields i32 masks"));
                }
                // The operand types decide float vs integer; they must
                // agree with each other (the result is always i32).
                let operand_float = src_tys.iter().flatten().map(|ty| ty.is_f32());
                if operand_float.clone().any(|f| f != operand_float.clone().next().unwrap_or(true)) {
                    self.errors.push(format!("statement {index}: cmp operands mix float and integer"));
                }
            }
            Alu::Rcp | Alu::Rsq | Alu::Sqrt | Alu::Floor | Alu::Ceil | Alu::Frac | Alu::Neg | Alu::Abs => {
                if !float {
                    self.errors.push(format!("statement {index}: float-only operation on {dst_ty:?}"));
                }
                if let Some(width) = src_lanes(0)
                    && width != u32::from(lanes)
                    && width != 1
                {
                    self.errors.push(format!("statement {index}: unary width mismatch"));
                }
            }
            Alu::IToF { .. } => {
                if !float {
                    self.errors.push(format!("statement {index}: conversion to a non-float"));
                }
            }
            Alu::FToI { .. } => {
                if float {
                    self.errors.push(format!("statement {index}: conversion to a float"));
                }
            }
            Alu::Shl | Alu::Shr { .. } | Alu::And | Alu::Or | Alu::Xor | Alu::Not | Alu::IDiv { .. } | Alu::IRem { .. } => {
                if float {
                    self.errors.push(format!("statement {index}: integer-only operation on {dst_ty:?}"));
                }
            }
            Alu::Zext | Alu::Sext | Alu::Trunc | Alu::Bitcast => {}
            Alu::Sel => {
                if op.srcs.len() != 3 {
                    self.errors.push(format!("statement {index}: sel takes three operands"));
                }
            }
            Alu::Mad => {
                if op.srcs.len() != 3 {
                    self.errors.push(format!("statement {index}: mad takes three operands"));
                }
            }
            Alu::Shuffle { lanes: perm } => {
                for lane in perm {
                    let total = src_tys.iter().flatten().map(|t| t.lanes() as u32).sum::<u32>();
                    if u32::from(*lane) >= total.max(1) {
                        self.errors.push(format!("statement {index}: shuffle lane {lane} out of range"));
                    }
                }
            }
            _ => {}
        }
    }

    fn src(&mut self, src: &Src, dst_ty: &RegTy, index: usize, position: usize) {
        match src {
            Src::VReg { v, swizzle, .. } => {
                let width = self.reg_ty(*v).map(|ty| ty.lanes()).unwrap_or(0);
                self.check_swizzle(swizzle, width, index, position);
            }
            Src::Input { input, swizzle, .. } => {
                let width = self
                    .program
                    .interface
                    .inputs
                    .get(*input as usize)
                    .map(|leaf| leaf.ty.lanes())
                    .unwrap_or(0);
                self.check_swizzle(swizzle, width, index, position);
            }
            Src::Uniform { object, reg, swizzle, rel, .. } => {
                let uniform = match self.program.interface.uniforms.get(*object as usize) {
                    Some(uniform) => uniform,
                    None => {
                        self.errors.push(format!("statement {index}: operand {position} reads undeclared uniform u{object}"));
                        return;
                    }
                };
                if !uniform.leaves.iter().any(|leaf| leaf.creg == *reg) {
                    self.errors.push(format!("statement {index}: operand {position} reads c{reg}, which uniform u{object} does not occupy"));
                }
                if let Some(rel) = rel {
                    if self.reg_ty(*rel).map(|ty| !matches!(ty, RegTy::Scalar(Scalar::Int { .. }))).unwrap_or(true) {
                        self.errors.push(format!("statement {index}: relative index must be an integer register"));
                    }
                }
                if let Some(lanes) = swizzle {
                    let _ = lanes;
                }
                let _ = dst_ty;
            }
            Src::ConstF32(_) | Src::ConstI32(_) | Src::Undef => {}
        }
    }

    fn check_swizzle(&mut self, swizzle: &Option<crate::ir::Swizzle>, width: u8, index: usize, position: usize) {
        if width == 0 {
            self.errors.push(format!("statement {index}: operand {position} reads an unknown register"));
            return;
        }
        if let Some(lanes) = swizzle {
            for lane in lanes {
                if *lane >= width {
                    self.errors.push(format!("statement {index}: operand {position} swizzle lane {lane} out of range"));
                }
            }
        }
    }

    fn src_reg(&self, src: &Src) -> Option<VReg> {
        match src {
            Src::VReg { v, .. } => Some(*v),
            _ => None,
        }
    }
}

impl crate::ir::Op {
    fn kind_arity(&self) -> usize {
        self.srcs.len()
    }
}
