//! The class solver.
//!
//! Goals are `C a1 ... an` obligations over classes; `args[0]` is the
//! dispatching argument. A goal is discharged from the enclosing
//! environment's bounds (instance contexts, for generic bodies) or by
//! matching instance heads, whose own bounds become recursive subgoals.
//! Because class parameters are higher-kinded, matching uses the same
//! spine unification as inference, so `Functor f` with `f` known better
//! than a bare constructor still resolves.
//!
//! The solver is used at check time (method resolution and instance
//! bodies) and at lowering time, where monomorphization makes the
//! deferred goals concrete.

use std::collections::HashMap;

use crate::defs::{Bound, ClassId, Defs};
use crate::infer::InferCtx;
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

const DEPTH_LIMIT: usize = 64;

#[derive(Debug, Clone)]
pub struct Obligation {
    pub class_id: ClassId,
    /// Class parameters in declaration order.
    pub args: Vec<Ty>,
}

#[derive(Debug, Clone)]
pub enum Solution {
    /// An instance applies; `args` are its generic parameters'
    /// instantiations, in declaration order.
    Instance {
        index: usize,
        args: Vec<Ty>,
    },
    /// Discharged by a bound in the environment.
    Bound,
    /// More than one instance applies.
    Ambiguous {
        instances: Vec<usize>,
    },
    Unsatisfied,
    /// Some argument is not known yet; the goal must be resolved after
    /// monomorphization.
    Deferred,
}

impl Solution {
    pub fn is_failure(&self) -> bool {
        matches!(self, Solution::Unsatisfied | Solution::Ambiguous { .. })
    }
}

pub struct Solver<'d> {
    defs: &'d Defs,
}

impl<'d> Solver<'d> {
    pub fn new(defs: &'d Defs) -> Solver<'d> {
        Solver { defs }
    }

    pub fn defs(&self) -> &'d Defs {
        self.defs
    }

    pub fn prove(&self, infer: &mut InferCtx, env: &[Bound], goal: &Obligation, span: Span) -> KResult<Solution> {
        let mut stack = Vec::new();
        self.prove_at(infer, env, goal, span, 0, &mut stack)
    }

    pub fn describe_failure(&self, goal: &Obligation, solution: &Solution, span: Span) -> Diag {
        let goal_ty = self.goal_display(goal);
        match solution {
            Solution::Unsatisfied => Diag::new(format!("the bound `{goal_ty}` is not satisfied"), span),
            Solution::Ambiguous { instances } => Diag::new(format!("multiple instances apply to `{goal_ty}` ({} candidates)", instances.len()), span),
            Solution::Deferred => Diag::new(format!("type annotations needed: cannot decide `{goal_ty}` yet"), span),
            _ => Diag::new("class obligation failed", span),
        }
    }

    pub fn goal_display(&self, goal: &Obligation) -> String {
        let class = &self.defs.classes[goal.class_id.index()];
        let args = goal.args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
        format!("{} {args}", class.name)
    }

    fn resolved_goal_display(&self, infer: &InferCtx, goal: &Obligation) -> String {
        let class = &self.defs.classes[goal.class_id.index()];
        let args = goal.args.iter().map(|t| infer.resolve(t).to_string()).collect::<Vec<_>>().join(" ");
        format!("{} {args}", class.name)
    }

    /// Tries the environment's bounds for the goal. On success the
    /// argument unifications are kept, so callers see the match.
    fn prove_bound(&self, infer: &mut InferCtx, env: &[Bound], goal: &Obligation, span: Span) -> KResult<bool> {
        for bound in env {
            if bound.class_id != goal.class_id || bound.args.len() != goal.args.len() {
                continue;
            }
            let mark = infer.mark();
            let ok = bound.args.iter().zip(goal.args.iter()).all(|(b, g)| infer.unify(b, g, span).is_ok());
            if ok {
                return Ok(true);
            }
            infer.rollback(mark);
        }
        Ok(false)
    }

    fn prove_at(&self, infer: &mut InferCtx, env: &[Bound], goal: &Obligation, span: Span, depth: usize, stack: &mut Vec<String>) -> KResult<Solution> {
        if depth > DEPTH_LIMIT {
            return Err(Diag::new(format!("class solver recursion limit reached while proving `{}`", self.goal_display(goal)), span));
        }

        let args: Vec<Ty> = goal.args.iter().map(|a| infer.resolve(a)).collect();
        let resolved = Obligation { class_id: goal.class_id, args };

        // Bounds discharge goals on rigid parameters before anything
        // else; they are the assumptions an instance context provides.
        if self.prove_bound(infer, env, &resolved, span)? {
            return Ok(Solution::Bound);
        }

        // With any argument still unknown, matching instances would
        // commit the unknowns to one candidate; leave the decision to
        // monomorphization instead.
        if resolved.args.iter().any(|a| a.contains_var()) {
            return Ok(Solution::Deferred);
        }

        let key = self.resolved_goal_display(infer, &resolved);
        if stack.contains(&key) {
            return Ok(Solution::Unsatisfied);
        }
        stack.push(key);

        let mut proven: Vec<usize> = Vec::new();
        let mut saw_deferred = false;
        // Every candidate is tried from the same pristine state.
        let base = infer.mark();
        for (index, instance) in self.defs.instances.iter().enumerate() {
            if instance.class_id != goal.class_id {
                continue;
            }
            infer.rollback(base);
            let mark = infer.mark();
            let mut subst: HashMap<String, Ty> = HashMap::new();
            for (name, _) in &instance.generics {
                subst.insert(name.clone(), infer.fresh());
            }
            let head_ok = instance.args.len() == resolved.args.len()
                && instance
                    .args
                    .iter()
                    .zip(resolved.args.iter())
                    .all(|(i, g)| infer.unify(&i.substitute(&subst), g, span).is_ok());
            if !head_ok {
                infer.rollback(mark);
                continue;
            }
            let mut ok = true;
            let mut deferred = false;
            for bound in &instance.bounds {
                let sub_goal = Obligation {
                    class_id: bound.class_id,
                    args: bound.args.iter().map(|a| a.substitute(&subst)).collect(),
                };
                match self.prove_at(infer, env, &sub_goal, span, depth + 1, stack)? {
                    Solution::Unsatisfied | Solution::Ambiguous { .. } => {
                        ok = false;
                        break;
                    }
                    Solution::Deferred => {
                        deferred = true;
                        break;
                    }
                    Solution::Instance { .. } | Solution::Bound => {}
                }
            }
            if ok && !deferred {
                proven.push(index);
                continue;
            }
            if deferred {
                saw_deferred = true;
            }
            infer.rollback(mark);
        }
        infer.rollback(base);
        stack.pop();

        match proven.len() {
            0 if saw_deferred => Ok(Solution::Deferred),
            0 if resolved.args.iter().any(|a| a.contains_param()) => Ok(Solution::Deferred),
            0 => Ok(Solution::Unsatisfied),
            1 => self.apply_instance(infer, env, &resolved, proven[0], span, depth, stack),
            _ => Ok(Solution::Ambiguous { instances: proven }),
        }
    }

    /// Re-applies the winning instance after the candidate scan, so its
    /// unifications survive in the caller's state and callers can read
    /// the instantiated generic arguments.
    fn apply_instance(&self, infer: &mut InferCtx, env: &[Bound], goal: &Obligation, index: usize, span: Span, depth: usize, stack: &mut Vec<String>) -> KResult<Solution> {
        let instance = &self.defs.instances[index];
        let mut subst: HashMap<String, Ty> = HashMap::new();
        for (name, _) in &instance.generics {
            subst.insert(name.clone(), infer.fresh());
        }
        for (i, g) in instance.args.iter().zip(goal.args.iter()) {
            infer.unify(&i.substitute(&subst), g, span)?;
        }
        for bound in &instance.bounds {
            let sub_goal = Obligation {
                class_id: bound.class_id,
                args: bound.args.iter().map(|a| a.substitute(&subst)).collect(),
            };
            self.prove_at(infer, env, &sub_goal, span, depth + 1, stack)?;
        }
        let args = instance.generics.iter().map(|(name, _)| infer.resolve(&subst[name])).collect();
        Ok(Solution::Instance { index, args })
    }
}
