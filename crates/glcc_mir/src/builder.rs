use smallvec::SmallVec;

use crate::ir::{Block, Const, Def, Func, Function, Inst, InstData, InstKind, MemFlags, Module, Terminator, Value, ValueData};
use crate::ty::{Signature, Ty};

/// Ergonomic builder over a function under construction.
pub struct FunctionBuilder<'f> {
    pub func: &'f mut Function,
    pub block: Block,
}

impl<'f> FunctionBuilder<'f> {
    pub fn new(func: &'f mut Function, block: Block) -> Self {
        FunctionBuilder { func, block }
    }

    pub fn switch_block(&mut self, block: Block) {
        self.block = block;
    }

    pub fn iconst(&mut self, bits: u32, value: i128) -> Value {
        self.func.push_const(Const::Int { bits, value })
    }

    fn emit(&mut self, kind: InstKind, result_ty: Option<Ty>) -> ValueOrUnit {
        let result = result_ty.map(|ty| {
            self.func.values.push(ValueData {
                ty,
                def: Def::Inst(Inst::from_usize(self.func.insts.len())),
            })
        });
        let inst = self.func.insts.push(InstData {
            block: self.block,
            kind,
            result,
            loc: None,
        });
        self.func.blocks[self.block].insts.push(inst);
        match result {
            Some(v) => ValueOrUnit::Value(v),
            None => ValueOrUnit::Unit(inst),
        }
    }

    fn emit_typed(&mut self, kind: InstKind, ty: Ty) -> Value {
        match self.emit(kind, Some(ty)) {
            ValueOrUnit::Value(v) => v,
            ValueOrUnit::Unit(_) => unreachable!("typed emit always yields a value"),
        }
    }

    pub fn add(&mut self, ty: Ty, a: Value, b: Value) -> Value {
        self.emit_typed(InstKind::Add(a, b), ty)
    }

    pub fn mul(&mut self, ty: Ty, a: Value, b: Value) -> Value {
        self.emit_typed(InstKind::Mul(a, b), ty)
    }

    pub fn call(&mut self, callee: Func, args: Vec<Value>, ret: Ty) -> Option<Value> {
        let has_result = !ret.is_void();
        let out = self.emit(
            InstKind::Call {
                callee,
                args: args.into_iter().collect(),
            },
            has_result.then_some(ret),
        );
        match out {
            ValueOrUnit::Value(v) => Some(v),
            ValueOrUnit::Unit(_) => None,
        }
    }

    pub fn alloca(&mut self, pointee: Ty, count: u64) -> Value {
        let ty = Ty::ptr(pointee.clone());
        self.emit_typed(InstKind::Alloca { pointee, count }, ty)
    }

    pub fn gep(&mut self, base: Value, indices: Vec<crate::ty::GepIndex>, result_ty: Ty) -> Value {
        self.emit_typed(
            InstKind::GetElementPtr {
                base,
                indices: indices.into_iter().collect(),
            },
            result_ty,
        )
    }

    pub fn load(&mut self, addr: Value, ty: Ty) -> Value {
        self.emit_typed(InstKind::Load { addr, flags: MemFlags::plain() }, ty)
    }

    pub fn global_addr(&mut self, global: crate::ir::Global, pointee: Ty) -> Value {
        self.emit_typed(InstKind::GlobalAddr { global }, Ty::ptr(pointee))
    }

    pub fn store(&mut self, addr: Value, val: Value) {
        let _ = self.emit(
            InstKind::Store {
                addr,
                val,
                flags: MemFlags::plain(),
            },
            None,
        );
    }

    pub fn ret(&mut self, val: Option<Value>) {
        self.func.blocks[self.block].term = Some(Terminator::Return { val });
    }

    pub fn br(&mut self, dest: Block, args: Vec<Value>) {
        self.func.blocks[self.block].term = Some(Terminator::Br {
            dest,
            args: args.into_iter().collect(),
        });
    }

    pub fn cond_br(&mut self, cond: Value, then_block: Block, else_block: Block, then_args: Vec<Value>, else_args: Vec<Value>) {
        self.func.blocks[self.block].term = Some(Terminator::CondBr {
            cond,
            then_block,
            else_block,
            then_args: then_args.into_iter().collect(),
            else_args: else_args.into_iter().collect(),
        });
    }
}

pub enum ValueOrUnit {
    Value(Value),
    Unit(Inst),
}

/// Module-level builder handing out function ids.
pub struct ModuleBuilder {
    pub module: Module,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        ModuleBuilder { module: Module::new() }
    }

    pub fn declare(&mut self, name: &str, sig: Signature) -> (Func, Block) {
        let (func, entry) = Function::new(name, sig);
        let id = self.module.push_func(func);
        (id, entry)
    }

    pub fn func_mut(&mut self, id: Func) -> &mut Function {
        &mut self.module.funcs[id]
    }

    /// Intern a source path into the module file table, returning its
    /// [`Location::file`](crate::debug::Location::file) index.
    pub fn intern_file(&mut self, path: &str) -> u32 {
        if let Some(index) = self.module.files.iter().position(|f| f.path == path) {
            return index as u32;
        }
        self.module.files.push(crate::debug::File { path: path.to_string() });
        (self.module.files.len() - 1) as u32
    }

    pub fn finish(self) -> Module {
        self.module
    }
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused_imports)]
use SmallVec as _SmallVecUse;
