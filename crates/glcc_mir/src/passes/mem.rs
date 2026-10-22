//! Lower libc memory calls with a constant size to MIR memory operations.
//!
//! A call to `memcpy(dst, src, 8)` is otherwise opaque to `Sroa` and
//! `Mem2Reg`: the destination stays address-taken and the backend has to
//! call libc, even though the copy is one load and one store. Turning the
//! call into [`InstKind::Memcpy`] lets the existing local-copy expansion
//! promote scalar destinations and lets lowering emit the copy inline.

use crate::ir::{Block, Function, InstData, InstKind, Module, Value};

use super::int_const;
use crate::pass::ModulePass;

/// Rewrite constant-size `memcpy`/`memset` calls.
#[derive(Default)]
pub struct LowerLibcMemory;

impl ModulePass for LowerLibcMemory {
    fn name(&self) -> &'static str {
        "lower-libc-memory"
    }

    fn run(&mut self, module: &mut Module) -> bool {
        let mut copies = Vec::new();
        let mut sets = Vec::new();
        for (id, ext) in module.externs.iter_enumerated() {
            match ext.name.as_str() {
                "memcpy" | "__memcpy" => copies.push(id),
                "memset" | "__memset" => sets.push(id),
                _ => {}
            }
        }
        if copies.is_empty() && sets.is_empty() {
            return false;
        }

        let mut changed = false;
        for func in module.funcs.iter_mut() {
            for block_idx in 0..func.blocks.len() {
                let block = Block::from_usize(block_idx);
                let insts = func.blocks[block].insts.clone();
                let mut rebuilt = Vec::with_capacity(insts.len());
                let mut expanded = false;
                for inst in insts {
                    let call = match &func.insts[inst].kind {
                        InstKind::CallExtern { target, args }
                            if args.len() == 3 && (copies.contains(target) || sets.contains(target)) =>
                        {
                            Some((*target, args.clone()))
                        }
                        _ => None,
                    };
                    let Some((target, args)) = call else {
                        rebuilt.push(inst);
                        continue;
                    };
                    let Some(len) = const_u64(func, args[2]).filter(|len| *len > 0) else {
                        rebuilt.push(inst);
                        continue;
                    };
                    let dst = args[0];
                    let kind = if copies.contains(&target) {
                        InstKind::Memcpy {
                            dst,
                            src: args[1],
                            len,
                        }
                    } else {
                        let Some(byte) = const_u8(func, args[1]) else {
                            rebuilt.push(inst);
                            continue;
                        };
                        InstKind::Memset { dst, byte, len }
                    };
                    if let Some(result) = func.insts[inst].result {
                        func.replace_uses(result, dst);
                    }
                    let loc = func.insts[inst].loc;
                    let replacement = func.insts.push(InstData {
                        block,
                        kind,
                        result: None,
                        loc,
                    });
                    rebuilt.push(replacement);
                    expanded = true;
                    changed = true;
                }
                if expanded {
                    func.blocks[block].insts = rebuilt;
                }
            }
        }
        changed
    }
}

fn const_u64(func: &Function, value: Value) -> Option<u64> {
    let (_, value) = int_const(func, value)?;
    u64::try_from(value).ok()
}

fn const_u8(func: &Function, value: Value) -> Option<u8> {
    let (_, value) = int_const(func, value)?;
    u8::try_from(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Const, Def, Function, Inst, Terminator, ValueData};
    use crate::ty::{Signature, Ty};
    use crate::verify::verify_func;

    fn push_value(func: &mut Function, block: Block, kind: InstKind, ty: Ty) -> Value {
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

    fn int_to_ptr(func: &mut Function, block: Block, value: i128, ty: Ty) -> Value {
        let c = iconst(func, 64, value);
        push_value(func, block, InstKind::IntToPtr(c), ty)
    }

    fn push_call(func: &mut Function, block: Block, target: crate::ir::Extern, args: &[Value], ty: Ty) -> (Value, Inst) {
        let result = func.values.push(ValueData {
            ty,
            def: Def::Inst(Inst::from_usize(func.insts.len())),
        });
        let call = func.insts.push(InstData {
            block,
            kind: InstKind::CallExtern {
                target,
                args: args.iter().copied().collect(),
            },
            result: Some(result),
            loc: None,
        });
        func.values[result].def = Def::Inst(call);
        func.blocks[block].insts.push(call);
        (result, call)
    }

    fn module_with(extern_name: &str) -> (Module, crate::ir::Func, Block, crate::ir::Extern) {
        let mut module = Module::new();
        let sig = Signature::new(
            vec![Ty::ptr(Ty::int(8)), Ty::ptr(Ty::int(8)), Ty::int(64)],
            Ty::ptr(Ty::int(8)),
        );
        let ext = module.push_extern(extern_name, sig);
        let (mut func, entry) = Function::new("caller", Signature::new(vec![Ty::int(64)], Ty::Void));
        let len = func.values.push(ValueData {
            ty: Ty::int(64),
            def: Def::Param(entry, 0),
        });
        func.blocks[entry].params.push(len);
        func.blocks[entry].term = Some(Terminator::Return { val: None });
        let id = module.funcs.push(func);
        (module, id, entry, ext)
    }

    #[test]
    fn lowers_constant_memcpy() {
        let (mut module, func_id, entry, ext) = module_with("memcpy");
        let ptr_ty = Ty::ptr(Ty::int(8));
        let (result, call) = {
            let f = &mut module.funcs[func_id];
            let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
            let src = int_to_ptr(f, entry, 2, ptr_ty.clone());
            let len = iconst(f, 64, 8);
            push_call(f, entry, ext, &[dst, src, len], ptr_ty)
        };

        assert!(LowerLibcMemory.run(&mut module));
        let f = &module.funcs[func_id];
        assert!(!f.blocks[entry].insts.contains(&call));
        let kinds: Vec<_> = f
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .map(|inst| &f.insts[*inst].kind)
            .collect();
        assert!(kinds.iter().any(|kind| matches!(kind, InstKind::Memcpy { len: 8, .. })));
        assert!(!kinds.iter().any(|kind| matches!(kind, InstKind::CallExtern { .. })));
        assert_eq!(f.use_count(result), 0, "the call result is replaced by dst");
        assert!(verify_func(f).is_empty());
    }

    #[test]
    fn keeps_non_constant_memcpy() {
        let (mut module, func_id, entry, ext) = module_with("memcpy");
        let ptr_ty = Ty::ptr(Ty::int(8));
        let (_, call) = {
            let f = &mut module.funcs[func_id];
            let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
            let src = int_to_ptr(f, entry, 2, ptr_ty.clone());
            let len = f.blocks[entry].params[0];
            push_call(f, entry, ext, &[dst, src, len], ptr_ty)
        };

        assert!(!LowerLibcMemory.run(&mut module));
        assert!(matches!(
            module.funcs[func_id].insts[call].kind,
            InstKind::CallExtern { .. }
        ));
    }

    #[test]
    fn lowers_constant_memset() {
        let (mut module, func_id, entry, ext) = module_with("memset");
        let ptr_ty = Ty::ptr(Ty::int(8));
        {
            let f = &mut module.funcs[func_id];
            let dst = int_to_ptr(f, entry, 1, ptr_ty.clone());
            let byte = iconst(f, 32, 0xab);
            let len = iconst(f, 64, 16);
            push_call(f, entry, ext, &[dst, byte, len], ptr_ty);
        }

        assert!(LowerLibcMemory.run(&mut module));
        let f = &module.funcs[func_id];
        let kinds: Vec<_> = f
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .map(|inst| &f.insts[*inst].kind)
            .collect();
        assert!(kinds
            .iter()
            .any(|kind| matches!(kind, InstKind::Memset { byte: 0xab, len: 16, .. })));
        assert!(!kinds.iter().any(|kind| matches!(kind, InstKind::CallExtern { .. })));
        assert!(verify_func(f).is_empty());
    }
}
