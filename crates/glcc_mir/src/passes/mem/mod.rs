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
                        InstKind::CallExtern { target, args } if args.len() == 3 && (copies.contains(target) || sets.contains(target)) => Some((*target, args.clone())),
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
                        InstKind::Memcpy { dst, src: args[1], len }
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
                    let replacement = func.insts.push(InstData { block, kind, result: None, loc });
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
mod tests;

