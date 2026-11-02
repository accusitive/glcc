
use cranelift_jit::{JITBuilder, JITModule};

use glcc_mir as mir;

use super::inst::dense_switch_span;
use super::lower_module;

#[test]
fn dense_switch_span_fills_small_gaps() {
    assert!(dense_switch_span(0, 185, 184));
    assert!(dense_switch_span(0, 9, 3));
    assert!(!dense_switch_span(0, 1000, 2));
    assert!(!dense_switch_span(0, 4096, 1));
}

fn param(mir_func: &mut mir::Function, block: mir::Block, ty: mir::Ty, index: u32) -> mir::Value {
    let v = mir_func.values.push(mir::ValueData {
        ty,
        def: mir::Def::Param(block, index),
    });
    mir_func.blocks[block].params.push(v);
    v
}

fn iconst(mir_func: &mut mir::Function, bits: u32, value: i128) -> mir::Value {
    mir_func.push_const(mir::Const::Int { bits, value })
}

fn emit(mir_func: &mut mir::Function, block: mir::Block, kind: mir::InstKind, ty: mir::Ty) -> mir::Value {
    let result = mir_func.values.push(mir::ValueData {
        ty,
        def: mir::Def::Inst(mir::Inst::from_usize(mir_func.insts.len())),
    });
    let inst = mir_func.insts.push(mir::InstData {
        block,
        kind,
        result: Some(result),
        loc: None,
    });
    mir_func.values[result].def = mir::Def::Inst(inst);
    mir_func.blocks[block].insts.push(inst);
    result
}

fn emit_unit(mir_func: &mut mir::Function, block: mir::Block, kind: mir::InstKind) {
    let inst = mir_func.insts.push(mir::InstData {
        block,
        kind,
        result: None,
        loc: None,
    });
    mir_func.blocks[block].insts.push(inst);
}

fn ret(mir_func: &mut mir::Function, block: mir::Block, val: Option<mir::Value>) {
    mir_func.blocks[block].term = Some(mir::Terminator::Return { val });
}

fn jit_module(mir_module: &mir::Module) -> (JITModule, std::collections::HashMap<mir::Func, cranelift_module::FuncId>) {
    jit_module_with(mir_module, &[])
}

/// [`jit_module`] plus extra runtime symbols (soft-float trampolines,
/// getret shims) resolved to their addresses before lowering.
fn jit_module_with(mir_module: &mir::Module, symbols: &[(&'static str, *const u8)]) -> (JITModule, std::collections::HashMap<mir::Func, cranelift_module::FuncId>) {
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("jit builder");
    for (name, addr) in symbols {
        builder.symbol(*name, *addr);
    }
    let mut jit = JITModule::new(builder);
    let ids = lower_module(mir_module, &mut jit, crate::abi::Target::native());
    jit.finalize_definitions().expect("jit finalize");
    (jit, ids)
}

fn func_addr(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize) -> *const u8 {
    let addr = jit.get_finalized_function(ids[&mir::Func::from_usize(index)]);
    assert!(!addr.is_null(), "missing jit function {index}");
    addr
}

fn run_i32(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: i32) -> i32 {
    unsafe {
        let func: extern "C" fn(i32) -> i32 = std::mem::transmute(func_addr(jit, ids, index));
        func(arg)
    }
}

fn run_i32_0(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize) -> i32 {
    unsafe {
        let func: extern "C" fn() -> i32 = std::mem::transmute(func_addr(jit, ids, index));
        func()
    }
}

fn run_i32_2(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, a: i32, b: i32) -> i32 {
    unsafe {
        let func: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(func_addr(jit, ids, index));
        func(a, b)
    }
}

fn run_f64(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: f64) -> f64 {
    unsafe {
        let func: extern "C" fn(f64) -> f64 = std::mem::transmute(func_addr(jit, ids, index));
        func(arg)
    }
}

fn run_u32(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, arg: u32) -> u32 {
    unsafe {
        let func: extern "C" fn(u32) -> u32 = std::mem::transmute(func_addr(jit, ids, index));
        func(arg)
    }
}

fn i32_sig(params: usize) -> mir::Signature {
    mir::Signature::new(vec![mir::Ty::int(32); params], mir::Ty::int(32))
}

#[test]
fn scalar_arithmetic_executes() {
    // f(x) = x * 2 + 1.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let x = param(&mut f, entry, mir::Ty::int(32), 0);
    let two = iconst(&mut f, 32, 2);
    let one = iconst(&mut f, 32, 1);
    let doubled = emit(&mut f, entry, mir::InstKind::Mul(x, two), mir::Ty::int(32));
    let result = emit(&mut f, entry, mir::InstKind::Add(doubled, one), mir::Ty::int(32));
    ret(&mut f, entry, Some(result));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 0, 20), 41);
}

/// `f(base, i)` = `gep(base, steps(i)) - base` in bytes, with `i`
/// feeding every [`mir::GepIndex::Index`] step. Pins the byte stride
/// each step form advances against the result type `gep_step` walks.
/// The base is never dereferenced, so any aligned address works.
fn run_gep_offset(pointee: mir::Ty, steps: impl FnOnce(mir::Value) -> Vec<mir::GepIndex>, idx_arg: i64) -> i64 {
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::ptr(pointee.clone()), mir::Ty::int(64)], mir::Ty::int(64));
    let (mut f, entry) = mir::Function::new("gepoff", sig);
    let base = param(&mut f, entry, mir::Ty::ptr(pointee.clone()), 0);
    let idx = param(&mut f, entry, mir::Ty::int(64), 1);
    let indices = steps(idx);
    let walked = indices.iter().fold(pointee, |ty, step| mir::Signature::gep_step(&ty, step).expect("gep step walks the pointee"));
    let addr = emit(
        &mut f,
        entry,
        mir::InstKind::GetElementPtr {
            base,
            indices: indices.into_iter().collect(),
        },
        mir::Ty::ptr(walked),
    );
    let base_int = emit(&mut f, entry, mir::InstKind::PtrToInt(base), mir::Ty::int(64));
    let addr_int = emit(&mut f, entry, mir::InstKind::PtrToInt(addr), mir::Ty::int(64));
    let diff = emit(&mut f, entry, mir::InstKind::Sub(addr_int, base_int), mir::Ty::int(64));
    ret(&mut f, entry, Some(diff));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    let buf = [0u8; 64];
    unsafe {
        let func: extern "C" fn(*const u8, i64) -> i64 = std::mem::transmute(func_addr(&jit, &ids, 0));
        func(buf.as_ptr(), idx_arg)
    }
}

#[test]
fn gep_step_strides_execute() {
    let row = mir::Ty::Array { elem: Box::new(mir::Ty::int(32)), len: 3 };
    let grid = mir::Ty::Array { elem: Box::new(row.clone()), len: 2 };
    // Index/ConstIndex over an array pointee strides by the ELEMENT
    // size: it indexes the elements of the pointed-to array,
    // including backwards (dynamic indices are sign-extended).
    assert_eq!(run_gep_offset(row.clone(), |_| vec![mir::GepIndex::ConstIndex(2)], 0), 8);
    assert_eq!(run_gep_offset(row.clone(), |i| vec![mir::GepIndex::Index(i)], 5), 20);
    assert_eq!(run_gep_offset(row.clone(), |i| vec![mir::GepIndex::Index(i)], -1), -4);
    // A row step (+12 per row) is spelled on the UNdecayed array
    // base, where the stepped element is the row itself: the end
    // pointer of `int m[2][3]` is `m + 2` at +24, never
    // +2 * sizeof(i32).
    assert_eq!(run_gep_offset(grid.clone(), |_| vec![mir::GepIndex::ConstIndex(2)], 0), 24);
    assert_eq!(run_gep_offset(grid.clone(), |i| vec![mir::GepIndex::Index(i)], 2), 24);
    // Nested subscript `m[i][j]`: one step per dimension.
    assert_eq!(run_gep_offset(grid.clone(), |i| vec![mir::GepIndex::Index(i), mir::GepIndex::ConstIndex(2)], 1), 20);
    // Scalar pointees stride by the pointee size.
    assert_eq!(run_gep_offset(mir::Ty::int(32), |i| vec![mir::GepIndex::Index(i)], 4), 16);
    assert_eq!(run_gep_offset(mir::Ty::ptr(mir::Ty::int(32)), |_| vec![mir::GepIndex::ConstIndex(3)], 0), 24);
}

#[test]
fn sqrt_executes() {
    // sqrt(x) on f32/f64 lowers to the Clif sqrt instruction.
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::Float(mir::FloatKind::F64)], mir::Ty::Float(mir::FloatKind::F64));
    let (mut f, entry) = mir::Function::new("dsqrt", sig);
    let x = param(&mut f, entry, mir::Ty::Float(mir::FloatKind::F64), 0);
    let r = emit(&mut f, entry, mir::InstKind::Sqrt(x), mir::Ty::Float(mir::FloatKind::F64));
    ret(&mut f, entry, Some(r));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_f64(&jit, &ids, 0, 25.0), 5.0);
    assert_eq!(run_f64(&jit, &ids, 0, 2.0), 2.0f64.sqrt());
}

#[test]
fn icmp_and_select_execute() {
    // max(a, b) via signed compare and select.
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(32), mir::Ty::int(32)], mir::Ty::int(32));
    let (mut f, entry) = mir::Function::new("max", sig);
    let a = param(&mut f, entry, mir::Ty::int(32), 0);
    let b = param(&mut f, entry, mir::Ty::int(32), 1);
    let gt = emit(
        &mut f,
        entry,
        mir::InstKind::Icmp {
            pred: mir::IcmpPred::Sgt,
            lhs: a,
            rhs: b,
        },
        mir::Ty::Bool,
    );
    let result = emit(
        &mut f,
        entry,
        mir::InstKind::Select {
            cond: gt,
            then_val: a,
            else_val: b,
        },
        mir::Ty::int(32),
    );
    ret(&mut f, entry, Some(result));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32_2(&jit, &ids, 0, 3, 7), 7);
    assert_eq!(run_i32_2(&jit, &ids, 0, 9, 2), 9);
}

#[test]
fn struct_gep_store_load_executes() {
    // Point { x, y } in a slot; return x + y.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("point", mir::Signature::new(vec![], mir::Ty::int(32)));
    let point_ty = mir::Ty::Struct {
        fields: vec![mir::Ty::int(32), mir::Ty::int(32)],
        offsets: vec![0, 4],
        size: 8,
    };
    let slot = emit(
        &mut f,
        entry,
        mir::InstKind::Alloca {
            pointee: point_ty.clone(),
            count: 1,
        },
        mir::Ty::ptr(point_ty.clone()),
    );
    let six = iconst(&mut f, 32, 6);
    let seven = iconst(&mut f, 32, 7);
    for (field, val) in [(0u32, six), (1u32, seven)] {
        let addr = emit(
            &mut f,
            entry,
            mir::InstKind::GetElementPtr {
                base: slot,
                indices: [mir::GepIndex::Field(field)].into_iter().collect(),
            },
            mir::Ty::ptr(mir::Ty::int(32)),
        );
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr,
                val,
                flags: mir::MemFlags::plain(),
            },
        );
    }
    let mut loaded = Vec::new();
    for field in [0u32, 1u32] {
        let addr = emit(
            &mut f,
            entry,
            mir::InstKind::GetElementPtr {
                base: slot,
                indices: [mir::GepIndex::Field(field)].into_iter().collect(),
            },
            mir::Ty::ptr(mir::Ty::int(32)),
        );
        loaded.push(emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr,
                flags: mir::MemFlags::plain(),
            },
            mir::Ty::int(32),
        ));
    }
    let sum = emit(&mut f, entry, mir::InstKind::Add(loaded[0], loaded[1]), mir::Ty::int(32));
    ret(&mut f, entry, Some(sum));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32_0(&jit, &ids, 0), 13);
}

#[test]
fn call_between_functions_executes() {
    let mut module = mir::Module::new();
    let (mut double, dentry) = mir::Function::new("double", i32_sig(1));
    let x = param(&mut double, dentry, mir::Ty::int(32), 0);
    let two = iconst(&mut double, 32, 2);
    let result = emit(&mut double, dentry, mir::InstKind::Mul(x, two), mir::Ty::int(32));
    ret(&mut double, dentry, Some(result));
    let callee = module.push_func(double);

    let (mut caller, centry) = mir::Function::new("caller", i32_sig(1));
    let arg = param(&mut caller, centry, mir::Ty::int(32), 0);
    let call_ty = mir::Ty::int(32);
    let call_result = mir::Function::new("tmp", i32_sig(0));
    let _ = call_result;
    let call_val = emit(
        &mut caller,
        centry,
        mir::InstKind::Call {
            callee,
            args: [arg].into_iter().collect(),
        },
        call_ty,
    );
    ret(&mut caller, centry, Some(call_val));
    module.push_func(caller);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 1, 21), 42);
}

#[test]
fn global_load_executes() {
    let mut module = mir::Module::new();
    let g = module.push_global("g", mir::Ty::int(32), Some(mir::Const::Int { bits: 32, value: 5 }));
    let (mut f, entry) = mir::Function::new("f", mir::Signature::new(vec![], mir::Ty::int(32)));
    let addr = emit(&mut f, entry, mir::InstKind::GlobalAddr { global: g }, mir::Ty::ptr(mir::Ty::int(32)));
    let loaded = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr,
            flags: mir::MemFlags::plain(),
        },
        mir::Ty::int(32),
    );
    let one = iconst(&mut f, 32, 1);
    let result = emit(&mut f, entry, mir::InstKind::Add(loaded, one), mir::Ty::int(32));
    ret(&mut f, entry, Some(result));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32_0(&jit, &ids, 0), 6);
}

#[test]
fn switch_executes() {
    // f(x): 0 -> 10, 1 -> 20, otherwise 30.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let x = param(&mut f, entry, mir::Ty::int(32), 0);
    let b0 = f.push_block(vec![]);
    let b1 = f.push_block(vec![]);
    let bdef = f.push_block(vec![]);
    f.blocks[entry].term = Some(mir::Terminator::Switch {
        scrutinee: x,
        cases: vec![(0, b0), (1, b1)],
        default: bdef,
    });
    for (block, val) in [(b0, 10), (b1, 20), (bdef, 30)] {
        let c = iconst(&mut f, 32, val);
        f.blocks[block].term = Some(mir::Terminator::Return { val: Some(c) });
    }
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 0, 0), 10);
    assert_eq!(run_i32(&jit, &ids, 0, 1), 20);
    assert_eq!(run_i32(&jit, &ids, 0, 99), 30);
}

#[test]
fn large_copies_call_libc_memcpy() {
    use cranelift_codegen::ir::Opcode;

    fn opcodes(len: u64) -> Vec<Opcode> {
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::Void), mir::Ty::ptr(mir::Ty::Void)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("copy", sig);
        let dst = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 0);
        let src = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 1);
        emit_unit(&mut f, entry, mir::InstKind::Memcpy { dst, src, len });
        ret(&mut f, entry, None);
        module.push_func(f);

        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, crate::abi::Target::native());
        let func = &sink.funcs[0];
        func.layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect()
    }

    let large = opcodes(1024);
    assert!(large.contains(&Opcode::Call), "{large:?}");
    assert!(!large.contains(&Opcode::Load), "{large:?}");

    let small = opcodes(64);
    assert!(!small.contains(&Opcode::Call), "{small:?}");
    assert!(small.contains(&Opcode::Load), "{small:?}");
}

#[test]
fn volatile_aggregate_copies_stay_inline() {
    use cranelift_codegen::ir::Opcode;

    let mut module = mir::Module::new();
    let ty = mir::Ty::Array {
        elem: Box::new(mir::Ty::int(8)),
        len: 1024,
    };
    let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::Void), mir::Ty::ptr(mir::Ty::Void)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("copy", sig);
    let dst = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 0);
    let src = param(&mut f, entry, mir::Ty::ptr(mir::Ty::Void), 1);
    let value = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: src,
            flags: mir::MemFlags::vol(),
        },
        ty,
    );
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: dst,
            val: value,
            flags: mir::MemFlags::vol(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);

    let mut sink = RecordingModule::new();
    lower_module(&module, &mut sink, crate::abi::Target::native());
    let func = &sink.funcs[0];
    let opcodes: Vec<_> = func
        .layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
        .map(|inst| func.dfg.insts[inst].opcode())
        .collect();
    assert!(!opcodes.contains(&Opcode::Call), "{opcodes:?}");
    assert!(opcodes.contains(&Opcode::Load), "{opcodes:?}");
}

#[test]
fn non_variadic_extern_calls_are_direct() {
    use cranelift_codegen::ir::Opcode;

    let mut module = mir::Module::new();
    let ext = module.push_extern("ext", mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32)));
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let arg = param(&mut f, entry, mir::Ty::int(32), 0);
    let value = emit(
        &mut f,
        entry,
        mir::InstKind::CallExtern {
            target: ext,
            args: [arg].into_iter().collect(),
        },
        mir::Ty::int(32),
    );
    ret(&mut f, entry, Some(value));
    module.push_func(f);

    let mut sink = RecordingModule::new();
    lower_module(&module, &mut sink, crate::abi::Target::native());
    let func = &sink.funcs[0];
    let opcodes: Vec<_> = func
        .layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
        .map(|inst| func.dfg.insts[inst].opcode())
        .collect();
    assert!(opcodes.contains(&Opcode::Call), "{opcodes:?}");
    assert!(!opcodes.contains(&Opcode::FuncAddr), "{opcodes:?}");
}

#[test]
fn variadic_extern_calls_stay_indirect_off_x86() {
    use cranelift_codegen::ir::Opcode;

    let mut module = mir::Module::new();
    let mut sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::int(8))], mir::Ty::int(32));
    sig.variadic = true;
    let ext = module.push_extern("printf", sig);
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let fmt = param(&mut f, entry, mir::Ty::ptr(mir::Ty::int(8)), 0);
    let one = iconst(&mut f, 32, 1);
    let value = emit(
        &mut f,
        entry,
        mir::InstKind::CallExtern {
            target: ext,
            args: [fmt, one].into_iter().collect(),
        },
        mir::Ty::int(32),
    );
    ret(&mut f, entry, Some(value));
    module.push_func(f);

    // The declared import has one named parameter, but the actuals
    // outnumber it, so the call must use its own signature. Only
    // x86-64 has the `%al` trampoline to route that through; other
    // targets call the address directly.
    for triple in ["aarch64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"] {
        let target = crate::abi::Target::from_triple(triple).expect("target");
        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, target);
        let func = &sink.funcs[0];
        let opcodes: Vec<_> = func
            .layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect();
        assert!(opcodes.contains(&Opcode::FuncAddr), "{triple}: {opcodes:?}");
        assert!(opcodes.contains(&Opcode::CallIndirect), "{triple}: {opcodes:?}");
        assert!(!opcodes.contains(&Opcode::Call), "{triple}: {opcodes:?}");
    }
}

#[test]
fn dense_switches_lower_to_a_jump_table() {
    use cranelift_codegen::ir::Opcode;

    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let x = param(&mut f, entry, mir::Ty::int(32), 0);
    let mut cases = Vec::new();
    let default = f.push_block(vec![]);
    for value in 0..4 {
        let block = f.push_block(vec![]);
        cases.push((value, block));
        let c = iconst(&mut f, 32, 10 + value as i128);
        f.blocks[block].term = Some(mir::Terminator::Return { val: Some(c) });
    }
    f.blocks[entry].term = Some(mir::Terminator::Switch { scrutinee: x, cases, default });
    let fallback = iconst(&mut f, 32, 0);
    f.blocks[default].term = Some(mir::Terminator::Return { val: Some(fallback) });
    module.push_func(f);

    let mut sink = RecordingModule::new();
    lower_module(&module, &mut sink, crate::abi::Target::native());
    let func = &sink.funcs[0];
    let opcodes: Vec<_> = func
        .layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
        .map(|inst| func.dfg.insts[inst].opcode())
        .collect();
    assert!(opcodes.contains(&Opcode::BrTable), "{opcodes:?}");
}

#[test]
fn condbr_loop_executes() {
    // Sum 1..=n with an explicit loop over block params.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let n = param(&mut f, entry, mir::Ty::int(32), 0);
    let header = f.push_block(vec![mir::Ty::int(32), mir::Ty::int(32)]);
    let body = f.push_block(vec![]);
    let exit = f.push_block(vec![mir::Ty::int(32)]);
    let zero = iconst(&mut f, 32, 0);
    let one_entry = iconst(&mut f, 32, 1);
    f.blocks[entry].term = Some(mir::Terminator::Br {
        dest: header,
        args: [one_entry, zero].into_iter().collect(),
    });
    let i = f.blocks[header].params[0];
    let acc = f.blocks[header].params[1];
    let done = emit(
        &mut f,
        header,
        mir::InstKind::Icmp {
            pred: mir::IcmpPred::Sgt,
            lhs: i,
            rhs: n,
        },
        mir::Ty::Bool,
    );
    f.blocks[header].term = Some(mir::Terminator::CondBr {
        cond: done,
        then_block: exit,
        else_block: body,
        then_args: [acc].into_iter().collect(),
        else_args: Default::default(),
    });
    let one = iconst(&mut f, 32, 1);
    let acc_next = emit(&mut f, body, mir::InstKind::Add(acc, i), mir::Ty::int(32));
    let i_next = emit(&mut f, body, mir::InstKind::Add(i, one), mir::Ty::int(32));
    f.blocks[body].term = Some(mir::Terminator::Br {
        dest: header,
        args: [i_next, acc_next].into_iter().collect(),
    });
    f.blocks[exit].term = Some(mir::Terminator::Return {
        val: Some(f.blocks[exit].params[0]),
    });
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 0, 10), 55);
}

#[test]
fn shared_constants_materialize_per_block() {
    // One constant used in sibling branches: a single definition
    // in either branch would not dominate the other's use, so
    // every block rebuilds its own copy.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(1));
    let p = param(&mut f, entry, mir::Ty::int(32), 0);
    let then_block = f.push_block(vec![]);
    let else_block = f.push_block(vec![]);
    let merge = f.push_block(vec![mir::Ty::int(32)]);
    let zero = iconst(&mut f, 32, 0);
    let nonzero = emit(
        &mut f,
        entry,
        mir::InstKind::Icmp {
            pred: mir::IcmpPred::Ne,
            lhs: p,
            rhs: zero,
        },
        mir::Ty::Bool,
    );
    f.blocks[entry].term = Some(mir::Terminator::CondBr {
        cond: nonzero,
        then_block,
        else_block,
        then_args: Default::default(),
        else_args: Default::default(),
    });
    let seven = iconst(&mut f, 32, 7);
    f.blocks[then_block].term = Some(mir::Terminator::Br {
        dest: merge,
        args: [seven].into_iter().collect(),
    });
    f.blocks[else_block].term = Some(mir::Terminator::Br {
        dest: merge,
        args: [seven].into_iter().collect(),
    });
    f.blocks[merge].term = Some(mir::Terminator::Return {
        val: Some(f.blocks[merge].params[0]),
    });
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 0, 1), 7);
    assert_eq!(run_i32(&jit, &ids, 0, 0), 7);
}

#[test]
fn definitions_may_live_in_later_indexed_dominators() {
    // `late` loads a value `mid` defines, and `mid` has the higher
    // arena index: filling in arena order would use the value
    // before its definition.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", i32_sig(0));
    let late = f.push_block(vec![]);
    let mid = f.push_block(vec![]);
    f.blocks[entry].term = Some(mir::Terminator::Br {
        dest: mid,
        args: Default::default(),
    });
    let two = iconst(&mut f, 32, 2);
    let forty = iconst(&mut f, 32, 40);
    let value = emit(&mut f, mid, mir::InstKind::Add(two, forty), mir::Ty::int(32));
    f.blocks[mid].term = Some(mir::Terminator::Br {
        dest: late,
        args: Default::default(),
    });
    f.blocks[late].term = Some(mir::Terminator::Return { val: Some(value) });
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32_0(&jit, &ids, 0), 42);
}

#[test]
fn undef_lowers_to_zero_for_floats_and_wide_ints() {
    // Undef has no bits, and `iconst` covers neither f64 nor i128,
    // so both need their own zero forms.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("returns_undef", mir::Signature::new(vec![], mir::Ty::Float(mir::FloatKind::F64)));
    let undef = f.values.push(mir::ValueData {
        ty: mir::Ty::Float(mir::FloatKind::F64),
        def: mir::Def::Const(mir::Const::Undef),
    });
    ret(&mut f, entry, Some(undef));
    module.push_func(f);

    let (mut g, entry) = mir::Function::new("wide_undef", i32_sig(0));
    let wide_undef = g.values.push(mir::ValueData {
        ty: mir::Ty::int(128),
        def: mir::Def::Const(mir::Const::Undef),
    });
    let one = iconst(&mut g, 128, 1);
    let sum = emit(&mut g, entry, mir::InstKind::Add(wide_undef, one), mir::Ty::int(128));
    let low = emit(&mut g, entry, mir::InstKind::Trunc(sum), mir::Ty::int(32));
    ret(&mut g, entry, Some(low));
    module.push_func(g);

    let (jit, ids) = jit_module(&module);
    let addr = func_addr(&jit, &ids, 0);
    let returns_undef: extern "C" fn() -> f64 = unsafe { std::mem::transmute(addr) };
    assert_eq!(returns_undef(), 0.0);
    assert_eq!(run_i32_0(&jit, &ids, 1), 1);
}

#[test]
fn float_converts_execute() {
    // f(x: i32) = sftoi(sitof(x) + 0.5).
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32));
    let (mut f, entry) = mir::Function::new("f", sig);
    let x = param(&mut f, entry, mir::Ty::int(32), 0);
    let as_f = emit(&mut f, entry, mir::InstKind::SItoF(x), mir::Ty::Float(mir::FloatKind::F64));
    let half = f.push_const(mir::Const::Float {
        kind: mir::FloatKind::F64,
        bits: 0x3FE0000000000000,
    });
    let bumped = emit(&mut f, entry, mir::InstKind::FAdd(as_f, half), mir::Ty::Float(mir::FloatKind::F64));
    let back = emit(&mut f, entry, mir::InstKind::SFtoI(bumped), mir::Ty::int(32));
    ret(&mut f, entry, Some(back));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32(&jit, &ids, 0, 3), 3);

    // Unsigned roundtrip preserves large values a signed path would not.
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(32)], mir::Ty::int(32));
    let (mut f, entry) = mir::Function::new("u", sig);
    let x = param(&mut f, entry, mir::Ty::int(32), 0);
    let as_f = emit(&mut f, entry, mir::InstKind::UItoF(x), mir::Ty::Float(mir::FloatKind::F64));
    let back = emit(&mut f, entry, mir::InstKind::UFtoI(as_f), mir::Ty::int(32));
    ret(&mut f, entry, Some(back));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_u32(&jit, &ids, 0, 4000000000), 4000000000);
}

#[test]
fn float_converts_to_narrow_ints() {
    // Sub-word destinations convert at 32 bits and narrow: x64 has
    // no 16-bit float conversion.
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("to_u16", i32_sig(0));
    let v = f.push_const(mir::Const::Float {
        kind: mir::FloatKind::F64,
        bits: u128::from(5.25f64.to_bits()),
    });
    let conv = emit(&mut f, entry, mir::InstKind::UFtoI(v), mir::Ty::int(16));
    let wide = emit(&mut f, entry, mir::InstKind::Zext(conv), mir::Ty::int(32));
    ret(&mut f, entry, Some(wide));
    module.push_func(f);

    let (mut g, entry) = mir::Function::new("to_i16", i32_sig(0));
    let v = g.push_const(mir::Const::Float {
        kind: mir::FloatKind::F64,
        bits: u128::from((-7.9f64).to_bits()),
    });
    let conv = emit(&mut g, entry, mir::InstKind::SFtoI(v), mir::Ty::int(16));
    let wide = emit(&mut g, entry, mir::InstKind::Sext(conv), mir::Ty::int(32));
    ret(&mut g, entry, Some(wide));
    module.push_func(g);

    let (jit, ids) = jit_module(&module);
    assert_eq!(run_i32_0(&jit, &ids, 0), 5);
    assert_eq!(run_i32_0(&jit, &ids, 1), -7);
}

#[test]
fn conversions_within_one_clif_width_respect_mir_bits() {
    // `Bool` and `i8` are both `i8` in Clif, so a `zext`/`sext`/`trunc`
    // between them has no Cranelift width change even though the MIR
    // widths differ.
    let mut module = mir::Module::new();

    let (mut f, entry) = mir::Function::new("z", mir::Signature::new(vec![mir::Ty::Bool], mir::Ty::int(8)));
    let x = param(&mut f, entry, mir::Ty::Bool, 0);
    let z = emit(&mut f, entry, mir::InstKind::Zext(x), mir::Ty::int(8));
    ret(&mut f, entry, Some(z));
    module.push_func(f);

    let (mut f, entry) = mir::Function::new("s", mir::Signature::new(vec![mir::Ty::Bool], mir::Ty::int(8)));
    let x = param(&mut f, entry, mir::Ty::Bool, 0);
    let s = emit(&mut f, entry, mir::InstKind::Sext(x), mir::Ty::int(8));
    ret(&mut f, entry, Some(s));
    module.push_func(f);

    let (mut f, entry) = mir::Function::new("t", mir::Signature::new(vec![mir::Ty::int(8)], mir::Ty::Bool));
    let x = param(&mut f, entry, mir::Ty::int(8), 0);
    let t = emit(&mut f, entry, mir::InstKind::Trunc(x), mir::Ty::Bool);
    ret(&mut f, entry, Some(t));
    module.push_func(f);

    // A Clif width change still starts from the MIR width: `sext`
    // from a one-bit boolean must produce -1, not 1.
    let (mut f, entry) = mir::Function::new("s32", mir::Signature::new(vec![mir::Ty::Bool], mir::Ty::int(32)));
    let x = param(&mut f, entry, mir::Ty::Bool, 0);
    let s = emit(&mut f, entry, mir::InstKind::Sext(x), mir::Ty::int(32));
    ret(&mut f, entry, Some(s));
    module.push_func(f);

    let (mut f, entry) = mir::Function::new("s64", mir::Signature::new(vec![mir::Ty::Bool], mir::Ty::int(64)));
    let x = param(&mut f, entry, mir::Ty::Bool, 0);
    let s = emit(&mut f, entry, mir::InstKind::Sext(x), mir::Ty::int(64));
    ret(&mut f, entry, Some(s));
    module.push_func(f);

    let (jit, ids) = jit_module(&module);
    let zext: extern "C" fn(i8) -> i8 = unsafe { std::mem::transmute(func_addr(&jit, &ids, 0)) };
    let sext: extern "C" fn(i8) -> i8 = unsafe { std::mem::transmute(func_addr(&jit, &ids, 1)) };
    let trunc: extern "C" fn(i8) -> i8 = unsafe { std::mem::transmute(func_addr(&jit, &ids, 2)) };
    let sext32: extern "C" fn(i8) -> i32 = unsafe { std::mem::transmute(func_addr(&jit, &ids, 3)) };
    let sext64: extern "C" fn(i8) -> i64 = unsafe { std::mem::transmute(func_addr(&jit, &ids, 4)) };
    assert_eq!((zext(0), zext(1)), (0, 1));
    assert_eq!((sext(0), sext(1)), (0, -1));
    assert_eq!((trunc(2), trunc(3)), (0, 1));
    assert_eq!((sext32(0), sext32(1)), (0, -1));
    assert_eq!((sext64(0), sext64(1)), (0, -1));
}

#[test]
fn f64_arithmetic_executes() {
    let sig = mir::Signature::new(vec![mir::Ty::Float(mir::FloatKind::F64)], mir::Ty::Float(mir::FloatKind::F64));
    let mut module = mir::Module::new();
    let (mut f, entry) = mir::Function::new("f", sig);
    let x = param(&mut f, entry, mir::Ty::Float(mir::FloatKind::F64), 0);
    let two = f.push_const(mir::Const::Float {
        kind: mir::FloatKind::F64,
        bits: 0x4000000000000000,
    });
    let result = emit(&mut f, entry, mir::InstKind::FMul(x, two), mir::Ty::Float(mir::FloatKind::F64));
    ret(&mut f, entry, Some(result));
    module.push_func(f);
    let (jit, ids) = jit_module(&module);
    assert_eq!(run_f64(&jit, &ids, 0, 2.5), 5.0);
}

/// A foreign `long double` callee returns in st(0), not xmm0: the
/// call declares no Clif return and fetches through getret. Uses
/// the host libm `fabsl` (address only; the declared prototype is
/// deliberately wrong and never called from Rust) and a
/// one-instruction x87 shim for getret (Rust never touches st(0),
/// so it survives from the foreign return intact).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn foreign_f80_return_comes_from_st0() {
    #[link(name = "m")]
    unsafe extern "C" {
        fn fabsl(x: f64) -> f64;
    }
    // Pops the live st(0) into memory: the exact getret contract.
    unsafe extern "C" fn getret_shim(out: *mut u8) {
        unsafe {
            core::arch::asm!(
                "fstpt (%rdi)",
                in("rdi") out,
                options(nostack, preserves_flags, att_syntax),
            );
        }
    }
    let f80 = mir::Ty::Float(mir::FloatKind::F80);
    let mut module = mir::Module::new();
    let ext = module.push_extern("fabsl", mir::Signature::new(vec![f80.clone()], f80.clone()));
    let (mut run, entry) = mir::Function::new("run", mir::Signature::new(vec![mir::Ty::int(64)], mir::Ty::Void));
    let out = param(&mut run, entry, mir::Ty::int(64), 0);
    // -2.5L: sign + exp 0x4000 over mantissa 0xA000....
    let arg = run.push_const(mir::Const::Float {
        kind: mir::FloatKind::F80,
        bits: 0xC000_A000_0000_0000_0000u128,
    });
    let got = emit(
        &mut run,
        entry,
        mir::InstKind::CallExtern {
            target: ext,
            args: [arg].into_iter().collect(),
        },
        f80,
    );
    emit_unit(
        &mut run,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: got,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut run, entry, None);
    module.push_func(run);
    let (jit, ids) = jit_module_with(
        &module,
        &[
            ("fabsl", fabsl as *const () as usize as *const u8),
            ("__glccrt_f80_getret", getret_shim as *const () as usize as *const u8),
        ],
    );
    let addr = func_addr(&jit, &ids, 0);
    let mut buf = [0u8; 16];
    unsafe {
        let run_fn: extern "C" fn(*mut u8) = std::mem::transmute(addr);
        run_fn(buf.as_mut_ptr());
    }
    // +2.5L low 10 bytes; the carrier's pad is indeterminate.
    assert_eq!(&buf[..10], &[0, 0, 0, 0, 0, 0, 0, 0xA0, 0x00, 0x40], "fabsl(-2.5L) retrieved from st(0)");
}

/// Quad arithmetic lowers to the toolchain trampolines on
/// AArch64/RISC-V (same out-pointer convention as the x87
/// helpers): object emission succeeds and references the symbols.
#[test]
fn f128_arith_calls_quad_trampolines() {
    use cranelift_codegen::settings::Configurable;
    let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str)] = &[
        (mir::InstKind::FAdd, "__glcc_quad_add"),
        (mir::InstKind::FSub, "__glcc_quad_sub"),
        (mir::InstKind::FMul, "__glcc_quad_mul"),
        (mir::InstKind::FDiv, "__glcc_quad_div"),
    ];
    for triple in ["aarch64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"] {
        for (op, helper) in cases {
            let f128 = mir::Ty::Float(mir::FloatKind::F128);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![f128.clone(), f128.clone()], f128.clone());
            let (mut f, entry) = mir::Function::new("f", sig);
            let a = param(&mut f, entry, f128.clone(), 0);
            let b = param(&mut f, entry, f128.clone(), 1);
            let sum = emit(&mut f, entry, op(a, b), f128.clone());
            ret(&mut f, entry, Some(sum));
            module.push_func(f);
            let mut flag_builder = cranelift_codegen::settings::builder();
            flag_builder.set("opt_level", "none").expect("flag");
            flag_builder.set("is_pic", "true").expect("flag");
            let flags = cranelift_codegen::settings::Flags::new(flag_builder);
            let isa = cranelift_codegen::isa::lookup(triple.parse().expect("triple"))
                .expect("isa")
                .finish(flags)
                .expect("isa flags");
            let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
            let mut obj = cranelift_object::ObjectModule::new(obj_builder);
            let target = crate::abi::Target::from_triple(triple).expect("target");
            lower_module(&module, &mut obj, target);
            let bytes = obj.finish().emit().expect("object emit");
            assert!(bytes.windows(helper.len()).any(|w| w == helper.as_bytes()), "{triple} object references {helper}");
        }
    }
}

/// x86-64 `__float128` arithmetic calls the vendored runtime directly (no
/// adapter: the carrier already rides XMM registers): object
/// emission succeeds and references the symbols.
#[test]
fn f128_arith_calls_libgcc_on_x86_64() {
    use cranelift_codegen::settings::Configurable;
    let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str)] = &[
        (mir::InstKind::FAdd, "__glcc_addtf3"),
        (mir::InstKind::FSub, "__glcc_subtf3"),
        (mir::InstKind::FMul, "__glcc_multf3"),
        (mir::InstKind::FDiv, "__glcc_divtf3"),
    ];
    for (op, helper) in cases {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![f128.clone(), f128.clone()], f128.clone());
        let (mut f, entry) = mir::Function::new("f", sig);
        let a = param(&mut f, entry, f128.clone(), 0);
        let b = param(&mut f, entry, f128.clone(), 1);
        let sum = emit(&mut f, entry, op(a, b), f128.clone());
        ret(&mut f, entry, Some(sum));
        module.push_func(f);
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "none").expect("flag");
        flag_builder.set("is_pic", "true").expect("flag");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
            .expect("isa")
            .finish(flags)
            .expect("isa flags");
        let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(obj_builder);
        let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
        lower_module(&module, &mut obj, target);
        let bytes = obj.finish().emit().expect("object emit");
        assert!(bytes.windows(helper.len()).any(|w| w == helper.as_bytes()), "x86_64 object references {helper}");
    }
}

/// Entry-param stack slots are assigned in param order:
/// `map_entry_params` collects per-owner pieces in a `BTreeMap`
/// so iteration (which creates stack slots and emits stores)
/// cannot permute incoming-register spills via a hash random
/// seed. Lowering the same module repeatedly must give identical
/// object bytes.
#[test]
fn entry_params_lower_deterministically() {
    use cranelift_codegen::settings::Configurable;
    let pair = || mir::Ty::Struct {
        fields: vec![mir::Ty::int(64), mir::Ty::int(64)],
        offsets: vec![0, 8],
        size: 16,
    };
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![pair(), pair(), pair(), pair()], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("f", sig);
    for i in 0..4 {
        param(&mut f, entry, pair(), i);
    }
    ret(&mut f, entry, None);
    module.push_func(f);
    let mut outputs = Vec::new();
    for _ in 0..8 {
        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("opt_level", "none").expect("flag");
        flag_builder.set("is_pic", "true").expect("flag");
        let flags = cranelift_codegen::settings::Flags::new(flag_builder);
        let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
            .expect("isa")
            .finish(flags)
            .expect("isa flags");
        let obj_builder = cranelift_object::ObjectBuilder::new(isa, "determtest", cranelift_module::default_libcall_names()).expect("object builder");
        let mut obj = cranelift_object::ObjectModule::new(obj_builder);
        let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
        lower_module(&module, &mut obj, target);
        outputs.push(obj.finish().emit().expect("object emit"));
    }
    for other in &outputs[1..] {
        assert_eq!(&outputs[0], other, "same MIR module lowered to different object bytes");
    }
}

/// Build `run(out, a, b)`: load an `F128` at each address, one
/// binary op, store the `F128` result at `out`.
fn qbinop_module(op: fn(mir::Value, mir::Value) -> mir::InstKind) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let a_ptr = param(&mut f, entry, mir::Ty::int(64), 1);
    let b_ptr = param(&mut f, entry, mir::Ty::int(64), 2);
    let a = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: a_ptr,
            flags: mir::MemFlags::plain(),
        },
        f128.clone(),
    );
    let b = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: b_ptr,
            flags: mir::MemFlags::plain(),
        },
        f128.clone(),
    );
    let r = emit(&mut f, entry, op(a, b), f128);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

/// x86-64 `__float128` arithmetic executes through the vendored runtime: exact
/// small cases, probed through the f64 oracles on both sides.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn f128_arith_executes_x86_64() {
    // Addresses only, borrowed from the system toolchain for the
    // JIT linker (the vendored `rt/softfloat` copies are
    // bitwise-faithful; never called from Rust, and the
    // prototypes are deliberately integer-shaped).
    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __addtf3(a: u128, b: u128) -> u128;
        fn __subtf3(a: u128, b: u128) -> u128;
        fn __multf3(a: u128, b: u128) -> u128;
        fn __divtf3(a: u128, b: u128) -> u128;
    }
    // (op, helper symbol, a as f64, b as f64, expected as f64).
    let cases: &[(fn(mir::Value, mir::Value) -> mir::InstKind, &str, f64, f64, f64)] = &[
        (mir::InstKind::FAdd, "__glcc_addtf3", 1.5, 2.5, 4.0),
        (mir::InstKind::FSub, "__glcc_subtf3", 1.0, 3.0, -2.0),
        (mir::InstKind::FMul, "__glcc_multf3", 2.0, 1.5, 3.0),
        (mir::InstKind::FDiv, "__glcc_divtf3", 7.0, 2.0, 3.5),
    ];
    for (op, helper, fa, fb, want) in cases {
        let module = qbinop_module(*op);
        let (jit, ids) = jit_module_with(
            &module,
            &[
                ("__glcc_addtf3", __addtf3 as *const () as usize as *const u8),
                ("__glcc_subtf3", __subtf3 as *const () as usize as *const u8),
                ("__glcc_multf3", __multf3 as *const () as usize as *const u8),
                ("__glcc_divtf3", __divtf3 as *const () as usize as *const u8),
            ],
        );
        let (alo, ahi) = crate::quad::ref_f64_to_quad(fa.to_bits());
        let (blo, bhi) = crate::quad::ref_f64_to_quad(fb.to_bits());
        let abuf = qbytes(alo, ahi);
        let bbuf = qbytes(blo, bhi);
        let mut out = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, *const u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(out.as_mut_ptr(), abuf.as_ptr(), bbuf.as_ptr());
        }
        assert_eq!(qhalves(&out), crate::quad::ref_f64_to_quad(want.to_bits()), "{helper} of {fa} and {fb}");
    }
}

/// `long double` <-> `__float128` widening through the vendored runtime
/// (`__extendxftf2`, exact): x87 bit patterns become their exact
/// quad halves, NaN stays NaN.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn f80_extend_executes() {
    // Address only, borrowed from the system toolchain (the
    // vendored copy is bitwise-faithful); deliberately
    // integer-shaped, never called from Rust.
    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __extendxftf2(a: u128, b: u128) -> u128;
    }
    // (F80 bits, expected quad lo/hi or `None` for NaN).
    let cases: &[(u128, Option<(u64, u64)>)] = &[
        (0x3fff_8000_0000_0000_0000, Some((0, 0x3fff_0000_0000_0000))),                          // 1.0
        (0xbfff_c000_0000_0000_0000, Some(crate::quad::ref_f64_to_quad(0xbff8_0000_0000_0000))), // -1.5
        (0x8000_0000_0000_0000_0000, Some((0, 0x8000_0000_0000_0000))),                          // -0.0
        (0x7fff_8000_0000_0000_0000, Some((0, 0x7fff_0000_0000_0000))),                          // +inf
        (0x0000_0000_0000_0000_0001, Some((0x0002_0000_0000_0000, 0))),                          // min denormal -> 2^-16445
        (0x7fff_c000_0000_0000_0000, None),                                                      // qnan
    ];
    for (bits, want) in cases {
        let _f80 = mir::Ty::Float(mir::FloatKind::F80);
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let c = f.push_const(mir::Const::Float {
            kind: mir::FloatKind::F80,
            bits: *bits,
        });
        let q = emit(&mut f, entry, mir::InstKind::Fpext(c), f128);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: q,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        let (jit, ids) = jit_module_with(&module, &[("__glcc_extendxftf2", __extendxftf2 as *const () as usize as *const u8)]);
        let mut buf = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(buf.as_mut_ptr());
        }
        match want {
            Some((elo, ehi)) => assert_eq!(qhalves(&buf), (*elo, *ehi), "extend of {bits:#x}"),
            None => {
                let (lo, hi) = qhalves(&buf);
                assert!(crate::quad::ref_quad_is_nan(lo, hi), "extend of {bits:#x} stays NaN");
            }
        }
    }
}

/// Quad -> `long double` narrowing through the vendored runtime
/// (`__trunctfxf2`, single rounding): exact values land on their
/// x87 patterns (low 10 bytes; pad indeterminate), wide quads
/// saturate to infinity, NaN stays NaN.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn f80_trunc_executes() {
    // Address only, borrowed from the system toolchain like above.
    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __trunctfxf2(a: u128) -> u128;
    }
    // Pops the live st(0): the exact getret contract (see above).
    unsafe extern "C" fn trunc_getret_shim(out: *mut u8) {
        unsafe {
            core::arch::asm!(
                "fstpt (%rdi)",
                in("rdi") out,
                options(nostack, preserves_flags, att_syntax),
            );
        }
    }
    // (quad lo/hi, expected low 10 x87 bytes or `None` for NaN).
    let cases: &[((u64, u64), Option<[u8; 10]>)] = &[
        ((0, 0x3fff_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f])), // 1.0
        (crate::quad::ref_f64_to_quad(0xbff8_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xbf])), // -1.5
        ((0, 0x3bcd_0000_0000_0000), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xcd, 0x3b])), // 2^-1074, exact in x87
        ((0xffff_ffff_ffff_ffff, 0x7ffe_ffff_ffff_ffff), Some([0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x7f])), // quad max overflows to +inf
        ((0, 0x7fff_8000_0000_0000), None),                                          // qnan
    ];
    for ((ilo, ihi), want) in cases {
        let f128 = mir::Ty::Float(mir::FloatKind::F128);
        let f80 = mir::Ty::Float(mir::FloatKind::F80);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f128.clone(),
        );
        let n = emit(&mut f, entry, mir::InstKind::Fptrunc(v), f80.clone());
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: n,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        let (jit, ids) = jit_module_with(
            &module,
            &[
                ("__glcc_trunctfxf2", __trunctfxf2 as *const () as usize as *const u8),
                ("__glccrt_f80_getret", trunc_getret_shim as *const () as usize as *const u8),
            ],
        );
        let inp = qbytes(*ilo, *ihi);
        let mut buf = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(buf.as_mut_ptr(), inp.as_ptr());
        }
        match want {
            Some(bytes) => assert_eq!(&buf[..10], bytes, "trunc of {ilo:#x}/{ihi:#x}"),
            None => {
                let exp = u16::from_le_bytes([buf[8], buf[9]]) & 0x7fff;
                let mut mantissa = [0u8; 8];
                mantissa.copy_from_slice(&buf[..8]);
                assert_eq!(exp, 0x7fff, "trunc stays infinite-exponent");
                assert_ne!(u64::from_le_bytes(mantissa), 0x8000_0000_0000_0000, "trunc stays NaN, not infinity");
            }
        }
    }
}

/// int128 -> `long double` through the vendored runtime
/// (`__glcc_floattixf`/`__glcc_floatuntixf`, single rounding):
/// exact values land on their x87 patterns (low 10 bytes; pad
/// indeterminate), `u128::MAX` rounds up to 2^128.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn f80_from_int128_executes() {
    // Addresses only, borrowed from the system toolchain like
    // above; deliberately integer-shaped, never called from Rust.
    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __floattixf(a: u128, b: u128) -> u128;
        fn __floatuntixf(a: u128, b: u128) -> u128;
    }
    // Pops the live st(0): the exact getret contract (see above).
    unsafe extern "C" fn from_getret_shim(out: *mut u8) {
        unsafe {
            core::arch::asm!(
                "fstpt (%rdi)",
                in("rdi") out,
                options(nostack, preserves_flags, att_syntax),
            );
        }
    }
    // (value, unsigned, expected low 10 x87 bytes).
    let cases: &[(u128, bool, [u8; 10])] = &[
        (1, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]),
        (1, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]),
        (u128::MAX, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0xbf]), // -1.0
        ((1 << 64) - 1, true, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x3e, 0x40]),
        (1 << 64, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x3f, 0x40]),
        (i64::MIN as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x3e, 0xc0]), // -2^63
        ((1 << 113) - 1, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x70, 0x40]),   // rounds up to 2^113
        (1 << 113, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x70, 0x40]),
        (u128::MAX, true, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7f, 0x40]),          // 2^128
        (i128::MIN as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7e, 0xc0]), // -2^127
        (i128::MAX as u128, false, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7e, 0x40]), // 2^127
    ];
    for (v, unsigned, want) in cases {
        let f80 = mir::Ty::Float(mir::FloatKind::F80);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(128)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let x = param(&mut f, entry, mir::Ty::int(128), 1);
        let kind = if *unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
        let r = emit(&mut f, entry, kind, f80);
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        let (jit, ids) = jit_module_with(
            &module,
            &[
                ("__glcc_floattixf", __floattixf as *const () as usize as *const u8),
                ("__glcc_floatuntixf", __floatuntixf as *const () as usize as *const u8),
                ("__glccrt_f80_getret", from_getret_shim as *const () as usize as *const u8),
            ],
        );
        let mut buf = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, u64, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(buf.as_mut_ptr(), *v as u64, (*v >> 64) as u64);
        }
        assert_eq!(&buf[..10], want, "from_int128({v:#x}, unsigned={unsigned})");
    }
}

/// `long double` -> int128 through the vendored runtime
/// (`__glcc_fixxfti`/`__glcc_fixunsxfti`), matching the toolchain
/// exactly (vectors probed against it, including out-of-range).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn f80_to_int128_executes() {
    // Addresses only, borrowed from the system toolchain like above.
    #[link(name = "gcc_s")]
    unsafe extern "C" {
        fn __fixxfti(a: u128) -> u128;
        fn __fixunsxfti(a: u128) -> u128;
    }
    // (F80 bits, signed expectation, unsigned expectation).
    // NaN (last row) asserts structurally below.
    let cases: &[(u128, u128, u128)] = &[
        (0x3fff_8000_0000_0000_0000, 1, 1),
        (0xbfff_c000_0000_0000_0000, u128::MAX, 0),       // -1.5
        (0x3ffe_8000_0000_0000_0000, 0, 0),               // 0.5
        (0x7fff_8000_0000_0000_0000, 0, 0),               // +inf
        (0xffff_8000_0000_0000_0000, 0, 0),               // -inf
        (0x0000_0000_0000_0000_0000, 0, 0),               // +0.0
        (0x407e_8000_0000_0000_0000, 1 << 127, 1 << 127), // +2^127
        (0xc07e_8000_0000_0000_0000, 1 << 127, 0),        // -2^127
    ];
    for unsigned in [false, true] {
        for (bits, s, u) in cases {
            let f80 = mir::Ty::Float(mir::FloatKind::F80);
            let mut module = mir::Module::new();
            let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
            let (mut f, entry) = mir::Function::new("run", sig);
            let out = param(&mut f, entry, mir::Ty::int(64), 0);
            let inp = param(&mut f, entry, mir::Ty::int(64), 1);
            let v = emit(
                &mut f,
                entry,
                mir::InstKind::Load {
                    addr: inp,
                    flags: mir::MemFlags::plain(),
                },
                f80.clone(),
            );
            let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
            let r = emit(&mut f, entry, kind, mir::Ty::int(128));
            emit_unit(
                &mut f,
                entry,
                mir::InstKind::Store {
                    addr: out,
                    val: r,
                    flags: mir::MemFlags::plain(),
                },
            );
            ret(&mut f, entry, None);
            module.push_func(f);
            let (jit, ids) = jit_module_with(
                &module,
                &[
                    ("__glcc_fixxfti", __fixxfti as *const () as usize as *const u8),
                    ("__glcc_fixunsxfti", __fixunsxfti as *const () as usize as *const u8),
                ],
            );
            let mut lo_b = [0u8; 8];
            let mut hi_b = [0u8; 8];
            lo_b.copy_from_slice(&bits.to_le_bytes()[..8]);
            hi_b.copy_from_slice(&bits.to_le_bytes()[8..]);
            let mut inp = [0u8; 16];
            inp[..8].copy_from_slice(&lo_b);
            inp[8..].copy_from_slice(&hi_b);
            let mut buf = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(buf.as_mut_ptr(), inp.as_ptr());
            }
            let want = if unsigned { *u } else { *s };
            assert_eq!(u128::from_le_bytes(buf), want, "to_int128({bits:#x}, unsigned={unsigned})");
        }
    }
    // NaN converts structurally (exact payload is toolchain detail).
    for unsigned in [false, true] {
        let bits = 0x7fff_c000_0000_0000_0000u128;
        let f80 = mir::Ty::Float(mir::FloatKind::F80);
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
        let (mut f, entry) = mir::Function::new("run", sig);
        let out = param(&mut f, entry, mir::Ty::int(64), 0);
        let inp = param(&mut f, entry, mir::Ty::int(64), 1);
        let v = emit(
            &mut f,
            entry,
            mir::InstKind::Load {
                addr: inp,
                flags: mir::MemFlags::plain(),
            },
            f80.clone(),
        );
        let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
        let r = emit(&mut f, entry, kind, mir::Ty::int(128));
        emit_unit(
            &mut f,
            entry,
            mir::InstKind::Store {
                addr: out,
                val: r,
                flags: mir::MemFlags::plain(),
            },
        );
        ret(&mut f, entry, None);
        module.push_func(f);
        let (jit, ids) = jit_module_with(
            &module,
            &[
                ("__glcc_fixxfti", __fixxfti as *const () as usize as *const u8),
                ("__glcc_fixunsxfti", __fixunsxfti as *const () as usize as *const u8),
            ],
        );
        let inp = bits.to_le_bytes();
        let mut buf = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(buf.as_mut_ptr(), inp.as_ptr());
        }
        assert_eq!(
            u128::from_le_bytes(buf),
            0x8000_0000_0000_0000_8000_0000_0000_0000u128,
            "to_int128(nan, unsigned={unsigned})"
        );
    }
}

/// Lower one `F128` op for a cross target without executing:
/// catches ISLE gaps (like `scalar_size` for stray `F128`s) that
/// pass x86-64-only episodically.
fn cross_lower(op: &str, triple: &str, build: &dyn Fn(&mut mir::Function, mir::Block) -> mir::Value) {
    use cranelift_codegen::settings::Configurable;
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![], mir::Ty::int(64));
    let (mut f, entry) = mir::Function::new("f", sig);
    let v = build(&mut f, entry);
    f.sig.ret = f.values[v].ty.clone();
    ret(&mut f, entry, Some(v));
    module.push_func(f);
    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("opt_level", "none").expect("flag");
    flag_builder.set("is_pic", "true").expect("flag");
    let flags = cranelift_codegen::settings::Flags::new(flag_builder);
    let isa = cranelift_codegen::isa::lookup(triple.parse().expect("triple"))
        .expect("isa")
        .finish(flags)
        .expect("isa flags");
    let obj_builder = cranelift_object::ObjectBuilder::new(isa, "quadtest", cranelift_module::default_libcall_names()).expect("object builder");
    let mut obj = cranelift_object::ObjectModule::new(obj_builder);
    let target = crate::abi::Target::from_triple(triple).expect("target");
    lower_module(&module, &mut obj, target);
    obj.finish().emit().unwrap_or_else(|_| panic!("{op} lowers on {triple}"));
}

/// `FNeg`/`Fcmp`/conversions on `F128` inline as integer IR, so
/// they lower on every target (only arithmetic needs trampolines).
#[test]
fn f128_nonarith_lowers_cross() {
    use mir::FcmpPred::*;
    let one = 0x3fff_0000_0000_0000u64;
    for triple in ["aarch64-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"] {
        // Negation of 1.0.
        cross_lower("fneg", triple, &|f, entry| {
            let c = f.push_const(mir::Const::Float {
                kind: mir::FloatKind::F128,
                bits: (one as u128) << 64,
            });
            let n = emit(f, entry, mir::InstKind::FNeg(c), mir::Ty::Float(mir::FloatKind::F128));
            emit(f, entry, mir::InstKind::SFtoI(n), mir::Ty::int(32))
        });
        // Each comparison against itself.
        for pred in [Eq, Ne, Lt, Le, Gt, Ge, Uno, Ord] {
            cross_lower("fcmp", triple, &|f, entry| {
                let c = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F128,
                    bits: (one as u128) << 64,
                });
                let t = emit(f, entry, mir::InstKind::Fcmp { pred, lhs: c, rhs: c }, mir::Ty::Bool);
                emit(f, entry, mir::InstKind::Zext(t), mir::Ty::int(32))
            });
        }
        // int -> quad -> int roundtrip.
        cross_lower("int-roundtrip", triple, &|f, entry| {
            let seven = iconst(f, 64, 7);
            let q = emit(f, entry, mir::InstKind::SItoF(seven), mir::Ty::Float(mir::FloatKind::F128));
            emit(f, entry, mir::InstKind::SFtoI(q), mir::Ty::int(64))
        });
        // 128-bit int -> quad -> int roundtrip (rounding path,
        // multi-piece scalar calling convention included).
        cross_lower("int128-roundtrip", triple, &|f, entry| {
            let seven = iconst(f, 128, 7);
            let q = emit(f, entry, mir::InstKind::SItoF(seven), mir::Ty::Float(mir::FloatKind::F128));
            emit(f, entry, mir::InstKind::SFtoI(q), mir::Ty::int(128))
        });
        // ceil/floor roundtrip through ints.
        for op in [mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind, mir::InstKind::Floor] {
            cross_lower("round", triple, &|f, entry| {
                let c = f.push_const(mir::Const::Float {
                    kind: mir::FloatKind::F128,
                    bits: (one as u128) << 64,
                });
                let r = emit(f, entry, op(c), mir::Ty::Float(mir::FloatKind::F128));
                emit(f, entry, mir::InstKind::SFtoI(r), mir::Ty::int(64))
            });
        }
        // double -> quad -> double roundtrip (bit-exact through ints).
        cross_lower("float-roundtrip", triple, &|f, entry| {
            let half = f.push_const(mir::Const::Float {
                kind: mir::FloatKind::F64,
                bits: 0x3FE0_0000_0000_0000,
            });
            let q = emit(f, entry, mir::InstKind::Fpext(half), mir::Ty::Float(mir::FloatKind::F128));
            let back = emit(f, entry, mir::InstKind::Fptrunc(q), mir::Ty::Float(mir::FloatKind::F64));
            emit(f, entry, mir::InstKind::Bitcast(back), mir::Ty::int(64))
        });
    }
}

/// `F80` unordered/ordered lower on x86-64 through the `ne`
/// self-comparison (no new asm): object emission succeeds with
/// the helper imports left unresolved.
#[test]
fn f80_unordered_lowers_x86_64() {
    use mir::FcmpPred::*;
    let one_bits = ((0x3fffu128) << 64) | 0x8000_0000_0000_0000u128;
    for pred in [Uno, Ord] {
        cross_lower("f80-fcmp", "x86_64-unknown-linux-gnu", &|f, entry| {
            let c = f.push_const(mir::Const::Float {
                kind: mir::FloatKind::F80,
                bits: one_bits,
            });
            let t = emit(f, entry, mir::InstKind::Fcmp { pred, lhs: c, rhs: c }, mir::Ty::Bool);
            emit(f, entry, mir::InstKind::Zext(t), mir::Ty::int(32))
        });
    }
}

/// `F80` ceil/floor lower on x86-64 through the frndint helpers:
/// object emission succeeds with the imports left unresolved.
#[test]
fn f80_round_lowers_x86_64() {
    let one_bits = ((0x3fffu128) << 64) | 0x8000_0000_0000_0000u128;
    for op in [mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind, mir::InstKind::Floor] {
        cross_lower("f80-round", "x86_64-unknown-linux-gnu", &|f, entry| {
            let c = f.push_const(mir::Const::Float {
                kind: mir::FloatKind::F80,
                bits: one_bits,
            });
            let r = emit(f, entry, op(c), mir::Ty::Float(mir::FloatKind::F80));
            emit(f, entry, mir::InstKind::SFtoI(r), mir::Ty::int(64))
        });
    }
}

/// Quad differential tests: builders against the `quad::ref_*`
/// oracles through memory. `F128` values cross into Rust as
/// 16-byte buffers (the carrier never names an `f128` in a Rust
/// signature); every builder here is integer IR plus moves, so
/// x86-64 JIT executes them exactly.
fn qbytes(lo: u64, hi: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&lo.to_le_bytes());
    b[8..].copy_from_slice(&hi.to_le_bytes());
    b
}

fn qhalves(buf: &[u8]) -> (u64, u64) {
    (u64::from_le_bytes(buf[..8].try_into().unwrap()), u64::from_le_bytes(buf[8..16].try_into().unwrap()))
}

fn run_q1(jit: &JITModule, ids: &std::collections::HashMap<mir::Func, cranelift_module::FuncId>, index: usize, out: *mut u8, inp: *const u8) {
    unsafe {
        let func: extern "C" fn(*mut u8, *const u8) = std::mem::transmute(func_addr(jit, ids, index));
        func(out, inp);
    }
}

/// Build `run(out, inp)`: load an `F128` at `inp`, apply one
/// unary op, store the `F128` result at `out`.
fn qunary_module(op: fn(mir::Value) -> mir::InstKind) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let inp = param(&mut f, entry, mir::Ty::int(64), 1);
    let v = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: inp,
            flags: mir::MemFlags::plain(),
        },
        f128.clone(),
    );
    let r = emit(&mut f, entry, op(v), f128.clone());
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_neg_executes() {
    let module = qunary_module(mir::InstKind::FNeg);
    let (jit, ids) = jit_module(&module);
    // (input lo/hi, expected lo/hi): sign flip, payload kept.
    let cases: &[((u64, u64), (u64, u64))] = &[
        ((0, 0x3fff_8000_0000_0000), (0, 0xbfff_8000_0000_0000)),                                         // 1.5
        ((0, 0x8000_0000_0000_0000), (0, 0x0000_0000_0000_0000)),                                         // -0 -> +0
        ((0, 0x0000_0000_0000_0000), (0, 0x8000_0000_0000_0000)),                                         // +0 -> -0
        ((0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0), (0x1234_5678_9abc_def0, 0xffff_badc_0ffe_e0d0)), // nan payload kept
        ((0, 0x7fff_0000_0000_0000), (0, 0xffff_0000_0000_0000)),                                         // inf
    ];
    for ((ilo, ihi), (elo, ehi)) in cases {
        let inp = qbytes(*ilo, *ihi);
        let mut out = [0u8; 16];
        run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
        assert_eq!(qhalves(&out), (*elo, *ehi), "neg of {ilo:#x}/{ihi:#x}");
    }
}

#[test]
fn quad_round_executes() {
    let q = crate::quad::ref_f64_to_quad;
    let inputs: &[(u64, u64)] = &[
        (0, 0x0000_0000_0000_0000),                     // +0
        (0, 0x8000_0000_0000_0000),                     // -0
        q(0x3ff0_0000_0000_0000),                       // 1.0
        q(0xbff0_0000_0000_0000),                       // -1.0
        q(0x3ff8_0000_0000_0000),                       // 1.5
        q(0xbff8_0000_0000_0000),                       // -1.5
        q(0x3fe0_0000_0000_0000),                       // 0.5
        q(0xbfe0_0000_0000_0000),                       // -0.5
        q(0x40c8_0000_0000_0000),                       // 10000.75
        q(0xc0c8_0000_0000_0000),                       // -10000.75
        (1, 0x0000_0000_0000_0000),                     // tiny denormal
        (1, 0x8000_0000_0000_0000),                     // tiny negative denormal
        (0, 0x403e_0000_0000_0000),                     // 2^63 integral
        (0, 0x3bcd_0000_0000_0000),                     // 2^-1074 (f64 min-denormal widened)
        (0, 0xbbcd_0000_0000_0000),                     // -2^-1074
        q(0x7ff0_0000_0000_0000),                       // +inf
        q(0xfff0_0000_0000_0000),                       // -inf
        q(0x7ff8_0000_0000_0000),                       // qnan
        (0xa000_0000_0000_0000, 0x3ffb_9999_9999_9999), // 0.1-ish
    ];
    for (op, oracle) in [
        (
            mir::InstKind::Ceil as fn(mir::Value) -> mir::InstKind,
            crate::quad::ref_quad_ceil as fn(u64, u64) -> (u64, u64),
        ),
        (
            mir::InstKind::Floor as fn(mir::Value) -> mir::InstKind,
            crate::quad::ref_quad_floor as fn(u64, u64) -> (u64, u64),
        ),
    ] {
        let module = qunary_module(op);
        let (jit, ids) = jit_module(&module);
        for (ilo, ihi) in inputs {
            let inp = qbytes(*ilo, *ihi);
            let mut out = [0u8; 16];
            run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
            assert_eq!(qhalves(&out), oracle(*ilo, *ihi), "round of {ilo:#x}/{ihi:#x}");
        }
    }
}

/// Build `run(out, a, b)`: load an `F128` at each address, one
/// `Fcmp`, store the `I8` result at `out`.
fn qcmp_module(pred: mir::FcmpPred) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let a_ptr = param(&mut f, entry, mir::Ty::int(64), 1);
    let b_ptr = param(&mut f, entry, mir::Ty::int(64), 2);
    let a = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: a_ptr,
            flags: mir::MemFlags::plain(),
        },
        f128.clone(),
    );
    let b = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: b_ptr,
            flags: mir::MemFlags::plain(),
        },
        f128.clone(),
    );
    let t = emit(&mut f, entry, mir::InstKind::Fcmp { pred, lhs: a, rhs: b }, mir::Ty::Bool);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: t,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_cmp_executes() {
    use mir::FcmpPred::*;
    let one = crate::quad::ref_f64_to_quad(0x3ff0_0000_0000_0000);
    let two = crate::quad::ref_f64_to_quad(0x4000_0000_0000_0000);
    let negtwo = crate::quad::ref_f64_to_quad(0xc000_0000_0000_0000);
    let pzero = (0u64, 0u64);
    let nzero = (0u64, 0x8000_0000_0000_0000u64);
    let inf = crate::quad::ref_f64_to_quad(0x7ff0_0000_0000_0000);
    let nan = crate::quad::ref_f64_to_quad(0x7ff8_0000_0000_0000);
    let pairs: &[((u64, u64), (u64, u64))] = &[(one, one), (one, two), (negtwo, one), (pzero, nzero), (nan, one), (one, inf), (negtwo, nzero)];
    for (pred, check) in [
        (Eq, crate::quad::ref_quad_eq as fn(u64, u64, u64, u64) -> bool),
        (Ne, (|a, b, c, d| !crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool),
        (Lt, crate::quad::ref_quad_lt as fn(u64, u64, u64, u64) -> bool),
        (
            Le,
            (|a, b, c, d| crate::quad::ref_quad_lt(a, b, c, d) || crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool,
        ),
        (Gt, (|a, b, c, d| crate::quad::ref_quad_lt(c, d, a, b)) as fn(u64, u64, u64, u64) -> bool),
        (
            Ge,
            (|a, b, c, d| crate::quad::ref_quad_lt(c, d, a, b) || crate::quad::ref_quad_eq(a, b, c, d)) as fn(u64, u64, u64, u64) -> bool,
        ),
        (
            Uno,
            (|a, b, c, d| crate::quad::ref_quad_is_nan(a, b) || crate::quad::ref_quad_is_nan(c, d)) as fn(u64, u64, u64, u64) -> bool,
        ),
        (
            Ord,
            (|a, b, c, d| !(crate::quad::ref_quad_is_nan(a, b) || crate::quad::ref_quad_is_nan(c, d))) as fn(u64, u64, u64, u64) -> bool,
        ),
    ] {
        let module = qcmp_module(pred);
        let (jit, ids) = jit_module(&module);
        for ((alo, ahi), (blo, bhi)) in pairs {
            let abuf = qbytes(*alo, *ahi);
            let bbuf = qbytes(*blo, *bhi);
            let mut out = [0u8; 1];
            unsafe {
                let func: extern "C" fn(*mut u8, *const u8, *const u8) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), abuf.as_ptr(), bbuf.as_ptr());
            }
            assert_eq!(out[0], u8::from(check(*alo, *ahi, *blo, *bhi)), "cmp {pred:?} of {alo:#x}/{ahi:#x} vs {blo:#x}/{bhi:#x}");
        }
    }
}

/// Build `run(out, x)`: convert one `I64` to `F128` (signed or
/// unsigned), store the halves at `out`.
fn qfrom_int_module(unsigned: bool) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let x = param(&mut f, entry, mir::Ty::int(64), 1);
    let kind = if unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
    let r = emit(&mut f, entry, kind, f128);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_from_int_executes() {
    for unsigned in [false, true] {
        let module = qfrom_int_module(unsigned);
        let (jit, ids) = jit_module(&module);
        for v in [0u64, 1, 2, 123456789, 0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff] {
            let mut out = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), v);
            }
            assert_eq!(qhalves(&out), crate::quad::ref_int_to_quad(v, unsigned), "from_int({v:#x}, unsigned={unsigned})");
        }
    }
}

/// Build `run(out, inp)`: load an `F128` at `inp`, convert to
/// `I64` (signed or unsigned), store at `out`.
fn qto_int_module(unsigned: bool) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let inp = param(&mut f, entry, mir::Ty::int(64), 1);
    let v = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: inp,
            flags: mir::MemFlags::plain(),
        },
        f128,
    );
    let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
    let r = emit(&mut f, entry, kind, mir::Ty::int(64));
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_to_int_executes() {
    // (lo, hi) inputs spanning exact, fractional, saturation, and
    // NaN/Inf edges; expectations from the oracles.
    let quads: &[(u64, u64)] = &[
        (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000), // 1.0
        (0x0000_0000_0000_0000, 0xbfff_0000_0000_0000), // -1.0
        (0x0000_0000_0000_0000, 0x403e_0000_0000_0000), // 2^63
        (0xffff_ffff_ffff_ffff, 0x403e_ffff_ffff_ffff), // 2^64-1
        (0x0000_0000_0000_0000, 0x403f_0000_0000_0000), // 2^64
        (0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // +inf
        (0x0000_0000_0000_0000, 0xffff_0000_0000_0000), // -inf
        (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000), // nan
        (0x8000_0000_0000_0000, 0x4005_c000_0000_0000), // 123.456-ish fractional
        (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
    ];
    for unsigned in [false, true] {
        let module = qto_int_module(unsigned);
        let (jit, ids) = jit_module(&module);
        for (lo, hi) in quads {
            let inp = qbytes(*lo, *hi);
            let mut out = [0u8; 8];
            run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
            let got = u64::from_le_bytes(out);
            let want = if unsigned {
                crate::quad::ref_quad_to_u64(*lo, *hi)
            } else {
                crate::quad::ref_quad_to_s64(*lo, *hi)
            };
            assert_eq!(got, want, "to_int({lo:#x}/{hi:#x}, unsigned={unsigned})");
        }
    }
}

/// Build `run(out, x)`: convert one `I128` to `F128` (signed or
/// unsigned), store the halves at `out`. The 128-bit argument
/// arrives as a register pair under the hood.
fn qfrom_int128_module(unsigned: bool) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(128)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let x = param(&mut f, entry, mir::Ty::int(128), 1);
    let kind = if unsigned { mir::InstKind::UItoF(x) } else { mir::InstKind::SItoF(x) };
    let r = emit(&mut f, entry, kind, f128);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_from_int128_executes() {
    // (value, unsigned): exact, halfway, saturation-adjacent.
    let cases: &[(u128, bool)] = &[
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        ((1 << 64) - 1, false),
        ((1 << 64) - 1, true),
        (1 << 64, true),
        (u64::MAX as u128, true),
        (i64::MAX as u128, false),
        (i64::MIN as u128, false),
        ((1 << 113) - 1, false),
        (1 << 113, false),
        ((1 << 113) + 1, false),
        ((1 << 113) + 3, false),
        (u128::MAX, false),
        (u128::MAX, true),
        (u128::MAX - 1, true),
        (i128::MAX as u128, false),
        (i128::MIN as u128, false),
        ((1u128 << 113) + 1, true),
        (0, true),
    ];
    for unsigned in [false, true] {
        let module = qfrom_int128_module(unsigned);
        let (jit, ids) = jit_module(&module);
        for (v, _u) in cases.iter().filter(|(_, u)| *u == unsigned) {
            let (lo, hi) = ((*v as u64), ((*v >> 64) as u64));
            let mut out = [0u8; 16];
            unsafe {
                let func: extern "C" fn(*mut u8, u64, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
                func(out.as_mut_ptr(), lo, hi);
            }
            assert_eq!(qhalves(&out), crate::quad::ref_int_to_quad128(*v, unsigned), "from_int128({v:#x}, unsigned={unsigned})");
        }
    }
}

/// Build `run(out, inp)`: load an `F128` at `inp`, convert to
/// `I128` (signed or unsigned), store the 16 bytes at `out`.
fn qto_int128_module(unsigned: bool) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let inp = param(&mut f, entry, mir::Ty::int(64), 1);
    let v = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: inp,
            flags: mir::MemFlags::plain(),
        },
        f128,
    );
    let kind = if unsigned { mir::InstKind::UFtoI(v) } else { mir::InstKind::SFtoI(v) };
    let r = emit(&mut f, entry, kind, mir::Ty::int(128));
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_to_int128_executes() {
    // (lo, hi) inputs spanning exact, fractional, saturation, and
    // NaN/Inf edges; expectations from the oracles.
    let quads: &[(u64, u64)] = &[
        (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000), // 1.0
        (0x0000_0000_0000_0000, 0xbfff_0000_0000_0000), // -1.0
        (0x0000_0000_0000_0000, 0x3ffe_0000_0000_0000), // 0.5
        (0x0000_0000_0000_0000, 0x403e_0000_0000_0000), // 2^63
        (0x0000_0000_0000_0000, 0x403f_0000_0000_0000), // 2^64
        (0x0000_0000_0000_0000, 0x4063_0000_0000_0000), // 2^100
        (0x0000_0000_0000_0000, 0x407d_0000_0000_0000), // 2^126
        (0x0000_0000_0000_0000, 0x407e_0000_0000_0000), // 2^127
        (0x0000_0000_0000_0000, 0xc07e_0000_0000_0000), // -2^127
        (0x0000_0000_0000_0000, 0x407f_0000_0000_0000), // 2^128
        (0x0000_0000_0000_0000, 0x7fff_0000_0000_0000), // +inf
        (0x0000_0000_0000_0000, 0xffff_0000_0000_0000), // -inf
        (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000), // nan
        (0x8000_0000_0000_0000, 0x4005_c000_0000_0000), // 123.456-ish fractional
        (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
    ];
    for unsigned in [false, true] {
        let module = qto_int128_module(unsigned);
        let (jit, ids) = jit_module(&module);
        for (lo, hi) in quads {
            let inp = qbytes(*lo, *hi);
            let mut out = [0u8; 16];
            run_q1(&jit, &ids, 0, out.as_mut_ptr(), inp.as_ptr());
            let got = u128::from_le_bytes(out);
            let want = if unsigned {
                crate::quad::ref_quad_to_u128(*lo, *hi)
            } else {
                crate::quad::ref_quad_to_s128(*lo, *hi)
            };
            assert_eq!(got, want, "to_int128({lo:#x}/{hi:#x}, unsigned={unsigned})");
        }
    }
}

/// Build `run(out, x)`: widen one `F32`/`F64` value to `F128`,
/// store the halves at `out`.
fn qwiden_module(bits: u32) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let (pty, fty) = if bits == 32 {
        (mir::Ty::int(32), mir::Ty::Float(mir::FloatKind::F32))
    } else {
        (mir::Ty::int(64), mir::Ty::Float(mir::FloatKind::F64))
    };
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), pty.clone()], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let x = param(&mut f, entry, pty, 1);
    let as_f = emit(&mut f, entry, mir::InstKind::Bitcast(x), fty);
    let r = emit(&mut f, entry, mir::InstKind::Fpext(as_f), f128);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: r,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_widen_executes() {
    for v in [
        0x0000_0000u32,
        0x8000_0000,
        0x3fc0_0000,
        0x3dcc_cccd,
        0x0080_0000,
        0x0000_0001,
        0x7f80_0000,
        0x7fc0_0000,
        0x7f80_0001,
    ] {
        let module = qwiden_module(32);
        let (jit, ids) = jit_module(&module);
        let mut out = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, u32) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(out.as_mut_ptr(), v);
        }
        assert_eq!(qhalves(&out), crate::quad::ref_f32_to_quad(v), "widen f32 {v:#x}");
    }
    for v in [
        0u64,
        0x8000_0000_0000_0000,
        0x3ff8_0000_0000_0000,
        0x3fb9_9999_9999_999a,
        0x0000_0000_0000_0001,
        0x0010_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0x7ff0_0000_0000_0001,
    ] {
        let module = qwiden_module(64);
        let (jit, ids) = jit_module(&module);
        let mut out = [0u8; 16];
        unsafe {
            let func: extern "C" fn(*mut u8, u64) = std::mem::transmute(func_addr(&jit, &ids, 0));
            func(out.as_mut_ptr(), v);
        }
        assert_eq!(qhalves(&out), crate::quad::ref_f64_to_quad(v), "widen f64 {v:#x}");
    }
}

/// Build `run(out, inp)`: load an `F128` at `inp`, narrow to
/// `F32`/`F64`, store the bits at `out`.
fn qnarrow_module(bits: u32) -> mir::Module {
    let f128 = mir::Ty::Float(mir::FloatKind::F128);
    let (fty, sty) = if bits == 32 {
        (mir::Ty::Float(mir::FloatKind::F32), mir::Ty::int(32))
    } else {
        (mir::Ty::Float(mir::FloatKind::F64), mir::Ty::int(64))
    };
    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![mir::Ty::int(64), mir::Ty::int(64)], mir::Ty::Void);
    let (mut f, entry) = mir::Function::new("run", sig);
    let out = param(&mut f, entry, mir::Ty::int(64), 0);
    let inp = param(&mut f, entry, mir::Ty::int(64), 1);
    let v = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr: inp,
            flags: mir::MemFlags::plain(),
        },
        f128,
    );
    let n = emit(&mut f, entry, mir::InstKind::Fptrunc(v), fty);
    let b = emit(&mut f, entry, mir::InstKind::Bitcast(n), sty);
    emit_unit(
        &mut f,
        entry,
        mir::InstKind::Store {
            addr: out,
            val: b,
            flags: mir::MemFlags::plain(),
        },
    );
    ret(&mut f, entry, None);
    module.push_func(f);
    module
}

#[test]
fn quad_narrow_executes() {
    let quads: &[(u64, u64)] = &[
        (0x0000_0000_0000_0000, 0x3fff_0000_0000_0000),       // 1.0
        (0xa000_0000_0000_0000, 0x3ffb_9999_9999_9999),       // 0.1-ish
        (0xffff_ffff_ffff_ffff, 0x7ffe_ffff_ffff_ffff),       // quad max -> inf
        (0x0000_0000_0000_0000, 0x0001_0000_0000_0000),       // tiny -> 0
        (0x0000_0000_0000_0001, 0x7fff_8000_0000_0000),       // nan
        (0x1234_5678_9abc_def0, 0x7fff_badc_0ffe_e0d0),       // nan payload
        (1u64 << 59, 0x3fff_0000_0000_0000),                  // halfway, rounds down
        ((1u64 << 59) | (1u64 << 52), 0x3fff_0000_0000_0000), // halfway + sticky, rounds up
        (0x0000_0000_0000_0000, 0x3bcd_0000_0000_0000),       // 2^-1074 -> f64 min-denormal
        (0x0000_0000_0000_0001, 0x3bcd_0000_0000_0000),       // 2^-1074 + 2^-1186 -> f64 min-denormal
        (0xffff_ffff_ffff_ffff, 0x3bcd_0000_0000_0000),       // just below 2^-1073 -> f64 min-denormal
    ];
    let module32 = qnarrow_module(32);
    let (jit32, ids32) = jit_module(&module32);
    let module64 = qnarrow_module(64);
    let (jit64, ids64) = jit_module(&module64);
    for (lo, hi) in quads {
        let inp = qbytes(*lo, *hi);
        let mut out32 = [0u8; 4];
        run_q1(&jit32, &ids32, 0, out32.as_mut_ptr(), inp.as_ptr());
        assert_eq!(u32::from_le_bytes(out32), crate::quad::ref_quad_to_f32(*lo, *hi), "narrow f32 of {lo:#x}/{hi:#x}");
        let mut out64 = [0u8; 8];
        run_q1(&jit64, &ids64, 0, out64.as_mut_ptr(), inp.as_ptr());
        assert_eq!(u64::from_le_bytes(out64), crate::quad::ref_quad_to_f64(*lo, *hi), "narrow f64 of {lo:#x}/{hi:#x}");
    }
}

/// Lower one MIR module to x86-64 ELF object bytes for structural
/// inspection (linkage, sections, alignment) the JIT cannot show.
fn emit_object(mir_module: &mir::Module) -> Vec<u8> {
    use cranelift_codegen::settings::Configurable;
    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("opt_level", "none").expect("flag");
    flag_builder.set("is_pic", "true").expect("flag");
    flag_builder.set("tls_model", "elf_gd").expect("tls model");
    let flags = cranelift_codegen::settings::Flags::new(flag_builder);
    let isa = cranelift_codegen::isa::lookup("x86_64-unknown-linux-gnu".parse().expect("triple"))
        .expect("isa")
        .finish(flags)
        .expect("isa flags");
    let obj_builder = cranelift_object::ObjectBuilder::new(isa, "linktest", cranelift_module::default_libcall_names()).expect("object builder");
    let mut obj = cranelift_object::ObjectModule::new(obj_builder);
    let target = crate::abi::Target::from_triple("x86_64-unknown-linux-gnu").expect("target");
    lower_module(mir_module, &mut obj, target);
    obj.finish().emit().expect("object emit")
}

/// Kind, weakness, and address of one symbol in emitted bytes.
fn symbol_info(object: &[u8], name: &str) -> (object::SymbolKind, bool, u64) {
    use object::read::{Object, ObjectSymbol};
    let file = object::File::parse(object).expect("parse object");
    for sym in file.symbols() {
        if sym.name().ok() == Some(name) {
            return (sym.kind(), sym.is_weak(), sym.address());
        }
    }
    panic!("no symbol {name}");
}

#[test]
fn weak_definitions_emit_stb_weak() {
    let sig = mir::Signature::new(vec![], mir::Ty::int(32));
    let mut module = mir::Module::new();
    let (mut weak, wentry) = mir::Function::new("weak_fn", sig.clone());
    weak.weak = true;
    let five = iconst(&mut weak, 32, 5);
    ret(&mut weak, wentry, Some(five));
    module.push_func(weak);
    let (mut strong, sentry) = mir::Function::new("strong_fn", sig);
    let six = iconst(&mut strong, 32, 6);
    ret(&mut strong, sentry, Some(six));
    module.push_func(strong);

    let object = emit_object(&module);
    let (_, is_weak, _) = symbol_info(&object, "weak_fn");
    assert!(is_weak, "weak function emits STB_WEAK");
    let (_, is_weak, _) = symbol_info(&object, "strong_fn");
    assert!(!is_weak, "plain definition stays strong");
}

#[test]
fn weak_imports_emit_stb_weak() {
    // `extern ... __attribute__((weak))` with no definition: the
    // undefined symbol is `STB_WEAK`, so the linker resolves it to
    // zero (tcctest.c's `weak_fpa`/`weak_fpc` declarations). Only
    // referenced imports are declared, so call both.
    let mut module = mir::Module::new();
    let weak = module.push_extern("weak_ext", mir::Signature::new(vec![], mir::Ty::int(32)));
    module.externs[weak].weak = true;
    let plain = module.push_extern("plain_ext", mir::Signature::new(vec![], mir::Ty::int(32)));
    let sig = mir::Signature::new(vec![], mir::Ty::int(32));
    let (mut caller, entry) = mir::Function::new("caller", sig);
    let w = emit(
        &mut caller,
        entry,
        mir::InstKind::CallExtern {
            target: weak,
            args: Default::default(),
        },
        mir::Ty::int(32),
    );
    let p = emit(
        &mut caller,
        entry,
        mir::InstKind::CallExtern {
            target: plain,
            args: Default::default(),
        },
        mir::Ty::int(32),
    );
    let sum = emit(&mut caller, entry, mir::InstKind::Add(w, p), mir::Ty::int(32));
    ret(&mut caller, entry, Some(sum));
    module.push_func(caller);
    let object = emit_object(&module);
    let (_, is_weak, _) = symbol_info(&object, "weak_ext");
    assert!(is_weak, "weak import emits STB_WEAK");
    let (_, is_weak, _) = symbol_info(&object, "plain_ext");
    assert!(!is_weak, "plain import stays strong");
}

#[test]
fn tls_globals_emit_tls_symbols() {
    let mut module = mir::Module::new();
    let g = module.push_global("tls_counter", mir::Ty::int(32), Some(mir::Const::Int { bits: 32, value: 7 }));
    module.globals[g].tls = true;
    module.globals[g].align = Some(4);
    let sig = mir::Signature::new(vec![], mir::Ty::int(32));
    let (mut f, entry) = mir::Function::new("read_tls", sig);
    let addr = emit(&mut f, entry, mir::InstKind::GlobalAddr { global: g }, mir::Ty::ptr(mir::Ty::int(32)));
    let v = emit(
        &mut f,
        entry,
        mir::InstKind::Load {
            addr,
            flags: mir::MemFlags::plain(),
        },
        mir::Ty::int(32),
    );
    ret(&mut f, entry, Some(v));
    module.push_func(f);

    let object = emit_object(&module);
    let (kind, _, _) = symbol_info(&object, "tls_counter");
    assert_eq!(kind, object::SymbolKind::Tls, "STT_TLS symbol");
}

#[test]
fn read_only_globals_land_in_rodata() {
    use object::read::{Object, ObjectSection, ObjectSymbol};

    let mut module = mir::Module::new();
    let g = module.push_global(
        "literal",
        mir::Ty::Array {
            elem: Box::new(mir::Ty::int(8)),
            len: 1,
        },
        Some(mir::Const::Bytes(vec![0])),
    );
    let global = &mut module.globals[g];
    global.local = true;
    global.writable = false;
    global.align = Some(1);

    let object = emit_object(&module);
    let file = object::File::parse(&*object).expect("parse object");
    let symbol = file.symbol_by_name("literal").expect("literal symbol");
    let object::SymbolSection::Section(index) = symbol.section() else {
        panic!("literal has no section");
    };
    let section = file.section_by_index(index).expect("literal section");
    assert_eq!(section.name().unwrap(), ".rodata");
    assert_eq!(section.data().expect("literal data"), &[0], "an all-zero literal stays in .rodata, not .bss");
}

#[test]
fn alignment_reaches_the_object() {
    let mut module = mir::Module::new();
    let g = module.push_global(
        "dbl",
        mir::Ty::Float(mir::FloatKind::F64),
        Some(mir::Const::Float {
            kind: mir::FloatKind::F64,
            bits: 1.5f64.to_bits() as u128,
        }),
    );
    module.globals[g].align = Some(8);
    let (mut f, entry) = mir::Function::new("f", mir::Signature::new(vec![], mir::Ty::Void));
    ret(&mut f, entry, None);
    module.push_func(f);

    let object = emit_object(&module);
    let (_, _, addr) = symbol_info(&object, "dbl");
    assert_eq!(addr % 8, 0, "double lands 8-aligned");
}

#[test]
fn custom_section_defines_hook_array() {
    use object::read::{Object, ObjectSection};

    let mut module = mir::Module::new();
    let sig = mir::Signature::new(vec![], mir::Ty::Void);
    let (mut hook, entry) = mir::Function::new("hook", sig);
    ret(&mut hook, entry, None);
    let hook_id = module.push_func(hook);

    let g = module.push_global(
        "__glcc_init_array",
        mir::Ty::Array {
            elem: Box::new(mir::Ty::ptr(mir::Ty::Void)),
            len: 1,
        },
        Some(mir::Const::Bytes(vec![0u8; 8])),
    );
    let global = &mut module.globals[g];
    global.local = true;
    global.align = Some(8);
    global.section = Some(".init_array".to_string());
    global.relocs = vec![(0, mir::DataReloc::Func { target: hook_id })];

    let object = emit_object(&module);
    let file = object::File::parse(&*object).expect("parse object");
    let section = file.section_by_name(".init_array").expect(".init_array section");
    assert_eq!(section.align(), 8);
    assert_eq!(section.data().expect("section data").len(), 8);
    assert!(section.relocations().next().is_some(), "hook address is a link-time reloc");
}

/// Volatile MIR accesses must be opaque to Cranelift's alias
/// analysis, which happily forwards a plain load from a dominating
/// store: they lower to atomic ops it treats as fences. 16-byte
/// scalars have no portable atomic form and keep a plain access
/// fenced on both sides instead.
#[test]
fn volatile_accesses_lower_to_atomics() {
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{Function, InstBuilder, Opcode, Type};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    fn access_opcodes(ty: Type) -> Vec<Opcode> {
        let mut func = Function::new();
        {
            let mut ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func, &mut ctx);
            let entry = builder.create_block();
            builder.switch_to_block(entry);
            let addr = builder.append_block_param(entry, types::I64);
            let value = super::volatile_load(&mut builder, ty, addr);
            super::volatile_store(&mut builder, ty, value, addr);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
        }
        func.layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect()
    }

    for ty in [types::I8, types::I16, types::I32, types::I64, types::F32, types::F64] {
        let opcodes = access_opcodes(ty);
        assert!(opcodes.contains(&Opcode::AtomicLoad), "{ty} load: {opcodes:?}");
        assert!(opcodes.contains(&Opcode::AtomicStore), "{ty} store: {opcodes:?}");
    }
    let opcodes = access_opcodes(types::F128);
    assert!(!opcodes.contains(&Opcode::AtomicLoad), "{opcodes:?}");
    assert!(!opcodes.contains(&Opcode::AtomicStore), "{opcodes:?}");
    assert!(opcodes.contains(&Opcode::Fence), "{opcodes:?}");
    assert!(opcodes.contains(&Opcode::Load) && opcodes.contains(&Opcode::Store), "{opcodes:?}");
}

/// A `Module` that compiles nothing: it keeps the finalized Clif of
/// each defined function so tests can inspect what lowering
/// emitted. Only enough of the trait for modules of plain functions.
struct RecordingModule {
    isa: std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>,
    declarations: cranelift_module::ModuleDeclarations,
    funcs: Vec<cranelift_codegen::ir::Function>,
}

impl RecordingModule {
    fn new() -> Self {
        let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        let isa = cranelift_native::builder().expect("native isa").finish(flags).expect("native isa");
        Self {
            isa,
            declarations: cranelift_module::ModuleDeclarations::default(),
            funcs: Vec::new(),
        }
    }
}

impl cranelift_module::Module for RecordingModule {
    fn isa(&self) -> &dyn cranelift_codegen::isa::TargetIsa {
        &*self.isa
    }

    fn declarations(&self) -> &cranelift_module::ModuleDeclarations {
        &self.declarations
    }

    fn declare_function(
        &mut self,
        name: &str,
        linkage: cranelift_module::Linkage,
        signature: &cranelift_codegen::ir::Signature,
    ) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
        self.declarations.declare_function(name, linkage, signature).map(|(id, _)| id)
    }

    fn declare_anonymous_function(&mut self, signature: &cranelift_codegen::ir::Signature) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
        self.declarations.declare_anonymous_function(signature)
    }

    fn declare_data(&mut self, name: &str, linkage: cranelift_module::Linkage, writable: bool, tls: bool) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
        self.declarations.declare_data(name, linkage, writable, tls).map(|(id, _)| id)
    }

    fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
        self.declarations.declare_anonymous_data(writable, tls)
    }

    fn define_function_with_control_plane(
        &mut self,
        _func: cranelift_module::FuncId,
        ctx: &mut cranelift_codegen::Context,
        _ctrl_plane: &mut cranelift_codegen::control::ControlPlane,
    ) -> cranelift_module::ModuleResult<()> {
        self.funcs.push(ctx.func.clone());
        Ok(())
    }

    fn define_function_bytes(
        &mut self,
        _func_id: cranelift_module::FuncId,
        _alignment: u64,
        _bytes: &[u8],
        _relocs: &[cranelift_module::ModuleReloc],
    ) -> cranelift_module::ModuleResult<()> {
        unreachable!("a recording module has no object bytes")
    }

    fn define_data(&mut self, _data_id: cranelift_module::DataId, _data: &cranelift_module::DataDescription) -> cranelift_module::ModuleResult<()> {
        unreachable!("a recording module defines no data")
    }
}

/// The wiring, not just the helpers: `emit_inst` must route a
/// volatile MIR access through the atomic path (which Cranelift's
/// alias analysis never forwards a store into) and leave plain
/// accesses plain.
#[test]
fn volatile_mir_accesses_lower_to_atomic_clif() {
    use cranelift_codegen::ir::Opcode;

    fn lowered_opcodes(volatile: bool) -> Vec<Opcode> {
        let mut module = mir::Module::new();
        let sig = mir::Signature::new(vec![mir::Ty::ptr(mir::Ty::int(32))], mir::Ty::int(32));
        let (mut f, entry) = mir::Function::new("f", sig);
        let p = param(&mut f, entry, mir::Ty::ptr(mir::Ty::int(32)), 0);
        let flags = if volatile { mir::MemFlags::vol() } else { mir::MemFlags::plain() };
        let load = emit(&mut f, entry, mir::InstKind::Load { addr: p, flags }, mir::Ty::int(32));
        let seven = iconst(&mut f, 32, 7);
        emit_unit(&mut f, entry, mir::InstKind::Store { addr: p, val: seven, flags });
        ret(&mut f, entry, Some(load));
        module.push_func(f);

        let mut sink = RecordingModule::new();
        lower_module(&module, &mut sink, crate::abi::Target::native());
        let func = &sink.funcs[0];
        func.layout
            .blocks()
            .flat_map(|block| func.layout.block_insts(block))
            .map(|inst| func.dfg.insts[inst].opcode())
            .collect()
    }

    let volatile_ops = lowered_opcodes(true);
    assert!(volatile_ops.contains(&Opcode::AtomicLoad), "{volatile_ops:?}");
    assert!(volatile_ops.contains(&Opcode::AtomicStore), "{volatile_ops:?}");
    let plain_ops = lowered_opcodes(false);
    assert!(plain_ops.contains(&Opcode::Load), "{plain_ops:?}");
    assert!(plain_ops.contains(&Opcode::Store), "{plain_ops:?}");
    assert!(!plain_ops.contains(&Opcode::AtomicLoad), "{plain_ops:?}");
    assert!(!plain_ops.contains(&Opcode::AtomicStore), "{plain_ops:?}");
}

fn eh_test_module() -> mir::Module {
    glcc_mir::parse_module(
        r#"
.extern @thrower sig=(i32) -> i32
.extern @__gxx_personality_v0 sig=() -> i32
.dataextern @_ZTIi ty=ptr(void)

.func @catcher sig=(i32) -> i32 personality=@__gxx_personality_v0
.params %0:i32
b0:
  invoke.ext @thrower sig=(i32) -> i32, i32 %0 to b1 unwind b2
b1(%1:i32):
  ret %1
b2(%2:ptr(void), %3:i64) .lpad [catch @_ZTIi]:
  %4:i32 = eh.typeid @_ZTIi
  ret %4

.func @selector sig=() -> i32 personality=@__gxx_personality_v0
b0:
  %0:i32 = eh.typeid @_ZTIi
  ret %0
b1(%1:ptr(void), %2:i64) .lpad [catch @_ZTIi]:
  unreachable
"#,
    )
    .expect("EH test module parses")
}

extern "C" fn jit_thrower(x: i32) -> i32 {
    x + 1
}

#[test]
fn invoke_normal_edge_returns_callee_result() {
    let module = eh_test_module();
    let (jit, ids) = jit_module_with(&module, &[("thrower", jit_thrower as *const u8)]);
    assert_eq!(run_i32(&jit, &ids, 0, 5), 6, "invoke normal edge carries the result");
    assert_eq!(run_i32(&jit, &ids, 0, 0), 1);
}

#[test]
fn eh_typeid_lowers_to_type_table_selector() {
    let module = eh_test_module();
    let (jit, ids) = jit_module_with(&module, &[("thrower", jit_thrower as *const u8)]);
    // One catch clause: `int` is type table slot 1 (0 is the cleanup selector).
    assert_eq!(run_i32_0(&jit, &ids, 1), 1);
}
