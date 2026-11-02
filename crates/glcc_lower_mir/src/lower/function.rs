use super::*;

impl<'s, 'i> Lowerer<'s, 'i> {
    pub(crate) fn lower_function(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) {
        let body = self.sema.bodies[&id].block.clone();
        let binding = &self.sema.decls[id.0 as usize];
        let entry = func.entry;
        frame.current = Some(entry);
        // Entry params are typed by the signature — the ABI's view —
        // while the body's binding may be narrower (K&R declarations
        // pass promoted values); the spill converts.
        let fe::parse::Ty::Func { params: sig_params, .. } = binding.ty.unqualified() else {
            panic!("function definition without a Func type");
        };
        // Entry block params correspond positionally to signature params.
        for (i, param_id) in self.sema.bodies[&id].params.iter().enumerate() {
            let param_binding = &self.sema.decls[param_id.0 as usize];
            if let Some(loc) = self.debug_location(module, param_binding.span) {
                frame.loc = Some(loc);
            }
            let sig_fe = sig_params.get(i);
            let param_ty = match sig_fe {
                Some(ty) => lower_ty(ty, self.sema, self.interner),
                None => lower_ty(&param_binding.ty, self.sema, self.interner),
            };
            let param = func.values.push(mir::ValueData {
                ty: param_ty,
                def: mir::Def::Param(entry, i as u32),
            });
            func.blocks[entry].params.push(param);
            // A `va_list` parameter arrives as a pointer to the
            // caller's tag object (the array-typed parameter decayed).
            // Alias it, don't spill the pointer into a slot (which
            // `va_arg` would misread as cursors): the tag itself is
            // the caller's object, so cursors stay shared as C
            // requires. Mirrors the legacy backend. MSVC's `va_list`
            // is a plain pointer: the parameter is the cursor itself,
            // spilled like any scalar so `va_arg` advances the copy.
            if self.va_list_is_array() && matches!(param_binding.ty.unqualified(), fe::parse::Ty::BuiltinVaList) {
                frame.slots.insert(*param_id, param);
                continue;
            }
            // -O0 frame: every parameter is spilled to a slot on entry.
            // Volatility rides the spill: a `volatile` parameter's
            // slot stays volatile for every later load.
            let body_ty = lower_ty(&param_binding.ty, self.sema, self.interner);
            let slot = self.alloca(func, frame, body_ty);
            let vol = ty_is_volatile(&param_binding.ty);
            let value = match sig_fe {
                Some(from) => self.convert(func, frame, param, from, &param_binding.ty),
                None => param,
            };
            self.store(func, frame, slot, value, vol);
            frame.slots.insert(*param_id, slot);
        }
        self.lower_compound(module, func, frame, funcs, externs, globals, &body);
        // Falling off the end: void returns, `main` returns 0 per C23,
        // anything else returns an undefined value. Returning garbage
        // like GCC does (rather than trapping) keeps legacy code that
        // ignores the result working; the value is UB either way.
        // An async body returns its *declared* result type (the coroutine
        // pass turns these returns into result-slot stores).
        let result_ty = func.coroutine.clone().unwrap_or_else(|| func.sig.ret.clone());
        if frame.current.is_some() {
            if result_ty.is_void() {
                self.terminate(func, frame, mir::Terminator::Return { val: None });
            } else if binding.name.is_some_and(|n| self.interner.resolve(n) == Some("main")) {
                let zero = self.iconst(func, 32, 0);
                self.terminate(func, frame, mir::Terminator::Return { val: Some(zero) });
            } else {
                let undef = func.values.push(mir::ValueData {
                    ty: result_ty,
                    def: mir::Def::Const(mir::Const::Undef),
                });
                self.terminate(func, frame, mir::Terminator::Return { val: Some(undef) });
            }
        }
        // Computed gotos dispatch here: one switch over the function's
        // labels, keyed by the ids `&&label` computes. Filled after the
        // body so every label block exists; labels whose address was
        // never taken get cases too (harmless, and it keeps label
        // addresses in static initializers working, since those fold
        // before the body is lowered).
        self.drain_allocas(module, func, frame);
        if let Some((dispatch, param)) = frame.computed_goto.take() {
            let dead = func.push_block(vec![]);
            func.blocks[dead].term = Some(mir::Terminator::Unreachable);
            let mut cases: Vec<(i128, mir::Block)> = Vec::new();
            for (name, target) in &frame.labels {
                if let Some(id) = self.sema.label_ids.get(name) {
                    cases.push((*id as i128, *target));
                }
            }
            cases.sort_by_key(|(id, _)| *id);
            func.blocks[dispatch].term = Some(mir::Terminator::Switch {
                scrutinee: param,
                cases,
                default: dead,
            });
        }
        frame.loc = self.debug_location(module, binding.span);
        func.debug = Some(self.build_function_debug(module, &frame.slots, &self.sema.bodies[&id], id, globals));
        let _ = funcs;
    }

    pub(crate) fn lower_compound(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        compound: &fe::parse::Compound,
    ) {
        for item in compound.items {
            match item {
                fe::parse::BlockItem::Decl { ids, .. } => {
                    for id in *ids {
                        self.lower_local(module, func, frame, funcs, externs, globals, *id);
                    }
                }
                fe::parse::BlockItem::Stmt(stmt) => self.lower_stmt(module, func, frame, funcs, externs, globals, stmt),
            }
        }
    }

    pub(crate) fn lower_local(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        id: fe::sema::DeclId,
    ) {
        let binding = &self.sema.decls[id.0 as usize];
        if let Some(loc) = self.debug_location(module, binding.span) {
            frame.loc = Some(loc);
        }
        if binding.block_static {
            // Static storage: the module global was folded upfront, so
            // uses already resolve through `place_of_var` to a
            // `GlobalAddr`. No slot and no per-call initialization.
            debug_assert!(globals.contains_key(&id), "block-static global missing");
            return;
        }
        // A pure `extern` declaration (including a block-scope one with
        // no file-scope entity) introduces no storage: it reuses the
        // import declared upfront, so no slot is allocated. Function
        // declarations are the same: calls resolve through
        // `funcs`/`externs`, never through a slot.
        if matches!(binding.kind, fe::sema::BindingKind::Func) {
            return;
        }
        if binding.linkage == fe::sema::Linkage::External && !binding.defined && !binding.tentative {
            debug_assert!(globals.contains_key(&id), "extern import missing");
            return;
        }
        if Self::ty_is_vla_object(&binding.ty) {
            self.lower_vla_object(module, func, frame, funcs, externs, globals, id);
            return;
        }
        let mir_ty = lower_ty(&binding.ty, self.sema, self.interner);
        let natural = self.sema.type_align(&binding.ty, self.interner).unwrap_or(1);
        let requested = binding.align.unwrap_or(0).max(natural);
        let slot = if requested > natural {
            self.alloca_aligned(func, frame, mir_ty, requested)
        } else {
            self.alloca(func, frame, mir_ty)
        };
        frame.slots.insert(id, slot);
        if let Some(init) = self.sema.inits.get(&id) {
            self.lower_init(module, func, frame, funcs, externs, globals, slot, &binding.ty, init);
        }
    }

    /// Lower one `asm` statement into a [`mir::InstKind::InlineAsm`]
    /// block: inputs evaluate in order, then output addresses. The
    /// backend turns the block into a call to a generated wrapper that
    /// runs the template (see `glcc_lower_cl::asm`). A bit-field output
    /// round-trips through a unit-typed temp: the wrapper stores whole
    /// units, and the field merges back afterwards.
    pub(crate) fn lower_asm_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        asm: &fe::parse::AsmStmt,
    ) {
        let mut values = Vec::with_capacity(asm.inputs.len());
        for input in asm.inputs {
            let arch = self.sema.target.arch;
            let value = if fe::check::memory_only_constraint(&input.constraint, arch) {
                self.place(module, func, frame, funcs, externs, globals, &input.expr).0
            } else if fe::check::has_immediate_constraint(&input.constraint, arch) {
                // `"i"`/`"n"`/`"I"`/... take integer constant
                // expressions (`-1`, `sizeof x`, `1 << 3`), not just
                // literals: fold here so the wrapper substitutes a
                // constant. Anything else (including a runtime value
                // for a `"Ir"`-style operand) falls through to the
                // normal path, where a register or memory alternative
                // carries it.
                match fe::eval::eval_const(&input.expr, self.sema, self.interner) {
                    Some(fe::eval::Value::Int(n)) => {
                        let mir_ty = lower_ty(&input.expr.ty, self.sema, self.interner);
                        let bits = match mir_ty {
                            mir::Ty::Int { bits } => bits.max(8),
                            _ => 64,
                        };
                        func.push_const(mir::Const::Int { bits, value: n })
                    }
                    _ => self.expr_value(module, func, frame, funcs, externs, globals, &input.expr),
                }
            } else {
                self.expr_value(module, func, frame, funcs, externs, globals, &input.expr)
            };
            values.push(value);
        }
        // Every output lvalue's address is evaluated, tie or not:
        // `asm("" : "=m"(a[i++]))` still bumps `i`.
        struct BitfieldOut {
            temp: mir::Value,
            addr: mir::Value,
            decl_ty: fe::parse::Ty,
            bit: u32,
            width: u32,
            volatile: bool,
        }
        let mut operands = Vec::with_capacity(asm.outputs.len() + asm.inputs.len());
        let mut conversion: Vec<(fe::parse::Ty, Option<fe::parse::Ty>)> = Vec::with_capacity(asm.outputs.len());
        let mut bitfield_outs: Vec<BitfieldOut> = Vec::new();
        for output in asm.outputs {
            let (addr, _) = self.place(module, func, frame, funcs, externs, globals, &output.expr);
            let name = output.name.and_then(|sym| self.interner.resolve(sym)).map(str::to_string);
            let bitfield = match &output.expr.kind {
                fe::check::CheckedExprKind::Member {
                    bitfield: Some((bit, width)),
                    decl_ty,
                    ..
                } => Some((*bit, *width, decl_ty.clone())),
                _ => None,
            };
            match bitfield {
                Some((bit, width, decl_ty)) => {
                    let unit_mir = lower_ty(&decl_ty, self.sema, self.interner);
                    let temp = self.alloca(func, frame, unit_mir.clone());
                    let volatile = access_volatile(&output.expr, &self.sema.decls);
                    if output.constraint.contains('+') {
                        let field = self.bitfield_load(func, frame, addr, &decl_ty, &output.expr.ty, bit, width, is_signed(&decl_ty, self.sema), volatile);
                        let unit = self.convert(func, frame, field, &output.expr.ty, &decl_ty);
                        self.store(func, frame, temp, unit, false);
                    }
                    bitfield_outs.push(BitfieldOut {
                        temp,
                        addr,
                        decl_ty: decl_ty.clone(),
                        bit,
                        width,
                        volatile,
                    });
                    conversion.push((output.expr.ty.clone(), Some(decl_ty.clone())));
                    operands.push(mir::AsmOperand {
                        value: temp,
                        constraint: output.constraint.clone(),
                        ty: unit_mir,
                        tie: None,
                        name,
                    });
                }
                None => {
                    let ty = lower_ty(&output.expr.ty, self.sema, self.interner);
                    conversion.push((output.expr.ty.clone(), None));
                    operands.push(mir::AsmOperand {
                        value: addr,
                        constraint: output.constraint.clone(),
                        ty,
                        tie: None,
                        name,
                    });
                }
            }
        }
        for (i, input) in asm.inputs.iter().enumerate() {
            let name = input.name.and_then(|sym| self.interner.resolve(sym)).map(str::to_string);
            let mir_ty = lower_ty(&input.expr.ty, self.sema, self.interner);
            let aggregate = mir_ty.is_aggregate();
            let memory = fe::check::memory_only_constraint(&input.constraint, self.sema.target.arch);
            let value = match input.tie() {
                Some(n) if aggregate => {
                    // An aggregate shares only a memory output: its value
                    // is copied into the output's location, and both
                    // operands name that one address.
                    if !fe::check::memory_only_constraint(&asm.outputs[n].constraint, self.sema.target.arch) {
                        panic!("asm register operand is wider than a register");
                    }
                    self.store(func, frame, operands[n].value, values[i], false);
                    values[i]
                }
                Some(n) => {
                    let (member_ty, unit_ty) = &conversion[n];
                    let value = self.convert(func, frame, values[i], &input.expr.ty, member_ty);
                    match unit_ty {
                        Some(unit_ty) => self.convert(func, frame, value, member_ty, unit_ty),
                        None => value,
                    }
                }
                None => values[i],
            };
            // An aggregate in a register constraint crosses as its
            // address, like an array operand in GCC.
            let ty = if aggregate && !memory { mir::Ty::ptr(mir::Ty::Void) } else { mir_ty };
            operands.push(mir::AsmOperand {
                value,
                constraint: input.constraint.clone(),
                ty,
                tie: input.tie().map(|n| n as u32),
                name,
            });
        }
        let labels: Vec<String> = asm.labels.iter().map(|label| self.interner.resolve(*label).unwrap_or("?").to_string()).collect();
        let block = mir::AsmBlock {
            template: asm.template.clone(),
            operands,
            outputs: asm.outputs.len() as u32,
            clobbers: asm.clobbers.to_vec(),
            labels,
            is_volatile: asm.is_volatile(),
            is_basic: asm.is_basic(),
        };
        // `asm goto` needs the wrapper's selector as an instruction
        // result; plain asm ignores it.
        let selector = if asm.is_goto() {
            Some(self.emit(func, frame, mir::InstKind::InlineAsm { block: Box::new(block) }, mir::Ty::int(64)))
        } else {
            self.emit_unit(func, frame, mir::InstKind::InlineAsm { block: Box::new(block) });
            None
        };
        for out in bitfield_outs {
            let unit_mir = lower_ty(&out.decl_ty, self.sema, self.interner);
            let value = self.load(func, frame, out.temp, unit_mir, false);
            self.bitfield_store(func, frame, out.addr, &out.decl_ty, value, out.bit, out.width, out.volatile);
        }
        if let Some(selector) = selector {
            // The wrapper returns 0 when the template fell through and
            // `i + 1` when it jumped to `labels[i]`. Dispatch after the
            // outputs merge so both paths observe the stored results.
            let mut cases = Vec::with_capacity(asm.labels.len());
            for (i, name) in asm.labels.iter().enumerate() {
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                cases.push(((i + 1) as i128, target));
            }
            let fallthrough = func.push_block(vec![]);
            self.terminate(
                func,
                frame,
                mir::Terminator::Switch {
                    scrutinee: selector,
                    cases,
                    default: fallthrough,
                },
            );
            frame.current = Some(fallthrough);
        }
    }

    /// Lower a statement whose value becomes an enclosing statement
    /// expression's result: an expression statement, or a chain of
    /// labels ending in one. Each label defines its jump block exactly
    /// like `lower_stmt`'s label arm, so computed gotos still find it.
    /// Anything else lowers for effect and yields no value.
    pub(crate) fn lower_value_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        stmt: &fe::parse::Stmt,
    ) -> Option<mir::Value> {
        if let Some(loc) = self.debug_location(module, stmt.span()) {
            frame.loc = Some(loc);
        }
        match stmt {
            fe::parse::Stmt::Expr { expr, .. } => Some(self.expr_value(module, func, frame, funcs, externs, globals, expr)),
            fe::parse::Stmt::Label { name, names_loop, stmt, .. } => {
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                let depth = frame.pending_loop_labels.len();
                if *names_loop {
                    frame.pending_loop_labels.push(*name);
                }
                let value = self.lower_value_stmt(module, func, frame, funcs, externs, globals, stmt);
                frame.pending_loop_labels.truncate(depth);
                value
            }
            _ => {
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
                None
            }
        }
    }

    pub(crate) fn lower_stmt(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        stmt: &fe::parse::Stmt,
    ) {
        if let Some(loc) = self.debug_location(module, stmt.span()) {
            frame.loc = Some(loc);
        }
        match stmt {
            fe::parse::Stmt::Decl { ids, .. } => {
                for id in *ids {
                    self.lower_local(module, func, frame, funcs, externs, globals, *id);
                }
            }
            fe::parse::Stmt::Empty { .. } => {}
            fe::parse::Stmt::Expr { expr, .. } => {
                // Ensure a current block exists even for dead code.
                self.live_block(func, frame);
                self.expr_value(module, func, frame, funcs, externs, globals, expr);
            }
            fe::parse::Stmt::Asm(asm) => {
                self.lower_asm_stmt(module, func, frame, funcs, externs, globals, asm);
            }
            fe::parse::Stmt::Return { expr, .. } => {
                let val = expr.as_ref().map(|e| self.expr_value(module, func, frame, funcs, externs, globals, e));
                // A void return carries no value (a `return f();` of a
                // void `f`), and a bare `return;` in a value-returning
                // function (C89 legacy code) returns an undefined value,
                // like falling off the end. An async body returns its
                // declared result type, not the handle its symbol yields
                // (the coroutine pass rewrites the return).
                let result_ty = func.coroutine.clone().unwrap_or_else(|| func.sig.ret.clone());
                let val = if result_ty.is_void() {
                    None
                } else {
                    Some(val.unwrap_or_else(|| {
                        func.values.push(mir::ValueData {
                            ty: result_ty,
                            def: mir::Def::Const(mir::Const::Undef),
                        })
                    }))
                };
                self.terminate(func, frame, mir::Terminator::Return { val });
            }
            fe::parse::Stmt::Compound(inner) => self.lower_compound(module, func, frame, funcs, externs, globals, inner),
            fe::parse::Stmt::If { cond, then, els, .. } => {
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                let then_block = func.push_block(vec![]);
                let else_block = func.push_block(vec![]);
                let merge = func.push_block(vec![]);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block,
                        else_block,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(then_block);
                self.lower_stmt(module, func, frame, funcs, externs, globals, then);
                // Every open path needs an explicit edge to the merge:
                // MIR blocks never fall through.
                let mut live = false;
                if frame.current.is_some() {
                    self.branch(func, frame, merge, vec![]);
                    live = true;
                }
                frame.current = Some(else_block);
                if let Some(els) = els {
                    self.lower_stmt(module, func, frame, funcs, externs, globals, els);
                }
                if frame.current.is_some() {
                    self.branch(func, frame, merge, vec![]);
                    live = true;
                }
                self.join(func, frame, merge, live);
            }
            fe::parse::Stmt::While { cond, body, .. } => {
                let labels = core::mem::take(&mut frame.pending_loop_labels);
                let header = func.push_block(vec![]);
                let loop_body = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, header, vec![]);
                frame.current = Some(header);
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block: loop_body,
                        else_block: exit,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.loops.push(LoopTarget {
                    cont: header,
                    labels: labels.clone(),
                });
                frame.breaks.push(BreakTarget { block: exit, labels });
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, header, vec![]);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::DoWhile { body, cond, .. } => {
                let labels = core::mem::take(&mut frame.pending_loop_labels);
                let loop_body = func.push_block(vec![]);
                let check = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, loop_body, vec![]);
                frame.loops.push(LoopTarget {
                    cont: check,
                    labels: labels.clone(),
                });
                frame.breaks.push(BreakTarget { block: exit, labels });
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, check, vec![]);
                }
                frame.current = Some(check);
                let test = self.bool_of(module, func, frame, funcs, externs, globals, cond);
                self.terminate(
                    func,
                    frame,
                    mir::Terminator::CondBr {
                        cond: test,
                        then_block: loop_body,
                        else_block: exit,
                        then_args: Default::default(),
                        else_args: Default::default(),
                    },
                );
                frame.current = Some(exit);
            }
            fe::parse::Stmt::For { init, cond, step, body, .. } => {
                let labels = core::mem::take(&mut frame.pending_loop_labels);
                match init {
                    fe::parse::ForInit::Decl(ids) => {
                        for id in *ids {
                            self.lower_local(module, func, frame, funcs, externs, globals, *id);
                        }
                    }
                    fe::parse::ForInit::Expr(e) => {
                        self.live_block(func, frame);
                        self.expr_value(module, func, frame, funcs, externs, globals, e);
                    }
                    fe::parse::ForInit::Empty => {}
                }
                let header = func.push_block(vec![]);
                let loop_body = func.push_block(vec![]);
                let step_block = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                self.branch(func, frame, header, vec![]);
                frame.current = Some(header);
                match cond {
                    Some(c) => {
                        let test = self.bool_of(module, func, frame, funcs, externs, globals, c);
                        self.terminate(
                            func,
                            frame,
                            mir::Terminator::CondBr {
                                cond: test,
                                then_block: loop_body,
                                else_block: exit,
                                then_args: Default::default(),
                                else_args: Default::default(),
                            },
                        );
                    }
                    None => self.branch(func, frame, loop_body, vec![]),
                }
                frame.loops.push(LoopTarget {
                    cont: step_block,
                    labels: labels.clone(),
                });
                frame.breaks.push(BreakTarget { block: exit, labels });
                frame.current = Some(loop_body);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                frame.loops.pop();
                if frame.current.is_some() {
                    self.branch(func, frame, step_block, vec![]);
                }
                frame.current = Some(step_block);
                if let Some(s) = step {
                    self.expr_value(module, func, frame, funcs, externs, globals, s);
                }
                // The step block always falls into the header: it is only
                // reachable via branches above, never fallen into, so an
                // open step always ends here.
                if frame.current.is_some() {
                    self.branch(func, frame, header, vec![]);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::Break { label, .. } => {
                // Loops and `switch`es share one stack so `break`
                // takes whichever encloses most tightly; the checker
                // rejects strays, so an empty stack is a lowering bug.
                let target = match label {
                    Some(name) => frame
                        .breaks
                        .iter()
                        .rev()
                        .find(|target| target.labels.contains(name))
                        .map(|target| target.block)
                        .expect("break label target missing"),
                    None => frame.breaks.last().map(|target| target.block).expect("break outside a loop or switch"),
                };
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::Continue { label, .. } => {
                let target = match label {
                    Some(name) => frame
                        .loops
                        .iter()
                        .rev()
                        .find(|target| target.labels.contains(name))
                        .map(|target| target.cont)
                        .expect("continue label target missing"),
                    None => frame.loops.last().expect("continue outside a loop").cont,
                };
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::Switch { cond, body, .. } => {
                let labels = core::mem::take(&mut frame.pending_loop_labels);
                // C compares in the promoted controlling type, and the
                // checker's duplicate-`case` detection ran in exactly
                // that type: promote the scrutinee explicitly (the
                // tree carries no conversion node) so dispatch agrees
                // with it. A narrow `char` switch would otherwise
                // miscompare high cases.
                let promoted = fe::eval::promoted(&cond.ty);
                let scrutinee = self.expr_value(module, func, frame, funcs, externs, globals, cond);
                let scrutinee = self.convert(func, frame, scrutinee, &cond.ty, &promoted);
                let dispatch = self.live_block(func, frame);
                let body_entry = func.push_block(vec![]);
                let exit = func.push_block(vec![]);
                frame.switches.push(SwitchTarget {
                    promoted,
                    scrutinee,
                    cases: Vec::new(),
                    ranges: Vec::new(),
                    default: None,
                });
                frame.breaks.push(BreakTarget { block: exit, labels });
                frame.current = Some(body_entry);
                self.lower_stmt(module, func, frame, funcs, externs, globals, body);
                frame.breaks.pop();
                let ctx = frame.switches.pop().expect("switch pushed by this arm");
                // A trailing open path falls out of the switch.
                if frame.current.is_some() {
                    self.branch(func, frame, exit, vec![]);
                }
                // Patch the dispatch now that every case block exists.
                // Cases may hide in nested compounds, selections, and
                // loops (Duff's device); all recorded into this
                // context while the body lowered.
                frame.current = Some(dispatch);
                let default = ctx.default.unwrap_or(exit);
                let bits = int_bits(&lower_ty(&ctx.promoted, self.sema, self.interner));
                // GNU case ranges test first, each in its own block
                // (`lo <= x <= hi`, compared in the promoted type's
                // signedness); the exact-value dispatch follows.
                let signed = is_signed(&ctx.promoted, self.sema);
                let mut cur = dispatch;
                for (lo, hi, target) in ctx.ranges.iter() {
                    let next = func.push_block(vec![]);
                    frame.current = Some(cur);
                    let lo_v = self.iconst(func, bits, *lo);
                    let hi_v = self.iconst(func, bits, *hi);
                    let ge = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: if signed { mir::IcmpPred::Sge } else { mir::IcmpPred::Uge },
                            lhs: ctx.scrutinee,
                            rhs: lo_v,
                        },
                        mir::Ty::Bool,
                    );
                    let le = self.emit(
                        func,
                        frame,
                        mir::InstKind::Icmp {
                            pred: if signed { mir::IcmpPred::Sle } else { mir::IcmpPred::Ule },
                            lhs: ctx.scrutinee,
                            rhs: hi_v,
                        },
                        mir::Ty::Bool,
                    );
                    let inside = self.emit(func, frame, mir::InstKind::And(ge, le), mir::Ty::Bool);
                    self.terminate(
                        func,
                        frame,
                        mir::Terminator::CondBr {
                            cond: inside,
                            then_block: *target,
                            else_block: next,
                            then_args: Default::default(),
                            else_args: Default::default(),
                        },
                    );
                    cur = next;
                }
                frame.current = Some(cur);
                if bits <= 64 {
                    self.terminate(
                        func,
                        frame,
                        mir::Terminator::Switch {
                            scrutinee: ctx.scrutinee,
                            cases: ctx.cases,
                            default,
                        },
                    );
                } else {
                    self.switch_128(func, frame, ctx, default);
                }
                frame.current = Some(exit);
            }
            fe::parse::Stmt::Goto { label, .. } => {
                // Labels resolve during the single pass: a forward
                // jump creates its block early, and the later `Label`
                // reuses it. The checker guarantees the label exists
                // in this function.
                let target = match frame.labels.get(label) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*label, block);
                        block
                    }
                };
                self.branch(func, frame, target, vec![]);
            }
            fe::parse::Stmt::GotoComputed { expr, .. } => {
                // Dispatch on the label id: branch to the function's
                // dispatch block with the computed value, which the
                // switch there routes to the label. Created lazily so
                // functions without computed gotos pay nothing.
                let value = self.expr_value(module, func, frame, funcs, externs, globals, expr);
                let (dispatch, _) = match frame.computed_goto {
                    Some(pair) => pair,
                    None => {
                        let block = func.push_block(vec![mir::Ty::int(64)]);
                        let param = func.blocks[block].params[0];
                        frame.computed_goto = Some((block, param));
                        (block, param)
                    }
                };
                self.branch(func, frame, dispatch, vec![value]);
            }
            fe::parse::Stmt::Label { name, names_loop, stmt, .. } => {
                // Execution falls through into a label; only an
                // already-terminated path (after `return`/`goto`)
                // arrives solely by jump.
                let target = match frame.labels.get(name) {
                    Some(block) => *block,
                    None => {
                        let block = func.push_block(vec![]);
                        frame.labels.insert(*name, block);
                        block
                    }
                };
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                let depth = frame.pending_loop_labels.len();
                if *names_loop {
                    frame.pending_loop_labels.push(*name);
                }
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
                frame.pending_loop_labels.truncate(depth);
            }
            fe::parse::Stmt::Case { expr, hi, stmt, .. } => {
                // Like a label, but recorded into the innermost open
                // `switch` for the dispatch terminator. The checker
                // rejects strays and duplicates, so those are backstop
                // panics; values fold through the same `const_as` the
                // duplicate check used.
                let target = func.push_block(vec![]);
                let value = match expr {
                    None => None,
                    Some(label) => {
                        let ctx = frame.switches.last().expect("case outside a switch");
                        Some(fe::eval::const_as(&ctx.promoted, label, self.sema, self.interner).expect("case label must fold"))
                    }
                };
                let upper = hi.as_ref().map(|label| {
                    let ctx = frame.switches.last().expect("case outside a switch");
                    fe::eval::const_as(&ctx.promoted, label, self.sema, self.interner).expect("case range bound must fold")
                });
                {
                    let ctx = frame.switches.last_mut().expect("case outside a switch");
                    match (value, upper) {
                        (None, _) => {
                            if ctx.default.replace(target).is_some() {
                                panic!("duplicate `default` in `switch`");
                            }
                        }
                        (Some(v), None) => ctx.cases.push((v, target)),
                        (Some(lo), Some(hi)) => ctx.ranges.push((lo, hi, target)),
                    }
                }
                if frame.current.is_some() {
                    self.branch(func, frame, target, vec![]);
                }
                frame.current = Some(target);
                self.lower_stmt(module, func, frame, funcs, externs, globals, stmt);
            }
        }
    }

    /// Join two branch paths at `merge`: live when either side falls
    /// through, otherwise sealed as unreachable so it still verifies.
    pub(crate) fn join(&self, func: &mut mir::Function, frame: &mut Frame, merge: mir::Block, live: bool) {
        if live {
            frame.current = Some(merge);
        } else {
            frame.current = Some(merge);
            self.terminate(func, frame, mir::Terminator::Unreachable);
        }
    }

    pub(crate) fn branch(&self, func: &mut mir::Function, frame: &mut Frame, dest: mir::Block, args: Vec<mir::Value>) {
        self.terminate(
            func,
            frame,
            mir::Terminator::Br {
                dest,
                args: args.into_iter().collect(),
            },
        );
    }

    /// A 128-bit `switch` as an if-else chain: the MIR `Switch`
    /// terminator only covers scrutinees the Clif emission can
    /// compare directly (64 bits and under), so a wide controlling
    /// expression compares through ordinary conditional branches.
    /// Enters with the dispatch block current; every test compares
    /// full 128-bit values for equality.
    pub(crate) fn switch_128(&self, func: &mut mir::Function, frame: &mut Frame, ctx: SwitchTarget, default: mir::Block) {
        if ctx.cases.is_empty() {
            self.branch(func, frame, default, vec![]);
            return;
        }
        let tests: Vec<mir::Block> = ctx.cases.iter().map(|_| func.push_block(vec![])).collect();
        self.branch(func, frame, tests[0], vec![]);
        for (i, (value, dest)) in ctx.cases.into_iter().enumerate() {
            frame.current = Some(tests[i]);
            let probe = self.iconst(func, 128, value);
            let same = self.emit(
                func,
                frame,
                mir::InstKind::Icmp {
                    pred: mir::IcmpPred::Eq,
                    lhs: ctx.scrutinee,
                    rhs: probe,
                },
                mir::Ty::Bool,
            );
            let otherwise = if i + 1 < tests.len() { tests[i + 1] } else { default };
            self.terminate(
                func,
                frame,
                mir::Terminator::CondBr {
                    cond: same,
                    then_block: dest,
                    else_block: otherwise,
                    then_args: Default::default(),
                    else_args: Default::default(),
                },
            );
        }
    }

    /// Head slot of the `alloca` block list, created empty in the entry
    /// block on first use: one list per call, however many times a loop
    /// re-runs the `alloca` that fills it.
    pub(crate) fn alloca_head(&self, func: &mut mir::Function, frame: &mut Frame) -> mir::Value {
        if let Some(slot) = frame.alloca_head {
            return slot;
        }
        let slot = self.alloca(func, frame, mir::Ty::ptr(mir::Ty::Void));
        let null = func.push_const(mir::Const::Null);
        let entry = func.entry;
        let inst = func.insts.push(mir::InstData {
            block: entry,
            kind: mir::InstKind::Store {
                addr: slot,
                val: null,
                flags: mir::MemFlags::plain(),
            },
            result: None,
            loc: frame.loc,
        });
        func.blocks[entry].insts.push(inst);
        frame.alloca_head = Some(slot);
        slot
    }

    /// `alloca`: a heap block whose first word links the previous one,
    /// so every return can drain the list. Sixteen bytes of header keep
    /// malloc's 16-byte alignment for the caller's bytes.
    pub(crate) fn lower_alloca(
        &self,
        module: &mut mir::ModuleBuilder,
        func: &mut mir::Function,
        frame: &mut Frame,
        funcs: &HashMap<fe::sema::DeclId, mir::Func>,
        externs: &HashMap<fe::sema::DeclId, mir::Extern>,
        globals: &HashMap<fe::sema::DeclId, mir::Global>,
        size: &fe::check::CheckedExpr,
    ) -> mir::Value {
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let byte_ty = mir::Ty::ptr(mir::Ty::int(8));
        let head = self.alloca_head(func, frame);
        let size = self.expr_value(module, func, frame, funcs, externs, globals, size);
        let header = self.iconst(func, 64, 16);
        let total = self.emit(func, frame, mir::InstKind::Add(size, header), mir::Ty::int(64));
        let malloc = self.extern_fn(module, "malloc", vec![mir::Ty::int(64)], ptr_ty.clone());
        let block = self.emit(
            func,
            frame,
            mir::InstKind::CallExtern {
                target: malloc,
                args: [total].into_iter().collect(),
            },
            ptr_ty.clone(),
        );
        let prev = self.load(func, frame, head, ptr_ty.clone(), false);
        self.store(func, frame, block, prev, false);
        self.store(func, frame, head, block, false);
        let bytes = self.emit(func, frame, mir::InstKind::Bitcast(block), byte_ty);
        let sixteen = self.iconst(func, 64, 16);
        self.emit(
            func,
            frame,
            mir::InstKind::GetElementPtr {
                base: bytes,
                indices: [mir::GepIndex::Index(sixteen)].into_iter().collect(),
            },
            ptr_ty,
        )
    }

    /// Free the `alloca` list at every return. Returns keep their
    /// value through a block parameter; the walk itself is a loop over
    /// the links, so any number of allocations drains.
    pub(crate) fn drain_allocas(&self, module: &mut mir::ModuleBuilder, func: &mut mir::Function, frame: &mut Frame) {
        let head = frame.alloca_head;
        let slots = frame.vla_slots.clone();
        if head.is_none() && slots.is_empty() {
            return;
        }
        let ptr_ty = mir::Ty::ptr(mir::Ty::Void);
        let free = self.extern_fn(module, "free", vec![ptr_ty.clone()], mir::Ty::Void);
        let null = func.push_const(mir::Const::Null);
        let returns: Vec<mir::Block> = func
            .blocks
            .iter_enumerated()
            .filter(|(_, b)| matches!(b.term, Some(mir::Terminator::Return { .. })))
            .map(|(id, _)| id)
            .collect();
        for block in returns {
            let Some(mir::Terminator::Return { val }) = func.blocks[block].term.take() else {
                unreachable!("filtered to returns");
            };
            // VLA objects die with the function: free each site's live
            // allocation before the return.
            if !slots.is_empty() {
                frame.current = Some(block);
                for slot in &slots {
                    let p = self.load(func, frame, *slot, ptr_ty.clone(), false);
                    self.emit_unit(
                        func,
                        frame,
                        mir::InstKind::CallExtern {
                            target: free,
                            args: [p].into_iter().collect(),
                        },
                    );
                }
            }
            let Some(head) = head else {
                // No `alloca` list: restore the plain return.
                frame.current = Some(block);
                self.terminate(func, frame, mir::Terminator::Return { val });
                continue;
            };
            let ret_ty: Vec<mir::Ty> = val.map(|v| func.values[v].ty.clone()).into_iter().collect();
            let has_val = !ret_ty.is_empty();
            let args: Vec<mir::Value> = val.into_iter().collect();
            let check = func.push_block(ret_ty.clone());
            let body = func.push_block(Vec::new());
            let done = func.push_block(ret_ty);
            func.blocks[block].term = Some(mir::Terminator::Br {
                dest: check,
                args: args.iter().copied().collect(),
            });
            frame.current = Some(check);
            let head_val = self.load(func, frame, head, ptr_ty.clone(), false);
            let live = self.icmp(func, frame, mir::IcmpPred::Ne, head_val, null);
            self.terminate(
                func,
                frame,
                mir::Terminator::CondBr {
                    cond: live,
                    then_block: body,
                    else_block: done,
                    then_args: Default::default(),
                    else_args: args.iter().copied().collect(),
                },
            );
            frame.current = Some(body);
            let head_val = self.load(func, frame, head, ptr_ty.clone(), false);
            let next = self.load(func, frame, head_val, ptr_ty.clone(), false);
            self.emit_unit(
                func,
                frame,
                mir::InstKind::CallExtern {
                    target: free,
                    args: [head_val].into_iter().collect(),
                },
            );
            self.store(func, frame, head, next, false);
            let args: Vec<mir::Value> = if has_val { vec![func.blocks[check].params[0]] } else { Vec::new() };
            self.branch(func, frame, check, args);
            frame.current = Some(done);
            let val = if has_val { Some(func.blocks[done].params[0]) } else { None };
            self.terminate(func, frame, mir::Terminator::Return { val });
        }
    }

    pub(crate) fn place_of_var(&self, func: &mut mir::Function, frame: &mut Frame, globals: &HashMap<fe::sema::DeclId, mir::Global>, id: fe::sema::DeclId) -> mir::Value {
        if let Some(slot) = frame.slots.get(&id) {
            // A VLA object's home is the slot holding its heap pointer:
            // the address of the object is the loaded value.
            if Self::ty_is_vla_object(&self.sema.decls[id.0 as usize].ty) {
                return self.load(func, frame, *slot, mir::Ty::ptr(mir::Ty::Void), false);
            }
            return *slot;
        }
        let Some(global) = globals.get(&id) else {
            panic!("variable has no address in this frame");
        };
        let pointee = lower_ty(&self.sema.decls[id.0 as usize].ty, self.sema, self.interner);
        self.emit(func, frame, mir::InstKind::GlobalAddr { global: *global }, mir::Ty::ptr(pointee))
    }

}
