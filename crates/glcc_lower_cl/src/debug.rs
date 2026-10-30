//! Debug record collection for the Clif lowering.
//!
//! While a MIR function lowers, `emit_inst` interns each instruction's
//! source location into a [`LoweredDebug`]; those indices become the
//! Cranelift `SourceLoc`s the compiled code carries back. Once the
//! function is defined, its compiled size, source map, and frame layout
//! resolve the MIR debug records into the machine ranges and
//! frame-relative homes `glcc_dwarf` emits.

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::{SourceLoc, StackSlot};
use cranelift_codegen::{CompiledCode, MachBufferFrameLayout};
use cranelift_module::{DataId, FuncId};
use glcc_dwarf::{FuncDebug, LoweredDebug, SrcLocRow, VarLoc};
use glcc_mir as mir;

/// CFA bias on SysV: the CFA sits two words above the frame pointer on
/// all supported targets (the pushed frame-pointer/link pair)
/// regardless of where within the frame the slots live.
const CFA_BIAS: i64 = 16;

/// Snapshot one just-defined function into `debug`.
pub(crate) fn collect_function_debug(
    debug: &mut LoweredDebug,
    func_id: FuncId,
    func: &mir::Function,
    slots: &HashMap<mir::Value, StackSlot>,
    code: &CompiledCode,
    data_ids: &HashMap<mir::Global, DataId>,
    tls_data: &HashSet<mir::Global>,
) {
    let frame = code.buffer.frame_layout();
    let ctx = HomeCtx {
        slots,
        frame,
        fb_base: frame.map(|l| l.frame_to_fp_offset as i64 + CFA_BIAS),
        data_ids,
        tls_data,
    };
    let (params, vars, scopes) = match &func.debug {
        Some(mir_debug) => (
            mir_debug.params.iter().map(|var| lower_var(var, &ctx)).collect(),
            mir_debug.vars.iter().map(|var| lower_var(var, &ctx)).collect(),
            mir_debug.scopes.iter().map(|scope| lower_scope(scope, &ctx)).collect(),
        ),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };
    debug.funcs.push(FuncDebug {
        func_id,
        name: func.name.clone(),
        decl: func.debug.as_ref().and_then(|d| d.decl),
        size: code.code_buffer().len() as u32,
        srclocs: code
            .buffer
            .get_srclocs_sorted()
            .iter()
            .filter(|row| row.loc.is_abs() && row.loc.as_abs() != SourceLoc::default())
            .map(|row| SrcLocRow {
                start: row.start,
                end: row.end,
                loc: row.loc.as_abs().bits(),
            })
            .collect(),
        external: !func.local,
        params,
        vars,
        scopes,
    });
}

/// Collect one module-level object (file-scope variable or block-scope
/// static) into `debug`. Thread-locals are skipped: a plain
/// `DW_OP_addr` would be wrong for them. `extern` declarations have no
/// storage here, so they become location-less declaration DIEs — and
/// an undeclared-unused import (`None`) still yields that DIE.
pub(crate) fn collect_global_debug(debug: &mut LoweredDebug, global: &mir::GlobalData, data: Option<DataId>) {
    let Some(record) = &global.debug else { return };
    if global.tls {
        return;
    }
    debug.globals.push(glcc_dwarf::GlobalDebug {
        name: record.name.clone(),
        decl: record.decl,
        debug_ty: record.debug_ty,
        data: if global.external { None } else { data },
        external: !global.local,
    });
}

/// Everything a variable home resolves against in one function.
struct HomeCtx<'a> {
    slots: &'a HashMap<mir::Value, StackSlot>,
    frame: Option<&'a MachBufferFrameLayout>,
    fb_base: Option<i64>,
    data_ids: &'a HashMap<mir::Global, DataId>,
    tls_data: &'a HashSet<mir::Global>,
}

/// Resolve one MIR variable record: its slot's frame offset from the
/// CFA, or its data symbol. A record whose home no longer exists in the
/// emitted code (the optimizer promoted its slot, and the value now
/// moves between registers or was folded away) becomes
/// [`VarLoc::OptimizedOut`] rather than being dropped, so the DIE still
/// names the variable and describes its type.
fn lower_var(var: &mir::VarDebug, ctx: &HomeCtx<'_>) -> glcc_dwarf::VarDebug {
    let loc = match var.home {
        mir::VarHome::Slot(value) => ctx
            .slots
            .get(&value)
            .and_then(|slot| ctx.frame?.stackslots.get(*slot))
            .and_then(|slot| ctx.fb_base.map(|base| i64::from(slot.offset) - base))
            .map(VarLoc::Fbreg)
            .unwrap_or(VarLoc::OptimizedOut),
        mir::VarHome::Global(global) => match ctx.data_ids.get(&global) {
            Some(data) if !ctx.tls_data.contains(&global) => VarLoc::Static(*data),
            _ => VarLoc::OptimizedOut,
        },
    };
    glcc_dwarf::VarDebug {
        name: var.name.clone(),
        decl: var.decl,
        debug_ty: var.debug_ty,
        loc,
    }
}

fn lower_scope(scope: &mir::ScopeDebug, ctx: &HomeCtx<'_>) -> glcc_dwarf::ScopeDebug {
    glcc_dwarf::ScopeDebug {
        start: scope.start,
        end: scope.end,
        vars: scope.vars.iter().map(|var| lower_var(var, ctx)).collect(),
        children: scope.children.iter().map(|child| lower_scope(child, ctx)).collect(),
    }
}
