//! Scheduling model and the (currently program-order) scheduler. The
//! model is pure data: per-opcode latency and throughput, with trivial
//! defaults, so targets improve codegen by filling a table. The
//! scheduling pass itself lands as a post-RA list scheduler; until then
//! this module defines the vocabulary and preserves program order.

use crate::machine::{MFunction, OpId};

#[derive(Clone, Copy)]
pub struct SchedEntry {
    pub op: OpId,
    /// Cycles until the result is usable.
    pub latency: u8,
}

#[derive(Clone, Copy)]
pub struct SchedModel {
    pub entries: &'static [SchedEntry],
}

impl SchedModel {
    /// Program order, latency 1: the correct default for any target
    /// that has not filled in its model yet.
    pub const TRIVIAL: SchedModel = SchedModel { entries: &[] };

    pub fn latency(&self, op: OpId) -> u8 {
        self.entries.iter().find(|e| e.op == op).map(|e| e.latency).unwrap_or(1)
    }
}

pub fn schedule(f: &mut MFunction, _model: &SchedModel) {
    // Program order is always correct; the list scheduler that reorders
    // within dependency bounds is a later milestone.
    let _ = f;
}
