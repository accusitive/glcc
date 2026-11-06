# Modularize builtins + implement legacy `__sync`

## Context

Compiling a triangle example for the mingw target fails on missing `__sync_*` builtins. The audit found builtin-function knowledge scattered across three name-matching sites in two crates: `parse.rs::declare_builtin_if_called` (prototype table), `check.rs` (~1000 lines of `check_*` helpers plus the `AtomicOp`/`RmwOp`/`FpPredicate`/`OverflowOp` enums at check.rs:53–160), and `glcc_lower_mir/src/lower/builtin.rs` (re-matching names to lower). Meanwhile `ext/` already models the right pattern for keyword spellings, and `__atomic_*` already flows through a clean pipeline (inline check → `AtomicOp` node → `lower/atomic.rs` → MIR atomic instructions).

Plan: restructure the frontend's builtin *checking* into a family-split `check/` directory (mirroring the `lower/` module pattern), then implement `__sync` as a new family that maps onto the existing `AtomicOp` pipeline — per the GNU docs it is defined in terms of `__atomic`, and the lowerer already ignores memory orders (everything seq-cst), so no new lowering exists beyond one CAS variant. ia32 builtins are explicitly a later step (`check/ia32.rs` placeholder only).

## Phase 1 — restructure (mechanical, keep green)

1. `mkdir crates/glcc_frontend/src/check/` + `git mv .../check.rs .../check/mod.rs` (preserves history/blame).
2. Move code out of `check/mod.rs` into `use super::*; impl Sema` submodules (descendant modules see `Sema`'s private methods, so no visibility churn):
   - `check/atomic.rs`: `check_atomic`, `check_c11_atomic`, all `check_atomic_*` / `check_c11_atomic_*` helpers (check.rs:3270–3884), plus `AtomicOp` + `RmwOp`.
   - `check/gnu.rs`: `check_clzg`, `check_rotate`, `check_constant_p`, `check_fp_class`, `check_fpclassify`, `check_classify_type`, `check_choose_expr`, `check_prefetch`, `check_alloca`, `check_await`, `check_frame_address`, `check_overflow`, plus `OverflowOp`, `FpPredicate`, and `typed_overflow_builtin`.
   - `check/msvc.rs`: `check_interlocked_cmpxchg` (check.rs:3646).
3. `check/mod.rs` re-exports the four enums (`pub use atomic::{AtomicOp, RmwOp}; ...`) so external users (`glcc_lower_mir`'s `lower/atomic.rs`, `lower/fp.rs`, plus in-crate `dump.rs`/`eval.rs` via `fe::check::*`) keep compiling unchanged.
4. Slim `check_call`'s dispatch (check.rs:3103–3237): the `__atomic_`/`__c11_atomic_`/`_InterlockedCompareExchange` prefix gates and the one-off name match become short delegations to the family modules.
5. `cargo build && cargo test -p glcc_frontend` — must be green before Phase 2.

## Phase 2 — centralize the prototype table

Move the name match in `parse.rs::declare_builtin_if_called` (parse.rs:3432–3579) into a pure `check/prototypes.rs`: `pub(super) fn builtin_prototype(name: &str) -> Option<(Ty, Vec<Ty>, bool)>`. `parse.rs` keeps only the `Binding` construction and symbol lookup. (The `__sync_synchronize` entry is removed in Phase 3.)

## Phase 3 — `check/sync.rs`: legacy `__sync` builtins

Checked inline at the top of `check_call` (`text.starts_with("__sync_")`), producing existing/new `AtomicOp` nodes — no declared prototypes needed:

| Name | Node |
|---|---|
| `__sync_synchronize` | `AtomicOp::Fence { signal: false }` (lowering already emits `AtomicFence`; drop its parse-table + `ScalarBuiltin::Sync` entries, keep `_ReadWriteBarrier` there) |
| `__sync_fetch_and_{add,sub,and,or,xor,nand}` | `Rmw { op, fetch: true, sized: None }` |
| `__sync_{add,sub,and,or,xor,nand}_and_fetch` | `Rmw { op, fetch: false, sized: None }` |
| `__sync_val_compare_and_swap` | **new** `AtomicOp::SyncCmpXchg { want_bool: false }` |
| `__sync_bool_compare_and_swap` | `SyncCmpXchg { want_bool: true }` |
| `__sync_lock_test_and_set` | `ExchangeN { sized: None }` (2 args; lowerer already ignores trailing order args) |
| `__sync_lock_release` | `StoreN { sized: None }` with a synthesized constant `0` converted to the pointee type |

Checking rules (per GNU docs): `ptr` must point to a scalar of size 1/2/4/8 that is integer or pointer type (reject `_Bool`/`bool` and other sizes); the value operand converts to the pointee type — reuse the pointee-validation helpers in `check/atomic.rs` (`atomic_n_value_ty` / `check_atomic_sized`). Pointer pointees work as-is since atomic ops are unscaled. NAND uses the existing `RmwOp::Nand` (GCC 4.4 semantics, which the MIR `AtomicRmwOp` lowering already implements).

Lowering: add the `SyncCmpXchg` arm in `glcc_lower_mir/src/lower/atomic.rs` — same CAS as `InterlockedCmpxchg` (takes expected/replacement by value, returns the old value), then for `want_bool` an `Icmp` against the expected value → `convert_bool`.

`glcc_pp/src/expand.rs` `has_builtin` (closed table, expand.rs:178): add the `__sync_*` spellings so `__has_builtin` in mingw headers takes the right path.

### Tests

- Frontend (`glcc_frontend/src/tests/mod.rs`, pattern at :6696): `__sync_fetch_and_add` type-checks against `char`/`short`/`int`/`long long`/pointer pointees; errors for `_Bool` pointee, struct pointee, wrong arity.
- Lower/exec (`glcc_lower_mir/src/lower/tests.rs`, JIT pattern at :2663): fetch-add returns old; add-and-fetch returns new; NAND = `~(tmp & v)`; val CAS returns old value and stores only on match; bool CAS returns success flag; lock_test_and_set + lock_release round-trip.

## Verification

1. `cargo build` + `cargo test -p glcc_frontend` + `cargo test -p glcc_lower_mir` after each phase (small steps, always green; no workspace-wide `cargo fmt`).
2. End-to-end: build the user's triangle example — `~/data/p/glcc/target/debug/glcc --target=x86_64-windows-w64-mingw -I/usr/x86_64-w64-mingw32/include/ -fms-extension tri.c -o /tmp/tri` — and confirm the `__sync` errors are gone; run under wine if available.

## Out of scope (next steps, recorded not implemented)

`check/ia32.rs` for `__builtin_ia32_*` (what currently blocks windows.h), the remaining `_Interlocked*` RMW spellings (`_InterlockedAdd`/`_InterlockedExchange` — trivially `Rmw` nodes once ia32 lands), and splitting the lower-side `builtin.rs` name matches.