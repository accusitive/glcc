# Exception handling for glcc_mir (plan)

Working plan for the MIR exception story. Written to be resumable after a
context loss: exact files, types, crate anchors, and acceptance tests. The
design mirrors LLVM IR's Itanium-EH subset where possible; the one place we
cannot mirror LLVM freely is codegen, because we lower to Cranelift — but
Cranelift 0.136 turns out to cover the hard part (see "Verified Cranelift
capabilities").

**Status:** not started. Spike first (S0); everything below it is sequenced
behind its outcome.

## How to resume

1. Read this file's status header and the phase list; continue at the first
   unchecked phase.
2. `cargo build`, then the owning package's tests
   (`cargo test -p glcc_mir`, `cargo test -p glcc_lower_cl`) and
   `sh tests/cpp/run.sh` for end-to-end.
3. Build after every edit; small steps, always green (AGENTS.md).

## Goal and requirements

Exceptions must round-trip the Itanium C++ ABI, because glcc-compiled C++
interops with libstdc++ in both directions:

- our frames can throw and catch (tests/cpp/exceptions.cpp),
- exceptions thrown by libstdc++ code (e.g. `std::bad_alloc` from
  `operator new`) unwind *through* our frames and can be caught in them
  (`std_vector.cpp` compiles `std::vector` code whose allocation is wrapped
  in an internal try block today),
- cleanups (destructors) run during unwind in our frames.

Concretely that means: `.eh_frame` FDEs for our functions, LSDAs
(`.gcc_except_table`) describing call sites → landing pads, and the
standard personality (`__gxx_personality_v0`) referenced from the CIE
augmentation. The unwinder (`libgcc_s`) does the stack walk; we never
implement one.

Explicitly out of scope: Windows (SEH/funclets — the existing "no story
yet" note stands), exception specifications/filters (`throw(int)` is gone
in C++17; `noexcept` lowers to a catch-all that calls `std::terminate`),
coroutine/async interplay.

## Verified Cranelift capabilities (anchors, 2026-11)

Cranelift is crates.io `0.136.x` (workspace Cargo.toml:18). Everything
below was checked in the vendored registry source
(`~/.cargo/registry/src/*/cranelift-{codegen,cranelift-frontend}-0.136.5`):

- `try_call` / `try_call_indirect` exist as branch/call instructions with an
  `exception_table` entity (`cranelift-codegen-meta/src/shared/instructions.rs`,
  the `try_call` definition). Semantics are LLVM-`invoke`-shaped: normal
  return → first target with return values as block args; exceptional return
  → the handler matching the thrown tag, with payload values as block args.
- `CallConv::supports_exceptions()` returns true for `SystemV` (among
  others) and `exception_payload_types()` is **two** word-sized values
  (`cranelift-codegen/src/isa/call_conv.rs:96,109`). Both native backends
  implement it with the exact Itanium landing-pad registers:
  `exception_payload_regs` is `rax`/`rdx` on x86-64
  (`isa/x64/abi.rs:979`) and `x0`/`x1` on aarch64
  (`isa/aarch64/abi.rs:1302`); all three ISAs (x64, aarch64, riscv64)
  have `try_call` lowering rules. No Cranelift fork is needed for the
  CLIF layer.
- Per-callsite unwind metadata survives compilation:
  `MachBufferFinalized.call_sites: &[MachCallSite]` where each
  `MachCallSite` has `ret_addr: CodeOffset` and
  `exception_handlers: &[FinalizedMachExceptionHandler]` with
  `Tag(ExceptionTag, CodeOffset)` / `Default(CodeOffset)` entries
  (`cranelift-codegen/src/machinst/buffer.rs:2243,2056`). That is the raw
  material for synthesizing the LSDA: call-site ranges from consecutive
  `ret_addr`s, landing-pad offsets from the handler records. Tags are our
  entities (`glcc_lower_cl` creates the `ExceptionTable`), so we never need
  Cranelift to explain them.
- "Actual unwinding is outside the purview of the core Cranelift compiler"
  (the `try_call` docs) — `.eh_frame`/LSDA emission and the personality are
  ours. Cranelift-object writes `.eh_frame` from its unwind info (the
  `unwind` feature is already enabled, `crates/glcc_lower_cl/Cargo.toml:13`;
  `isa/unwind/systemv.rs` exists), but its CIE does not carry a personality
  augmentation — see "CIE/FDE personality" below.
- `cranelift-frontend` exposes `try_call` directly
  (`cranelift-frontend/src/frontend.rs:2008`).

## Design: MIR mirrors LLVM's Itanium subset

The LLVM-EH pieces that matter for Itanium zero-cost EH are `personality`,
`invoke`, `landingpad` (clauses), `resume`, and `llvm.eh.typeid.for`. MIR
mirrors them as follows.

### IR (`crates/glcc_mir/src/ir/mod.rs`)

- `Func` gains `personality: Option<Func>` (an extern reference, e.g.
  `@__gxx_personality_v0`). C functions and `-fno-exceptions` code leave it
  unset; the C frontend never emits EH ops, so this is C++-only.
- `Terminator::Invoke { callee, kind (internal/extern/indirect), sig, args,
  normal: Block, normal_args, unwind: Block }` — a call that can unwind is
  a terminator with two CFG edges, exactly like LLVM. Plain `Call` keeps
  its meaning (no unwind edge: the callee does not throw — C code, or a
  `noexcept` leaf we trust). Every call emitted inside a live EH scope must
  be an `Invoke`.
- Landing pads are ordinary blocks with two block params — `(ptr(void), i32)`
  (exception object, selector) — distinguished by a per-block `lpad` flag
  carrying the clause list: `clauses: SmallVec<[LpadClause; 4]>` where
  `LpadClause = Catch(Func/typeinfo global) | CatchAll | Cleanup`, in source
  order. Clause order defines the function's type table (LLVM: the
  `landingpad` clause list is the type table; `llvm.eh.typeid.for` indexes
  it). The block's two params receive the unwinder's payload registers via
  the `try_call` catch-target mechanism.
- `Terminator::Resume { exn: Value, sel: Value }` — LLVM `resume`.
- `InstKind::EhTypeId { typeinfo } -> i32` — `llvm.eh.typeid.for`: the
  selector constant for a typeinfo, i.e. its 1-based position in this
  function's type table (0 is reserved for cleanup). Lowered to a constant.

### MIR text (`crates/glcc_mir/src/parse.rs`, `display/`)

```
.func @f sig=(i32) -> i32 personality=@__gxx_personality_v0

b0():
  ...
  %5:i32 = invoke.ext @g sig=(i32) -> i32, i32 %1 to b_norm unwind b_lpad
  ...
b_norm():
  ...
b_lpad(%exn:ptr(void), %sel:i32) .lpad [catch @_ZTIi, catch @_ZTIPKc, cleanup]:
  %tid:i32 = eh.typeid @_ZTIi
  %c:bool = icmp.eq %sel, %tid
  brif %c b_catch_int b_cleanup
b_catch_int():
  %obj:ptr(void) = call.ext @__cxa_begin_catch ptr(void) %exn
  ...                                   # copy out, then
  %ignored:ptr(void) = call.ext @__cxa_end_catch
  ret ...
b_cleanup():
  call.ext @__cxa_end_catch
  resume %exn, %sel
```

- `invoke` / `invoke.ext` / `invoke.ind` mirror the three existing call
  spellings; `to b_norm unwind b_lpad` mirrors LLVM's `to label ... unwind
  label`.
- The `.lpad [...]` clause list is a directive-line variant on the block
  label; a block with `.lpad` must have exactly the two payload params and
  may only be targeted by an `unwind` edge.
- `catch @_ZTIi` references a typeinfo symbol declared with `.dataextern`
  (the emitter declares them; at link time they come from libstdc++/the
  runtime). `catchall` spells the `catch(...)` clause.

### Verifier (`crates/glcc_mir/src/verify/mod.rs`)

- New roots: the set of `.lpad` blocks joins the entry block as dominance
  roots — a landing pad's params are definitions; values defined in a
  landing pad dominate its EH subtree only (LLVM has the same rule via
  EH-pad dominance; the simple reverse-postorder fill in
  `crates/glcc_lower_cl/src/lower/body.rs:block_fill_order` must add the
  lpad roots, which also fixes the "uses value from non-dominating inst"
  Cranelift panics seen with the setjmp shape).
- `invoke`'s `unwind` target must be a `.lpad` block; `Resume` is legal only
  inside an EH subtree of a function with a personality; `EhTypeId`'s
  typeinfo must appear in some `.lpad` clause list of the same function.
- `.lpad` blocks: exactly 2 params, `ptr(void)` + `i32`, and must be
  single-use as `unwind` targets (no normal branch into them).

### Optimizer (`crates/glcc_mir/src/passes/`, `optimize/`)

v1 rules, deliberately conservative:

- Treat `Invoke` as a memory barrier with side effects everywhere
  `Call` already is; additionally no pass may move code between an
  `Invoke` and its `normal` edge target.
- `DeadModuleItems` roots extend to: personality-referenced functions,
  `__cxa_*`/`_Unwind_Resume` externs, everything reachable from `.lpad`
  blocks (they are roots like `init_funcs`).
- No inlining across `Invoke` in v1 (inlining a callee with cleanups
  requires the LLVM "EH prep" normalization; defer).
- `sccp`/`licm`/`sroa` treat `.lpad` blocks as roots too (nothing to
  hoist/sink across them anyway since they are unwind-only).

## EDG side: switch to the zero-cost IL

Today cpfe runs with `DO_FULL_PORTABLE_EH_LOWERING=TRUE`
(`edg/src/targ_def.h:5030`): EH is fully lowered to the portable setjmp
form — `__curr_eh_stack_entry` bookkeeping globals, `_setjmp` calls,
`__internal_rethrow`, per-TU `__dso_handle`/EH bookkeeping data symbols.
That is why the emitted MIR contains all of it (and why `std_vector`
crashes on aarch64: the shape is unsupported there).

Switch to `DO_FULL_PORTABLE_EH_LOWERING=FALSE` (i.e.
`GENERATE_EH_TABLES=FALSE`) in `edg-sys/build.rs`'s define set. The IL the
backend sees then keeps the structured shape
(`edg/src/il_def.h:15834` `stmk_try_block`, `a_try_supplement` with
guarded statement + `a_handler` list, `a_handler.typeinfo_var` guarded on
`DO_IL_LOWERING && !DO_FULL_PORTABLE_EH_LOWERING` at `il_def.h:16336`,
throw as an expression/statement, object lifetimes for cleanups). Note
`c_gen_be` requires portable mode — acceptable: it is only the
`GLCPP_EDG_BACKEND=cgen` debugging path; keep cgen on portable mode by
leaving the define flip mir-backend-only (dispatch on the existing
backend switch before setting defines, or document that cgen + EH is
broken).

`mir_gen_be` (`crates/edg-sys/bridge/mir_gen_be.cpp`) gains:

- `stmk_try_block` walking: emit the guarded statement with a cleanup/catch
  scope pushed; each call emitted inside the scope becomes an `invoke`
  whose `unwind` targets the scope's landing pad.
- One landing pad per try block (merged: identical clause sets share one
  `.lpad` block), body: clause dispatch on `%sel` vs `eh.typeid` of each
  `handler.typeinfo_var`, then `__cxa_begin_catch` + catch-parameter init
  (the `a_handler.dynamic_init` copy) + handler body + `__cxa_end_catch`.
  The cleanup path runs enclosing destructor chains (the object-lifetime
  walk `mir_gen_be` already does for normal scope exits) and ends in
  `Resume`.
- `throw`: `__cxa_allocate_exception` + copy-construct + `__cxa_throw`
  (noreturn; the IL's throw expression carries the typeinfo variable and
  copy routine — same pieces the portable lowering uses to call
  `__throw`). `throw;` (rethrow) → `Resume` of the current `%exn`.
- The setjmp machinery disappears from the output: no `__curr_eh_stack_entry`
  stores, no `_setjmp`, no `__internal_rethrow` (which is what breaks
  libstdc++-only linking on non-x86 today).

`-fno-exceptions`: no `stmk_try_block` appears; nothing changes. If an EH
construct appears in a function anyway (throw inside noexcept, etc.) the
emitter keeps its current contract: fail with a clear diagnostic, never
emit wrong code.

## Lowering and object emission (`crates/glcc_lower_cl`)

- `Invoke` → cranelift `try_call` / `try_call_indirect` with a
  `SystemV`-convention callee; the `unwind` block becomes the catch target
  with the two payload block args; the `normal` edge is the first target.
- `.lpad` blocks lower as normal blocks (their two params are the payload
  block args); `Resume` → `call.ext @_Unwind_Resume(ptr %exn)` (noreturn) +
  `unreachable`; `EhTypeId` → the precomputed selector constant.
- Exception tables: one `Default(lpad)` entry per function (or per
  clause-set group) — catch-vs-cleanup dispatch happens in MIR via the
  selector, exactly like gcc-style lowering, so Cranelift's tag machinery
  stays trivial.
- **LSDA synthesis**: after compiling a function with a personality, walk
  `CompiledCode`'s `buffer.call_sites`: each `MachCallSite` with a
  non-empty `exception_handlers` contributes a call-site record whose range
  is `[previous ret_addr, this ret_addr)` (the standard conservative
  encoding) and whose landing-pad offset comes from the `Default` handler;
  the type table comes from the function's `.lpad` clause lists (typeinfo
  symbol order), the action table from clause adjacency. Emit as
  `.gcc_except_table.<func>` + an `@_GCC_except_table<n>` symbol. This is a
  new module next to the existing object patchup code in
  `crates/glcc_lower_cl/src/lower/mod.rs` (which already post-processes
  `ObjectProduct` for narrow relocations, `.init_array`, and DWARF).
- **CIE/FDE personality**: the FDE's CIE must carry the `P` (personality)
  and `L` (LSDA encoding) augmentation or `_Unwind_RaiseException` will not
  consult the LSDA. Two options, decide in S0:
  1. Post-process the `.eh_frame` Cranelift wrote: inject a supplementary
     CIE with `P`/`L` augmentation and repoint EH functions' FDEs at it
     (byte surgery precedented by the winunwind/narrow-reloc work), or
  2. Suppress Cranelift's `.eh_frame` for personality functions and emit
     our own CIE+FDE pairs from the same `MachineUnwindInfo` (CFA rule +
     saved-register list Cranelift already reports).
- Non-EH functions (all C code) keep Cranelift's plain FDEs — the unwinder
  needs only the CFA rule to unwind through them.

## Runtime pieces

Provided by libstdc++/libgcc_s, nothing to write: `__cxa_throw`,
`__cxa_allocate_exception`, `__cxa_free_exception`, `__cxa_begin_catch`,
`__cxa_end_catch`, `__cxa_rethrow` (unused once rethrow is a `Resume`),
`_Unwind_Resume`, `__gxx_personality_v0`, typeinfo symbols. EDG's
`libC.a` keeps `operator new`/arrays/RTTI helpers; its setjmp EH machinery
(`throw.o` etc.) simply stops being referenced.

## Phases

- **S0 — spike (do first).** Hand-written CLIF (not through MIR): a
  function with `try_call` under `SystemV`, compiled via cranelift-object,
  plus a hand-written LSDA + CIE augmentation; call it from a libstdc++-
  throwing frame and catch. Proves the unwinding path end to end
  (`.eh_frame` augmentation approach, option 1 vs 2 above, and the
  LSDA-from-`call_sites` encoding — the payload registers themselves are
  already verified, see the anchors above). Timebox this; it de-risks
  everything.
- **P0 — MIR plumbing.** `ir/mod.rs` (`Terminator::Invoke`, `Resume`,
  `EhTypeId`, `Func::personality`, lpad flag + clauses), `parse.rs`,
  `display/`, `verify/mod.rs`, optimizer guards. Text round-trip tests in
  `glcc_mir`. `cargo test -p glcc_mir` green.
- **P1 — lower_cl lowering.** `Invoke`/`Resume`/`EhTypeId`/`.lpad` → CLIF
  `try_call` etc. + `block_fill_order` lpad roots. Lowering works with
  *no* unwind metadata yet (throwing into our frames still fails; normal
  paths and MIR-level selector dispatch are testable via the MIR API).
  `cargo test -p glcc_lower_cl` green.
- **P2 — unwind metadata.** LSDA synthesis + CIE/FDE personality.
  Acceptance: `glcpp tests/cpp/exceptions.cpp -o ex && ./ex` prints the
  g++-matching line with the *setjmp* EDG mode still in place is NOT
  required — P2 can be tested with a hand-written MIR function that calls
  `__cxa_throw` (libstdc++) and catches in an `.lpad`.
- **P3 — EDG zero-cost mode.** Define flip + `mir_gen_be` EH emission as
  above. Acceptance: `sh tests/cpp/run.sh` — `exceptions` leaves xfail;
  `std_vector` stops needing `__internal_rethrow` (removes the last
  non-x86 libC.a dependency).
- **P4 — polish.** `noexcept` → terminate, function-try-blocks,
  rethrow ordering, optimizer EH-awareness beyond the barriers, and the
  aarch64 re-check of everything above (the plan is written target-neutral
  but only x86-64 has working EH-shaped code today).

## Risks / open decisions

- `try_call` under `SystemV` on x86-64/aarch64: `supports_exceptions()`
  says yes, but the backends' payload-register handling for native ABIs is
  the least-exercised path in Cranelift — that is exactly what S0 exists to
  prove before any MIR work lands.
- CIE augmentation strategy (post-process vs own FDEs) — decide in S0.
- If a target ABI ever lacks two payload registers for `SystemV`, the
  fallback is the Cranelift fork the old doc assumed; keep that door noted
  but don't build it ahead of need.
- Interaction with `fern_exec`/JIT paths: JIT-registered code needs the
  same FDE registration (`__register_frame`) — flag for the fern targets
  when EH lands there; out of scope until then.
