# Inline asm implementation plan (P0 → P2)

This file is the working plan for the inline-asm feature. It is written to be
resumable after a context loss: it names exact files, types, line anchors and
acceptance tests. Read `docs/inline-asm.md` (design) and
`docs/inline-asm-catalogue.md` (what the torture tests do) first.

**Status:** P0, P1a and P1b are done, plus `asm goto` and `%=` from P2.
Torture 1442/1514, ctestsuite 220/220 on x86-64/aarch64/riscv64. The
only unbuilt piece is the T2 stack-shared wrapper (decision below).

## How to resume

1. Read this file's progress log; continue at the first unchecked step.
2. `cargo build`, then `cargo test --workspace`.
3. Build after **every** edit; small steps, always green. Commit per step.

## Ground rules

- GNU C only: `asm`/`__asm__`/`__asm` are keywords here.
- Parser errors flow as `Result` then emit via codespan and panic
  (`Parser::emit` / `unwrap_diag`); `#[should_panic]` tests match message
  substrings, keep prefixes stable.
- `parse_statement`'s frame size is load-bearing (large translation
  units overflow the 2 MB test thread); new parser code must live in
  helpers that take only references, not grow `parse_statement`'s frame.
- No chain-of-thought comments; self-documenting code first.
- `glcc` on PATH is stale: use `target/debug/glcc`.

## Verified anchors (tree at `a81f64f`, 2026-10-16)

Frontend:
- `parse_statement` `parse.rs:1031`; first line calls `skip_gnu_suffixes()`
  (`1036`); `skip_gnu_suffixes` `3509`; `parse_gnu_annotations` `3519` — it
  **eats `__asm__(...)` when followed by `(`** (`3528`), so a statement-start
  `__asm__("")` is currently swallowed as an annotation.
- `peek_is_gnu` `1766`; `string_literal_body` `3723` (raw escapes);
  `eval::string_bytes` `eval.rs:644` (decodes escapes).
- `parse_translation_unit` `449`; top-level `__asm__` skip `453–462`.
- `Stmt` enum `5932`; `HasSpan` `6005`; `stmt_has_label` `6031`;
  `fold_constant_if` statement match starts `1330`; `check_flow_stmt` `1475`;
  `Stmt::Goto` goto collection `1554`; `LabelAddr` use `1618`.
- `sema.bodies: HashMap<DeclId, FuncBody>` `sema.rs:239`; `FuncBody.block.items`
  holds `BlockItem::Stmt(Stmt)`; `remap_stmt_ident` `sema.rs:1076`.
- `GnuAttributes` `parse.rs:5056`; attribute list parser `3589`.
- `dump.rs` matches `Stmt` (15 sites).
- `Sema::is_lvalue` `check.rs:672`; `readonly_error` `check.rs:954`;
  `check_expr` `1718`; `check_assign` `2224` (lvalue/const pattern).

Lowering:
- `glcc_lower_mir` `lower_stmt` `lower.rs:1689`; `expr_value` `2156`;
  `place(...)` (lvalue address) used at `2732`; `store(...)` `2735`;
  `convert(...)`; `ty_is_volatile` `7494`; `access_volatile` `7509`.
- Test harness `run_main(src) -> i32` (C → MIR → JIT) `lower.rs:7596`;
  example tests `assignment_forms` `9346`.
- MIR: `InstKind` `glcc_mir/src/ir.rs:133`; `Terminator` `469`;
  `InstData { block, kind, result }` `543`; `InstKind` is matched in
  `builder.rs`, `display.rs`, `ir.rs`, `passes.rs`, `verify.rs`,
  `glcc_lower_cl/src/lib.rs`, `glcc_lower_cl/src/lower.rs`.

Driver:
- `compile_source` `driver.rs:891` returns `Vec<u8>`; pipeline: parse `906`,
  `glcc_lower_mir::lower_translation_unit(&sema,&interner,&tu)` `908`, MIR
  passes `914–923`, `glcc_lower_cl::lower_module(&mir_module,&mut obj,cl_target)`
  `968`, `obj.finish().emit()` `969`.
- Callers `1455`, `1613`; `-c` temp-object handling `1773`;
  `assemble_cmd` (`cc -c`) `994`; `link_command` `1137`; `emit_output` `1164`.
- Only the driver calls `lower_translation_unit`/`lower_module` (checked);
  changing their signatures cannot break `fern`/`fernc`.

Tests:
- Frontend tests inspect `parser.sema.bodies[&fid]` after
  `parser.parse_translation_unit()` (e.g. `tests/mod.rs:3906`).
- Torture subset: `./tools/torture.bash --no-build <basename-substring...>`.
- Cross-target recipe (from prior work): `clang --target=$t-linux-gnu -c
  rt/glccrt.c -o rt-$t.o`; `GLCC_CC=$ROOT/tools/zig-cc
  GLCC_RT=rt-$t.o target/debug/glcc --target=$t -w src.c -lm -o bin -static`;
  `qemu-$t-static bin`.

## P0 — parse + empty-template semantics (no backend work)

Scope: `asm` statements with an empty (or whitespace-only) template.
Non-empty templates are a **parse-time error** in P0
(`"non-empty asm templates are not supported yet"`), removed in P1b.
`asm goto` parses, records its labels, and falls through.

### P0.1 AST (`parse.rs`)

Add next to `Stmt` (`5932`):

```rust
pub struct AsmOperand {
    pub name: Option<SymbolU32>,   // [name]
    pub constraint: String,        // raw: "=" / "+" / "&" prefix + letters/digits
    pub expr: CheckedExpr,
    pub span: Span,
}

pub struct AsmStmt {
    pub template: String,          // decoded (escapes processed)
    pub outputs: Vec<AsmOperand>,
    pub inputs: Vec<AsmOperand>,
    pub clobbers: Vec<String>,
    pub labels: Vec<SymbolU32>,    // asm goto targets
    pub is_volatile: bool,
    pub is_goto: bool,
    pub span: Span,
}
```
`Stmt::Asm(AsmStmt)`; helper methods on `AsmOperand`:
`fn is_output(&self)`, `fn is_inout(&self)` (starts with `+`),
`fn is_early(&self)` (`&`), `fn tie(&self) -> Option<usize>` (constraint is
`0`–`9` after the `=`/`+`/`&` prefix).

### P0.2 Parser (`parse.rs`)

- `parse_statement` (`1031`): before `skip_gnu_suffixes()` add
  `if let Some(stmt) = self.try_parse_asm_statement(start) { return stmt; }`.
- `try_parse_asm_statement(&mut self, start: Mark) -> Option<Stmt>`:
  1. advance over any leading `__extension__` markers;
  2. if the token is `Identifier` resolving to `asm`/`__asm__`/`__asm` **and**
     the next token after optional qualifiers (`volatile`/`__volatile__`,
     `inline`/`__inline__`, `goto`/`__goto__`) is `(`, parse it; else return
     `None` (consumed `__extension__` markers are no-ops).
  3. qualifiers: `volatile` → `is_volatile`; `goto` → `is_goto`; `inline` ignored.
  4. template: collect consecutive `Token::StringLiteral`s, decode with
     `eval::string_bytes`; error on undecodable; P0 gate: reject when
     `template.trim()` is non-empty.
  5. sections, each optional: `:` outputs, `:` inputs, `:` clobbers,
     `:` goto-labels, then `)` and `;`.
- `parse_asm_operands(&mut self, outputs: bool) -> Vec<AsmOperand>`: loop
  `[name]` (optional) + string-literal constraint + `( expr )`; outputs use
  `sema.check_asm_output`, inputs `sema.check_expr`; comma-separated; stops at
  `:`/`)`.
- `parse_asm_string_list` (clobbers) and `parse_asm_label_list` (identifiers).
- `parse_translation_unit` (`453`): extend the top-level `__asm__` skip to all
  three spellings (P1a replaces the skip with collection).
- Update every exhaustive `Stmt` match: `HasSpan` (`6005`),
  `stmt_has_label` (`6031`, asm defines no labels → `false`), `fold_constant_if`
  (`1330`), `check_flow_stmt` (`1475`: walk operand exprs with
  `check_flow_expr`, push `asm.labels` into `gotos` like `Stmt::Goto` `1554`),
  `dump.rs`, `sema.rs remap_stmt_ident` (`1076`: remap operand exprs).
  The compiler will list any remaining matches.

### P0.3 Checker (`check.rs`)

```rust
pub fn check_asm_output(&self, expr: &Expr, interner: &Interner)
    -> Result<CheckedExpr, CompileError>
```
mirrors `check_assign` (`2224`): check, require `is_lvalue`, reject `const`
(`readonly_error`). Bit-field outputs: test `gcc -c` behaviour; if GCC accepts,
handle in lowering via `bitfield_store` (copy the `Assign` arm at `2704`); if
GCC rejects, reject in the checker.

### P0.4 Lowering (`glcc_lower_mir`)

`lower_stmt` (`1689`) arm `Stmt::Asm(asm)` → new helper:
1. evaluate every input via `expr_value` in order (side effects are the point);
2. for every output, evaluate the lvalue address via `place(...)`;
3. if the output has `tie(n)`: `store` the tied input's value converted to the
   output type (`convert`), with `access_volatile`;
4. `+` and untied `=` outputs: nothing beyond step 2 (value unchanged);
5. `is_goto`/`clobbers`/`is_volatile`: no code in P0 (documented gap; the MIR
   node in P1b provides barrier semantics).

No MIR changes in P0. Empty templates emit nothing, matching the model
experiment in the catalogue.

### P0.5 Tests

Frontend (`crates/glcc_frontend/src/tests/mod.rs`, pattern at `3906`):
- `asm("")`, `asm volatile("" ::: "memory")`, `asm("" : "=r"(y) : "r"(x))`,
  `asm("" : "+r"(x))`, named operands, `asm goto("" :::: lab)`;
- `asm("" : "=r"(1))` errors (not an lvalue);
- `asm("nop")` errors (non-empty, P0);
- `void f(void){ __asm__("" ::: "memory"); }` is parsed as a statement, not
  swallowed by `skip_gnu_suffixes` (regression for `3528`).

Lowering (`run_main`, `lower.rs:7596`):
- `asm_plus_identity`: `int main(){int x=7; asm("" : "+r"(x)); return x;}` → 7;
- `asm_tie_assigns`: `int main(){int y; asm("" : "=r"(y) : "r"(x))` with
  `x=7` → 7;
- `asm_untied_output_keeps_value`: `int y=5; asm("" : "=r"(y)); return y;` → 5.

### P0.6 Suite check

`cargo build`, then `./tools/torture.bash` and compare with the catalogue appendix:
the 65 asm tests should reach the model's ~57 passes (remaining failures are
the ones the catalogue lists as needing out-of-scope features). `tools/ctestsuite.bash`
must stay 220/220.

## P1a — the `.s` pipeline, file-scope asm, naked functions

1. **File-scope asm**: replace the `parse_translation_unit` skip (`453`) with
   collection into `Sema.toplevel_asm: Vec<(String, Span)>` (decoded template).
2. **Thread the text**: `compile_source` (`891`) returns the object bytes plus
   the per-TU `.s` text. Collect `sema.toplevel_asm` + text returned by
   `lower_cl`. Change `lower_module` to a collecting variant
   (`lower_module_collecting(&mir, &mut obj, target, &mut String)`) and keep
   `lower_module` delegating with a throwaway buffer (fern/fernc call neither,
   but this keeps the API stable).
3. **Driver**: when the text is non-empty, write `<stem>-asm.s` in the scratch
   dir, assemble with `assemble_cmd`-style `cc -c` (`994`), and
   - link flow: pass the extra object to `link_command` (`1137`);
   - `-c` flow (`1773`): merge with `cc -r` / `ld -r` (mold may lack `-r`;
     fall back to bfd/lld);
   - `-save-temps`: keep the `.s`.
4. **Naked functions**: parse `naked` in `GnuAttributes` (`5056`) and the
   attribute list parser (`3589`); store on the function declaration like
   `aligned`. In `lower_translation_unit`, a naked definition becomes an
   extern MIR declaration plus a text block; the body must consist only of
   asm statements (error otherwise). Emit in the `.s`:
   `.globl name`/`.local`, `.type name,@function`, `name:`, body, `.size`.
5. **Tests**: file-scope asm defining a function called from C; naked function
   returning a constant; both through the driver (a new driver integration
   test); `-c` produces a single object.

## P1b — real templates (out-of-line wrappers)

Design: `docs/inline-asm.md` Design A. Steps:

1. MIR: `InstKind::InlineAsm { block: Box<AsmBlock> }` with
   `AsmBlock { template, operands: Vec<AsmOperand>, clobbers, is_volatile }`,
   `AsmOperand { value: Value, is_addr: bool, constraint: String, tie: Option<u32>, ty: Ty }`
   (`value` = input value, or output address). Update every `InstKind` match
   (list in anchors); make DCE and every pass treat it as side-effecting;
   `escaping_addresses` must treat its address operands as escaping.
2. `glcc_lower_mir`: emit the block instead of P0's direct stores; remove the
   non-empty parse-time gate.
3. `glcc_lower_cl`: new `src/asm.rs`:
   - area stack slot (16-aligned, sized for operands + clobber saves);
   - store inputs / load outputs around a call to the wrapper;
   - wrapper emission: static per-site register allocation (outputs first,
     then inputs; ties share; `&` forbids overlap; named constraints pin;
     clobbered registers excluded), reserved register chosen from callee-saved
     registers the block does not name (x86-64 rbx/r12–r15), clobber
     save/restore for callee-saved clobbers, even stack alignment;
   - template substitution: `%%`, `%N`, `%cN`, `%nN`, `%[name]`, and x86
     `%b/%w/%k/%q/%h`; `"m"` → address in a scratch register, `%N` expands to
     `(%scratch)`; `"i"`/`"n"` → textual constant (via `eval_const`); `"g"` →
     `"r"`;
   - wrapper symbol `__glcc_asm_<hash(tu)>_<n>`, declared as an import
     (`Linkage::Import`) so the call links against the `.s` object;
   - per-target constraint tables; x86-64 first, then aarch64/riscv64.
4. Tests: `run_main` for `rdtsc`/`cpuid`/`bswap`/`lock xadd`, memory operands,
   ties, multi-output, clobber saves, `%c0` immediates; driver-level link test.

## P2 — stack-shared wrappers and `asm goto`

Design: `docs/inline-asm.md` Design B.
- Wrapper entry `pop` → stash return address in the area; template runs with
  the caller's `%rsp`; exit `mov stash,%r11; jmp *%r11`; clobber saves use the
  area (never `%rsp`).
- Use for templates that mention the stack pointer; keep plain wrappers
  otherwise.
- `asm goto` — **implemented** (see below); it turned out not to need the
  stack-shared wrapper at all.
- Tests: context-switch-shaped asm that reads/writes `%rsp`.

### P2 notes: `asm goto` (implemented)

- Parse: the non-empty-template gate is gone. `asm goto` requires a
  non-empty label list ("asm goto needs at least one label"); a label list
  on a plain `asm` is rejected ("asm label list requires asm goto", GCC's
  grammar has no fourth colon there). `%l[name]` references are scanned
  out of the template (skipping `%%`) and must name a listed label, else
  "undefined named operand '<name>'" (GCC's wording); the flow checker
  already treats the label list as gotos, so an undefined *function*
  label still reports "label '<x>' used but not defined".
- MIR: `AsmBlock.labels: Vec<String>` (selector order). A goto block's
  `InlineAsm` carries a result (the i64 selector); lowering then merges
  bit-field outputs and terminates with
  `Terminator::Switch { scrutinee, cases: [(i+1, label_i)], default:
  fallthrough }`, so the fallthrough continues in a fresh block. Output
  stores happen on every path before the dispatch.
- Wrapper: signature `(area: i64) -> i64` for goto blocks. The template's
  `%l[name]` expands to `.Lasm<hash>_<i>`. The epilogue gains a merge
  label: fallthrough writes selector 0, then stores, then a move into the
  return register; one stub per label follows the `ret`, writing
  `i + 1` and jumping back to the merge. The selector lives in the first
  pool register no operand uses (it must survive the stores; it is never
  pushed, so the pops cannot clobber it). The wrapper hash covers the
  label list, since it fixes the selector numbering.
- Per-target jump mnemonics for the stubs: `jmp` (x86-64), `b` (aarch64),
  `j` (riscv64) — `jmp` is not accepted by the other two assemblers.
- Tests: `asm::tests` (selector text per target, hash covers labels),
  `passes::tests::inlines_callee_with_inline_asm`,
  `lower_mir::tests::asm_goto_dispatches_on_the_selector`,
  `driver::tests::asm_goto_wrapper_returns_a_selector`; end-to-end
  one/two-label, output-on-both-paths, loop-back, unconditional-jump and
  shared-label programs on x86-64/aarch64/riscv64 at `-O0`/`-O1`/`-O2`.

## Test commands

```
cargo build
cargo test -p glcc_frontend -p glcc_lower_mir -p glcc_lower_cl -p glcc_mir
./tools/torture.bash --no-build <names...>        # subset by basename substring
./tools/torture.bash -j"$(nproc)"                 # full, ~1384/1514 before asm
./tools/ctestsuite.bash                           # must stay 220/220
```

## Progress log

- [x] P0.1 AST
- [x] P0.2 parser
- [x] P0.3 checker
- [x] P0.4 lowering
- [x] P0.5 tests
- [x] P0.6 suite check
- [x] P1a.1 file-scope asm collection
- [x] P1a.2 text threading
- [x] P1a.3 driver `.s` pipeline
- [x] P1a.4 naked functions
- [x] P1a.5 tests
- [x] P1b.1 MIR node
- [x] P1b.2 lowering
- [x] P1b.3 wrapper generation
- [x] P1b.4 tests
- [x] P2 asm goto
- [ ] P2 stack-shared wrappers — not built; naked functions cover the
  stack-switching case (see "P2 decision")

### P2 decision (stack-shared wrappers): not built

Only templates that *read or write* `%rsp`/`sp` inside a non-naked
`asm` need Design B. The bare-metal use — switch stacks, call C,
switch back — works today through a **naked** function, whose body is
emitted verbatim. Verified end to end on all three targets with the
same shape:

```c
__attribute__((naked)) void run_on_stack(void *sp, void *fn) {
    __asm__ volatile(
        "movq %rsp, %rax\n"     // x86-64: the return address is on the stack
        "movq %rdi, %rsp\n"
        "pushq %rax\n"
        "call *%rsi\n"
        "popq %rsp\n"
        "ret\n");
}
```

AArch64 and RISC-V must also save the *link register* across the inner
call: `blr`/`jalr` overwrites `x30`/`ra`, and the final `ret` then
jumps back into the wrapper (an infinite loop, not a crash). Their
bodies:

```
mov x9, sp;  mov sp, x0;  stp x30, x9, [sp, #-16]!   // aarch64
blr x1;      ldp x30, x9, [sp], #16;  mov sp, x9;  ret

mv t0, sp;   mv sp, a0;   addi sp, sp, -16           // riscv64
sd ra, 0(sp); sd t0, 8(sp); jalr ra, a1
ld ra, 0(sp); ld t0, 8(sp); addi sp, sp, 16; mv sp, t0; ret
```

Reading `%rsp` inside a Design A wrapper yields the wrapper's frame, a
few dozen bytes below the caller's. That is harmless for the common
kernel idiom (`andq $-16384, %rsp` stack-base masking: same thread
stack) and wrong for capturing an exact stack pointer, which needs a
naked function anyway; *writing* `%rsp` in a wrapper is never safe, and
naming it as a clobber is already a panic. So Design B stays on the
shelf until something concrete needs it.

### `aligned` on objects (bare-metal prerequisite, fixed)

Globals now honor `__attribute__((aligned(n)))` exactly like GCC: the
attribute *sets* the object's alignment, raising or lowering it
(`long x __attribute__((aligned(1)))` is 1-aligned and `__alignof__(x)`
reports 1, matching gcc). `declare_global_with_name` takes
`binding.align` when present, else the type's alignment, and
`eval::align_of_expr` reports a declaration's alignment for objects as
well as functions. Before this, `aligned` only reached `__alignof__`
and placement silently used the type's alignment, so static page
tables, DMA buffers and secondary stacks came out misaligned.

Known gap: the same attribute on a *local* is still ignored (locals
are 16-byte aligned at best — `stack_slot`'s fixed exponent). Fixing
it means an alignment field on `InstKind::Alloca`, which the `fern`
frontend's lowering constructs; deferred rather than break it.

### `section` attributes (data objects and naked functions)

`__attribute__((section("name")))` places an object (or a naked
function) in that section, so a linker script can put vector tables,
`.noinit` state or DMA buffers where they belong:

```c
__attribute__((section(".vectors"))) const unsigned long vec[2] = {1, 2};
__attribute__((section(".noinit"))) char scratch[16];
__attribute__((naked, section(".mystart"))) void boot(void) { ... }
```

Data objects go through the object writer, which already had a
`custom_section` field (used for `.init_array`); naked functions get a
`.section name,"ax"` / `.previous` pair around their text. A *normal*
function cannot honor it — the object writer has no per-function
section — so the parser warns ("attribute `section` on a function is
ignored") instead of silently ignoring it, like GCC's own warnings for
attributes it cannot apply.

Known limitation: an uninitialized object in a custom section is
emitted as zero-filled `PROGBITS`, not `@nobits` (the object writer
picks the section flags and ignores a zero initializer when a custom
section is set), so a `.noinit` buffer is zeroed by the loader rather
than preserving RAM. Use file-scope asm (`.section .noinit,"aw",@nobits`)
if that matters.

### Bare-metal verification (all three targets, by hand)

The suites are hosted programs, so the bare-metal path is checked
manually (see `/tmp/opencode/free-*.c`, `naked1*.c`, `m3*.c`):

- freestanding link with no libc — `-nostdlib -nostartfiles -static
  -Wl,-e,_start` — and `_start` written as a naked function that exits
  with a raw syscall (x86-64 `syscall`, aarch64 `svc #0`, riscv64
  `ecall`), all exit 42;
- stack switch: naked body saves the old stack pointer and link
  register, runs C on a static stack, switches back;
- `aligned(4096)` statics, custom sections, `%=`-unique local labels,
  `asm goto`, memory constraints and atomics (`ldxr`/`stxr` with
  `"Ir"`, `lock cmpxchg`, riscv `"+A"`);
- an `interrupt`-style handler is written as a naked function with
  explicit save/restore: GCC's `interrupt` attribute is not implemented
  (it changes the prologue/epilogue, which a Cranelift body cannot
  express) and now warns instead of being ignored silently.

### P1b notes (implemented)

- MIR: `InstKind::InlineAsm { block: Box<AsmBlock> }` (no result for
  plain asm; an `asm goto` block's result is its selector).
  `AsmBlock { template, operands, outputs: u32, clobbers, labels,
  is_volatile }`, `AsmOperand { value, constraint, ty, tie, name }` in
  GCC order (outputs first).
  `for_each_operand`/`remap_inst`/`display` updated;
  the pass list needs nothing else (`is_pure` does not list it, so DCE
  keeps it; `escaping_addresses` walks operands).
- Wrapper symbol: FNV-1a over arch, template, output count, clobbers and
  per-operand (kind, size, tie, immediate) → `__glcc_asm_<16 hex>`,
  emitted `.weak` so identical blocks in different TUs coalesce, and
  deduped per module. The call is an import, so the wrapper must be in
  the TU's `.s`.
- Wrapper ABI: `(area: i64) -> ()`, or `-> i64` for `asm goto` (the
  selector). The caller allocates a 16-byte
  aligned stack slot and stores one 8-byte entry per operand: an
  output's address (memory constraints) or its current value (`+`, or
  an input tied to it), an input's value or address. After the call it
  loads each register output back and stores it through the output's
  address; memory outputs are written in place by the template.
- Allocation is static, outputs first: ties share the target's kind and
  register; named letters pin (a/b/c/d/S/D → rax/rbx/rcx/rdx/rsi/rdi);
  `"Q"` restricts to the high-byte pool (rax/rbx/rcx/rdx); the pool is
  caller-saved (x86-64 rax,rcx,rdx,rsi,r8–r11; aarch64 x1–x17;
  riscv64 a1–a7,t0–t6) minus clobbers and the reserved register. The
  reserved register carries the area address: the first callee-saved
  candidate (x86-64 rbx,r12–r15; aarch64 x19–x28; riscv64 s1–s11) that
  is not clobbered, not named by a constraint and not mentioned in the
  template.
- Callee-saved registers the wrapper writes (named operand registers
  and clobbers) are pushed at entry and popped before `ret`; x86-64
  adds `sub $8,%rsp` when the push count is even so the template runs
  with a 16-byte aligned stack.
- Template substitution: `%N` prints the mode-sized register of the
  operand's type (char→`al`, short→`ax`, int→`eax`, long/pointer→`rax`),
  `%b/%w/%k/%q/%h` sub-registers (x86), `%cN`/`%nN` immediates,
  `%[name]`, `%%`, `%=` (a decimal derived from the wrapper hash, so
  local-label templates `N:`/`Nf`/`Nb` are unique per block and stable
  across TUs), `%l[name]` (an `asm goto` stub label, see the P2 notes);
  unknown `%` sequences are errors. A memory operand expands to
  `(%scratch)` on x86-64 and RISC-V (GAS's `offset(reg)` form, zero
  offset) and `[scratch]` on AArch64. An immediate `%N` carries the
  target's syntax (`$5` x86-64, `#5` aarch64, bare riscv64) like GCC;
  `%cN`/`%nN` strip it. Immediate operands fold any integer constant
  expression in lowering (`eval_const`), not just literals.
- Constraint letters are per-target (`check::constraint_letter`, mirrored
  in `asm::letter_class` for the wrapper side; `check` decides whether
  an input is an address or a value, so the two tables must agree):
  register `r q g X` everywhere, `m o` memory everywhere, `i n s` and
  the range classes `I J K L M N` everywhere; x86-64 adds named
  `a b c d S D A`, high-byte `Q`, `e` and `Z`; AArch64 adds `Q` and the
  `U..` addressing-mode qualifiers as *memory* (`Ump`, `Ush`, ...;
  `Uci`/`Ucj` are registers), and `Z`; RISC-V adds `A` (an address in a
  register) as memory. Letters the target has but the backend cannot
  honor (`S` symbolic addresses, `k` the stack pointer, `w`/`x`/`y`
  FP/SIMD, `Upl`/`Upa` SVE predicates, `f`) panic with what they select.
- Constraints with several alternatives resolve like GCC's matching:
  register beats memory beats immediate for the *first* matching
  alternative. An immediate class followed by a register or memory one
  (`"Ir"`, `"im"`) takes the immediate only when the value is an integer
  constant; otherwise the register (or memory) alternative carries it,
  which is what makes AArch64's `"Ir"` atomic templates work with both
  constant and runtime operands.
- Aggregate operands: an aggregate input under a register constraint
  crosses as its address (what GCC does for arrays); an aggregate tied
  to a memory output is copied into the output's location by MIR (an
  aggregate `Store` becomes a memcpy) and the two operands share the
  address. An aggregate register *output* still panics (impossible
  constraints, as GCC).
- Bit-field outputs round-trip through a unit-typed temp: initialized
  with the field's value for `+`, and merged back with `bitfield_store`
  after the call. Memory constraints on bit-fields are rejected at
  parse time ("cannot take address of bit-field"), as GCC does.
- Parse gates: memory-only constraints require an addressable operand
  (`"m"(x + 1)` → "memory input is not directly addressable");
  `is_lvalue`, `designates_bitfield` and the letter classification
  (`constraint_letters`, `constraint_letter`, `memory_only_constraint`,
  `has_immediate_constraint`) are free functions in `check.rs` so the
  parser and lowering share them.
- Panics: "asm register operand is wider than a register",
  "clobbering the stack pointer is not supported", "asm constraint
  names the clobbered register <r>", "unsupported asm constraint '<c>'"
  (x87 `t`/`u` land here), "asm block needs more registers than the
  target provides".
- Optimized builds: the MIR inliner remaps `InlineAsm` operand values
  (`remap_operands_materialized`), so `-O1`+ no longer panics on
  "inline of InlineAsm ... needs operand mapping" when a function with
  asm gets inlined; `passes::tests::inlines_callee_with_inline_asm`
  covers it. The asm torture subset passes 58/59 at `-O2` (only
  `bitfld-5`, a frontend typing gap, fails).
- Known gaps: x87 register-stack constraints (`990413-2`); register
  variables (`pr51447`); `"S"` symbolic addresses (used by `adrp`/`add`
  pairs); FP/SIMD operands; `aligned` on locals (see the note above).
  Independent failures still open: `bitfld-5`
  (a bit-field expression keeps its declared type, not GCC's
  reduced-precision arithmetic), `pr49218` (Cranelift `udiv.i128`),
  `pr53645`/`-2` (vector literals), `20061220-1` (nested functions).
- Tests: `asm::tests` in `glcc_lower_cl/src/asm.rs` cover wrapper text
  per target (registers, ties, memory, immediates, reserved-register
  choice, clobber saves, sub-registers, mode names, `%=`, dedupe);
  `inline_asm_wrappers_become_assembler_text` in `src/driver.rs` checks
  the collected text; the lower_mir `asm_empty_templates` test now
  inspects MIR instead of JIT (the JIT cannot resolve wrappers).

### P1a notes (implemented)

- Commits `4cf2f66` (collect), `2021025` (`.s` pipeline), `c5d4181`
  (naked). Torture stays 1441/1514; ctestsuite 220/220 ×3.
- File-scope asm is basic asm only (GCC rejects operands there):
  `Sema.toplevel_asm: Vec<(String, Span)>`.
- `compile_source` returns `(Vec<u8>, String)`: bytes plus the per-TU
  assembler text (file-scope templates, then `naked_asm_text`). The
  driver writes `<stem>-<i>-asm.s` in scratch, assembles it with
  `assemble_cmd`, and either adds the object to the link (`link_command`
  gained an `asm_objects` parameter; `do_compile` returns
  `(produced, asm_objects)`) or merges it with `cc -r` for `-c`.
  `-save-temps` writes `<stem>-asm.s`.
- `naked` lives on `Binding` (merges with OR, like `inline`); the
  parser allows non-empty templates while a naked body parses and
  rejects anything but basic asm/empty statements there. Lowering turns
  a naked definition into an MIR *extern* and skips it in the body
  loop; `glcc_lower_mir::naked_asm_text` emits `.globl name`,
  `.type`/label/body/`.size` for it (always `.globl`: the definition
  is a separate object, so a local symbol could not resolve; a
  same-named `static` naked function in two TUs would collide, noted
  as a gap).
- A naked prototype carrying the attribute makes the later definition
  naked: the parser reads the merged binding (`sema.decls[id].naked`)
  after `declare`, and `naked_asm_text` dedupes by `DeclId` because
  prototype and definition share one.

### P0 notes (implemented)

- Commits `b71da2b` (parse + semantics), `ac74bf4` (tests).
- Matching constraints are digits on **inputs** (`"0"(x)` matches output
  0), not outputs: `AsmOperand::tie` is used on `asm.inputs`; outputs
  must start with `=` or `+`.
- Statement-start `__asm__` used to be swallowed by `parse_gnu_annotations`
  (`skip_gnu_suffixes`) both in `parse_block_item` and `parse_statement`.
  `parse_gnu_annotations_impl(skip_asm_labels)` now gates that skip;
  `skip_gnu_markers` (no asm labels) is what `parse_block_item` uses.
- The parsed statement comes back as `Box<Stmt>` and `parse_block_item`
  adds no second `Stmt` temporary: a plain `Stmt` return grew the
  recursive parser's stack frames enough to overflow the 2 MB test
  thread on large translation units. Keep it boxed.
- `volatile`/`inline`/`goto` lex as `Token::Keyword`; the GNU spellings
  are identifiers. `asm_qualifier(&Token)` handles both.
- Result: 70 torture files matching `asm` → 62 pass; failures are
  `990413-2` (needs P1b real templates), `pr49218` (Cranelift i128),
  `pr51447` (register vars), `pr53645`/`-2` (vector literals),
  `20061220-1` (pre-existing), `bitfld-5` and `pr85156` (unrelated
  pre-existing bugs).
- Full torture suite: **1441/1514** (baseline 1384; +57, matching the
  model experiment). `tools/ctestsuite.bash`: 220/220 on x86_64, aarch64,
  riscv64. `eeprof-1` and `noinit-attribute` run-fail under gcc too and
  contain no asm (pre-existing).
