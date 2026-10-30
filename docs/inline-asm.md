# Inline asm for glcc — design for bare-metal code

`inline-asm-catalogue.md` covers what the failing torture tests do with asm.
This note is about the real feature: enough inline asm to write bare-metal
applications with full access to the CPU.

**Update (native path):** the wasmtime fork at `inline-asm-prototype` now
ships a native Cranelift `inline_asm` instruction for x86-64 (see
`cranelift/docs/inline-asm.md` in the fork). On x86-64, glcc lowers blocks
whose constraints and templates fit that surface in place
(`glcc_lower_cl::asm_native`); everything else — and every block on
aarch64/riscv64 — still uses the out-of-line wrapper below, with a
one-line stderr note per fallback reason. The wrapper stays the
reference design for the other targets.

## Goal

Not "the tests pass". The target is C that can do:

- port I/O, MSR/CSR writes, control registers, segment/`gs`-relative access;
- fences, cache/TLB maintenance, atomics, `cpuid`/`rdtsc`;
- page-table setup, interrupt entry/exit, context switches, stack setup;
- register-pinned variables (`register ... asm("r10")`), naked functions,
  file-scope asm.

## Backend reality

Verified against the pinned Cranelift 0.136.2 and cg_clif's implementation:

- Cranelift has **no inline asm**: no IR instruction, no meta instruction, no
  raw-bytes escape hatch. wasmtime#1041 (2018) is still the tracking issue.
- cg_clif ships the workaround for x86-64/aarch64/riscv64: every asm block
  becomes a **separate function** in a per-CGU assembly blob, assembled by
  the system assembler and linked. Operand protocol (from
  `src/inline_asm.rs`): one stack slot per asm block holds all operand
  values; the wrapper is called with a pointer to that slot; the wrapper
  saves clobbered callee-saved registers, loads input registers, runs the
  template with statically allocated registers substituted, stores output
  registers, restores clobbers, and returns. Cranelift's register allocator
  is not involved in the asm block at all.
- glcc already has the plumbing: `.s`/`.S` inputs are assembled by the
  system driver and linked (`assemble_cmd`), `-T`/`-Xlinker`/`-Wl,`/`-e`/
  `-nostdlib` pass through, and labels-as-values are already integer ids
  (`sema.label_ids`) — which is exactly what `asm goto` dispatch needs.
- Cranelift x86-64 never uses the red zone (nothing in the backend mentions
  it). This is what makes Design B below safe.

## Capability tiers

| tier | capability | needs |
|---|---|---|
| T0 | empty templates, barriers, operand pinning | parse + semantics, no code |
| T1 | real instructions, `r`/`m`/`i`/named-register constraints, clobbers, ties, multi-output, `volatile`, `asm goto` | out-of-line wrapper |
| T2 | `%rsp` read/write | stack-shared wrapper |
| T3 | register variables, per-asm performance | true in-place codegen (backend fork) |

## Design A — out-of-line wrapper (T0/T1)

Per asm site:

1. The frontend parses the template, constraints, clobbers, `volatile`,
   `goto`.
2. Lowering allocates one 16-aligned stack slot (the **area**) sized for all
   operands plus clobber saves. Inputs are stored, the wrapper is called
   with the area address, outputs are loaded.
3. The wrapper (emitted into a per-TU `.s`) is:
   - save clobbered callee-saved registers into the area;
   - load each input operand into its statically allocated register
     (`"m"` operands get their address loaded into a scratch register and
     the placeholder expands to `(%scratch)`; `"i"`/`"n"` are substituted
     textually; `"g"` is treated as `"r"`);
   - the template with `%0`/`%1`/`%c0`/`%%`/`%b0`/`%w0`/`%k0`/`%q0`
     substituted;
   - store outputs into the area;
   - restore clobbers; `ret`.
4. Operand registers are allocated **statically**, per site, from a
   per-target table, following cg_clif: outputs first (more constrained),
   then inputs; `"0"` ties share; `&` early-clobber forbids overlap; named
   constraints pin; clobbered registers are excluded and naming one in a
   constraint is an error.
5. The wrapper keeps stack alignment even (16 bytes) so templates that
   `call` work.

One register per site is reserved to address the area across the template
(the template may clobber anything else). cg_clif reserves `rbx`; C code
cannot, because `cpuid` needs `"=b"` — so the reserved register is chosen
per site from the callee-saved set the asm does not name anywhere
(x86-64: rbx, r12–r15; aarch64: x19–x28; riscv64: s1–s11), erroring only if
all are named. The wrapper saves/restores the caller's value of that
register, so the call ABI still holds.

Covers T0/T1 completely. Gaps: templates that read/write `%rsp` see the
wrapper's frame (masking-style stack-base computation still works — it
is the same thread stack — but capturing an exact `%rsp` needs a naked
function); `asm goto` cannot jump to C labels; every asm is a call.

## Design B — stack-shared wrapper (T2) — not built

Only templates that manipulate `%rsp` need this. The wrapper runs the
template with the *caller's* `%rsp` and returns with a stashed return
address:

```
wrapper:
    pop  %r11                    ; rsp = caller's rsp at the asm site
    mov  %r11, RET(%area_reg)    ; stash the return address in the area
    ...save clobbers, load inputs...
    <template>
    ...store outputs, restore clobbers...
    mov  RET(%area_reg), %r11
    jmp  *%r11                   ; the return address is not on the stack,
                                 ; so this is the "jump back"
```

The template now sees exactly the stack GCC's inline asm would: `%rsp` is
the caller's value, everything below it is free (pushes and calls inside
the template work), and a normal fall-through returns correctly. Templates
that switch stacks and jump away simply never reach the epilogue — that is
how context-switch code behaves.

`asm goto` needs no stack sharing: the template's `%l[name]` expands to a
wrapper-local stub label; the stub sets a selector (0 = fallthrough,
`i + 1` for `labels[i]`) in a register no operand uses and rejoins the
epilogue, and the caller dispatches with a `switch` over the selector to
the function's existing label blocks. No C label addresses cross the call,
so the plain Design A wrapper is enough. This is implemented (T1); the
stack-sharing wrapper below is only needed for templates that read or
write `%rsp`.

Caveats: one reserved register (same per-site choice as Design A); a
template that executes `ret`/`leave` itself sees a stashed-away return
address (rare, and broken under any wrapper scheme); the scheme is new —
cg_clif does not have it — so it needs its own tests.

**Decision: not built.** Nothing needs it yet. Stack switches belong in
naked functions (body emitted verbatim, real stack, no wrapper), and
reading `%rsp` in a wrapper is only wrong when the *exact* value
matters. Kept as the design if that changes; the mechanism is
target-independent (`pop`/`mov`/`jmp *` on x86-64, `ldr`/`br` on
aarch64, `ld`/`jr` on riscv64).

## Design C — true in-place (T3) — deferred

Real in-place asm means adding an instruction to Cranelift's IR, fixed
register constraints and clobbers in the register allocator, and — the hard
part — an **in-process assembler** per ISA, because the template can only
be assembled after register allocation and Cranelift emits bytes directly.
There is no mature Rust text assembler for aarch64/riscv64 at all. This is
a fork-scale project (months). It is only needed for register variables and
for removing the call overhead; T0–T2 are achievable without it. Revisit
only if the wrapper overhead proves unacceptable in practice.

## Why not naked wrappers / globals / jumps

- The call *is* the jump back; the wrapper needs no naked attribute and no
  globals. Globals are strictly worse than the area pointer: not
  reentrant across threads or interrupts, extra memory traffic, and the
  values must reach registers inside the template anyway.
- `__attribute__((naked))` is still valuable, but as its own feature:
  whole-function asm (interrupt vectors, `_start`, context-switch routines)
  emitted verbatim, no wrapper. It is not a way around the call frame — a
  naked function called with `call` has the same return-address question,
  answered the same way as Design B.
- The "somehow jump back" instinct is right, and it is Design B's stashed
  return address + `jmp *`: needed only when the template must see the
  caller's stack.

## Plan

- **P0** Frontend: parse `asm`/`__asm__`/`__asm` statements (templates,
  constraints, clobbers, `volatile`, `goto`) and implement empty-template
  semantics: evaluate operands, `"0"` ties as assignments, `+` identity,
  barrier, no code. Unlocks ~57 torture tests and the barrier idioms bare
  metal needs everywhere. No backend work.
- **P1a** The `.s` pipeline, plus the non-statement forms while it is
  fresh: file-scope asm (verbatim into the `.s`) and naked functions (body
  emitted verbatim, symbol defined by the `.s`). Cheap, and already enough
  to write vectors, entry stubs, and context switches.
- **P1b** Plain wrappers (Design A) with the per-target constraint tables,
  x86-64 first, then aarch64/riscv64. This is the bulk of bare-metal asm:
  port I/O, MSRs/CSRs, cpuid/rdtsc, fences, cache/TLB, atomics, syscalls.
- **P2** `asm goto` (done: selector-returning wrappers). Stack-shared
  wrappers (Design B) are **not** built — naked functions cover stack
  switches, and only an exact `%rsp` capture in non-naked asm would
  need them.
- **P3** Optional: in-place codegen (Design C) for performance and
  register variables.

## glcc integration

- `glcc_frontend`: `StmtKind::Asm` with parsed template pieces, operand
  expressions, constraint strings, clobbers, `volatile`/`goto` flags;
  file-scope asm as an external declaration.
- `glcc_mir`: an `InlineAsm` instruction carrying the parsed asm, opaque to
  every pass (like a call with side effects). This is the shared seam;
  `glcc_lower_cl` (below) consumes it.
- `glcc_lower_cl`: allocate the area, emit the call, collect wrapper text;
  `lower_module` returns the collected `.s` text alongside its `FuncId` map.
- driver: write the `.s` next to the object, assemble it with the same
  machinery as user `.s` inputs, then link both objects; under `-c`, merge
  with `ld -r` (mold may lack `-r`; fall back to bfd/lld). `-save-temps`
  should keep the generated `.s` for inspection.
- per-target tables: constraint letters, register classes, sub-register
  naming, reserved-register sets, memory-operand substitution.

## Adjacent bare-metal work

- `__attribute__((naked))` and file-scope asm — in the plan above.
- `__attribute__((section(...)))`, `interrupt`, `aligned` on functions:
  needed for vectors and custom sections; mostly frontend + emission.
- `-ffreestanding` semantics (the flag is currently accepted and ignored),
  `-nostdlib` (already works), linker scripts (`-T` already passes
  through).
- Register variables (`register ... asm("r10")`): T3, deferred. Kernels
  use them; most bare-metal applications do not.
- x87 constraints (`"t"`, `"u"`, `st(1)`): niche (one torture test); error
  until someone needs them.

## Testing

- The 65 torture tests (T0/T1; 57 pass with P0 semantics alone).
- New asm tests per target: `cpuid`, `rdtsc`, `bsf`/`bswap`/`lock` ops,
  memory operands, ties, multi-output, clobbers, `asm goto`.
- A freestanding smoke test: `_start` with raw syscalls via asm, built with
  `-nostdlib`, run natively and under qemu for aarch64/riscv64.
- Object-level: `-c` produces one merged object; `-save-temps` keeps the
  generated `.s` for inspection.
