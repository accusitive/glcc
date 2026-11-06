# Plan: native inline-asm support for SSE (xmm) register operands

Status: drafted 2026-11-06. Execute in order; each phase ends green.

## Goal

Make glcc's *native* inline-asm path handle xmm-class operands so the
remaining asm-stub builtins stop falling back to the out-of-line
wrapper, and fix a latent correctness bug found on the way: literal
`%%xmmN` scratch registers in native templates are currently **not**
reported to regalloc (glcc's clobber pass drops non-GPR names), so a
live vector value in that register can be silently clobbered.

Today (post `%%`-escape and probe-fidelity fixes, 2026-11-06):
- Native: `movmskps` stub, all pure-MIR builtins, GPR/mem/imm operands.
- Wrapper fallback, by cause:
  - `rcpss`/`rcpps`/`rsqrtss`/`rsqrtps` stubs — `=m` outputs (staged
    through memory because the fork cannot render `%xmmN` operands).
  - `vec_init_v2si`/`cvtps2pi`/`pmovmskb` stubs — MMX registers as
    literal text work, but their 32-bit GPR inputs need `%k`-style
    width modifiers on non-fixed operands (separate, optional phase).
  - Anything wanting an allocated `%xmmN` operand — impossible: the
    fork's operand substitution renders GPRs only.

## Repos and current architecture

Fork: `/home/gl/data/p/wasmtime-asm`, branch `inline-asm-prototype`
(clean). glcc consumes it from GitHub (`Cargo.toml:18-21`,
`crates/glcc_lower_cl/Cargo.toml:13-17`), no `[patch]` section.

Fork, `cranelift/codegen/src/isa/x64/inline_asm.rs`:
- `Constraint::parse` (~:95): GPR letters only (`r q l a b c d S D Q R
  i n m o V`); unknown letter → `unsupported()` panic. Non-Reg output
  kinds rejected (`=m` unsupported by design).
- `InlineAsmOperands { uses: [AsmUse], defs: [AsmDef] }`; `RegConstraint
  { Any, Fixed(Reg), Limit(u16) }`.
- `lookup_clobber_reg` (~:225): GPR names only, panics otherwise.
- `reg_name` / `reg_name_sized` (~:315): render via
  `asm::gpr::enc::to_string` — **GPR only**; this is the root gap.
- `substitute` (~:469): `Source::Reg(preg)` → `reg_name`; `MemUse` →
  `(%reg)` (AT&T); `Const` → `$v`. `%%`→`%`, `%=`→unique number.

Fork, regalloc plumbing (mostly class-agnostic already):
- `lower/isle.rs` (~:415): `ConstraintKind::Reg` → use with
  `reg = lower_to_reg(value)`. The regalloc2 **class rides the vreg**
  (`add_operand` in `machinst/reg.rs:573` pushes `Operand::new(VReg,
  ...)`), so a vector-typed CLIF value already yields an xmm-class
  operand; only the constraint grammar and name rendering block it.
- `inst/mod.rs:1287` (`get_operands`): `reg_use`/`reg_fixed_use`/
  `reg_limit_use` — no class assumptions beyond the `Reg` passed in.
  `regs::xmm0()`.. constructors already exist in `inst/regs.rs`, so
  `Fixed(xmmN)` works once parse accepts it
  (`debug_assert_is_allocatable_preg` is satisfied: xmm PRegs are in
  the x64 allocatable set).

glcc, `crates/glcc_lower_cl/src/asm_native.rs` (~512 lines):
- `plan()`: parses constraints (`try_parse_constraint`, in
  `asm/mod.rs`), maps them to fork grammar (`fork_constraint`:
  `r q g X→r`, `a A→a`, `b c d S D Q R→same`, `i n→i`, `m o V→m`), then
  `translate_template` + a probe that renders concrete asm and calls
  `cranelift_assembler_glasm::encode_module(&probe, true)`. Any probe
  error → out-of-line wrapper (correct, slower, prints a note).
  Probe rules established 2026-11-06 (keep!): `%%`→`%`; `m` →
  `(%r11)`; `i` → the real `$const`; operand refs → `%r11`.
- `NativeKind { Reg, Mem, Imm }`; `NativeOperand { width, fixed, high,
  kind }`. **No register-class concept.**
- `plan()` rejects `width > 8 && Kind::Reg` (wide operands) — blocks
  16-byte vectors.
- Clobber pass: `find_reg(t, name)` over glcc's GPR table; non-GPR
  names (`xmm0`) are silently dropped → the latent bug above.
- `translate_template`: `%N` passes through for width-8 Reg / Mem /
  Imm; narrower Reg needs a fixed register; `%<mod>N` needs fixed
  (except the high/`Q` path).

## Phases

### Phase 1 — fork: xmm constraints and name rendering

`cranelift/codegen/src/isa/x64/inline_asm.rs`:

1. `Constraint::parse`: accept `'x' | 'v'` as
   `ConstraintKind::Reg(RegConstraint::Any)` — regalloc class comes
   from the vreg, no new variant needed. Document that the *value*
   must be vector-typed (an integer value under `"x"` would allocate a
   GPR and render wrong; acceptable, matches "class from value" model).
2. `reg_name_sized` (and therefore `reg_name`/`substitute`): branch on
   the PReg's class — vector-class PRegs render `xmm0`..`xmm15` from
   `hw_enc` (AT&T prefixes `%`). Add a small name helper next to
   `asm::gpr::enc::to_string` (an `asm::xmm::enc` or a local table).
   For size != Quadword on a vector PReg, panic — glcc must not send
   such modifiers (Phase 3 enforces).
3. `lookup_clobber_reg`: accept `xmm0`..`xmm15` (via the existing
   `regs::xmmN()` ctors → `.to_real_reg()`), so `"xmm3"` in a clobber
   list reaches `PRegSet` and regalloc reserves it.

Verify: `cargo test -p cranelift-codegen --lib` inside the fork; add a
unit test next to the existing inline-asm helpers for name rendering
(`reg_name` of an xmm PReg) if the module has a test seam; otherwise
cover via glcc (Phase 4).

### Phase 2 — glcc: constraint class plumbing

`crates/glcc_lower_cl/src/asm/mod.rs` + `asm_native.rs`:

1. Constraint parsing: recognize `x`/`v` as register class **Vec**
   (keep `Kind::Reg`, add class to the parsed result or to
   `NativeKind`: `NativeKind::Vec` is the smallest change).
2. `fork_constraint`: map `x | v → x`.
3. `NativeOperand` gains the class; `plan()`'s wide-operand gate
   (`width > 8 && Kind::Reg`) becomes: GPR-wide → still rejected,
   Vec (width 16) → allowed.
4. `translate_template`: plain `%N` on a Vec operand passes through
   (like width-8 GPR). A `%<mod>N` modifier on a Vec operand → error
   ("modifiers on SSE operands are not native yet") — this pairs with
   the fork panic in Phase 1.2.
5. Probe: operand refs with Vec class render `%xmm11` (GPR stand-in
   is `%r11`; pick any allocatable xmm, e.g. `%xmm5`).

### Phase 3 — glcc: clobbers (correctness fix)

1. glcc's clobber pass: recognize `xmm0`..`xmm15` (glcc's own register
   table or a literal list) and pass the canonical name through in
   `NativeAsm.clobbers` (the wrapper path needs no action — call ABI
   covers caller-saved xmm; the **native** path needs them in the
   fork's PRegSet, which Phase 1.3 enables).
2. Trace where `NativeAsm.clobbers` feeds the fork's
   `InlineAsmContents` clobber list (the native emission in
   `glcc_lower_cl`) and confirm literal `%%xmmN` scratch in a template
   plus a matching `"xmmN"` clobber keeps regalloc honest. The
   asm-stub templates in `glcc_lower_mir/src/lower/ia32.rs` all use
   `%%xmm0`/`%%mm0` as scratch — add `"xmm0"` to their clobber lists.
3. Regression test: extend `clobbers_are_canonicalized`
   (`asm_native.rs` tests) — `"xmm0"` must now survive into
   `native.clobbers`.

### Phase 4 — rewrite the SSE stubs, end-to-end

`crates/glcc_lower_mir/src/lower/ia32.rs`:

1. `RcpSs { rsqrt }` / `RcpPs { rsqrt }`: drop the memory staging.
   New form, pure register operands:
   `rcpss %1, %0` (scalar; note operand order src,dst for AT&T) /
   `rcpps %1, %0`, constraints `["=x", "x"]`, vector-typed MIR values.
   Lane semantics (lane 0 vs all lanes) come from the instruction, as
   today. This also removes the last `=m` users.
2. Exec test in `crates/glcc_lower_mir/src/lower/tests.rs`: call
   `__builtin_ia32_rcpss/__builtin_ia32_rcpps/__builtin_ia32_rsqrtps`
   through the JIT and assert approximate values (these are
   approximate instructions — use loose bounds, e.g. `abs(x*rcp(x) -
   1) < 2^-12`-style), and assert `plan` went native via the absence
   of the wrapper note or a direct `plan()` unit test.
3. `movmskps` stub: already native; add `"xmm0"` clobber (Phase 3).

### Phase 5 (optional, separable) — narrow GPR modifiers on variable registers

Unblocks the MMX stubs (`vec_init_v2si`, `cvtps2pi`, `pmovmskb`):
1. `translate_template`: allow `%k`/`%q`/`%w` modifiers on **non-fixed
   width-8 GPR** operands — the fork renders the width-selected name of
   the allocated register, which is correct for any GPR (a 32-bit value
   in `r11` is `r11d`). Keep `%b`/`%h` fixed-only (high byte needs
   a/b/c/d).
2. Rewrite `vec_init_v2si`'s stub inputs to `%k1`/`%k2`;
   `cvtps2pi` needs no change (its GPR output goes through `movq
   %%mm0, %0` — width-8, already fine once `pmovmskb`-style templates
   probe correctly); check each stub's actual operands.
3. Probe: the existing `%<mod>N` probe branch already renders
   width-matched stand-ins (`k` → `%r11d`) — extend for `w`.

## Testing the fork locally (required workflow)

glcc consumes the fork from GitHub, so fork changes are invisible to
glcc tests until pushed. To iterate locally, add to `glcc/Cargo.toml`
(temporary, DO NOT commit):

```toml
[patch."https://github.com/accusitive/wasmtime"]
cranelift-codegen = { path = "/home/gl/data/p/wasmtime-asm/cranelift/codegen" }
cranelift-assembler-glasm = { path = "/home/gl/data/p/wasmtime-asm/cranelift/assembler-glasm" }
cranelift-frontend = { path = "/home/gl/data/p/wasmtime-asm/cranelift/frontend" }
cranelift-module = { path = "/home/gl/data/p/wasmtime-asm/cranelift/module" }
cranelift-object = { path = "/home/gl/data/p/wasmtime-asm/cranelift/object" }
cranelift-native = { path = "/home/gl/data/p/wasmtime-asm/cranelift/native" }
cranelift-jit = { path = "/home/gl/data/p/wasmtime-asm/cranelift/jit" }
```

(All git deps in `Cargo.toml` and `crates/glcc_lower_cl/Cargo.toml`
must be covered or cargo errors on the unused patch — adjust the list
to what cargo reports.) When done: commit + push the fork to
`inline-asm-prototype` first, remove the patch section, then commit
glcc. Note the repo's CI gotcha: batched pushes only CI-test the HEAD
commit — if the fork push and glcc commit land separately, the glcc
commit may ship untested until the next CI run.

## Verification commands

```
cd /home/gl/data/p/wasmtime-asm && cargo test -p cranelift-codegen --lib
cd /home/gl/data/p/glcc
cargo test -p glcc_lower_cl --lib        # plan/probe unit tests
cargo test -p glcc_lower_mir --lib       # exec tests incl. new rcp tests
cargo test -p glcc_frontend --lib && cargo test -p glcc_pp --lib
# end-to-end, both opt levels, notes must be gone:
./target/debug/glcc /tmp/rcp.c -o /tmp/rcp && /tmp/rcp
./target/debug/glcc -O2 /tmp/rcp.c -o /tmp/rcp2 && /tmp/rcp2
```

`/tmp/rcp.c`: exercise `__builtin_ia32_rcpss/rcpps/rsqrtss/rsqrtps`
plus `movmskps` and a pmmintrin include, return 0 on success.

## Gotchas

- **Probe fidelity is a safety invariant**: the probe exists because a
  template the fork cannot encode panics mid-compile at emit time
  (`inline_asm.rs` `unwrap_or_else(panic!)`). Never make the probe
  more permissive than the emitter; when unsure, render exactly the
  shapes `substitute()` produces (`%xmmN`, `(%reg)`, `$const`).
- **`MemUse` addresses are GPRs** even for vector operands — a `"m"`
  vector input is an address in a GPR; only `"x"` inputs are xmm.
- **`RegConstraint::Any` + `"x"`**: the class comes from the value, so
  glcc must only pass `"x"` for vector-typed operands (its plan() knows
  the MIR type — validate there, error otherwise).
- **`=m` stays unsupported** in the fork (`Constraint::parse` rejects
  non-Reg outputs); don't try to add it in this plan — Phase 4 removes
  the need.
- **`%h`/`%b` on xmm**: GCC disallows; glcc errors (Phase 2.4), fork
  panics (Phase 1.2) — the glcc error is the user-visible one.
- AGENTS.md rules apply in both repos: no superfluous comments, build
  after every edit, small steps always green. Do not run workspace-wide
  `cargo fmt`. Never use `git stash` in the glcc repo.
- The wrapper path must keep working for anything that still falls
  back (e.g. `=m` users if any remain, asm goto) — the fallback notes
  should only appear for genuinely-unsupported shapes.

## Acceptance criteria

1. `__builtin_ia32_rcpss/rcpps/rsqrtss/rsqrtps` compile to native
   inline asm (no wrapper note) at `-O0` and `-O2`, values correct.
2. A template with literal `%%xmmN` scratch + `"xmmN"` clobber compiles
   natively and cannot clobber a live value in that register.
3. All glcc suites green: `glcc_lower_cl`, `glcc_lower_mir`,
   `glcc_frontend`, `glcc_pp` (+ fork's `cranelift-codegen` lib tests).
4. The empty `#include <pmmintrin.h>` compile stays note-free and
   single-function (no regression of the 2026-11-06 emission fix).
5. Fork changes committed and pushed to `inline-asm-prototype`; glcc
   commit references the new fork rev; no `[patch]` section left in
   `Cargo.toml`.
