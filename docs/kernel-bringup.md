# Kernel bring-up status

`tools/kernel.bash` builds linux-6.12.30 (tinyconfig, ARCH=x86_64, LP64) with
`CC=glcc -fbackend=native -fintegrated-as`, cross-compiling from an aarch64
host. Failure is expected; this note tracks what works, what breaks, and the
workarounds the shim applies (each one is a real glcc gap).

## Driver-side support added for this

- `-MD`/`-MMD` depfiles generate in-process: the native pp tracks every file
  a directive resolves to (`preprocess_with_all`) and the driver formats
  gcc-style depfiles (`-MM` filters system dirs, `-MP` adds phony targets).
  Previously depgen always spawned an external preprocessor.
- `-dM` implemented in the native pp (kbuild's `cc-version` probes use it).
- Named address spaces (`__seg_fs`/`__seg_gs`/`__seg_kfs`/`__seg_kgs`): GCC
  x86 builtin qualifiers the kernel's per-cpu code uses under
  `CONFIG_CC_HAS_NAMED_AS`. Parsed and erased — the segment-relative
  addressing they request is NOT modeled, so per-cpu codegen is wrong at
  runtime until the address space reaches the backend.
- `__builtin_object_size`/`__builtin_dynamic_object_size`: no tracking, so
  the GCC unknown constant (`(size_t)-1` for types 0/1, `0` for 2/3).
- Fixed-width bit-count builtins (`__builtin_clzll` & friends) fold in
  constant expressions (kernfs-style `63 - __builtin_clzll(x)` array bounds).
- `extern void` objects (linker-symbol markers) parse.
- Null-pointer casts see through one further cast (`(unsigned long)(T *)0`
  switch labels).
- `-fintegrated-as` and the integrated assembler now work cross-target:
  glasm is host-portable (the `object` crate carries per-arch endianness).

## glasm (integrated assembler) support added

- `.macro`/`.endm`/`.exitm`, `.rept`/`.irp`/`.endr`, `.if`/`.else`/`.elseif`
  /`.endif`/`.ifc`/`.ifnc`/`.ifdef`/`.ifndef`/`.ifb`/`.ifnb`, `.set`/`sym =`
  for conditional evaluation, `\param`/`\@`/`\()` substitution, keyword
  arguments with defaults, parenthesized argument lists, statement
  splitting before macro matching, backslash line continuations, bare
  instruction prefixes (`lock; addl`, `lock` on its own line) coalesced
  with the next instruction.
- Comparison/logical operators in expressions (`==`, `!=`, `<`, `<=`, `>`,
  `>=`, `&&`, `||`) for `.if` and `.skip` guards.
- `.skip`/`.space`/`.zero`/`.fill` take expression counts (negative clamps
  to zero); `.4byte`/`.2byte`/`.8byte`; `.align`/`.p2align` expressions.
- Cross-section `sym - .` (`RValue::RelDiff`): PC-relative relocation
  against the target, matching GNU as (`LOCK_PREFIX`, ALTERNATIVE
  descriptors).
- Undefined referenced symbols auto-extern (GNU as semantics).

## Native x86-64 backend support added

- Scalar loads/stores through any address value (the patterns only
  covered slot addresses).
- Aggregate `struct{i64,i64}` returns returned `[rax, rdx]` — the table
  defaulted both pieces to rax and killed regalloc2.
- Inline asm: spilled operands round-trip through the scratch register;
  symbol references assemble to relocations (module tables, or lazy
  externs for undeclared names); fragment-local labels retarget to the
  enclosing function; side sections (`.pushsection` in templates, e.g.
  ALTERNATIVE descriptors, `.smp_locks`) carry their own relocations;
  `%%` literal escapes and `%c[name]` modifier forms.

## Session of 2026-11-13/14: the assembler+pp wave (entry_64/head_64 unblocked)

The build was stuck behind a wall of `.S` failures that all turned out to be
glasm/glcc-pp gaps, each now fixed (each verified byte-for-byte against GNU as
where applicable):

glasm:
- `%fs:`/`%gs:` segment overrides: parsed in AT&T and Intel syntax, emitted
  via the nasm `PPS_SEG` prefix slot (`65`-prefixed `cmpq %gs:(%rsi)`,
  `lock addq %gs:sym(%rip)`, ...). Grouped displacements `(a + b)(%rip)`
  parse too.
- Registers in pp expressions: `depth_of` skipped `GTok::Reg`, so every
  `.if \base == %rsp` macro-argument check evaluated against an empty
  expression and took the wrong arm (UNWIND_HINT_REGS' "bad base register").
- The inactive-branch `.elseif` handler activated the frame without
  evaluating the condition, so alternating `.elseif`s all ran.
- `;` is a statement separator on x86, not a comment: the pp stripped
  everything after the first `;`, so `i = 0 ; .rept N ; ... ; .endr` blocks
  from kernel macros never reached the pp. Same-line `.rept`/`.irp` bodies
  are handled now; `<`/`>` no longer group macro arguments (they are shifts:
  `(3 << 16)` swallowed the following args).
- Macro params missing at a call bind to the empty string (GAS), not raw
  `\param` text (IBRS_ENTER without save_reg).
- `.code16/32/64` directives recognized (only `.code64` encodes); `lretX`
  translates to `retfX`; one-operand shifts (`shrl %edx`) fill the missing
  `$1`; `movq` with a vector register selects the SSE move instead of the
  integer alias; `.octa` carries >64-bit literals (`GTok::BigInt`); `ljmp`
  forms emit their raw bytes (`EA seg/off` direct, `FF /5` indirect
  `*sym(%rip)`); `.set name, %reg` register aliases stand in for the register
  in operands (GAS semantics, vgetrandom-chacha.S); negative `.times`/`.skip`
  counts clamp to zero (the kernel's `-((diff) > 0) * diff` padding guards);
  `.incbin "file"[, skip[, count]]` (dtb.S); the finalizer names the symbol
  in extern-in-small-field errors.
- Whitespace between `%` and the register name lexes (`(% rip)` from
  `_ASM_RIP`).

glcc pp (assembler mode):
- `$` no longer joins identifiers (`btsl $FOO` must expand `FOO`), for the
  main file AND included files (the include path lexed with the C rules —
  the root cause of every unexpanded `X86_*_BIT` inside kernel headers'
  macro bodies).
- `__ASSEMBLER__` is predefined in assembler mode, like GNU cpp
  (vdso-note.S: build-salt.h branches on it).

frontend:
- Integer constants survive a pointer-cast round trip in const contexts
  (`case (unsigned long) SEND_SIG_PRIV`, i.e. `(unsigned long)((T *)1)`).

The driver test `pp_selection_flag_and_env` was updated: `-dM` is native now,
so it no longer forces the external pp.

## Known remaining gaps (census after the 2026-11-14 session, tinyconfig,
## 333 TUs per pass, 7 error lines)

The whole C-side list below the old census (non-constant global
initializers, Int128 cmpxchg, "i"/"n" constraints, inline-asm shift counts,
`%/Str`, entry.S, dtb.S, vgetrandom-chacha.S) is gone from the log; rerun
`GLCC_NATIVE_CENSUS=1` for a survey of what still panics in codegen. What
still fails:

- `.code16`/`.code32` realmode encoding (reboot.S, trampoline_64.S): the
  encoder is hardwired to 64-bit; operand/address-size prefixes per `.code`
  state are the next milestone.
- vdso.lds preprocessing routes to an external pp (`clang`), which is not
  installed; the driver needs a native `-E` path for linker scripts.
- MIR verifier: `bad gep step Field(N)` on nested-struct geps (execmem).
- OOM: big TUs peak 1-2GB of glcc RSS; the shim caps at 3GB per process so
  a runaway dies in its own log.


## Shim workarounds (all fixable in glcc itself)

- `--target=x86_64 -fbackend=native -fintegrated-as` forced (kbuild CC=
  takes one word).
- stdin splice (`-` inputs), GCC-8 version masquerade.
- `-S` (asm-offsets etc.) delegates to `x86_64-linux-gnu-gcc` with a
  denylist of flags it rejects.
- `ARCH=x86_64` on every make invocation (kbuild derives the host arch).
- `ulimit -v 3GB` per glcc (runaway diagnostics).
- `GLCC_NATIVE_CENSUS=1` lowers every function, int3-stubs the failures,
  and tallies the panic messages — use it for surveys.
