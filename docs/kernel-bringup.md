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

## Known remaining gaps (census after the 2026-11-13 session, tinyconfig,
## ~250 TUs compiled before the make aborts on early failures)

- `non-constant global initializer` (data.rs, ~7 files: alternative.o,
  cpu/common.o, dma/coherent.o, ...): static initializers the data
  lowering cannot fold yet — the largest remaining C-side class.
- Segment prefixes in assembler operands (`%gs:(sym)(%rip)` in retpoline.S):
  the `Operand.seg` field exists but no encoder emits FS/GS prefix bytes.
- `Int{128}` values reach selection: 128-bit atomics (`cmpxchg128`);
  LegalizeWideInts does not cover these shapes.
- Inline asm `.times` with label-difference counts ("times count must be
  constant", vgetcpu.o); "shift amount -12 out of imm8 range" (a folded
  negative shift reaching the encoder).
- `asm operand constraint requires an integer constant` — "i"/"n"
  constraint folding.
- `.S` stragglers: entry.S (one `%/Str` quirk at line ~957), dtb.S,
  vgetrandom-chacha.S (`expected ')'`); vdso.lds preprocessing still routes
  to an external pp (`GLCC_PP`-independent `-E` path in the link flow).
- MIR verifier: `bad gep step Field(N)` on nested-struct geps (execmem).
- Diagnostics: `GLCC_DUMP_FAILED_ASM=1` writes the preprocessed `.S`
  (`/tmp/glcc-failed.s`) and the glasm-expanded text
  (`/tmp/glcc-pp-out.s`) for assembly failures.
- OOM: big TUs peak 1-2GB of glcc RSS (one pathological TU asked for 3.5GB
  in one allocation); `-j` > 8 OOMs a 27GB box — the shim caps at 3GB per
  process so a runaway dies in its own log.

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
