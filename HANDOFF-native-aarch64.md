# Handoff: native aarch64 backend — state (2026-11-12)

The native backend (`crates/glcc_codegen`, `-fbackend=native`) now has
two targets: x86-64 SysV and **aarch64 Linux (AAPCS64)**, integer +
F32/F64. c-testsuite aarch64-native: **219/220** — same score as the
x86-64 native backend (220/220 cranelift, 219 native). The one known
failure is `00204` (arm64-specific va_arg test; it reaches `long
double`/F128, which panics by design — see "next milestones").

This host **is** aarch64, so native binaries run directly — no qemu:

```bash
cargo build --release
./target/release/glcc -w --target=aarch64 -fbackend=native t.c -o out -lm
./out                     # links via host cc + ~/.local/lib/glcc/glccrt.o
```

`tools/ctestsuite.bash` runs aarch64 natively when `uname -m` is
aarch64 (mirrors its x86_64 native branch; zig/qemu/clang only for the
cross path):

```bash
GLCC=target/release/glcc CFLAGS_EXTRA="-fbackend=native" \
  ALLOW="aarch64/00204" ./tools/ctestsuite.bash --targets=aarch64
```

## Architecture notes (what differs from x86-64)

- **Pipeline is arch-neutral**: a target is `TargetIsa` + `AbiImpl`
  plus tables. aarch64 lives in `src/aarch64/{regs,tables,enc,abi,mod}.rs`.
- **Encoding** is fixed 32-bit words, no x86-style recipe layer.
  Golden tests (`aarch64/enc.rs` `#[cfg(test)]`) pin ~60 instructions
  to GNU-as-generated words — extend them from
  `cc -c` + `objdump -d` whenever adding an encoding.
- **Branch fields** are imm26; conditional branches encode as
  `b.!cond skip; b target; skip:` so every branch/call field is
  patched through `TargetIsa::patch_branch_field` (imm26 at [25:0]).
  No imm19 machinery.
- **Symbol addresses**: defined symbols use adrp+add with the full
  addend on both relocs (page + lo12 reconstruct S+A exactly);
  preemptible symbols (extern data, extern functions) use GOT-indirect
  `LeaGot` (adrp :got: + ldr :got_lo12:) — PIE linkers reject
  adrp+add against them. `object.rs` direct-patches only branch-style
  relocs (PcRel32/Plt32/Call26); everything else goes through ELF.
- **Small integers**: sub-32-bit values keep garbage upper halves
  (x86-style), but AArch64 has no sub-32-bit compare/divide — signed
  uses (sdiv/srem/asr, signed preds) canonicalize inputs with
  sxtb/sxth, unsigned/equality with uxtb/uxth, at the use site
  (`canonicalize` in `aarch64/mod.rs`). Spill/reload MOVEs at W64 are
  harmless because every 64-bit-reading consumer canonicalizes first.
- **AAPCS64** (`aarch64/abi.rs`): HFAs (≤4 same-width float members,
  16-byte alignment defeats) in V regs at member width; other ≤16B
  composites as 8-byte chunks in X regs; **>16B composites by
  reference** (call hook copies into a stack temp, passes the
  address); sret in **x8, outside the x0-x7 pool** (entry gp budget
  starts at 0). Piece arrays are `[PReg; 4]` (HFA upper bound).
- **Frame**: fp = x29 at the stp'd pair ([fp] = saved fp, [fp+8] =
  lr — stack args at fp+16). Prologue: `sub sp,#16; stp x29,x30,[sp];
  mov x29,sp; sub sp,#frame_size; str callee-saved at [fp,-8(i+1)]`.
  The sub is exactly `frame_size` — it counts from fp down; getting
  this wrong puts slots below sp and inside the callee's frame (this
  was the infinite-recursion bug). Deep frames decompose via
  `sp_adjust` (imm12<<12 shift form + low imm12).
- **Scratch registers**: x16 (IP0) = encoder-only scratch (big
  displacements, srem's internal sdiv); x17 (IP1) = regalloc2
  move-cycle scratch; v17 = vector scratch. All non-allocatable.
- **Div/rem**: `sdiv`/`udiv` + `msub` for remainder (no hardware
  remainder). Division by zero yields 0 (ARM semantics), never traps
  — C UB, accepted.
- **va_list** (AAPCS64, 32-byte tag, descending signed offsets):
  ported from `glcc_lower_cl/src/abi/mod.rs` (`arm_reg_arg`,
  `va_aggregate_whole`, 16-align skip). Save area 192B (GP +0..64
  stride 8, VR +64..192 stride 16).

## Known gaps (next milestones, roughly in order)

1. **F128 / `long double`** — panics (`check_no_quad`). Route through
   `__addtf3`-family libcalls (compiler-rt/libgcc, Q-register
   convention: load into q-regs, bl) like `glcc_lower_cl/src/quad/`
   does; `rt/glccrt.c`'s aarch64 block already has `__glcc_quad_*`
   trampolines if the pointer-based convention is preferred. 00204
   becomes the measuring stick; also unblocks `long double` math
   tests everywhere.
2. **bswap** (`rev`), atomic RMW/CAS (ldaddal/cas — aarch64 has clean
   LSE atomics), jump tables for switch (tbz/tbnz/cbz forms would also
   shrink the comparison chains).
3. **Direct b.cond** (drop the skip-trampoline; needs an imm19 field
   kind in `block_refs`), logical-bitmask immediates, and the small
   x64-style pattern table if selection ever needs the speed.
4. **setjmp/longjmp**: `rt/glccrt.c`'s aarch64 block lacks
   `__glccrt_setjmp/__glccrt_longjmp` (x86-64's x87 block and the
   Win64 block have them). c-testsuite has no setjmp tests, but the
   torture suite does.
5. **abi_probe** conformance run (needs zig + qemu-user installed;
   neither is on this VM).
6. Windows-aarch64 / riscv64 native — not started.

## Debugging techniques that worked

- `GLCC_NATIVE_DUMP=1` for before/after-RA machine text;
  `GLCC_NATIVE_TRACE=1` for pipeline phase timing;
  `GLCC_NATIVE_CENSUS=1` to survey feature gaps over a big TU.
- **objdump the actual object** — the printer is a debug view and can
  mislead (e.g. it prints "rip" for baseless memory operands).
- Differential against `cc -O0` ground truth; reduced repros compile
  in <1 s on this host, so bisect at the backend level freely.
- gdb works normally (native execution, no qemu): watchpoints on spill
  slots, `x/…i $pc`.
- When an encoding looks wrong, assemble the instruction with the host
  `cc` and compare words (the golden tests came from exactly that).
- Beware stale binaries: `cargo build --release | grep error` hides a
  failed build behind grep exit codes — check the "Finished" line.
