// glcc soft-float glue: rounding-mode query and inexact raising for
// the vendored quad-add path. Implemented against the x87 control
// and status words directly (no libc link dependency, so the runtime
// links clean): `fesetround` programs the x87 RC bits, which is what
// is read back, and the inexact divide sets SW.PE exactly like the
// `F80` helpers raise it through hardware. (The pure-integer quad
// paths elsewhere likewise leave MXCSR alone.)
//
// The `softfloat/` directory vendors llvm-project compiler-rt
// (release 19.1, Apache-2.0 WITH LLVM-exception; per-file license
// notices intact, files otherwise byte-identical upstream) for the
// operations with no hardware path on x86-64: quad add/sub/mul/div
// (`addtf3`, `subtf3`, `multf3`, `divtf3`), x87<->quad
// (`extendxftf2`, `trunctfxf2`), int128<->x87 (`floattixf`,
// `floatuntixf`, `fixxfti`, `fixunsxfti`), and 128-bit clz
// (`clzti2`).
//
// Entries are renamed to `__glcc_*` at compile time so our objects
// never collide with (or depend on) toolchain runtimes:
//
//   -D__addtf3=__glcc_addtf3 -D__subtf3=__glcc_subtf3
//   -D__multf3=__glcc_multf3 -D__divtf3=__glcc_divtf3
//   -D__extendxftf2=__glcc_extendxftf2 -D__trunctfxf2=__glcc_trunctfxf2
//   -D__floattixf=__glcc_floattixf -D__floatuntixf=__glcc_floatuntixf
//   -D__fixxfti=__glcc_fixxfti -D__fixunsxfti=__glcc_fixunsxfti
//   -D__clzti2=__glcc_clzti2
//
// (Internal cross-calls, e.g. `subtf3` -> `addtf3`, follow the same
// renames automatically.) Compile each `.c` plus this file with the
// target clang (`-Irt/softfloat` for the headers) and combine behind
// the `glccrt.o` runtime object; see `install.sh`.
#include "fp_mode.h"

__attribute__((visibility("hidden"))) CRT_FE_ROUND_MODE __fe_getround(void) {
  // x87 control word RC field: 00 nearest, 01 down, 10 up, 11 toward
  // zero. This is what `fesetround` programs, so soft-float follows
  // the live mode like the hardware paths do.
  unsigned short cw;
  __asm__("fnstcw %0" : "=m"(cw));
  switch ((cw >> 10) & 3) {
  case 1:
    return CRT_FE_DOWNWARD;
  case 2:
    return CRT_FE_UPWARD;
  case 3:
    return CRT_FE_TOWARDZERO;
  default:
    return CRT_FE_TONEAREST;
  }
}

__attribute__((visibility("hidden"))) int __fe_raise_inexact(void) {
  // One genuinely-inexact divide sets SW.PE; stack-balanced, scratch
  // memory only, no flags touched besides the exception bits the op
  // itself raises.
  unsigned val = 3;
  __asm__ volatile("fld1; fildl %0; fdivp %%st, %%st(1); fstp %%st(0)"
                   : "+m"(val)
                   :
                   : "st");
  return 0;
}
