// glcc soft-float glue for Windows targets: the x87-free companion of
// glcc-fe.c, for the quad arithmetic in this directory when it is
// compiled by glcc for `x86_64-windows` (see build.rs; Win64 `long
// double` is a double, so the runtime cannot use the x87 control
// word). The UCRT `fegetround` query (via its fenv API) and a
// genuinely-inexact divide's effect (UCRT has no `feraiseexcept`) are
// what the soft-float code reads instead, so quad arithmetic follows
// the environment the same way the x87 path does on ELF.
#include "fp_mode.h"

int fegetround(void);
int feraiseexcept(int);

// UCRT fenv.h: the rounding-mode values are the x87 control-word
// encodings (FE_TONEAREST 0x000, FE_DOWNWARD 0x100, FE_UPWARD 0x200,
// FE_TOWARDZERO 0x300), inlined so this TU needs no system headers
// under glcc.
CRT_FE_ROUND_MODE __fe_getround(void) {
  switch (fegetround()) {
  case 0x100:
    return CRT_FE_DOWNWARD;
  case 0x200:
    return CRT_FE_UPWARD;
  case 0x300:
    return CRT_FE_TOWARDZERO;
  default:
    return CRT_FE_TONEAREST;
  }
}

int __fe_raise_inexact(void) {
  // UCRT has no `feraiseexcept`: one genuinely-inexact divide sets the
  // sticky inexact flag in MXCSR, the Win64 counterpart of the x87
  // status word the ELF glue sets.
  volatile double one = 1.0;
  volatile double three = 3.0;
  volatile double inexact = one / three;
  (void)inexact;
  return 0;
}
