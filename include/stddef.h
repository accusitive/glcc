/* Common definitions (<stddef.h>) for glcc.
 *
 * Types come from the target-derived predefined macros; the MSVC
 * guard macros keep the Windows SDK headers from typedef'ing the same
 * names again.
 */
#ifndef __GLCC_STDDEF_H
#define __GLCC_STDDEF_H

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

#define NULL ((void *)0)
#define offsetof(T, m) __builtin_offsetof(T, m)

/* The alignment of the strictest alignment the target gives any
 * scalar type: `long double` covers it (16 on x86-64 System V and
 * AArch64/RISC-V, 8 on Windows where it is `double`). */
typedef struct {
    long long __ll;
    long double __ld;
} max_align_t;

#if defined(_MSC_VER)
#define _SIZE_T_DEFINED
#define _PTRDIFF_T_DEFINED
#define _WCHAR_T_DEFINED
#define _NULL_DEFINED
#endif

#endif
