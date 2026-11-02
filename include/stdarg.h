/* Variable arguments (<stdarg.h>) for glcc.
 *
 * The frontend lowers the __builtin_va_* forms to the target ABI
 * (SysV/AAPCS64 register-save areas, RISC-V and Windows pointer
 * lists), so this header only names them.
 */
#ifndef __GLCC_STDARG_H
#define __GLCC_STDARG_H

typedef __builtin_va_list va_list;

#define va_start(v, l) __builtin_va_start(v, l)
#define va_arg(v, T) __builtin_va_arg(v, T)
#define va_end(v) __builtin_va_end(v)
#define va_copy(d, s) __builtin_va_copy(d, s)

#endif
