/* Variable arguments (<stdarg.h>) for glcc.
 *
 * The frontend lowers the __builtin_va_* forms to the target ABI
 * (SysV/AAPCS64 register-save areas, RISC-V and Windows pointer
 * lists), so this header only names them.
 *
 * Follows GCC's stdarg.h protocol: glibc headers (stdio.h & co.)
 * predefine `__need___va_list` and include this header to obtain
 * `__gnuc_va_list` without leaking the user-level `va_list` name.
 */
#ifndef __GLCC_STDARG_H
#ifndef __need___va_list
#define __GLCC_STDARG_H
#endif
#undef __need___va_list

#ifndef __GLCC_GNUC_VA_LIST
#define __GLCC_GNUC_VA_LIST
typedef __builtin_va_list __gnuc_va_list;
#endif

#ifdef __GLCC_STDARG_H

#define va_start(v, l) __builtin_va_start(v, l)
#define va_arg(v, T) __builtin_va_arg(v, T)
#define va_end(v) __builtin_va_end(v)
#define va_copy(d, s) __builtin_va_copy(d, s)

typedef __gnuc_va_list va_list;

#endif

#endif
