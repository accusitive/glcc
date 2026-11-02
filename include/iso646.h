/* Alternative spellings (<iso646.h>) for glcc. C23 makes the
 * spellings keywords; before that they are macros.
 */
#ifndef __GLCC_ISO646_H
#define __GLCC_ISO646_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L
#define and &&
#define and_eq &=
#define bitand &
#define bitor |
#define compl ~
#define not !
#define not_eq !=
#define or ||
#define or_eq |=
#define xor ^
#define xor_eq ^=
#endif

#endif
