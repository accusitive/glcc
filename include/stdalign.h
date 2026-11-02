/* Alignment (<stdalign.h>) for glcc. C23 makes alignas/alignof
 * keywords; before that they are macros over _Alignas/_Alignof.
 */
#ifndef __GLCC_STDALIGN_H
#define __GLCC_STDALIGN_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L
#define alignas _Alignas
#define alignof _Alignof
#endif

#define __alignas_is_defined 1
#define __alignof_is_defined 1

#endif
