/* Boolean type (<stdbool.h>) for glcc. C23 makes bool/true/false
 * keywords; before that they are macros over _Bool.
 */
#ifndef __GLCC_STDBOOL_H
#define __GLCC_STDBOOL_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L
#define bool _Bool
#define true 1
#define false 0
#endif

#define __bool_true_false_are_defined 1

#endif
