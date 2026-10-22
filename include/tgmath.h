/* Type-generic math (<tgmath.h>) for glcc.
 *
 * The system header uses clang's `__attribute__((overloadable))`, which
 * glcc does not implement. These macros dispatch with the GNU type
 * builtins. They are an extension in pre-C11 modes, just as several
 * clang resource headers rely on compiler extensions regardless of the
 * selected dialect.
 */
#ifndef __GLCC_TGMATH_H
#define __GLCC_TGMATH_H

#include <math.h>

#ifndef __cplusplus
#include <complex.h>

#define __glcc_tg_is(t, x) __builtin_types_compatible_p(__typeof__(x), t)

#define __glcc_tg_call1(x, f, d, l) \
    __builtin_choose_expr(__glcc_tg_is(float, x), f(x), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), l(x), d(x)))

#define __glcc_tg_call2(x, y, f, d, l) \
    __builtin_choose_expr(__glcc_tg_is(float, (x) + (y)), f((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(long double, (x) + (y)), l((x), (y)), d((x), (y))))

#define __glcc_tg_call1x(x, y, f, d, l) \
    __builtin_choose_expr(__glcc_tg_is(float, x), f((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), l((x), (y)), d((x), (y))))

#define __glcc_tg_call2x(x, y, z, f, d, l) \
    __builtin_choose_expr(__glcc_tg_is(float, (x) + (y)), f((x), (y), (z)), \
    __builtin_choose_expr(__glcc_tg_is(long double, (x) + (y)), l((x), (y), (z)), d((x), (y), (z))))

#define __glcc_tg_call3(x, y, z, f, d, l) \
    __builtin_choose_expr(__glcc_tg_is(float, (x) + (y) + (z)), f((x), (y), (z)), \
    __builtin_choose_expr(__glcc_tg_is(long double, (x) + (y) + (z)), l((x), (y), (z)), d((x), (y), (z))))

#define __glcc_tg_callz1(x, f, d, l, cf, cd, cl) \
    __builtin_choose_expr(__glcc_tg_is(float, x), f(x), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), l(x), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), cf(x), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), cd(x), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), cl(x), d(x))))))

#define __glcc_tg_callz2(x, y, f, d, l, cf, cd, cl) \
    __builtin_choose_expr(__glcc_tg_is(float, (x) + (y)), f((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(long double, (x) + (y)), l((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, (x) + (y)), cf((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, (x) + (y)), cd((x), (y)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, (x) + (y)), cl((x), (y)), d((x), (y)))))))

#undef acos
#undef asin
#undef atan
#undef acosh
#undef asinh
#undef atanh
#undef cos
#undef sin
#undef tan
#undef cosh
#undef sinh
#undef tanh
#undef exp
#undef log
#undef pow
#undef sqrt
#undef fabs
#undef atan2
#undef cbrt
#undef ceil
#undef copysign
#undef erf
#undef erfc
#undef exp2
#undef expm1
#undef fdim
#undef floor
#undef fma
#undef fmax
#undef fmin
#undef fmod
#undef frexp
#undef hypot
#undef ilogb
#undef ldexp
#undef lgamma
#undef llrint
#undef llround
#undef log10
#undef log1p
#undef log2
#undef logb
#undef lrint
#undef lround
#undef nearbyint
#undef nextafter
#undef nexttoward
#undef remainder
#undef remquo
#undef rint
#undef round
#undef scalbn
#undef scalbln
#undef tgamma
#undef trunc
#undef carg
#undef cimag
#undef conj
#undef cproj
#undef creal

#define acos(x) __glcc_tg_callz1((x), acosf, (acos), acosl, cacosf, cacos, cacosl)
#define asin(x) __glcc_tg_callz1((x), asinf, (asin), asinl, casinf, casin, casinl)
#define atan(x) __glcc_tg_callz1((x), atanf, (atan), atanl, catanf, catan, catanl)
#define acosh(x) __glcc_tg_callz1((x), acoshf, (acosh), acoshl, cacoshf, cacosh, cacoshl)
#define asinh(x) __glcc_tg_callz1((x), asinhf, (asinh), asinhl, casinhf, casinh, casinhl)
#define atanh(x) __glcc_tg_callz1((x), atanhf, (atanh), atanhl, catanhf, catanh, catanhl)
#define cos(x) __glcc_tg_callz1((x), cosf, (cos), cosl, ccosf, ccos, ccosl)
#define sin(x) __glcc_tg_callz1((x), sinf, (sin), sinl, csinf, csin, csinl)
#define tan(x) __glcc_tg_callz1((x), tanf, (tan), tanl, ctanf, ctan, ctanl)
#define cosh(x) __glcc_tg_callz1((x), coshf, (cosh), coshl, ccoshf, ccosh, ccoshl)
#define sinh(x) __glcc_tg_callz1((x), sinhf, (sinh), sinhl, csinhf, csinh, csinhl)
#define tanh(x) __glcc_tg_callz1((x), tanhf, (tanh), tanhl, ctanhf, ctanh, ctanhl)
#define exp(x) __glcc_tg_callz1((x), expf, (exp), expl, cexpf, cexp, cexpl)
#define log(x) __glcc_tg_callz1((x), logf, (log), logl, clogf, clog, clogl)
#define pow(x, y) __glcc_tg_callz2((x), (y), powf, (pow), powl, cpowf, cpow, cpowl)
#define sqrt(x) __glcc_tg_callz1((x), sqrtf, (sqrt), sqrtl, csqrtf, csqrt, csqrtl)
#define fabs(x) __glcc_tg_callz1((x), fabsf, (fabs), fabsl, cabsf, cabs, cabsl)

#define atan2(x, y) __glcc_tg_call2((x), (y), atan2f, (atan2), atan2l)
#define cbrt(x) __glcc_tg_call1((x), cbrtf, (cbrt), cbrtl)
#define ceil(x) __glcc_tg_call1((x), ceilf, (ceil), ceill)
#define copysign(x, y) __glcc_tg_call2((x), (y), copysignf, (copysign), copysignl)
#define erf(x) __glcc_tg_call1((x), erff, (erf), erfl)
#define erfc(x) __glcc_tg_call1((x), erfcf, (erfc), erfcl)
#define exp2(x) __glcc_tg_call1((x), exp2f, (exp2), exp2l)
#define expm1(x) __glcc_tg_call1((x), expm1f, (expm1), expm1l)
#define fdim(x, y) __glcc_tg_call2((x), (y), fdimf, (fdim), fdiml)
#define floor(x) __glcc_tg_call1((x), floorf, (floor), floorl)
#define fma(x, y, z) __glcc_tg_call3((x), (y), (z), fmaf, (fma), fmal)
#define fmax(x, y) __glcc_tg_call2((x), (y), fmaxf, (fmax), fmaxl)
#define fmin(x, y) __glcc_tg_call2((x), (y), fminf, (fmin), fminl)
#define fmod(x, y) __glcc_tg_call2((x), (y), fmodf, (fmod), fmodl)
#define frexp(x, y) __glcc_tg_call1x((x), (y), frexpf, (frexp), frexpl)
#define hypot(x, y) __glcc_tg_call2((x), (y), hypotf, (hypot), hypotl)
#define ilogb(x) __glcc_tg_call1((x), ilogbf, (ilogb), ilogbl)
#define ldexp(x, y) __glcc_tg_call1x((x), (y), ldexpf, (ldexp), ldexpl)
#define lgamma(x) __glcc_tg_call1((x), lgammaf, (lgamma), lgammal)
#define llrint(x) __glcc_tg_call1((x), llrintf, (llrint), llrintl)
#define llround(x) __glcc_tg_call1((x), llroundf, (llround), llroundl)
#define log10(x) __glcc_tg_call1((x), log10f, (log10), log10l)
#define log1p(x) __glcc_tg_call1((x), log1pf, (log1p), log1pl)
#define log2(x) __glcc_tg_call1((x), log2f, (log2), log2l)
#define logb(x) __glcc_tg_call1((x), logbf, (logb), logbl)
#define lrint(x) __glcc_tg_call1((x), lrintf, (lrint), lrintl)
#define lround(x) __glcc_tg_call1((x), lroundf, (lround), lroundl)
#define nearbyint(x) __glcc_tg_call1((x), nearbyintf, (nearbyint), nearbyintl)
#define nextafter(x, y) __glcc_tg_call2((x), (y), nextafterf, (nextafter), nextafterl)
#define nexttoward(x, y) __glcc_tg_call1x((x), (y), nexttowardf, (nexttoward), nexttowardl)
#define remainder(x, y) __glcc_tg_call2((x), (y), remainderf, (remainder), remainderl)
#define remquo(x, y, z) __glcc_tg_call2x((x), (y), (z), remquof, (remquo), remquol)
#define rint(x) __glcc_tg_call1((x), rintf, (rint), rintl)
#define round(x) __glcc_tg_call1((x), roundf, (round), roundl)
#define scalbn(x, y) __glcc_tg_call1x((x), (y), scalbnf, (scalbn), scalbnl)
#define scalbln(x, y) __glcc_tg_call1x((x), (y), scalblnf, (scalbln), scalblnl)
#define tgamma(x) __glcc_tg_call1((x), tgammaf, (tgamma), tgammal)
#define trunc(x) __glcc_tg_call1((x), truncf, (trunc), truncl)

#define carg(x) \
    __builtin_choose_expr(__glcc_tg_is(float, x), atan2f(0.0f, (x)), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), atan2l(0.0L, (x)), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), cargf((x)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), (carg)((x)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), cargl((x)), atan2(0.0, (x)))))))
#define cimag(x) \
    __builtin_choose_expr(__glcc_tg_is(float, x), 0.0f, \
    __builtin_choose_expr(__glcc_tg_is(long double, x), 0.0L, \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), cimagf((x)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), (cimag)((x)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), cimagl((x)), 0.0)))))
#define conj(x) \
    __builtin_choose_expr(__glcc_tg_is(float, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), conjf((x)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), (conj)((x)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), conjl((x)), (double)(x))))))
#define cproj(x) \
    __builtin_choose_expr(__glcc_tg_is(float, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), cprojf((x)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), (cproj)((x)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), cprojl((x)), (double)(x))))))
#define creal(x) \
    __builtin_choose_expr(__glcc_tg_is(float, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(long double, x), (x), \
    __builtin_choose_expr(__glcc_tg_is(float _Complex, x), crealf((x)), \
    __builtin_choose_expr(__glcc_tg_is(double _Complex, x), (creal)((x)), \
    __builtin_choose_expr(__glcc_tg_is(long double _Complex, x), creall((x)), (double)(x))))))

#endif /* __cplusplus */
#endif /* __GLCC_TGMATH_H */
