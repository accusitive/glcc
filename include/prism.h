/* prism.h — the prism shader dialect's user surface.
 *
 * Requires `glcc -fprism` (the header refuses to compile elsewhere):
 * prism is a C subset for shader programs — see `docs/prism.md` and
 * `crates/glcc_prism`. Entry points are marked with `[[prism::vertex]]`
 * / `[[prism::fragment]]`; inputs and outputs are plain C structs
 * whose members carry `[[prism::semantic(NAME)]]`.
 *
 * Everything here is plain C: vector typedefs ride GCC's `vector_size`
 * attribute, math helpers are macros over swizzles, and the vector
 * constructors plus lane-wise min/max/abs are compiler builtins —
 * they lower to shader operations, never to library calls. Macros
 * keep to swizzles and calls on purpose: `__builtin_choose_expr`
 * branches are checked lazily, but a compound literal anywhere in an
 * expansion is checked eagerly at parse time (it would error in the
 * branches the dispatch did not select).
 */
#ifndef __GLCC_PRISM_H
#define __GLCC_PRISM_H

#ifndef __GLCC_PRISM__
#error "prism.h requires the prism dialect: compile with glcc -fprism"
#endif

/* ---- vector typedefs (GCC vector extensions) ---- */

typedef float float2 __attribute__((vector_size(8)));
typedef float float3 __attribute__((vector_size(12)));
typedef float float4 __attribute__((vector_size(16)));

typedef int int2 __attribute__((vector_size(8)));
typedef int int3 __attribute__((vector_size(12)));
typedef int int4 __attribute__((vector_size(16)));

typedef unsigned int uint2 __attribute__((vector_size(8)));
typedef unsigned int uint3 __attribute__((vector_size(12)));
typedef unsigned int uint4 __attribute__((vector_size(16)));

/* Column-major, like HLSL's default: `mul(m, v)` computes
 * `float4(dot(m.c0, v), dot(m.c1, v), dot(m.c2, v), dot(m.c3, v))`. */
typedef struct {
    float4 c0, c1, c2, c3;
} float4x4;

/* ---- resources ---- */

/* Opaque host-bound texture+sampler pair (DX9-style combined
 * sampler). File-scope objects of this type are resources: they
 * cannot be copied, passed, or returned — only named in `tex2D`. */
typedef struct __prism_sampler2d {
    unsigned int __prism_opaque[2];
} sampler2D;

/* ---- compiler builtins ----
 *
 * Prototypes for the operations the compiler expands to shader
 * instructions (never calls): declared here so ordinary C type
 * checking applies, and so the implicit-declaration path never
 * fires. The per-width constructors exist because a compound literal
 * (`(float4){...}`) is checked at parse time and cannot ride inside
 * a type-generic dispatch. */

float4 tex2D(sampler2D s, float2 uv);

float min(float a, float b);
float max(float a, float b);
float sqrtf(float x);

float2 splat2(float s);
float3 splat3(float s);
float4 splat4(float s);

float2 min2(float2 a, float2 b);
float3 min3(float3 a, float3 b);
float4 min4(float4 a, float4 b);
float2 max2(float2 a, float2 b);
float3 max3(float3 a, float3 b);
float4 max4(float4 a, float4 b);

float2 abs2(float2 a);
float3 abs3(float3 a);
float4 abs4(float4 a);

float4 mul4x4(float4x4 m, float4 v);

/* ---- macro helpers (swizzles and calls only) ---- */

#define dot2(a, b) ((a).x * (b).x + (a).y * (b).y)
#define dot3(a, b) (dot2(a, b) + (a).z * (b).z)
#define dot4(a, b) (dot3(a, b) + (a).w * (b).w)

#define length2(a) sqrtf(dot2(a, a))
#define length3(a) sqrtf(dot3(a, a))
#define length4(a) sqrtf(dot4(a, a))

#define normalize2(a) ((a) / splat2(length2(a)))
#define normalize3(a) ((a) / splat3(length3(a)))
#define normalize4(a) ((a) / splat4(length4(a)))

#define distance2(a, b) length2((b) - (a))
#define distance3(a, b) length3((b) - (a))
#define distance4(a, b) length4((b) - (a))

#define lerp2(a, b, t) ((a) + ((b) - (a)) * (t))
#define lerp3(a, b, t) ((a) + ((b) - (a)) * (t))
#define lerp4(a, b, t) ((a) + ((b) - (a)) * (t))

#define clamp2(a, lo, hi) min2(max2(a, splat2(lo)), splat2(hi))
#define clamp3(a, lo, hi) min3(max3(a, splat3(lo)), splat3(hi))
#define clamp4(a, lo, hi) min4(max4(a, splat4(lo)), splat4(hi))

#define saturate_f(a) min(max(a, 0.0f), 1.0f)

/* ---- type-generic spellings ----
 *
 * `__builtin_choose_expr` checks only the selected branch (compound
 * literals excepted — see the header comment), so the per-width
 * expansions never type-check against the wrong width. The scalar
 * branches reference the declared builtins directly (the macro name
 * inside its own expansion stays a function call). */

#define __prism_is(t, x) __builtin_types_compatible_p(__typeof__(x), t)

#define dot(a, b)                                                                                                  \
    __builtin_choose_expr(__prism_is(float2, a), dot2(a, b),                                                       \
    __builtin_choose_expr(__prism_is(float3, a), dot3(a, b),                                                       \
                                                       dot4(a, b)))

#define length(a)                                                                                                  \
    __builtin_choose_expr(__prism_is(float2, a), length2(a),                                                       \
    __builtin_choose_expr(__prism_is(float3, a), length3(a),                                                       \
                                                       length4(a)))

#define normalize(a)                                                                                               \
    __builtin_choose_expr(__prism_is(float2, a), normalize2(a),                                                    \
    __builtin_choose_expr(__prism_is(float3, a), normalize3(a),                                                    \
                                                       normalize4(a)))

#define distance(a, b)                                                                                             \
    __builtin_choose_expr(__prism_is(float2, a), distance2(a, b),                                                  \
    __builtin_choose_expr(__prism_is(float3, a), distance3(a, b),                                                  \
                                                       distance4(a, b)))

#define saturate(a)                                                                                                \
    __builtin_choose_expr(__prism_is(float, a), saturate_f(a),                                                     \
    __builtin_choose_expr(__prism_is(float2, a), clamp2(a, 0.0f, 1.0f),                                            \
    __builtin_choose_expr(__prism_is(float3, a), clamp3(a, 0.0f, 1.0f),                                            \
                                                       clamp4(a, 0.0f, 1.0f))))

/* Vectors only: `clamp` on scalars is `min(max(a, lo), hi)`. */
#define clamp(a, lo, hi)                                                                                           \
    __builtin_choose_expr(__prism_is(float2, a), clamp2(a, lo, hi),                                                \
    __builtin_choose_expr(__prism_is(float3, a), clamp3(a, lo, hi),                                                \
                                                       clamp4(a, lo, hi)))

/* Lane-wise min/max on vectors (`min`/`max` themselves stay the
 * scalar builtins). */
#define vmin(a, b)                                                                                                 \
    __builtin_choose_expr(__prism_is(float2, a), min2(a, b),                                                       \
    __builtin_choose_expr(__prism_is(float3, a), min3(a, b),                                                       \
                                                       min4(a, b)))

#define vmax(a, b)                                                                                                 \
    __builtin_choose_expr(__prism_is(float2, a), max2(a, b),                                                       \
    __builtin_choose_expr(__prism_is(float3, a), max3(a, b),                                                       \
                                                       max4(a, b)))

/* Matrix-vector product (column-major). */
#define mul(m, v) mul4x4(m, v)

#endif /* __GLCC_PRISM_H */
