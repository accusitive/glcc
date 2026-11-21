/* A fragment shader: the diffuse map blended toward its tint by an
 * accumulated detail-map glow. Shows tex2D over a loop-carried
 * accumulator, two samplers, and vector/scalar uniform packing
 * (detail_strength shares a c-register with nothing — it starts a
 * fresh register after the tint vector).
 *
 *   make      # compiles to build/frag.isa
 */
#include <prism.h>

struct FSIn {
    [[prism::semantic(TEXCOORD0)]] float2 uv;
};

struct FSOut {
    [[prism::semantic(COLOR0)]] float4 color;
};

sampler2D diffuse;
sampler2D detail;
float4 tint [[prism::uniform]];
float detail_strength [[prism::uniform]];

[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    float4 base = tex2D(diffuse, i.uv);
    float4 glow = splat4(0.0f);
    for (int j = 0; j < 4; j = j + 1) {
        float t = (float)j * 0.25f;
        float2 duv = splat2(t * detail_strength);
        glow = glow + tex2D(detail, i.uv + duv);
    }
    glow = glow * splat4(0.25f);
    o.color = lerp4(base, base * tint, glow);
    return o;
}
