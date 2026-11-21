/* Control flow in the dialect: early return, a do-while with break
 * and continue, and prism.h's length/saturate/lerp helpers. Around
 * the UV center, draws up to six concentric rings (skipping the
 * second) whose brightness falls off toward the disc edge; outside
 * it, the plain texture via an early return.
 *
 *   make      # compiles to build/rings.isa
 */
#include <prism.h>

struct FSIn {
    [[prism::semantic(TEXCOORD0)]] float2 uv;
};

struct FSOut {
    [[prism::semantic(COLOR0)]] float4 color;
};

sampler2D tex;
float4 ring_color [[prism::uniform]];
float ring_scale [[prism::uniform]];

[[prism::fragment]]
struct FSOut fs_main(struct FSIn i) {
    struct FSOut o;
    float2 p = i.uv - splat2(0.5f);
    float d = length2(p);
    if (d > 0.5f) {
        o.color = tex2D(tex, i.uv);
        return o;
    }
    float shell = 0.0f;
    int j = 0;
    do {
        j = j + 1;
        if (j == 2) { continue; }
        if (j > 6) { break; }
        float r = d * ring_scale;
        float band = saturate_f(1.0f - max(r - (float)j, (float)j - r));
        shell = max(shell, band);
    } while (j < 8);
    float4 c = tex2D(tex, i.uv);
    o.color = lerp4(c, ring_color, splat4(shell));
    return o;
}
