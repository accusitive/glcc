/* A vertex shader: MVP transform plus a per-vertex Lambert term.
 * Shows the interface-struct form (in/out members carry
 * [[prism::semantic]]), matrix uniforms packed into c-registers, and
 * a helper function — prep force-inlines helpers into the entry.
 *
 *   make      # compiles to build/vs.isa
 */
#include <prism.h>

struct VSIn {
    [[prism::semantic(POSITION)]] float4 pos;
    [[prism::semantic(NORMAL)]] float3 normal;
    [[prism::semantic(TEXCOORD0)]] float2 uv;
};

struct VSOut {
    [[prism::semantic(POSITION)]] float4 pos;
    [[prism::semantic(COLOR0)]] float4 diffuse;
    [[prism::semantic(TEXCOORD0)]] float2 uv;
};

float4x4 mvp [[prism::uniform]];      /* takes c0..c3 */
float4 light_dir [[prism::uniform]];  /* view-space direction */
float4 light_color [[prism::uniform]];
float4 ambient [[prism::uniform]];

static float4 lambert(float3 n) {
    float ndl = dot3(n, normalize3(light_dir.xyz));
    return light_color * splat4(max(ndl, 0.0f));
}

[[prism::vertex]]
struct VSOut vs_main(struct VSIn i) {
    struct VSOut o;
    o.pos = mul(mvp, i.pos);
    o.diffuse = lambert(normalize3(i.normal)) + ambient;
    o.uv = i.uv;
    return o;
}
