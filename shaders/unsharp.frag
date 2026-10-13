#version 330 core
#include "common.glsl"
// Unsharp masking (R13.1): orig + α·(orig − blur).
// u_blur es el gaussiano de u_src calculado antes (blur_1d.frag en dos pasadas o
// blur_2d.frag, según la tecla S). Solo lee el texel propio, así que el modo de
// borde no influye aquí; sí influye en el desenfoque previo.

uniform sampler2D u_src;
uniform sampler2D u_blur;
uniform float u_alpha;

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 s = fetch(u_src, p).rgb;
    vec3 b = fetch(u_blur, p).rgb;
    frag_color = vec4(s + u_alpha * (s - b), 1.0);  // alfa siempre 1 (S3)
}
