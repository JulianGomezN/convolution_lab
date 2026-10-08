#version 330 core
#include "common.glsl"
// Desenfoque 1D en la dirección u_direction: (1, 0) horizontal, (0, 1) vertical.
// Dos pasadas (H y V) dan un desenfoque separable con 2·(2r+1) lecturas por píxel.
// u_weights[i + r] es el peso del vecino p + i·u_direction, i = -r..r (R11.4).
// La longitud 61 es MAX_TAPS de kernels.py (r ≤ 30); un test comprueba que coinciden.

uniform sampler2D u_src;
uniform ivec2 u_direction;
uniform int u_radius;
uniform float u_weights[61];

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 acc = vec3(0.0);
    for (int i = -u_radius; i <= u_radius; ++i) {
        acc += u_weights[i + u_radius] * fetch(u_src, p + i * u_direction).rgb;
    }
    frag_color = vec4(acc, 1.0);  // alfa siempre 1 (S3)
}
