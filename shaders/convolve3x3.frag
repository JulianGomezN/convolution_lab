#version 330 core
#include "common.glsl"
// Correlación 3x3 (S1, R5.2): fila r → dy = r-1, columna c → dx = c-1, y hacia abajo.
// Los 9 pesos llegan como array de floats (R6.2), fila superior primero.

uniform sampler2D u_src;
uniform float u_kernel[9];

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 acc = vec3(0.0);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            acc += u_kernel[r * 3 + c] * fetch(u_src, p + ivec2(c - 1, r - 1)).rgb;
        }
    }
    frag_color = vec4(acc, 1.0);  // alfa siempre 1 (S3)
}
