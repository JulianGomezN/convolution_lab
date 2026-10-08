#version 330 core
#include "common.glsl"
// Sobel sobre el canal r (imagen en gris), como correlación con y hacia abajo (S1):
//   Gx = [-1 0 1; -2 0 2; -1 0 1]     Gy = [-1 -2 -1; 0 0 0; 1 2 1]
// Oscuro → claro hacia la derecha da Gx > 0; hacia abajo, Gy > 0.
// Salida en una sola textura (R12.8): (Gx, Gy, |G|, θ = atan(Gy, Gx)), θ en [-π, π].

uniform sampler2D u_src;

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float s[9];  // s[r*3 + c] = vecino (c-1, r-1)
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            s[r * 3 + c] = fetch(u_src, p + ivec2(c - 1, r - 1)).r;
        }
    }
    float gx = (s[2] + 2.0 * s[5] + s[8]) - (s[0] + 2.0 * s[3] + s[6]);
    float gy = (s[6] + 2.0 * s[7] + s[8]) - (s[0] + 2.0 * s[1] + s[2]);
    // atan(0, 0) no está definido en GLSL; numpy.arctan2(0, 0) da 0.
    float theta = (gx == 0.0 && gy == 0.0) ? 0.0 : atan(gy, gx);
    frag_color = vec4(gx, gy, length(vec2(gx, gy)), theta);
}
