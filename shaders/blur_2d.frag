#version 330 core
#include "common.glsl"
// Desenfoque 2D directo en una sola pasada: (2r+1)² lecturas por píxel (R14.1).
// El peso del vecino p + (i, j) es u_weights[i + r] · u_weights[j + r], con el mismo
// array 1D que blur_1d.frag. Un array de 61² = 3721 pesos superaría el mínimo
// garantizado de componentes uniform del fragment shader (1024).
// Como cada modo de borde remapea x e y por separado, el resultado coincide con las
// dos pasadas de blur_1d.frag salvo redondeo (R14.2). Lo que cambia es el coste,
// que es lo que mide el benchmark.

uniform sampler2D u_src;
uniform int u_radius;
uniform float u_weights[61];

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 acc = vec3(0.0);
    for (int j = -u_radius; j <= u_radius; ++j) {
        vec3 row = vec3(0.0);
        for (int i = -u_radius; i <= u_radius; ++i) {
            row += u_weights[i + u_radius] * fetch(u_src, p + ivec2(i, j)).rgb;
        }
        acc += u_weights[j + u_radius] * row;
    }
    frag_color = vec4(acc, 1.0);  // alfa siempre 1 (S3)
}
