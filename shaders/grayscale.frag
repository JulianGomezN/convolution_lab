#version 330 core
#include "common.glsl"
// Escala de grises Rec.601 sobre valores sRGB sin linealizar (S4, R12.6):
// Y = 0.299 R + 0.587 G + 0.114 B, salida (Y, Y, Y, 1).
// Solo lee el propio píxel, siempre dentro de la imagen; se usa fetch() para que
// todas las lecturas pasen por common.glsl.

uniform sampler2D u_src;

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float y = dot(fetch(u_src, p).rgb, vec3(0.299, 0.587, 0.114));
    frag_color = vec4(y, y, y, 1.0);
}
