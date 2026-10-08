#version 330 core
#include "common.glsl"
// Doble umbral con comprobación de vecinos (R12.10) sobre la salida de nms.frag (r).
// v = NMS · u_mag_gain (0.25: un escalón negro → blanco ideal da v = 1; D3, S7).
//   fuerte: v ≥ u_high                → borde
//   débil:  u_low ≤ v < u_high        → borde solo si alguno de sus 8 vecinos es fuerte
//   resto:  v < u_low                 → no borde
// Una sola pasada, sin histéresis iterativa (fuera de alcance).
// Salida: r = borde (0/1), g = clasificación (0 / 0.5 / 1), b = r, alfa = 1.

uniform sampler2D u_src;
uniform float u_low;
uniform float u_high;
uniform float u_mag_gain;

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float v = fetch(u_src, p).r * u_mag_gain;
    float cls = v >= u_high ? 1.0 : (v >= u_low ? 0.5 : 0.0);
    float edge = cls == 1.0 ? 1.0 : 0.0;
    if (cls == 0.5) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if ((dx != 0 || dy != 0) && fetch(u_src, p + ivec2(dx, dy)).r * u_mag_gain >= u_high) {
                    edge = 1.0;
                }
            }
        }
    }
    frag_color = vec4(edge, cls, edge, 1.0);
}
