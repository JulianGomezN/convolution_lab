#version 330 core
#include "common.glsl"
// Supresión de no máximos (R12.9) sobre la salida de sobel.frag: b = |G|, a = θ.
// θ se cuantiza a 0°, 45°, 90° o 135° (a = θ en grados, módulo 180) y |G| se
// conserva solo si es máximo frente a sus 2 vecinos en la dirección del gradiente.
// Con y hacia abajo, θ = 45° apunta a (+1, +1).
// Desigualdad asimétrica (m ≥ n1 y m > n2): en una meseta de 2 píxeles iguales
// solo sobrevive uno, sin líneas dobles. Salida (m, m, m, 1), |G| sin escalar (D3).

uniform sampler2D u_src;

out vec4 frag_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 g = fetch(u_src, p);
    float m = g.b;
    float a = mod(degrees(g.a) + 180.0, 180.0);
    ivec2 d;
    if (a < 22.5 || a >= 157.5) {
        d = ivec2(1, 0);   // 0°: vecinos izquierda y derecha
    } else if (a < 67.5) {
        d = ivec2(1, 1);   // 45°
    } else if (a < 112.5) {
        d = ivec2(0, 1);   // 90°: vecinos arriba y abajo
    } else {
        d = ivec2(-1, 1);  // 135°
    }
    float n1 = fetch(u_src, p + d).b;
    float n2 = fetch(u_src, p - d).b;
    float v = (m >= n1 && m > n2) ? m : 0.0;
    frag_color = vec4(v, v, v, 1.0);
}
