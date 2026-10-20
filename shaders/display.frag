#version 330 core
// Muestra una textura en pantalla o en el FBO de captura.
// Las texturas guardan la fila 0 arriba (D2); la pantalla tiene el origen abajo a
// la izquierda, así que con u_flip_y = 1 se remapea el índice de fila:
// y_tex = H - 1 - y. Es un remapeo de índices enteros: los vértices no cambian.
// La captura usa u_flip_y = 0 porque fbo.read() devuelve la fila 0 primero.
//
// Canal (u_channel), para ver por separado las etapas de Canny:
//   0: rgb    1: r    2: g    3: b   (1-3 se muestran en gris)
//   4: dirección del gradiente de sobel.frag: tono = (θ + π) / 2π con θ en a,
//      brillo = |G| (b) · u_gain recortado a [0, 1]; no usa u_map.
// Mapeo de valores (R9.2), sobre v = valor · u_gain:
//   u_map 0: recorte a [0, 1]
//   u_map 1: |v|
//   u_map 2: offset 0.5 + 0.5·v (0 se ve gris medio)
//
// Modo lupa (R21): con u_lens_on = 1 el resultado procesado solo se muestra dentro
// de un círculo bajo el cursor; fuera se ve la imagen original (u_original) con un
// borde suave. u_lens_center está en el mismo sistema de índices que `p` ya
// "desflipado" (fila 0 arriba), igual que el fetch de u_src y de u_original, así
// que la distancia al centro no depende de u_flip_y. Con u_lens_on = 0 el resultado
// es exactamente el de siempre.

uniform sampler2D u_src;
uniform sampler2D u_original;
uniform int u_flip_y;
uniform int u_channel;
uniform int u_map;
uniform float u_gain;
uniform int u_lens_on;
uniform vec2 u_lens_center;  // en píxeles, sistema fila 0 arriba
uniform float u_lens_radius;  // radio en píxeles
uniform float u_lens_edge;  // ancho del borde suave en píxeles

out vec4 frag_color;

const float PI = 3.14159265358979;

// hsv → rgb con h, s, v en [0, 1].
vec3 hsv2rgb(vec3 c) {
    vec3 k = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(k - 1.0, 0.0, 1.0), c.y);
}

void main() {
    ivec2 size = textureSize(u_src, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (u_flip_y == 1) {
        p.y = size.y - 1 - p.y;
    }
    // Color procesado final (el de siempre): canal, mapeo/ganancia y rama de dirección.
    vec4 t = texelFetch(u_src, p, 0);
    vec3 processed;
    if (u_channel == 4) {
        float hue = (t.a + PI) / (2.0 * PI);
        processed = hsv2rgb(vec3(hue, 1.0, clamp(t.b * u_gain, 0.0, 1.0)));
    } else {
        vec3 v;
        if (u_channel == 1) {
            v = vec3(t.r);
        } else if (u_channel == 2) {
            v = vec3(t.g);
        } else if (u_channel == 3) {
            v = vec3(t.b);
        } else {
            v = t.rgb;
        }
        v *= u_gain;
        if (u_map == 1) {
            v = abs(v);
        } else if (u_map == 2) {
            v = 0.5 + 0.5 * v;
        }
        processed = clamp(v, 0.0, 1.0);
    }
    if (u_lens_on == 1) {
        // Dentro del círculo (m = 1) se ve el procesado; fuera (m = 0) la original,
        // con una transición suave de u_lens_edge píxeles antes del radio.
        vec3 original = clamp(texelFetch(u_original, p, 0).rgb, 0.0, 1.0);
        float d = distance(vec2(p), u_lens_center);
        float m = 1.0 - smoothstep(u_lens_radius - u_lens_edge, u_lens_radius, d);
        frag_color = vec4(mix(original, processed, m), 1.0);
    } else {
        frag_color = vec4(processed, 1.0);
    }
}
