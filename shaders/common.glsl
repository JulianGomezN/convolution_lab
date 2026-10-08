// Funciones compartidas por los shaders que leen vecinos (R10.3).
// shader_loader lo inserta donde aparece la directiva include, justo después de
// #version. (No escribir la directiva literal en comentarios: moderngl también la busca.)
//
// Modos de borde (u_border), equivalentes a numpy.pad (R10.5):
//   0 zero      -> constant (0)   fuera de la imagen devuelve vec4(0)
//   1 replicate -> edge           aaaa|abcd|dddd
//   2 reflect   -> reflect        dcb|abcd|cba (sin repetir el borde), periódico
//   3 wrap      -> wrap           abcd|abcd|abcd
// Se remapean coordenadas enteras antes de texelFetch, así funcionan con
// desplazamientos mayores que la imagen (R10.4). El wrap del sampler no sirve:
// texelFetch lo ignora.

uniform int u_border;

// Módulo con resultado en [0, n) también para a < 0. En GLSL 3.30 el operador %
// no está definido con operandos negativos. Exacto para |a| < 2^24.
// La división en float de la GPU no tiene por qué estar correctamente redondeada
// (por ejemplo, 14/7 puede dar 1.9999999), así que se corrige un posible ±1.
int imod(int a, int n) {
    int m = a - n * int(floor(float(a) / float(n)));
    if (m < 0) {
        m += n;
    } else if (m >= n) {
        m -= n;
    }
    return m;
}

// Reflexión sin repetir el borde con periodo 2n-2; con n = 1 siempre es 0.
int reflect_idx(int p, int n) {
    if (n == 1) {
        return 0;
    }
    int period = 2 * n - 2;
    int m = imod(p, period);
    return m < n ? m : period - m;
}

// Lee el texel entero p de t aplicando el modo de borde u_border.
vec4 fetch(sampler2D t, ivec2 p) {
    ivec2 n = textureSize(t, 0);
    if (u_border == 1) {
        p = clamp(p, ivec2(0), n - 1);
    } else if (u_border == 2) {
        p = ivec2(reflect_idx(p.x, n.x), reflect_idx(p.y, n.y));
    } else if (u_border == 3) {
        p = ivec2(imod(p.x, n.x), imod(p.y, n.y));
    } else if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, n))) {
        return vec4(0.0);  // zero (0) y cualquier valor desconocido
    }
    return texelFetch(t, p, 0);
}
