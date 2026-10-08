#version 330 core
// Quad a pantalla completa: los vértices ya están en coordenadas de recorte (±1).
// Sin uniforms ni transformaciones geométricas (R2.1, R2.2).
in vec2 in_pos;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
