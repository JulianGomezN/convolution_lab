#version 330 core

uniform sampler2D u_status;

out vec4 frag_color;

void main() {
    ivec2 size = textureSize(u_status, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    p.y = size.y - 1 - p.y;
    frag_color = texelFetch(u_status, p, 0);
}
