#version 120

uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main() {
    gl_Position = u_projection * u_view * vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}

