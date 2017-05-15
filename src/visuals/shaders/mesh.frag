#version 120

uniform vec4 u_ambientk;
uniform vec4 u_light_color;
uniform vec4 u_base_color;

varying vec3 v_normal_vec;
varying vec3 v_light_vec;

varying vec4 v_base_color;

varying vec4 v_color;

void main() {
    // Diffuse shading
    float diffusek = dot(v_light_vec, v_normal_vec);
    diffusek = clamp(diffusek, 0.0, 1.0);
    vec4 diffuse_color = u_light_color * diffusek;

    gl_FragColor = u_base_color * (u_ambientk + diffuse_color);
}
