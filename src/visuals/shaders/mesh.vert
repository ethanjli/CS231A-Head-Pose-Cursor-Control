#version 120

uniform vec4 u_ambientk;
uniform vec4 u_light_color;
uniform vec3 u_light_dir;
uniform vec4 u_base_color;

varying vec3 v_normal_vec;
varying vec3 v_light_vec;

void main() {
    vec4 pos_doc = $visual_to_doc(vec4($position, 1.0));
    vec4 normal_doc = $visual_to_doc(vec4($normal, 1.0));
    vec4 origin_doc = $visual_to_doc(vec4(0.0, 0.0, 0.0, 1.0));

    normal_doc /= normal_doc.w;
    origin_doc /= origin_doc.w;

    vec3 normal = normalize(normal_doc.xyz - origin_doc.xyz);
    v_normal_vec = normal; // varying copy

    vec3 light = normalize(u_light_dir);
    v_light_vec = light; // varying copy

    gl_Position = $doc_to_render($visual_to_doc(vec4($position, 1.0)));
}
