#version 120

// Uniforms
// ------------------------------------
uniform float u_linewidth;
uniform float u_antialias;
uniform float u_size;

// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute vec3  a_color;

// Varyings
// ------------------------------------
varying vec4 v_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_size = u_size;
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;
    v_color  = vec4(a_color, 1.0);
    vec4 visual_pos = vec4(a_position, 1);
    vec4 doc_pos = $visual_to_doc(visual_pos);
    gl_Position = $doc_to_render(doc_pos);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}

