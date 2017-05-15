#version 120

varying vec2 v_texcoord;

void main() {
    gl_Position = $doc_to_render($visual_to_doc(vec4($position, 0.0, 1.0)));
    v_texcoord = $texcoord;
}

