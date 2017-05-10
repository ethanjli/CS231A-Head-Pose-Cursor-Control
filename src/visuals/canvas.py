import numpy as np
import vispy.gloo

import visuals

VERTEX_SHADER_FILENAME = 'canvas.vert'
FRAGMENT_SHADER_FILENAME = 'canvas.frag'

def checkerboard(grid_num=8, grid_size=32):
    row_even = int(grid_num / 2) * [0, 1]
    row_odd = int(grid_num / 2) * [1, 0]
    Z = np.row_stack(int(grid_num / 2) * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)

class CanvasVisual(visuals.CustomVisual):
    def __init__(self):
        super(CanvasVisual, self).__init__()
        self.program = visuals.load_shader_program(VERTEX_SHADER_FILENAME,
                                                   FRAGMENT_SHADER_FILENAME)
        self.program.vert['position'] = vispy.gloo.VertexBuffer([(-1, -1), (-1, +1),
                                                                 (+1, -1), (+1, +1)])
        self.program.vert['texcoord'] = vispy.gloo.VertexBuffer([(0, 0), (1, 0),
                                                                 (0, 1), (1, 1)])

    def draw(self, transforms):
        visual_to_doc = transforms.visual_to_document
        self.program.vert['visual_to_doc'] = visual_to_doc
        doc_to_render = transforms.framebuffer_to_render * transforms.document_to_framebuffer
        self.program.vert['doc_to_render'] = doc_to_render

        # Finally, draw the triangles.
        self.program.draw('triangle_strip')

    def set_param(self, key, value):
        self.program[key] = value

    def set_vert_param(self, key, value):
        self.program.vert[key] = value

    def set_frag_param(self, key, value):
        self.program.frag[key] = value

class CheckerboardVisual(CanvasVisual):
    def __init__(self):
        super(CheckerboardVisual, self).__init__()
        self.program['texture'] = checkerboard()

class TextureVisual(CanvasVisual):
    def __init__(self, texture_name='text.png'):
        super(TextureVisual, self).__init__()
        self.program['texture'] = visuals.load_texture(texture_name)

