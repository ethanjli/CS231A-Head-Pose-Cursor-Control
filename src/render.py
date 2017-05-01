#!/usr/bin/env python2
"""Classes for rendering an object."""
import os
import sys
import numpy as np
import vispy.gloo
import vispy.app
import vispy.util.transforms
import vispy.visuals

from ext import transform

SHADERS_FOLDER = 'shaders'
VERTEX_SHADER_FILENAME = 'canvas.vert'
FRAGMENT_SHADER_FILENAME = 'canvas.frag'

_PACKAGE_PATH = os.path.dirname(sys.modules[__name__].__file__)

with open(os.path.join(_PACKAGE_PATH, SHADERS_FOLDER, VERTEX_SHADER_FILENAME), 'r') as f:
    VERTEX_SHADER = f.read()
with open(os.path.join(_PACKAGE_PATH, SHADERS_FOLDER, FRAGMENT_SHADER_FILENAME), 'r') as f:
    FRAGMENT_SHADER = f.read()

def normalize(vector):
    return vector / np.linalg.norm(vector)

def checkerboard(grid_num=8, grid_size=32):
    row_even = int(grid_num / 2) * [0, 1]
    row_odd = int(grid_num / 2) * [1, 0]
    Z = np.row_stack(int(grid_num / 2) * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)

class Renderer(vispy.app.Canvas):
    """Manages rendering of a canvas."""
    def __init__(self):
        vispy.app.Canvas.__init__(self, keys='interactive', size=(512, 512))
        # Rendering
        self.program = vispy.gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER, count=4)
        self.program['position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.program['texture'] = checkerboard()

        self._configure_camera()
        self._configure_renderer()

        vispy.gloo.set_viewport(0, 0, 512, 512)

        self.show()

    def _configure_camera(self):
        self.xyzrpy = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        self.view = np.eye(4, dtype=np.float32)
        self.program['u_view'] = self.view
        self.projection = np.eye(4, dtype=np.float32)
        self._apply_zoom()

    def _configure_renderer(self):
        vispy.gloo.set_state('translucent', clear_color='white')

    def update_view(self):
        self.view = np.linalg.inv(transform.build_se3_transform(self.xyzrpy))
        self.view[3,0] = self.view[0,3]
        self.view[0,3] = 0
        self.view[3,1] = self.view[1,3]
        self.view[1,3] = 0
        self.view[3,2] = self.view[2,3]
        self.view[2,3] = 0
        self.program['u_view'] = self.view
        self.update()

    def on_draw(self, event):
        vispy.gloo.clear()
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        self._apply_zoom()

    def _apply_zoom(self):
        vispy.gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = vispy.util.transforms.perspective(
            90.0, self.size[0] / float(self.size[1]), 0.001, 100000.0)
        self.program['u_projection'] = self.projection

    def startRendering(self, timer_observer=None, interval=10):
        if timer_observer is not None:
            self._observer = timer_observer
            self.on_timer = timer_observer.execute
            self.timer = vispy.app.Timer('auto', connect=self.on_timer, start=True)
        self.show()
        vispy.app.run()

