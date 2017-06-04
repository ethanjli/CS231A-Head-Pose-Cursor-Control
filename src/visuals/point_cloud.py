import os
import sys

import numpy as np
import vispy.visuals
import vispy.gloo

from utilities import util
from utilities import profiling
import visuals

VERTEX_SHADER_FILENAME = 'point_cloud.vert'
FRAGMENT_SHADER_FILENAME = 'point_cloud.frag'

class PointCloudVisual(visuals.CustomVisual):
    def __init__(self):
        super(PointCloudVisual, self).__init__()

        self.program = visuals.load_shader_program(VERTEX_SHADER_FILENAME,
                                                   FRAGMENT_SHADER_FILENAME)
        self._initialize_rendering()
        self.framerate_counter = profiling.FramerateCounter()

    def initialize_data(self, num_points):
        self.data_size = num_points
        self.data = np.zeros(num_points, [('a_position', np.float32, 3),
                                          ('a_color', np.float32, 3)])
        self.data_vbo = vispy.gloo.VertexBuffer(self.data)
        self.program.bind(self.data_vbo)
        self.updated_state = False

    def update_grid_data(self, points, rgb):
        """Updates the point cloud data, given by one point per grid cell, and re-renders it."""
        self.update_list_data(util.grid_data_to_list_data(points),
                              util.grid_data_to_list_data(rgb))

    def update_list_data(self, points, rgb):
        """Updates the point cloud data, given by one point per row, and re-renders it.
        Data are all assumed to be of the same number of points.
        """
        num_points = points.shape[0]
        prev_num_points = self.data_size
        if num_points > self.data.shape[0]:
            print('Warning: Can only render ' + str(self.data.shape[0]) + ' of the ' +
                  str(num_points) + ' requested points!')
        # Set data
        self.data['a_position'][:num_points] = points
        self.data['a_color'][:num_points] = rgb
        # Clear out extraneous points from the previous frame
        if prev_num_points > num_points:
            self.data['a_position'][num_points:self.data_size] = \
                np.zeros((prev_num_points - num_points, 3))
            self.data['a_color'][num_points:self.data_size] = \
                np.zeros((prev_num_points - num_points, 3))
        self.data_size = num_points
        self.updated_state = True
        self.data_vbo.set_data(self.data)
        self.framerate_counter.tick()

    def redraw(self):
        if self.updated_state:
            self.program.bind(self.data_vbo)
        super(PointCloudVisual, self).redraw()

    def _initialize_rendering(self):
        u_linewidth = 1.0
        u_antialias = 1.0
        self.set_param('u_linewidth', u_linewidth)
        self.set_param('u_antialias', u_antialias)

    def draw(self, transforms):
        # Note we use the "additive" GL blending settings so that we do not
        # have to sort the mesh triangles back-to-front before each draw.
        # gloo.set_state('additive', cull_face=False)

        visual_to_doc = transforms.visual_to_document
        self.program.vert['visual_to_doc'] = visual_to_doc
        doc_to_render = transforms.framebuffer_to_render * transforms.document_to_framebuffer
        self.program.vert['doc_to_render'] = doc_to_render

        # Finally, draw the triangles.
        self.program.draw('points')

    def set_param(self, key, value):
        self.program[key] = value

    def update_scale(self, pixel_scale):
        self.set_param('u_size', 2 * pixel_scale)

class StereoReconstructionVisual(PointCloudVisual):
    def __init__(self):
        super(StereoReconstructionVisual, self).__init__()

    @staticmethod
    def base_transform():
        transform = vispy.visuals.transforms.AffineTransform()
        transform.scale((-1, 1, 1))
        transform.translate((0, 0.9, 0))
        transform.rotate(1.2, (1, 0, 0))
        return transform

class LIDARPointCloudVisual(PointCloudVisual):
    def __init__(self):
        super(LIDARPointCloudVisual, self).__init__()

    @staticmethod
    def base_transform():
        transform = vispy.visuals.transforms.AffineTransform()
        transform.rotate(90, [1,0,0])
        transform.rotate(90, [0,1,0])
        transform.translate((0, 1.0, -2.5))
        transform.rotate(-3, [1,0,0])
        return transform
