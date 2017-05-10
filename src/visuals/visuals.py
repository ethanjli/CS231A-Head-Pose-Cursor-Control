import os
import sys
import threading

import vispy.io
import vispy.visuals
import vispy.scene

_PACKAGE_PATH = os.path.dirname(sys.modules[__name__].__file__)
SHADERS_FOLDER = 'shaders'
TEXTURES_FOLDER = 'textures'

class CustomVisual(vispy.visuals.Visual):
    def __init__(self):
        self.__lock = threading.RLock()
        super(CustomVisual, self).__init__()

    @staticmethod
    def base_transform():
        return vispy.visuals.transforms.AffineTransform()

    def update_scale(self, pixel_scale):
        pass

    def update_transform(self, new_transform):
        with self.__lock:
            self.transform = new_transform

    def get_lock(self):
        return self.__lock

def load_shader(shader_filename):
    with open(os.path.join(_PACKAGE_PATH, SHADERS_FOLDER, shader_filename), 'r') as f:
        shader = f.read()
    return shader

def load_shader_program(vertex_shader_filename, fragment_shader_filename):
    vertex_shader = load_shader(vertex_shader_filename)
    fragment_shader = load_shader(fragment_shader_filename)
    return vispy.visuals.shaders.ModularProgram(vertex_shader, fragment_shader)

def load_texture(texture_name):
    texture_filename = os.path.join(_PACKAGE_PATH, TEXTURES_FOLDER, texture_name)
    return vispy.io.imread(texture_filename)

def create_visual_node(Visual):
    return vispy.scene.visuals.create_visual_node(Visual)
