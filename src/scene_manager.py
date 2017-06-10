import numpy as np

from visuals import canvas
from visuals import point_cloud
from visuals import text

from utilities import util
from utilities import profiling
import render

VISUAL_NAMES = ['axes', 'checkerboard', 'texture', 'face']

VIEW_PRESETS = {
    '1': {
        'camera': {   # Camera parameters
            'fov': 90,
            'elevation': 0,
            'azimuth': 90,
            'distance': 1080,
            'center': (0, 0, 80.0)
        },
        'visuals': {   # Visual visibility
            'axes': False
        }
    }
}

class SceneManager():
    """Manages customizable aspects of the visuals scene."""
    def __init__(self, presets):
        self._pipeline = None
        self._presets = presets

        self.axes = None
        self.checkerboard = None
        self.texture = None
        self.face_point_cloud = None

        self._visibilities = {visual_name: False
                              for visual_name in VISUAL_NAMES}

    def register_rendering_pipeline(self, pipeline, register_updater=True):
        self._pipeline = pipeline
        pipeline.register_key_press_observer(self)
        if register_updater:
            pipeline.register_timer_observer(self)

        self.framerate_counter = profiling.FramerateCounter()
        framerate_counter = text.FramerateCounter(
            self._pipeline, self.framerate_counter, 'render', 'redraws/sec')
        self._pipeline.add_text(framerate_counter)

        self.axes = pipeline.visual_nodes['axes']
        self._visibilities['axes'] = True

        self._camera = self._pipeline.camera

    # Visual Instantiation

    def add_checkerboard(self):
        self.checkerboard = self._pipeline.instantiate_visual(
            canvas.CheckerboardVisual, 'checkerboard')
        self._visibilities['checkerboard'] = True

    def add_texture(self):
        self.texture = self._pipeline.instantiate_visual(
            canvas.TextureVisual, 'texture')
        self._visibilities['texture'] = True

    def add_face(self):
        self.face_point_cloud = self._pipeline.instantiate_visual(
            point_cloud.FaceVisual, 'face')
        framerate_counter = text.FramerateCounter(
            self._pipeline, self.face_point_cloud.framerate_counter,
            'data', 'face updates/sec')
        self._pipeline.add_text(framerate_counter)
        self._visibilities['face'] = True

    # Visual Initialization

    def init_face(self, face):
        self.face_point_cloud.initialize_data(
            face.get_array_shape()[0])

    # Event Handlers

    def on_key_press(self, event):
        if event.text == 'c':
            self.toggle_visibility('checkerboard')
        if event.text == 't':
            self.toggle_visibility('texture')
        elif event.text == 'f':
            self.toggle_visibility('face')
        elif event.text == 'a':
            self.toggle_visibility('axes')
        elif event.text in self._presets:
            self.apply_preset(self._presets[event.text])

    def toggle_visibility(self, visual_name, new_visibility=None):
        try:
            if new_visibility is None:
                new_visibility = not self._visibilities[visual_name]
            self._pipeline.set_visibility(visual_name, new_visibility)
            self._visibilities[visual_name] = new_visibility
        except KeyError:
            print('SceneManager Warning: No "' + visual_name +
                  '" visual in the RenderingPipeline!')

    def apply_preset(self, preset):
        self._camera.reset_all(preset['camera'])
        for (visual_node, display) in preset['visuals'].items():
            self._pipeline.set_visibility(visual_node, display)

    def execute(self, event):
        updated = False

        # Update visuals
        if (self.axes is not None and self.axes.updated_state):
            self.axes.redraw()
            updated = True
        if self.checkerboard is not None and self.checkerboard.updated_state:
            self.checkerboard.redraw()
            updated = True
        if self.texture is not None and self.texture.updated_state:
            self.texture.redraw()
            updated = True
        if (self.face_point_cloud is not None and
                self.face_point_cloud.updated_state):
            self.face_point_cloud.redraw()
            updated = True

        # Update camera
        updated = self._camera.update_all() or updated

        if updated:
            self._pipeline.update()
            self.framerate_counter.tick()
