#!/usr/bin/env python2
"""Classes for visualizing reconstructions."""
import os
import sys
import threading
import numpy as np
import vispy.app
import vispy.scene
import vispy.visuals

import visuals

_PACKAGE_PATH = os.path.dirname(sys.modules[__name__].__file__)

_VIEW_PRESETS = {  # ((FOV, Elevation, Azimuth, Distance, Center), VisualNodeDisplayOverrides)
    '1': ((30, 6, 90, 30, (0, 1.8, 2.3)),
          {'car': True}),
    '2': ((30, -10, 90, 1.5, (1, 1.2, -2.3)),
          {'car': False})
}
_PIXEL_KEYS = {}

class AdjustableCamera(vispy.scene.cameras.TurntableCamera):
    """Adjustable turntable camera which can be further adjusted by the mouse.
    Mouse adjustments persist except when reset.
    """
    def __init__(self):
        self.__fov = 30
        self.elevation_offset = 6
        self.__elevation = self.elevation_offset
        self.azimuth_offset = 90
        self.__azimuth = self.azimuth_offset
        self.distance_offset = 30
        self.__distance = self.distance_offset
        self.center_offset = np.array([0, 1.8, 2.3])
        self.__center = self.center_offset
        self.__lock = threading.RLock()
        super(AdjustableCamera, self).__init__(
            fov=self.__fov, elevation=self.__elevation, azimuth=self.__azimuth,
            distance=self.__distance, up='y', center=self.__center)

    def update_center(self, new_center):
        """Adjust camera center while maintaining mouse adjustments."""
        with self.__lock:
            delta = self._view_camera.center - self._center
            self.__center = new_center + self.center_offset
            self.center = delta + self.__center

    def update_elevation(self, new_elevation):
        """Adjust camera elevation while maintaining mouse adjustments."""
        with self.__lock:
            delta = self.elevation - self.__elevation
            self.__elevation = new_elevation + self.elevation_offset
            self.elevation = delta + self.__elevation

    def reset_elevation(self):
        """Reset mouse adjustments to camera elevation."""
        with self.__lock:
            self.elevation = self.__elevation

    def update_azimuth(self, new_azimuth):
        """Adjust camera azimuth while maintaining mouse adjustments."""
        with self.__lock:
            delta = self.azimuth - self.__azimuth
            self.__azimuth = new_azimuth + self.azimuth_offset
            self.azimuth = delta + self.__azimuth

    def reset_azimuth(self):
        """Reset mouse adjustments to camera azimuth."""
        with self.__lock:
            self.azimuth = self.__azimuth

    def reset_all(self, new_values=None):
        with self.__lock:
            if new_values is not None:
                (self.__fov, self.__elevation, self.__azimuth,
                 self.__distance, self.__center) = new_values
            self.fov = self.__fov
            self.elevation = self.__elevation
            self.azimuth = self.__azimuth
            self.distance = self.__distance
            self.center = self.__center

    # Multi-threading support
    def view_changed(self):
        with self.__lock:
            super(AdjustableCamera, self).view_changed()

    def get_lock(self):
        return self.__lock

class RenderingPipeline(vispy.scene.SceneCanvas):
    """Manages rendering of 3-D point cloud data."""
    def __init__(self):
        self.camera = AdjustableCamera()
        super(RenderingPipeline, self).__init__(keys='interactive', size=(800, 600), bgcolor='white')

        self.visual_nodes = {}
        self._timers = []
        self._window_scale = 1.0
        self.transformSystem = vispy.visuals.transforms.TransformSystem(self)

        self._view = self.central_widget.add_view()
        self._view.camera = self.camera
        self._add_axes()

        # Custom text visuals
        self._texts = []
        self._available_text_position = np.array([5, 5])

    # Initialization
    def _add_axes(self):
        self._axes = vispy.scene.visuals.XYZAxis(parent=self.get_scene())
        self._axes.transform = vispy.visuals.transforms.AffineTransform()
        self._axes.transform.scale((5, 5, 5))

    # Rendering
    def start_rendering(self):
        for timer in self._timers:
            timer.start()
        self.update()
        self.show()
        vispy.app.run()

    def instantiate_visual(self, Visual, name):
        VisualNode = visuals.visuals.create_visual_node(Visual)
        visual_node = VisualNode(parent=self.get_scene())
        self.visual_nodes[name] = visual_node
        visual_node.transform = visual_node.base_transform()
        visual_node.update_scale(self.pixel_scale)
        return visual_node

    def add_text(self, text):
        self._texts.append(text)
        text.set_position(self._available_text_position)
        self._available_text_position[1] += 5 + text.get_size()


    # Utility

    def get_scene(self):
        return self._view.scene

    def get_scale(self):
        return (self._window_scale *
                30.0 / self.camera.fov *
                self.camera.distance_offset / self.camera.distance)

    def _update_scale(self):
        for (_, visual_node) in self.visual_nodes.items():
            visual_node.update_scale(self.get_scale())

    # Event loops
    def register_timer_observer(self, timer_observer):
        self._timer_observers.append(timer_observer)
        timer = vispy.app.Timer('auto', connect=timer_observer.execute)
        self._timers.append(timer)

    # Event Handlers
    def on_resize(self, event):
        with self.camera.get_lock():
            self._window_scale = self._window_scale * self.size[1] / self._central_widget.size[1]
            self._update_scale()
            super(RenderingPipeline, self).on_resize(event)

    def on_mouse_wheel(self, event):
        self._update_scale()

    def on_draw(self, event):
        for text in self._texts:
            text.update()
        with self.camera.get_lock():
            super(RenderingPipeline, self).on_draw(event)

    def on_key_press(self, event):
        if event.text == ' ':
            for timer in self._timers:
                if timer.running:
                    timer.stop()
                else:
                    timer.start()
                # TODO: also pause reconstruction
        elif event.text in _VIEW_PRESETS:
            self._on_key_press_view_preset(event)
        elif event.text in _PIXEL_KEYS:
            self._on_key_press_pixel_size(event)

    def _on_key_press_view_preset(self, event):
        (self.camera.fov_offset, self.camera.elevation_offset, self.camera.azimuth_offset,
         self.camera.distance_offset, self.camera.center_offset) = _VIEW_PRESETS[event.text][0]
        self.camera.reset_all(_VIEW_PRESETS[event.text][0])
        for (visual_node, display) in _VIEW_PRESETS[event.text][1].items():
            if display:
                self.visual_nodes[visual_node].add_parent(self.get_scene())
            else:
                try:
                    self.visual_nodes[visual_node].remove_parent(self.get_scene())
                except ValueError:
                    pass  # visual node is already not in the scene

    def _on_key_press_pixel_size(event):
        # TODO: implement this!
        pass
