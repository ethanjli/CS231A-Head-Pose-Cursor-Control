#!/usr/bin/env python2
"""Classes for visualizing reconstructions."""
import numpy as np
import vispy.app
import vispy.scene
import vispy.visuals

from utilities import computation_chains
import visuals
import visuals.axes

_PIXEL_KEYS = {}

class CameraParameter(computation_chains.Parameter):
    def __init__(self, parameter_name):
        self.camera = None
        self.parameter_name = parameter_name
        self._last_updated_value = None
        super(CameraParameter, self).__init__()

    def register_camera(self, camera):
        self.camera = camera

    def get(self):
        return getattr(self.camera, self.parameter_name)

    def set(self, value):
        return setattr(self.camera, self.parameter_name, value)

    def update(self):
        if self._last_updated_value is None:
            self._last_updated_value = self.get()
        if self.source is None:
            return
        delta = np.array(self.get()) - np.array(self._last_updated_value)
        self._last_updated_value = np.array(self.source.get())
        self.set(delta + self._last_updated_value)
        self.updated()

    def reset(self):
        self._last_updated_value = None

class CameraParametersManager():
    def __init__(self, camera, camera_parameters):
        self._camera = camera
        parameters = {
            'fov': [
                ('base', computation_chains.ParameterOffset(
                    camera_parameters['fov'])),
                ('end', CameraParameter('fov'))
            ],
            'elevation': [
                ('base', computation_chains.ParameterOffset(
                    camera_parameters['elevation'])),
                ('head', computation_chains.ParameterOffset(0)),
                ('end', CameraParameter('elevation'))
            ],
            'azimuth': [
                ('base', computation_chains.ParameterOffset(
                    camera_parameters['azimuth'])),
                ('head', computation_chains.ParameterOffset(0)),
                ('end', CameraParameter('azimuth'))
            ],
            'distance': [
                ('base', computation_chains.ParameterOffset(
                    camera_parameters['distance'])),
                ('end', CameraParameter('distance'))
            ],
            'center': [
                ('base', computation_chains.ParameterOffset(
                    np.array(camera_parameters['center']))),
                ('end', CameraParameter('center'))
            ]
        }
        for chain in parameters.values():
            computation_chains.chain(*zip(*chain)[1])
        self.parameters = {name: dict(chain)
                           for (name, chain) in parameters.items()}

    def make_camera(self):
        initial_values = {name: chain['end'].source.get()
                          for (name, chain) in self.parameters.items()}
        initial_values['up'] = 'y'
        camera = vispy.scene.cameras.TurntableCamera(**initial_values)

        for chain in self.parameters.values():
            chain['end'].register_camera(camera)

        return camera

    def reset_all(self, camera_parameters):
        for (name, value) in camera_parameters.items():
            self.parameters[name]['base'].offset = value
            self.parameters[name]['end'].reset()
            self.parameters[name]['end'].update()

    def update_all(self):
        updated = False
        for chain in self.parameters.values():
            if chain['end'].needs_update():
                chain['end'].update()
                updated = True
        return updated

class RenderingPipeline(vispy.scene.SceneCanvas):
    """Manages rendering of 3-D point cloud data."""
    def __init__(self, camera_parameters):
        super(RenderingPipeline, self).__init__(keys='interactive', size=(1920, 1080), fullscreen=True, bgcolor='white')

        self.visual_nodes = {}
        self._key_press_observers = []
        self.timer_observers = []
        self._timers = []
        self._window_scale = 1.0
        self.transformSystem = vispy.visuals.transforms.TransformSystem(self)

        self._view = self.central_widget.add_view()
        self.camera = CameraParametersManager(self._view.camera, camera_parameters)
        self._view.camera = self.camera.make_camera()
        self._add_axes()

        # Custom text visuals
        self._texts = []
        self._available_text_position = np.array([5, 5])

    # Initialization
    def _add_axes(self):
        self.axes = self.instantiate_visual(visuals.axes.AxesVisual, 'axes',
                                            create_visual_node=False)

    # Rendering
    def start_rendering(self):
        for timer in self._timers:
            timer.start()
        self.update()
        self.show()
        vispy.app.run()

    def instantiate_visual(self, Visual, name, create_visual_node=True):
        if create_visual_node:
            VisualNode = visuals.visuals.create_visual_node(Visual)
        else:
            VisualNode = Visual
        visual_node = VisualNode(parent=self.get_scene())
        self.visual_nodes[name] = visual_node
        visual_node.update_transform(visual_node.apply_final_transform(
            visual_node.base_transform()))
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
        return (self._window_scale * 90.0 /
                self.camera.parameters['fov']['end'].get() *
                self.camera.parameters['distance']['base'].offset /
                self.camera.parameters['distance']['end'].get())

    def _update_scale(self):
        for (_, visual_node) in self.visual_nodes.items():
            visual_node.update_scale(self.get_scale())

    # Event loops
    def register_timer_observer(self, timer_observer):
        self.timer_observers.append(timer_observer)
        timer = vispy.app.Timer('auto', connect=timer_observer.execute)
        self._timers.append(timer)

    # Event Handlers
    def on_resize(self, event):
        self._window_scale = self._window_scale * self.size[1] / self._central_widget.size[1]
        self._update_scale()
        super(RenderingPipeline, self).on_resize(event)

    def on_mouse_wheel(self, event):
        self._update_scale()

    def on_draw(self, event):
        for text in self._texts:
            text.update()
        super(RenderingPipeline, self).on_draw(event)

    def register_key_press_observer(self, key_press_observer):
        self._key_press_observers.append(key_press_observer)

    def on_key_press(self, event):
        for observer in self._key_press_observers:
            observer.on_key_press(event)
        if event.text == ' ':
            for timer in self._timers:
                if timer.running:
                    timer.stop()
                else:
                    timer.start()
        elif event.text in _PIXEL_KEYS:
            self._on_key_press_pixel_size(event)

    def set_visibility(self, visual_node, visibility):
        if visibility:
            self.visual_nodes[visual_node].add_parent(self.get_scene())
        else:
            try:
                self.visual_nodes[visual_node].remove_parent(self.get_scene())
            except ValueError:
                pass  # visual node is already not in the scene

    def _on_key_press_pixel_size(event):
        # TODO: implement this!
        pass
