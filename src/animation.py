import threading

import vispy.visuals

import util
import signal_processing
import head_pose
import visuals.text

_HEAD_POSE_POSTPROCESSORS = {
    'yaw': lambda value: value,
    'pitch': lambda value: -1 * value,
    'roll': lambda value: -1 * value,
    'x': lambda value: value,
    'y': lambda value: value,
    'z': lambda value: value
}

_CALIBRATION_FILTERS = {parameter: signal_processing.SlidingWindowFilter(
                            40, estimation_mode=('kernel', signal_processing.half_gaussian_window(40, 20.0)))
                        for parameter in head_pose.PARAMETERS}

class AsynchronousAnimator(object):
    """Abstract class for animators which run asynchronously in a thread."""
    def __init__(self, animator_name='AsynchronousAnimator'):
        self._animator_thread = None
        self._animate = True
        self.animator_name = animator_name

    def animate_sync(self):
        while self._animate:
            self.execute()

    def animate_async(self):
        if self._animator_thread is not None:
            return
        self._animator_thread = threading.Thread(target=self.animate_sync,
                                                 name=self.animator_name)
        self._animator_thread.start()

    def stop_animating(self):
        self._animate = False
        if self._animator_thread is not None:
            self._animator_thread.join()
            self._animator_thread = None

class HeadPoseAnimator(object):
    """Asynchronously updates a rendering pipeline with head pose tracking."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS):
        self._pipeline = None
        self._visual_node = None
        self._head_pose = head_pose.HeadPose()
        self.head_pose_postprocessors = head_pose_postprocessors
        self.framerate_counter = util.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def animate_async(self):
        self._head_pose.monitor_async(self._update_canvas)

    def clean_up(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._head_pose.stop_monitoring()

    def _update_canvas(self, parameters):
        transform = self._visual_node.base_transform()
        postprocessed = {parameter: self.head_pose_postprocessors[parameter](value)
                         for (parameter, value) in parameters.items()}
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        self._visual_node.transform = transform
        self.framerate_counter.tick()

class ScreenStabilizer(HeadPoseAnimator):
    """Asynchronously guides the user through screen stabilization calibration."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS):
        super(ScreenStabilizer, self).__init__(head_pose_postprocessors)
        self.state = 'ready'
        self.calibration = None
        self._head_pose = head_pose.HeadPose(_CALIBRATION_FILTERS)
        self._head_visual_node = None

    def register_rendering_pipeline(self, pipeline):
        """Starts updating a RenderingPipeline.

        Threading:
            Instantiates a head pose tracking thread.
        """
        super(ScreenStabilizer, self).register_rendering_pipeline(pipeline)
        self.instructions = visuals.text.Text(pipeline, initial_text='Press the space bar to start calibrating.', font_size=12)
        pipeline.add_text(self.instructions)
        pipeline.register_key_press_observer(self)

    def register_head_visual_node(self, visual_node):
        self._head_visual_node = visual_node

    def clean_up(self):
        super(ScreenStabilizer, self).clean_up()

    def on_key_press(self, event):
        if self.state == 'ready':
            if event.text == ' ':
                self._start_calibrating()
        elif self.state == 'calibrating':
            if event.text == ' ':
                self._start_stabilizing()

    def _start_calibrating(self):
        self.state = 'calibrating'
        self.instructions.visual.text = 'When you\'ve achieved a stable position, press the space bar to start stabilizing the screen.'
        self.instructions.update()
        self._head_pose.monitor_async(self._stabilize)

    def _start_stabilizing(self):
        self.state = 'stabilizing'
        self._head_pose.filters = head_pose.DEFAULT_FILTERS
        self.instructions.visual.text = ''
        self.instructions.update()
        self.framerate_counter.reset()
        print('Calibration parameters: {}'.format(self.calibration))

    def _stabilize(self, parameters):
        if self.state == 'calibrating':
            self._update_head_visual_node(parameters)
        elif self.state == 'stabilizing':
            calibrated_parameters = {parameter: value - self.calibration[parameter]
                                     for (parameter, value) in parameters.items()}
            self._update_canvas(calibrated_parameters)

    def _update_head_visual_node(self, parameters):
        transform = self._head_visual_node.base_transform()
        self.calibration = {parameter: value for (parameter, value) in parameters.items()}
        postprocessed = {parameter: self.head_pose_postprocessors[parameter](value)
                        for (parameter, value) in parameters.items()}
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        self._head_visual_node.transform = transform
        self.framerate_counter.tick()
