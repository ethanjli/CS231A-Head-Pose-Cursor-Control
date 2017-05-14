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

_CALIBRATION_FILTERS = {signal_processing.SlidingWindowFilter(
                            10, estimation_mode=('kernel', signal_processing.half_gaussian_window(10, 4.0)))
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
