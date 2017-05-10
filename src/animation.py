import threading

import head_pose
import util
import visuals.text

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

class HeadPoseAnimator():
    """Asynchronously updates a rendering pipeline with head pose tracking."""
    def __init__(self, yaw_multiplier=1, pitch_multiplier=-1, roll_multiplier=-1):
        self._pipeline = None
        self._visual_node = None
        self._head_pose = head_pose.HeadPose()
        self.yaw_multiplier = yaw_multiplier
        self.pitch_multiplier = pitch_multiplier
        self.roll_multiplier = roll_multiplier
        self.framerate_counter = util.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        """Starts updating a RenderingPipeline.

        Threading:
            Instantiates a head pose tracking thread.
        """
        self._head_pose.monitor_async(self._update_canvas)
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def clean_up(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._head_pose.stop_monitoring()

    def _update_canvas(self, yaw, pitch, roll, x, y, z):
        transform = self._visual_node.base_transform()
        transform.rotate(roll * self.roll_multiplier, (0, 0, 1))
        self._visual_node.transform = transform
        self.framerate_counter.tick()
