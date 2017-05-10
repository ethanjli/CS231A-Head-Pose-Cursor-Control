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
    def __init__(self, yaw_multiplier=1, pitch_multiplier=-1):
        self._pipeline = None
        self._head_pose = head_pose.HeadPose()
        self.yaw_multiplier = yaw_multiplier
        self.pitch_multiplier = pitch_multiplier
        self.framerate_counter = util.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        """Starts updating a RenderingPipeline.

        Threading:
            Instantiates a head pose tracking thread.
        """
        self._camera = pipeline.camera
        self._head_pose.monitor_async(self._update_camera)
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def clean_up(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._head_pose.stop_monitoring()

    def _update_camera(self, yaw, pitch):
        self._camera.update_azimuth(self._head_pose.yaw * self.yaw_multiplier)
        self._camera.update_elevation(self._head_pose.pitch * self.pitch_multiplier)
        self.framerate_counter.tick()
