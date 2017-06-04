import threading

import numpy as np
import transforms3d

from utilities import profiling
from utilities import signal_processing
import facial_landmarks
import head_pose
import stereo_cameras
import transform_util
import stereo_util
import visuals.text

_HEAD_POSE_POSTPROCESSORS = {
    'yaw': lambda value: -1 * value,
    'pitch': lambda value: -1 * value,
    'roll': lambda value: -1 * value,
    'x': lambda value: 63.5 * value,  # cm that the camera is to the left of the center of the head
    'y': lambda value: -93 * value,  # cm, just a scaling factor due to the calibration technique
    'z': lambda value: 58.28 * value - 1.7  # cm between the camera and the head
}

_CALIBRATION_FILTERS = {parameter: signal_processing.SlidingWindowFilter(
                            40, estimation_mode=('kernel', signal_processing.half_gaussian_window(40, 20.0)))
                        for parameter in head_pose.PARAMETERS}

STEREO_CALIBRATION = stereo_util.StereoModelCalibration(
    -stereo_cameras.TRANSLATION[0], stereo_cameras.K_LEFT, stereo_cameras.K_RIGHT,
    np.vstack([facial_landmarks.MODEL[parameter]
               for parameter in facial_landmarks.PARAMETERS]))

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

    def execute(self):
        pass

class CalibratedAnimator(object):
    """Mix-in for asynchronous animators which have a calibration phase before a response phase.
    This needs to be the leftmost base class of any subclass.

    Objects with this mix-in need to expose on_start_calibrating().
    """
    def __init__(self, *args, **kwargs):
        super(CalibratedAnimator, self).__init__(*args, **kwargs)
        self.state = 'ready'
        self.instructions = None

    def register_rendering_pipeline(self, pipeline):
        """Starts updating a RenderingPipeline."""
        self.instructions = visuals.text.Text(
            pipeline,initial_text='Press the space bar to start calibrating.', font_size=12)
        pipeline.add_text(self.instructions)
        pipeline.register_key_press_observer(self)

    def on_key_press(self, event):
        if self.state == 'ready':
            if event.text == ' ':
                self.start_calibrating()
        elif self.state == 'calibrating':
            if event.text == ' ':
                self.start_responding()

    def start_calibrating(self):
        self.state = 'calibrating'
        self.instructions.visual.text = 'When you\'ve achieved a stable position, press the space bar to start stabilizing the screen.'
        self.instructions.update()
        self.on_start_calibrating()

    def start_responding(self):
        self.state = 'stabilizing'
        self.instructions.visual.text = ''
        self.instructions.update()
        self.on_start_responding()

# MONOCULAR ANIMATION

class HeadPoseAnimator(object):
    """Asynchronously updates a rendering pipeline with head pose tracking."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS, filters=head_pose.DEFAULT_FILTERS):
        self._pipeline = None
        self._visual_node = None
        self._head_pose = head_pose.HeadPose(filters)
        self.head_pose_postprocessors = head_pose_postprocessors
        self.framerate_counter = profiling.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        self._pipeline = pipeline
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def animate_async(self, callback=None):
        if callback is not None:
            self._head_pose.monitor_async(callback)
        else:
            self._head_pose.monitor_async(self.on_update)

    def stop_animating(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._head_pose.stop_monitoring()

    def on_update(self, parameters):
        pass

    def postprocess(self, parameters):
        return {parameter: self.head_pose_postprocessors[parameter](value)
                for (parameter, value) in parameters.items()}

class HeadRollAnimator(HeadPoseAnimator):
    def __init__(self, *args, **kwargs):
        super(HeadRollAnimator, self).__init__(*args, **kwargs)

    def on_update(self, parameters):
        postprocessed = self.postprocess(parameters)
        transform = self._visual_node.base_transform()
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        self._visual_node.update_transform(transform)
        self.framerate_counter.tick()

class HeadVisualAnimator(HeadPoseAnimator):
    def __init__(self, *args, **kwargs):
        super(HeadVisualAnimator, self).__init__(*args, **kwargs)

    def on_update(self, parameters):
        postprocessed = self.postprocess(parameters)
        transform = self._visual_node.base_transform()
        transform.rotate(postprocessed['yaw'], (0, 1, 0))
        transform.rotate(postprocessed['pitch'], (1, 0, 0))
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        transform.translate((postprocessed['x'],
                             postprocessed['y'], 0))
        self._visual_node.update_transform(transform)
        self.framerate_counter.tick()

# CALIBRATED MONOCULAR ANIMATION

class ScreenStabilizer(CalibratedAnimator):
    """Asynchronously guides the user through screen stabilization calibration."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS):
        super(ScreenStabilizer, self).__init__()
        self.calibration = None
        self._calibration = None
        self._head_pose = None
        self.framerate_counter = None
        self._head_visual_node = None

    def register_rendering_pipeline(self, pipeline):
        super(ScreenStabilizer, self).register_rendering_pipeline(pipeline)
        self._pipeline = pipeline

    def register_head_visual_node(self, visual_node):
        self._head_visual_node = visual_node

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def on_start_calibrating(self):
        self._head_pose = HeadVisualAnimator(_HEAD_POSE_POSTPROCESSORS, _CALIBRATION_FILTERS)
        self.framerate_counter = self._head_pose.framerate_counter
        self._head_pose.register_rendering_pipeline(self._pipeline)
        self._head_pose.register_visual_node(self._head_visual_node)
        self._head_pose.animate_async(self._update_calibration)

    def on_start_responding(self):
        self._head_pose.stop_animating()
        self._head_pose = HeadPoseAnimator(_HEAD_POSE_POSTPROCESSORS)
        self.framerate_counter = self._head_pose.framerate_counter
        self._head_pose.animate_async(self._update_canvas)
        self.calibration = transform_util.Calibration(**self.calibration)

    def stop_animating(self):
        self._head_pose.stop_animating()

    def _update_calibration(self, parameters):
        postprocessed = self._head_pose.postprocess(parameters)
        #print('Postprocessed: {}'.format({parameter: round(value, 2)
        #                                  for (parameter, value) in postprocessed.items()}))
        self.calibration = {parameter: value for (parameter, value) in postprocessed.items()}
        transform = self._head_visual_node.base_transform()
        transform.rotate(postprocessed['yaw'], (0, 1, 0))
        transform.rotate(postprocessed['pitch'], (1, 0, 0))
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        transform.translate((postprocessed['x'], postprocessed['y'], 0.01))
        self._head_visual_node.transform = transform
        self.framerate_counter.tick()
        self._pipeline.update()

    def _update_canvas(self, parameters):
        postprocessed = self._head_pose.postprocess(parameters)
        #print('Postprocessed: {}'.format({parameter: round(value, 2)
        #                                  for (parameter, value) in postprocessed.items()}))

        base_vertices = self._visual_node.get_base_vertices()
        screen_base_vertices = [transform_util.render_xy_to_screen_xy(*vertex)
                                for vertex in base_vertices]
        screen_transformed_vertices = [self.calibration.transform(
                                           vertex[0], vertex[1], **postprocessed)
                                       for vertex in screen_base_vertices]
        render_transformed_vertices = [transform_util.screen_xy_to_render_xy(*vertex)
                                       for vertex in screen_transformed_vertices]
        self._visual_node.update_vertices(render_transformed_vertices)
        self.framerate_counter.tick()
        self._pipeline.update()

# STEREO ANIMATION

class FacialLandmarkAnimator(AsynchronousAnimator):
    """Asynchronously updates a rendering pipeline with facial landmarks."""
    def __init__(self):
        super(FacialLandmarkAnimator, self).__init__('FacialLandmarkAnimator')
        self._pipeline = None
        self._visual_node = None

        self._tracker_left = facial_landmarks.FacialLandmarks(camera_index=0)
        self._tracker_right = facial_landmarks.FacialLandmarks(camera_index=1)
        self._left_landmarks = None
        self._left_landmarks_updated = False
        self._right_landmarks = None
        self._right_landmarks_updated = False
        self.framerate_counter = profiling.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        self._pipeline = pipeline
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def animate_sync(self):
        self._tracker_left.monitor_async(self._update_left_landmarks)
        self._tracker_right.monitor_async(self._update_right_landmarks)
        super(FacialLandmarkAnimator, self).animate_sync()

    def _update_left_landmarks(self, parameters):
        self._left_landmarks = parameters
        self._left_landmarks_updated = True

    def _update_right_landmarks(self, parameters):
        self._right_landmarks = parameters
        self._right_landmarks_updated = True

    def stop_animating(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._tracker_left.stop_monitoring()
        self._tracker_right.stop_monitoring()
        super(FacialLandmarkAnimator, self).stop_animating()

    def execute(self):
        if self._left_landmarks_updated and self._right_landmarks_updated:
            keypoints = np.array([[self._left_landmarks[parameter], self._right_landmarks[parameter]]
                                for parameter in facial_landmarks.PARAMETERS])
            self.on_update(keypoints)
            self._left_landmarks_updated = False
            self._right_landmarks_updated = False

    def on_update(self, keypoints):
        pass

class FaceAxesAnimator(FacialLandmarkAnimator):
    def __init__(self):
        super(FaceAxesAnimator, self).__init__()

    def register_rendering_pipeline(self, pipeline):
        super(FaceAxesAnimator, self).register_rendering_pipeline(pipeline)
        self.register_visual_node(pipeline.axes)

    def on_update(self, keypoints):
        (rotation, translation) = STEREO_CALIBRATION.compute_RT(keypoints)
        transform = self._visual_node.base_transform()
        transform.translate(-translation)
        try:
            (axis, angle) = transforms3d.axangles.mat2axangle(rotation)
            transform.rotate(np.rad2deg(angle), axis)
        except ValueError:
            pass
        self._visual_node.transform = transform
        self.framerate_counter.tick()
        self._pipeline.update()
