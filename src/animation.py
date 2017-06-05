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

_HEAD_POSE_CALIBRATION_FILTERS = {parameter: signal_processing.SlidingWindowFilter(
                                      40, estimation_mode=('kernel', signal_processing.half_gaussian_window(40, 20.0)))
                                  for parameter in head_pose.PARAMETERS}

class AsynchronousAnimator(object):
    """Abstract class for animators which run asynchronously in a thread."""
    def __init__(self, animator_name='AsynchronousAnimator'):
        self._animator_thread = None
        self._animate = True
        self.animator_name = animator_name

    def animate_sync(self, callback=None):
        while self._animate:
            self.execute(callback)

    def animate_async(self, callback=None):
        if self._animator_thread is not None:
            return
        self._animator_thread = threading.Thread(target=self.animate_sync,
                                                 name=self.animator_name,
                                                 kwargs={'callback': callback})
        self._animator_thread.start()

    def stop_animating(self):
        self._animate = False
        if self._animator_thread is not None:
            self._animator_thread.join()
            self._animator_thread = None

    def execute(self, callback=None):
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
        self._head_pose = HeadVisualAnimator(_HEAD_POSE_POSTPROCESSORS, _HEAD_POSE_CALIBRATION_FILTERS)
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

def make_facial_calibration_filters():
    return [[signal_processing.SlidingWindowFilter(20, estimation_mode='mean')
             for j in range(2)] for i in range(facial_landmarks.NUM_KEYPOINTS)]

class FacialLandmarkAnimator(AsynchronousAnimator):
    """Asynchronously updates a rendering pipeline with facial landmarks."""
    def __init__(self, left_filters, right_filters):
        super(FacialLandmarkAnimator, self).__init__('FacialLandmarkAnimator')
        self._pipeline = None
        self._visual_node = None

        self._tracker_left = facial_landmarks.FacialLandmarks(
            camera_index=0, filters=left_filters)
        self._tracker_right = facial_landmarks.FacialLandmarks(
            camera_index=1, filters=right_filters)
        self._left_keypoints = None
        self._left_keypoints_updated = False
        self._right_keypoints = None
        self._right_keypoints_updated = False
        self.framerate_counter = profiling.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        self._pipeline = pipeline
        framerate_counter = visuals.text.FramerateCounter(
            pipeline, self.framerate_counter, 'headpose', 'updates/sec')
        pipeline.add_text(framerate_counter)

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def animate_sync(self, callback=None):
        self._tracker_left.monitor_async(self._update_left_keypoints)
        self._tracker_right.monitor_async(self._update_right_keypoints)
        super(FacialLandmarkAnimator, self).animate_sync(callback)

    def _update_left_keypoints(self, parameters):
        self._left_keypoints = parameters
        self._left_keypoints_updated = True

    def _update_right_keypoints(self, parameters):
        self._right_keypoints = parameters
        self._right_keypoints_updated = True

    def stop_animating(self):
        """Stops updating a RenderingPipeline.

        Threading:
            Joins a head pose tracking thread.
        """
        self._tracker_left.stop_monitoring()
        self._tracker_right.stop_monitoring()
        super(FacialLandmarkAnimator, self).stop_animating()

    def execute(self, callback=None):
        if self._left_keypoints_updated and self._right_keypoints_updated:
            keypoints = np.stack([self._left_keypoints, self._right_keypoints], axis=1)
            if callback is None:
                self.on_update(keypoints)
            else:
                callback(keypoints)
            self._left_keypoints_updated = False
            self._right_keypoints_updated = False

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

class FacePointsAnimator(FacialLandmarkAnimator):
    def __init__(self, *args, **kwargs):
        super(FacePointsAnimator, self).__init__(*args, **kwargs)
        self._camera_matrices = stereo_util.make_parallel_camera_matrices(
            stereo_cameras.K_LEFT, stereo_cameras.K_RIGHT, -stereo_cameras.TRANSLATION[0])


    def register_rendering_pipeline(self, pipeline):
        super(FacePointsAnimator, self).register_rendering_pipeline(pipeline)

    def on_update(self, keypoints):
        face = stereo_util.compute_3d_model(keypoints, self._camera_matrices)
        #print face
        self._visual_node.update_list_data(face)
        self.framerate_counter.tick()
        self._pipeline.update()

class CalibratedFaceAnimator(CalibratedAnimator):
    def __init__(self):
        super(CalibratedFaceAnimator, self).__init__()
        self.calibration = None
        self._calibration = None
        self.framerate_counter = None
        self._visual_node = None

    def register_rendering_pipeline(self, pipeline):
        super(CalibratedFaceAnimator, self).register_rendering_pipeline(pipeline)
        self._pipeline = pipeline

    def register_visual_node(self, visual_node):
        self._visual_node = visual_node

    def on_start_calibrating(self):
        self._facial_landmarks = FacePointsAnimator(
            make_facial_calibration_filters(), make_facial_calibration_filters())
        self.framerate_counter = self._facial_landmarks.framerate_counter
        self._facial_landmarks.register_rendering_pipeline(self._pipeline)
        self._facial_landmarks.register_visual_node(self._visual_node)
        self._facial_landmarks.animate_async(self._update_calibration)

    def on_start_responding(self):
        self._facial_landmarks.stop_animating()
        self.calibration = stereo_util.StereoModelCalibration(
            -stereo_cameras.TRANSLATION[0], stereo_cameras.K_LEFT, stereo_cameras.K_RIGHT,
            stereo_util.compute_3d_model(self._calibration, stereo_util.make_parallel_camera_matrices(
                stereo_cameras.K_LEFT, stereo_cameras.K_RIGHT, -stereo_cameras.TRANSLATION[0])))
        self._facial_landmarks = FacePointsAnimator(
            make_facial_calibration_filters(), make_facial_calibration_filters())
        self.framerate_counter = self._facial_landmarks.framerate_counter
        self._facial_landmarks.animate_async(self._update_head)

    def stop_animating(self):
        self._facial_landmarks.stop_animating()

    def _update_calibration(self, parameters):
        self._calibration = parameters
        self._facial_landmarks.on_update(parameters)

    def _update_head(self, parameters):
        (rotation, translation) = self.calibration.compute_RT(parameters)
        transformed = stereo_util.apply_transformation(self.calibration._model_3d, rotation, translation)
        self._visual_node.update_list_data(transformed)
        self.framerate_counter.tick()
        self._pipeline.update()
