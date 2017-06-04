import threading

import numpy as np

import facial_landmarks
import head_pose
import util
import transform_util
import stereo_util
import signal_processing
import vispy.visuals
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

INTRINSICS = np.array([[454.64682856, 0.00000, 160.0],
                       [0.00000, 454.64682856, 120.0],
                       [0.00000, 0.00000, 1.00]])
STEREO_CALIBRATION = stereo_util.StereoModelCalibration(
    23.0, INTRINSICS, INTRINSICS, np.vstack([facial_landmarks.MODEL[parameter]
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

class HeadPoseAnimator(object):
    """Asynchronously updates a rendering pipeline with head pose tracking."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS):
        self._pipeline = None
        self._visual_node = None
        self._head_pose = head_pose.HeadPose()
        self.head_pose_postprocessors = head_pose_postprocessors
        self.framerate_counter = util.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        self._pipeline = pipeline
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

class FacialLandmarkAnimator(AsynchronousAnimator):
    """Asynchronously updates a rendering pipeline with facial landmarks."""
    def __init__(self):
        super(FacialLandmarkAnimator, self).__init__('FacialLandmarkAnimator')
        self._pipeline = None
        self._visual_node = None
        self._facial_landmarks_left = facial_landmarks.FacialLandmarks(camera_index=0)
        self._facial_landmarks_right = facial_landmarks.FacialLandmarks(camera_index=1)
        self._left_landmarks = None
        self._left_landmarks_updated = False
        self._right_landmarks = None
        self._right_landmarks_updated = False
        self.framerate_counter = util.FramerateCounter()

    def register_rendering_pipeline(self, pipeline):
        pass

    def register_visual_node(self, visual_node):
         pass

    def animate_sync(self):
        self._facial_landmarks_left.monitor_async(self._update_left_landmarks)
        self._facial_landmarks_right.monitor_async(self._update_right_landmarks)
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
        self._facial_landmarks_left.stop_monitoring()
        self._facial_landmarks_right.stop_monitoring()
        super(FacialLandmarkAnimator, self).stop_animating()

    def execute(self):
        if self._left_landmarks_updated and self._right_landmarks_updated:
            self.framerate_counter.tick()
            self._triangulate()
            self._left_landmarks_updated = False
            self._right_landmarks_updated = False

    def _triangulate(self):
        keypoints = np.array([[self._left_landmarks[parameter], self._right_landmarks[parameter]]
                              for parameter in facial_landmarks.PARAMETERS])
        camera_matrices = STEREO_CALIBRATION._camera_matrices
        model = stereo_util.compute_3d_model(keypoints, camera_matrices)
        print(model[0])

class ScreenStabilizer(HeadPoseAnimator):
    """Asynchronously guides the user through screen stabilization calibration."""
    def __init__(self, head_pose_postprocessors=_HEAD_POSE_POSTPROCESSORS):
        super(ScreenStabilizer, self).__init__(head_pose_postprocessors)
        self.state = 'ready'
        self.calibration = None
        self._calibration = None
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
        self.calibration = transform_util.Calibration(**self.calibration)

    def _stabilize(self, parameters):
        if self.state == 'calibrating':
            self._update_head_visual_node(parameters)
        elif self.state == 'stabilizing':
            self._update_canvas(parameters)

    def _update_head_visual_node(self, parameters):
        transform = self._head_visual_node.base_transform()
        postprocessed = {parameter: self.head_pose_postprocessors[parameter](value)
                        for (parameter, value) in parameters.items()}
        self.calibration = {parameter: value for (parameter, value) in postprocessed.items()}
        print('Postprocessed: {}'.format({parameter: round(value, 2)
                                          for (parameter, value) in postprocessed.items()}))
        #print 'x:', parameters['x']
        #print 'y:', parameters['y']
        #print 'z:', parameters['z']
        #print 'x_post:', postprocessed['x']
        #print 'y_post:', postprocessed['y']
        #print 'z_post:', postprocessed['z']
        transform.rotate(postprocessed['yaw'], (0, 1, 0))
        transform.rotate(postprocessed['pitch'], (1, 0, 0))
        transform.rotate(postprocessed['roll'], (0, 0, 1))
        transform.translate((postprocessed['x'], postprocessed['y'], 0.01))
        self._head_visual_node.transform = transform
        self.framerate_counter.tick()
        self._pipeline.update()

    def _update_canvas(self, parameters):
        base_vertices = self._visual_node.get_base_vertices()
        postprocessed = {parameter: self.head_pose_postprocessors[parameter](value)
                        for (parameter, value) in parameters.items()}
        print('Postprocessed: {}'.format({parameter: round(value, 2)
                                          for (parameter, value) in postprocessed.items()}))
        #print 'x:', postprocessed['x']
        #print 'y:', postprocessed['y']
        #print 'z:', postprocessed['z']
        #transform = self._visual_node.base_transform()
        #transform.rotate(postprocessed['yaw'] - self.calibration.yaw, (0, 1, 0))
        #transform.rotate(postprocessed['pitch'] - self.calibration.pitch, (1, 0, 0))
        #transform.rotate(postprocessed['roll'] - self.calibration.roll, (0, 0, 1))
        #transform.translate((postprocessed['x'] - self.calibration.x,
        #                     postprocessed['y'] - self.calibration.y, 0))
        #self._visual_node.transform = transform

        screen_base_vertices = [transform_util.render_xy_to_screen_xy(*vertex)
                                for vertex in base_vertices]
        print screen_base_vertices
        screen_transformed_vertices = [self.calibration.transform(
                                           vertex[0], vertex[1], **postprocessed)
                                       for vertex in screen_base_vertices]
        render_transformed_vertices = [transform_util.screen_xy_to_render_xy(*vertex)
                                       for vertex in screen_transformed_vertices]
        self._visual_node.update_vertices(render_transformed_vertices)
        self.framerate_counter.tick()
        self._pipeline.update()
