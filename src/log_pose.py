import numpy as np
import transforms3d

import facial_landmarks
import render
import scene_manager
import animation

class CSVLogger(animation.CalibratedCursorAnimator):
    def __init__(self):
        print 't (sample #), x (cm), y (cm), z (cm), r (deg), p (deg), y (deg)'
        self.t = 0
        super(CSVLogger, self).__init__()

    def _update_head(self, parameters):
        (rotation, translation, _) = self.calibration.compute_RT_ransac(points=parameters, threshold=2, num_iter=50)
        try:
            (angle_z, angle_y, angle_x) = np.rad2deg(transforms3d.taitbryan.mat2euler(rotation))
        except ValueError:
            return
        print str(self.t) + ', ' + str(translation[0]) + ', ' + str(translation[1]) + ', ' + str(translation[2]) + ', ' + str(angle_z) + ', ' + str(angle_y) + ', ' + str(angle_x)
        self.t += 1

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_face()
scene_manager.face_point_cloud.initialize_data(facial_landmarks.NUM_KEYPOINTS)

facial_landmarks = CSVLogger()
facial_landmarks.register_rendering_pipeline(pipeline)
facial_landmarks.register_visual_node(scene_manager.face_point_cloud)

pipeline.start_rendering()

facial_landmarks.stop_animating()
