import numpy as np
import transforms3d

import facial_landmarks
import render
import scene_manager
import animation
import stereo_util
import transform_util

class CSVLogger(animation.CalibratedCursorAnimator):
    def __init__(self):
        print 't (sample #), x (px), y (px)'
        self.t = 0
        super(CSVLogger, self).__init__()

    def _update_head(self, parameters):
        try:
            points_3d = stereo_util.compute_3d_model(parameters, self.calibration._camera_matrices)
            points_3d_filtered = np.empty_like(points_3d)
            for i in range(facial_landmarks.NUM_KEYPOINTS):
                for j in range(3):
                    self.points_3d_filters[i][j].append(points_3d[i,j])
                    points_3d_filtered[i,j] = self.points_3d_filters[i][j].estimate_current()
            target = self.calibration.compute_gaze_location(points_3d=points_3d_filtered, use_ransac=True,
                                                            threshold=2, num_iter=50)
            target_px = transform_util.screen_xy_to_render_xy(*target)
            target_px = -2 * np.array([target_px[0], target_px[1]])
            for i in range(2):
                self.target_filters[i].append(target_px[i])
            target_px = np.array([[self.target_filters[0].estimate_current(),
                                   self.target_filters[1].estimate_current(), 0.0]],
                                 dtype='f')
            if not np.any(np.isnan(target_px)):
                self._visual_node.update_list_data(target_px)
                self.framerate_counter.tick()
                self._pipeline.update()
                print str(self.t) + ', ' + str(target_px[0][0]) + ', ' + str(target_px[0][1])
                self.t += 1
        except stereo_util.NoIntersectionException:
            pass

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_face()
scene_manager.face_point_cloud.initialize_data(facial_landmarks.NUM_KEYPOINTS)

tracker = CSVLogger()
tracker.register_rendering_pipeline(pipeline)
tracker.register_visual_node(scene_manager.face_point_cloud)

pipeline.start_rendering()

tracker.stop_animating()
