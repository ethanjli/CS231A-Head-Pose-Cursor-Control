import numpy as np
import transforms3d

import facial_landmarks
import stereo_cameras
import stereo_util
import animation

class CSVLogger(animation.FacialLandmarkAnimator):
    def __init__(self):
        header = 't (sample #)'
        for i in range(facial_landmarks.NUM_KEYPOINTS):
            header += ', x' + str(i) + ' (px), y' + str(i) + ' (px), z' + str(i) + ' (px)'
        print header
        self.t = 0
        self.camera_matrices = stereo_util.make_parallel_camera_matrices(stereo_cameras.K_LEFT,
                                                                         stereo_cameras.K_RIGHT,
                                                                         -stereo_cameras.TRANSLATION[0])
        super(CSVLogger, self).__init__(animation.make_facial_raw_filters(),
                                        animation.make_facial_raw_filters())

    def on_update(self, parameters):
        points_3d = stereo_util.compute_3d_model(parameters, self.camera_matrices)
        row = str(self.t)
        for i in range(facial_landmarks.NUM_KEYPOINTS):
            row += ', ' + str(points_3d[i,0]) + ', ' + str(points_3d[i,1]) + ', ' + str(points_3d[i,2])
        print row
        self.t += 1


tracker = CSVLogger()
tracker.animate_sync()
