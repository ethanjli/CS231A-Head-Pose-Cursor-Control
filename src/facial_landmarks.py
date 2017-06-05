from os import path
import sys
import subprocess
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

import numpy as np

from utilities import signal_processing
import monitoring

_PACKAGE_PATH = path.dirname(sys.modules[__name__].__file__)
_ROOT_PATH = path.dirname(_PACKAGE_PATH)
_GAZR_PATH = path.join(_ROOT_PATH, 'ext', 'gazr')
_FACIAL_LANDMARK_TRACKER_PATH = path.join(_GAZR_PATH, 'build', 'gazr_estimate_facial_landmarks')
_FACIAL_LANDMARK_MODEL_PATH = path.join(_ROOT_PATH, 'ext', 'dlib', 'shape_predictor_68_face_landmarks.dat')
_FACIAL_LANDMARK_TRACKER_ARGS = [_FACIAL_LANDMARK_TRACKER_PATH, '--show',
                                 '--model', _FACIAL_LANDMARK_MODEL_PATH]

NUM_KEYPOINTS = 69

class FacialLandmarks(monitoring.Monitor):
    """Consumes facial landmark tracking stream from stdin and updates."""
    def __init__(self, camera_index=0):
        super(FacialLandmarks, self).__init__()
        self.parameters = np.zeros((NUM_KEYPOINTS, 2))
        self.camera_index = camera_index

    def on_update(self, data):
        if "face_0" in data:
            self.parameters = np.array(data['face_0'])
            self.updated = True

    def get_tracker_args(self):
        args = list(_FACIAL_LANDMARK_TRACKER_ARGS)
        args.append('--camera')
        args.append(str(self.camera_index))
        return args
