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

# Anthropometric for male adult
# Relative position of various facial feature relative to sellion
# Values taken from https://en.wikipedia.org/wiki/Human_head
# X points forward
MODEL = {
    'sellion': np.array([0.0, 0.0, 0.0]),
    'right_eye': np.array([-20.0, -65.5, -5.0]),
    'left_eye': np.array([-20.0, 65.5, -5.0]),
    'right_ear': np.array([-100.0, -77.5, -6.0]),
    'left_ear': np.array([-100.0, 77.5, -6.0]),
    'nose': np.array([21.0, 0.0, -48.0]),
    'stommion': np.array([10.0, 0.0, -75.0]),
    'menton': np.array([0.0, 0.0, -133.0])
}
PARAMETERS = ['sellion', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'nose', 'stommion', 'menton']

class FacialLandmarks(monitoring.Monitor):
    """Consumes facial landmark tracking stream from stdin and updates."""
    def __init__(self, camera_index=0):
        super(FacialLandmarks, self).__init__()
        self.parameters = {parameter: None for parameter in MODEL.keys()}
        self.camera_index = camera_index

    def on_update(self, data):
        if "face_0" in data:
            self.parameters = {parameter: data['face_0'][parameter] for parameter in PARAMETERS}
            self.updated = True

    def get_tracker_args(self):
        args = list(_FACIAL_LANDMARK_TRACKER_ARGS)
        args.append('--camera')
        args.append(str(self.camera_index))
        return args
