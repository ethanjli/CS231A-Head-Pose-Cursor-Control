from os import path
import sys
import subprocess
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

import numpy as np

from utilities import profiling
from utilities import signal_processing
import monitoring

_PACKAGE_PATH = path.dirname(sys.modules[__name__].__file__)
_ROOT_PATH = path.dirname(_PACKAGE_PATH)
_GAZR_PATH = path.join(_ROOT_PATH, 'ext', 'gazr')
_HEAD_POSE_TRACKER_PATH = path.join(_GAZR_PATH, 'build', 'gazr_estimate_head_direction')
_HEAD_POSE_MODEL_PATH = path.join(_ROOT_PATH, 'ext', 'dlib', 'shape_predictor_68_face_landmarks.dat')
_HEAD_POSE_TRACKER_ARGS = [_HEAD_POSE_TRACKER_PATH, '--show',
                           '--model', _HEAD_POSE_MODEL_PATH]

PARAMETERS = ['yaw', 'pitch', 'roll', 'x', 'y', 'z']
DEFAULT_ANGLE_PARAMETERS = {
    'window_size': 10,
    'threshold': 0.5,
    'nonstationary_transition_smoothness': 4,
}
DEFAULT_FILTERS = {
    'yaw': signal_processing.SlidingWindowThresholdFilter(**DEFAULT_ANGLE_PARAMETERS),
    'pitch': signal_processing.SlidingWindowThresholdFilter(window_size=10, threshold=1,
                                                            nonstationary_transition_smoothness=4),
    'roll': signal_processing.SlidingWindowThresholdFilter(**DEFAULT_ANGLE_PARAMETERS),
    'x': signal_processing.SlidingWindowThresholdFilter(threshold=0.005),
    'y': signal_processing.SlidingWindowThresholdFilter(threshold=0.005),
    'z': signal_processing.SlidingWindowThresholdFilter()
}

class HeadPose(monitoring.Monitor):
    """Consumes head pose tracking stream from stdin and updates."""
    def __init__(self, filters=DEFAULT_FILTERS):
        super(HeadPose, self).__init__()
        self.parameters = {parameter: None for parameter in PARAMETERS}
        self.filters = filters

        self.update_rate_counter = profiling.FramerateCounter()

    def on_update(self, data):
        if "face_0" in data:
            raw_data = {
                'yaw': data['face_0']['yaw'] - 180,
                'pitch': data['face_0']['pitch'] - 180,
                'roll': data['face_0']['roll'] + 90,
                'x': data['face_0']['y'],
                'y': data['face_0']['z'],
                'z': data['face_0']['x']
            }
            for (parameter, raw_value) in raw_data.items():
                self.filters[parameter].append(raw_value)
                self.parameters[parameter] = self.filters[parameter].estimate_current()
            self.update_rate_counter.tick()
        self.updated = all(filtered_value is not None for filtered_value in self.parameters.values())

    def get_tracker_args(self):
        return _HEAD_POSE_TRACKER_ARGS
