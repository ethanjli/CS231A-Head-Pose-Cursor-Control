from os import path
import sys
import subprocess
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
import numpy as np

import signal_processing

_PACKAGE_PATH = path.dirname(sys.modules[__name__].__file__)
_ROOT_PATH = path.dirname(_PACKAGE_PATH)
_GAZR_PATH = path.join(_ROOT_PATH, 'ext', 'gazr')
_HEAD_POSE_TRACKER_PATH = path.join(_GAZR_PATH, 'build', 'gazr_estimate_head_direction')
_HEAD_POSE_MODEL_PATH = path.join(_ROOT_PATH, 'ext', 'dlib', 'shape_predictor_68_face_landmarks.dat')
_HEAD_POSE_TRACKER_ARGS = [_HEAD_POSE_TRACKER_PATH, '--show',
                           '--model', _HEAD_POSE_MODEL_PATH]

PARAMETERS = ['yaw', 'pitch', 'roll', 'x', 'y', 'z']
DEFAULT_FILTERS = {parameter: signal_processing.SlidingWindowThresholdFilter(
                       10, threshold=0.5, nonstationary_transition_smoothness=4,
                       smoothing_mode=('convolve', signal_processing.gaussian_window(7, 2.0)),
                       estimation_mode=('poly', 4))
                   for parameter in PARAMETERS}

class HeadPose():
    """Consumes head pose tracking stream from stdin and updates."""
    def __init__(self, filters=DEFAULT_FILTERS):
        self.parameters = {parameter: None for parameter in PARAMETERS}
        self.filters = filters
        self.updated = False

        self._tracker_process = None
        self._monitor_thread = None

    def update(self, line=None):
        """Synchronously updates parameters once from the stdin buffer.

        Arguments:
            line: the head pose tracking stdout line. If None, will read from stdin.
        """
        if line is None:
            line = sys.stdin.readline()
        data = eval(line)
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
        self.updated = all(filtered_value is not None for filtered_value in self.parameters.values())

    def _start_tracker(self):
        """Starts the external head pose tracking program.
        The program's stdout is piped to the current stdin.
        """
        on_posix = 'posix' in sys.builtin_module_names

        self._tracker_process = subprocess.Popen(_HEAD_POSE_TRACKER_ARGS,
                                                 stdout=subprocess.PIPE,
                                                 bufsize=1, close_fds=on_posix)

    def monitor_sync(self, callback=None):
        """Synchronously updates parameters continuously from stdin.

        Arguments:
            callback: If provided, calls callback after each update with the parameters.
        """
        self._start_tracker()
        for line in iter(self._tracker_process.stdout.readline, b''):
            self.update(line)
            if self.updated:
                callback(self.parameters)

    def monitor_async(self, callback=None):
        """Asynchronously updates parameters continuously from stdin.

        Arguments:
            callback: If provided, calls callback after each update with the parameters.
        Threading:
            Instantiates a singleton thread named HeadPose.
        """
        if self._monitor_thread is not None:
            return
        self._monitor_thread = threading.Thread(target=self.monitor_sync, name='HeadPose',
                                                kwargs={'callback': callback})
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stops head pose tracking.
        Stops the tracker process, if it exists.
        Stops the asynchronous monitor, if it was started.
        """
        if self._tracker_process is not None:
            self._tracker_process.terminate()
            self._tracker_process.wait()
            self._tracker_process = None
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None

