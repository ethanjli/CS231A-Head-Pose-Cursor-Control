from os import path
import sys
import subprocess
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
import scipy.signal
import numpy as np

import signal_processing

_PACKAGE_PATH = path.dirname(sys.modules[__name__].__file__)
_ROOT_PATH = path.dirname(_PACKAGE_PATH)
_GAZR_PATH = path.join(_ROOT_PATH, 'ext', 'gazr')
_HEAD_POSE_TRACKER_PATH = path.join(_GAZR_PATH, 'build', 'gazr_estimate_head_direction')
_HEAD_POSE_MODEL_PATH = path.join(_ROOT_PATH, 'ext', 'dlib', 'shape_predictor_68_face_landmarks.dat')
_HEAD_POSE_TRACKER_ARGS = [_HEAD_POSE_TRACKER_PATH, '--show',
                           '--model', _HEAD_POSE_MODEL_PATH]

class HeadPose():
    """Consumes head pose tracking stream from stdin and updates."""
    def __init__(self, smoothing_pitch=10, smoothing_yaw=10, smoothing_roll=10,
                 smoothing_x=10, smoothing_y=10, smoothing_z=10):
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.x = None
        self.y = None
        self.z = None
        self.updated = False

        half_gaussian_window = signal_processing.normalize_window(scipy.signal.gaussian(
            2 * smoothing_roll, 4.0)[:smoothing_roll])
        print half_gaussian_window
        self._yaw_smoothing_filter = signal_processing.SlidingWindowFilter(smoothing_yaw)
        self._pitch_smoothing_filter = signal_processing.SlidingWindowFilter(smoothing_pitch)
        self._roll_smoothing_filter = signal_processing.SlidingWindowFilter(
            smoothing_roll, estimation_mode=('kernel', half_gaussian_window))
        self._x_smoothing_filter = signal_processing.SlidingWindowFilter(smoothing_x)
        self._y_smoothing_filter = signal_processing.SlidingWindowFilter(smoothing_y)
        self._z_smoothing_filter = signal_processing.SlidingWindowFilter(smoothing_z)

        self._tracker_process = None
        self._monitor_thread = None

    def update(self, line=None):
        """Synchronously updates yaw and pitch and roll once from the stdin buffer.

        Arguments:
            line: the head pose tracking stdout line. If None, will read from stdin.
        """
        if line is None:
            line = sys.stdin.readline()
        data = eval(line)
        if "face_0" in data:
            raw_yaw = data['face_0']['yaw'] - 180
            self._yaw_smoothing_filter.append(raw_yaw)
            self.yaw = self._yaw_smoothing_filter.estimate_current()
            raw_pitch = data['face_0']['pitch'] - 180
            self._pitch_smoothing_filter.append(raw_pitch)
            self.pitch = self._pitch_smoothing_filter.estimate_current()
            raw_roll = data['face_0']['roll'] + 90
            self._roll_smoothing_filter.append(raw_roll)
            self.roll = self._roll_smoothing_filter.estimate_current()
            print '{},{}'.format(self._roll_smoothing_filter.get_head(), self.roll)
            raw_x = data['face_0']['y']
            self._x_smoothing_filter.append(raw_x)
            self.x = self._x_smoothing_filter.estimate_current()
            raw_y = data['face_0']['z']
            self._y_smoothing_filter.append(raw_y)
            self.y = self._y_smoothing_filter.estimate_current()
            raw_z = data['face_0']['x']
            self._z_smoothing_filter.append(raw_z)
            self.z = self._z_smoothing_filter.estimate_current()
        if (self.pitch is not None and self.yaw is not None and self.roll is not None and
                self.x is not None and self.y is not None and self.z is not None):
            self.updated = True

    def _start_tracker(self):
        """Starts the external head pose tracking program.
        The program's stdout is piped to the current stdin.
        """
        on_posix = 'posix' in sys.builtin_module_names

        self._tracker_process = subprocess.Popen(_HEAD_POSE_TRACKER_ARGS,
                                                 stdout=subprocess.PIPE,
                                                 bufsize=1, close_fds=on_posix)

    def monitor_sync(self, callback=None):
        """Synchronously updates yaw and pitch and roll continuously from stdin.

        Arguments:
            callback: If provided, calls callback after each update with the yaw and pitch and roll.
        """
        self._start_tracker()
        for line in iter(self._tracker_process.stdout.readline, b''):
            self.update(line)
            if self.updated:
                callback(self.yaw, self.pitch, self.roll, self.x, self.y, self.z)

    def monitor_async(self, callback=None):
        """Asynchronously updates yaw and pitch and roll continuously from stdin.

        Arguments:
            callback: If provided, calls callback after each update with the yaw and pitch and roll.
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

