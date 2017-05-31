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

class FacialLandmarks():
    """Consumes facial landmark tracking stream from stdin and updates."""
    def __init__(self):
        self.parameters = {parameter: None for parameter in PARAMETERS}
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
            raw_data = {data['face_0'][parameter] for parameter in MODEL.keys()}
            self.updated = True

    def _start_tracker(self):
        """Starts the external head pose tracking program.
        The program's stdout is piped to the current stdin.
        """
        on_posix = 'posix' in sys.builtin_module_names

        self._tracker_process = subprocess.Popen(_FACIAL_LANDMARK_TRACKER_ARGS,
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
        self._monitor_thread = threading.Thread(target=self.monitor_sync, name='FacialLandmarks',
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

