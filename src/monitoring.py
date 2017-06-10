import sys
import subprocess
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

class Monitor(object):
    """Monitors a stream from stdin."""
    def __init__(self):
        self.updated = False
        self._tracker_process = None
        self._monitor_thread = None

    def on_update(self, data):
        """Handles a new sample of data from stdin.
        If data is updated, needs to set self.updated to True."""
        pass

    def update(self, line=None):
        """Synchronously updates parameters once from the stdin buffer.

        Arguments:
            line: the head pose tracking stdout line. If None, will read from stdin.
        """
        if line is None:
            line = sys.stdin.readline()
        data = eval(line)
        self.on_update(data)

    def _start_tracker(self):
        """Starts the external head pose tracking program.
        The program's stdout is piped to the current stdin.
        """
        on_posix = 'posix' in sys.builtin_module_names

        self._tracker_process = subprocess.Popen(self.get_tracker_args(),
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
        self._monitor_thread = threading.Thread(target=self.monitor_sync, name=self.__class__.__name__,
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

