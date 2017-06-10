import time

import util

class FramerateCounter():
    """A frame rate counter."""
    def __init__(self, smoothing_window_size=20):
        self._buffer = util.RingBuffer(smoothing_window_size, dtype='d')

    def tick(self):
        current_time = time.time()
        self._buffer.append(current_time)

    def query(self):
        earliest_time = self._buffer.get_tail()
        current_time = self._buffer.get_head()
        frames = self._buffer.length
        if frames <= 1 or current_time == earliest_time:
            return None
        return frames / (current_time - earliest_time)

    def reset(self):
        self._buffer.reset()
