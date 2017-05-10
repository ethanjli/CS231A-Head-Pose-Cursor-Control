#!/usr/bin/env python2
"""Functions and classes for loading of data from Oxford datasets."""
from os import path
import time

import numpy as np

class RingBuffer(object):
    """A 1-D ring buffer."""
    def __init__(self, length, dtype='f'):
        self.data = np.zeros(length, dtype=dtype)
        self._index = -1
        self.length = 0

    def append(self, value):
        """Adds a value to the buffer, overwriting a stale entry if needed."""
        if self.length < self.data.size:
            self.length += 1
        self._index = (self._index + 1) % self.data.size
        self.data[self._index] = value

    def get_head(self):
        """Gets the most recently added value in the buffer."""
        return self.data[self._index]

    def get_tail(self):
        """Gets the least recently added value in the buffer."""
        if self.length:
            return self.data[(self._index + 1) % self.length]
        else:
            return None

    def get_continuous(self):
        """Returns a copy of the buffer with elements in correct time order."""
        if self.length < self.data.size:
            return np.concatenate((self.data[:self._index + 1],
                                   self.data[self._index + 1:]))[:self.length]
        else:
            return np.concatenate((self.data[self._index + 1:],
                                   self.data[:self._index + 1]))

# SIGNAL PROCESSING

class SlidingWindowFilter(RingBuffer):
    """A 1-D sliding window noise filter."""
    def __init__(self, window_size):
        super(SlidingWindowFilter, self).__init__(window_size)

    def get_mean(self):
        """Gets the mean of the values in the window."""
        if self.length:
            return np.mean(self.data[:self.length])
        else:
            return None

    def get_median(self):
        """Gets the median of the values in the window."""
        if self.length:
            return np.median(self.data[:self.length])
        else:
            return None

class FramerateCounter():
    """A frame rate counter."""
    def __init__(self, smoothing_window_size=120):
        self._buffer = RingBuffer(smoothing_window_size, dtype='d')

    def tick(self):
        current_time = time.time()
        self._buffer.append(current_time)

    def query(self):
        earliest_time = self._buffer.get_tail()
        current_time = self._buffer.get_head()
        frames = self._buffer.length
        if frames <= 1:
            return None
        return frames / (current_time - earliest_time)

