#!/usr/bin/env python2
"""Generic utility functions."""
import numpy as np

class RingBuffer(object):
    """A 1-D ring buffer."""
    def __init__(self, length, dtype='f'):
        if length == 0:
            raise ValueError('RingBuffer length must be a positive number!')
        self.data = np.zeros(length, dtype=dtype)
        self._index = -1
        self.length = 0

    def reset(self):
        self.data = np.zeros(self.data.shape[0], dtype=self.data.dtype)
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
