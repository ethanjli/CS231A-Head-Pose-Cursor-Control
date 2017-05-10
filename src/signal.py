import numpy as np
import sklearn

import util

class SlidingWindowFilter(util.RingBuffer):
    """A 1-D sliding window noise filter."""
    def __init__(self, window_size):
        super(SlidingWindowFilter, self).__init__(window_size)

    def get_timeseries(self):
        values = self.get_continuous()
        times = np.arange(len(values)) - len(values)
        return (times, values)

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
