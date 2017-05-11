import scipy.signal
import numpy as np

import util

def normalize_window(window):
    return window / window.sum()

def reflect_signal(values, length):
    return np.r_[values[length - 1:0:-1], values, values[-2:-length - 1:-1]]

def smooth(values, mode):
    if mode is None:
        return values
    elif isinstance(mode, tuple) and mode[0] == 'median':
        return scipy.signal.medfilt(values, mode[1])
    elif isinstance(mode, tuple) and mode[0] == 'convolve':
        reflected = reflect_signal(values, len(mode[1]) - int(len(mode[1]) / 2))
        smoothed = scipy.signal.convolve(mode[1], reflected, mode='valid')
        return smoothed

def estimate_poly(times, values, degree):
    coef = np.polyfit(times, values, degree)
    return coef[degree]

class SlidingWindowFilter(util.RingBuffer):
    """A 1-D sliding window noise filter."""
    def __init__(self, window_size, smoothing_mode=None, estimation_mode=('poly', 3)):
        super(SlidingWindowFilter, self).__init__(window_size)
        self.smoothing_mode = smoothing_mode
        self.estimation_mode = estimation_mode

    def estimate_current(self):
        if self.estimation_mode == 'median':
            return self.get_median()
        elif self.estimation_mode == 'mean':
            return self.get_mean()
        else:
            if self.length < self.data.size:
                return None
            (times, values) = self.get_timeseries()
            values = smooth(values, self.smoothing_mode)
            if isinstance(self.estimation_mode, tuple) and self.estimation_mode[0] == 'poly':
                return estimate_poly(times, values, self.estimation_mode[1])

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

    def get_timeseries(self):
        values = self.get_continuous()
        times = np.arange(len(values)) - len(values)
        return (times, values)

