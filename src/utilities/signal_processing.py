import time

import scipy.signal
import numpy as np
import cv2

import util

def normalize_window(window):
    return window / window.sum()

def gaussian_window(window_length, standard_deviation):
    return normalize_window(scipy.signal.gaussian(window_length, standard_deviation))

def half_gaussian_window(window_length, standard_deviation):
    return normalize_window(scipy.signal.gaussian(2 * window_length, standard_deviation)[:window_length])

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
        elif self.estimation_mode == 'raw':
            return self.get_head()
        else:
            if self.length < self.data.size:
                return None
            (times, values) = self.get_timeseries()
            values = smooth(values, self.smoothing_mode)
            if isinstance(self.estimation_mode, tuple):
                if self.estimation_mode[0] == 'poly':
                    return estimate_poly(times, values, self.estimation_mode[1])
                elif self.estimation_mode[0] == 'kernel':
                    return np.dot(self.estimation_mode[1], values)

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

    def get_min(self):
        """Gets the min of the values in the window."""
        return np.amin(self.data[:self.length])

    def get_max(self):
        """Gets the min of the values in the window."""
        return np.amax(self.data[:self.length])

    def get_timeseries(self):
        values = self.get_continuous()
        times = np.arange(len(values)) - len(values)
        return (times, values)

class SlidingWindowThresholdFilter(SlidingWindowFilter):
    def __init__(self, window_size=8, threshold=0, nonstationary_transition_smoothness=3,
                 smoothing_mode=('convolve', gaussian_window(7, 2.0)), estimation_mode=('poly', 4)):
        super(SlidingWindowThresholdFilter, self).__init__(window_size, smoothing_mode, estimation_mode)
        self.threshold = threshold
        self._stationary_value = None
        self.nonstationary_transition_smoothness = nonstationary_transition_smoothness
        self._nonstationary_duration = 0

    def estimate_current(self):
        if self.in_stationary_range(self.get_head()):
            if self._stationary_value is None:
                self._stationary_value = super(SlidingWindowThresholdFilter, self).estimate_current()
                self._nonstationary_duration = 0
            return self._stationary_value
        else:
            self._nonstationary_duration += 1
            if self._nonstationary_duration >= self.nonstationary_transition_smoothness:
                self._stationary_value = None
                return super(SlidingWindowThresholdFilter, self).estimate_current()
            else:
                return self._stationary_value

    def in_stationary_range(self, value):
        return (value >= self.estimate_stationary_value() - self.threshold and
                value <= self.estimate_stationary_value() + self.threshold)

    def estimate_stationary_value(self):
        return self.get_mean()

class KalmanFilter(object):
    def __init__(self):
        self.kalman = cv2.KalmanFilter(3, 1, 0)
        self.kalman.statePre = np.zeros((3, 1), np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0]], np.float32)
        self.kalman.measurementNoiseCov = np.array([10], np.float32)
        self.kalman.errorCovPost = np.eye(3, dtype=np.float32) * 4

        self.last_measurement_time = None
        self.estimated = None

    def append(self, x):
        current_time = time.time()
        if self.last_measurement_time is None:
            self.last_measurement_time = current_time
        else:
            dt = current_time - self.last_measurement_time
            self.last_measurement_time = current_time
            self.kalman.transitionMatrix = np.array(
                [[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]], np.float32)
            self.kalman.processNoiseCov = np.array(
                [[1.0 / 9 * dt ** 6, 1.0 / 6 * dt ** 5, 1.0 / 3 * dt ** 4],
                 [1.0 / 6 * dt ** 5, 1.0 / 4 * dt ** 4, 1.0 / 2 * dt ** 3],
                 [1.0 / 3 * dt ** 4, 1.0 / 2 * dt ** 3, dt]], np.float32) * 1000

        self.kalman.predict()
        if x is not None:
            self.estimated = self.kalman.correct(np.array([[x]], np.float32)).ravel()
        else:
            self.estimated = prediction.ravel()
        print self.estimated[1], self.estimated[2]

    def estimate_current(self):
        return self.estimated[0]

class ThresholdKalmanFilter(KalmanFilter):
    def __init__(self, position_from_stationary=5, velocity_from_stationary=5, acceleration_from_stationary=8,
                 velocity_to_stationary=5, acceleration_to_stationary=5):
        super(ThresholdKalmanFilter, self).__init__()
        self.position_from_stationary = position_from_stationary
        self.velocity_from_stationary = velocity_from_stationary
        self.acceleration_from_stationary = acceleration_from_stationary
        self.velocity_to_stationary = velocity_to_stationary
        self.acceleration_to_stationary = acceleration_to_stationary
        self._stationary_value = None

    def append(self, x):
        super(ThresholdKalmanFilter, self).append(x)
        abs_velocity = abs(self.estimated[1])
        abs_acceleration = abs(self.estimated[2])
        if self._stationary_value is None:
            if (abs_velocity < self.velocity_to_stationary and
                    abs_acceleration < self.acceleration_to_stationary):
                self._stationary_value = self.estimated[0]
        else:
            abs_position = abs(self.estimated[0] - self._stationary_value)
            if (abs_position > self.position_from_stationary
                    or abs_velocity > self.velocity_from_stationary or
                    abs_acceleration > self.acceleration_from_stationary):
                self._stationary_value = None

    def estimate_current(self):
        if self._stationary_value is not None:
            return self._stationary_value
        return super(ThresholdKalmanFilter, self).estimate_current()
