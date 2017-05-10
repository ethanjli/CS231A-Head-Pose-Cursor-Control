import numpy as np
import sklearn.linear_model

import util

class SlidingWindowFilter(util.RingBuffer):
    """A 1-D sliding window noise filter."""
    def __init__(self, window_size, estimation_mode='median'):
        super(SlidingWindowFilter, self).__init__(window_size)
        self.estimation_mode = estimation_mode

    def estimate_current(self):
        if self.estimation_mode == 'ransac_linear':
            return self.get_ransac_linear_estimation()
        elif self.estimation_mode == 'median':
            return self.get_median()
        elif self.estimation_mode == 'mean':
            return self.get_mean()

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

    def get_ransac_linear_fit(self):
        model = sklearn.linear_model.LinearRegression()
        model_ransac = sklearn.linear_model.RANSACRegressor(model)
        (times, values) = self.get_timeseries()
        times = np.reshape(times, (len(times), 1))
        model_ransac.fit(times, values)
        return model_ransac

    def get_ransac_linear_estimation(self):
        if self.length == self.data.size:
            model = self.get_ransac_linear_fit()
            return model.predict(np.array([[0]]))
        else:
            return None

