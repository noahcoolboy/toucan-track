# Filters for smoothing keypoints (2D and 3D)
import numpy as np

def get_filter(settings, fps, d):
    if settings is None:
        return RawFilter()
    elif settings.get("type") is None or settings.get("type").lower() == 'raw':
        return RawFilter()
    elif settings.get("type").lower() == 'movingaverage':
        return MovingAverageFilter(settings.get("window_size", 5), d)
    elif settings.get("type").lower() == 'oneeuro':
        return OneEuroFilter(fps,
                             settings.get("mincutoff", 0.05),
                             settings.get("beta", 80.0),
                             settings.get("dcutoff", 1.0),
                             d)
    else:
        raise ValueError('Unknown filter type: {}'.format(settings.get("type")))

class RawFilter:
    def filter(self, x, timestamp=None):
        return x

class MovingAverageFilter:
    def __init__(self, window_size, d=2):
        self.window_size = window_size
        self.window = np.zeros((window_size, d))
        self.idx = 0

    def filter(self, x, timestamp=None):
        self.window[self.idx] = x
        self.idx = (self.idx + 1) % self.window_size
        return np.mean(self.window, axis=0)


class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0, d=2):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

        self.x_prev = np.zeros(d)
        self.dx_prev = np.zeros(d)
        self.lasttime = None

    def filter(self, x, timestamp=None):
        if self.lasttime is None:
            self.lasttime = timestamp
            self.x_prev = x
            return x

        dt = (timestamp - self.lasttime) / 1000.0
        self.lasttime = timestamp

        dx = (x - self.x_prev) / dt
        edx = self._lowpass(dx, self.dx_prev, dt, self.dcutoff)
        self.dx_prev = dx

        cutoff = self.mincutoff + self.beta * np.abs(edx)
        filtered_x = self._lowpass(x, self.x_prev, dt, cutoff)
        self.x_prev = filtered_x

        return filtered_x

    def _lowpass(self, x, x_prev, dt, cutoff):
        # RC = 1 / (2 * pi * fc)
        RC = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (dt + RC)
        return x_prev + alpha * (x - x_prev)