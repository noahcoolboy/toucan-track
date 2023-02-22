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
    elif settings.get("type").lower() == 'kalman':
        return KalmanFilter(fps, settings.get("Q", 0.1), settings.get("R", 1), d)
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



class KalmanFilter:
    def __init__(self, fps, Q, R, d=2):
        """
        Initialize Kalman filter object.

        Parameters:
        - F: state transition matrix
        - H: measurement matrix
        - Q: process noise covariance matrix
        - R: measurement noise covariance matrix
        - x0: initial state estimate
        - P0: initial error covariance matrix
        """
        self.d = d
        self.F = np.eye(d * 2)
        for i in range(d):
            self.F[i, i + d] = 1.0 / fps
        
        self.H = np.eye(d * 2)[:d]
        self.Q = Q * np.eye(d * 2)
        self.R = R * np.eye(d)
        self.x = np.zeros(d * 2)
        self.P = np.eye(d * 2)

    def predict(self):
        """
        Predict next state and error covariance matrix.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Update state estimate and error covariance matrix with measurement.
        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    def filter(self, point, timestamp=None):
        # Predict the next state estimate based on the previous estimate
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Compute the Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)

        # Update the state estimate with the new measurement
        self.x = x_pred + K @ (point - self.H @ x_pred)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred

        return self.x[:self.d]