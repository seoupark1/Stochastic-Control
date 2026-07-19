import numpy as np
from numpy.typing import ArrayLike, NDArray

# linear kalman filter
class KalmanFilter:

    def __init__(self, 
                 state: ArrayLike, 
                 covariance: ArrayLike,
                 motion_jacobian: ArrayLike, 
                 control_jacobian: ArrayLike, 
                 measurement_jacobian: ArrayLike,
                 motion_noise_covariance: ArrayLike,
                 measurement_noise_covariance: ArrayLike):
        
        # state & covariance
        self.x = np.asarray(state, dtype = float).reshape(-1)
        self.P = np.asarray(covariance, dtype = float)

        # constant jacobians for linear system
        self.F = np.asarray(motion_jacobian, dtype = float)
        self.G = np.asarray(control_jacobian, dtype = float)
        self.H = np.asarray(measurement_jacobian, dtype = float)

        # noise covariances
        self.Q = np.asarray(motion_noise_covariance, dtype = float)
        self.R = np.asarray(measurement_noise_covariance, dtype = float)

    def prediction(self, 
                   control_vector:ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)

        # prediction stage
        self.x = self.F @ self.x + self.G @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) / 2

    def correction(self, 
                   measurement_vector: ArrayLike):
        
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)

        # find out the size
        n = len(self.x)

        # extract kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # correction stage
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(n) - K @ self.H) @ self.P
        self.P = (self.P + self.P.T) / 2

    def kalmanfilter(self,
                     control_vector:ArrayLike,
                     measurement_vector: ArrayLike):
        
        self.prediction(control_vector)
        self.correction(measurement_vector)

    @property
    def state(self):
        return self.x.copy()
    
    @property
    def covariance(self):
        return self.P.copy()