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
        
        self.x = np.asarray(state, dtype = float).reshape(-1)

        self.P = np.asarray(covariance, dtype = float)

        # constant jacobian for linear system
        self.F = np.asarray(motion_jacobian, dtype = float)

        # constant jacobian for linear system
        self.G = np.asarray(control_jacobian, dtype = float)

        # constant jacobian for linear system
        self.H = np.asarray(measurement_jacobian, dtype = float)

        self.Q = np.asarray(motion_noise_covariance, dtype = float)

        self.R = np.asarray(measurement_noise_covariance, dtype = float)


    def kalmanfilter(self,
                     control_vector: ArrayLike,
                     measurement_vector: ArrayLike):

        u = np.asarray(control_vector, dtype = float)

        y = np.asarray(measurement_vector, dtype = float)

        # find out the size
        n = len(self.x)

        # prediction stage
        x_check = self.F @ self.x + self.G @ u
        P_check = self.F @ self.P @ self.F.T + self.Q

        # extract kalman gain
        K = P_check @ self.H.T @ np.linalg.inv(self.H @ P_check @ self.H.T + self.R)

        # correction stage
        self.x = x_check + K @ (y - self.H @ x_check)
        self.P = (np.eye(n) - K @ self.H) @ P_check

    @property
    def state(self):
        return self.x.copy()
    
    @property
    def covariance(self):
        return self.P.copy()