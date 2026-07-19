import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

# only for additive noise
class ExtendedKalmanFilter:
    
    def __init__(self,
                state: ArrayLike,
                covariance: ArrayLike,
                motion_model: Callable,
                motion_jacobian: Callable,
                measurement_model: Callable,
                measurement_jacobian: ArrayLike,
                motion_noise_covariance: ArrayLike,
                measurement_noise_covariance: ArrayLike):
        
        # state & covariance
        self.x = np.asarray(state, dtype = float).reshape(-1)
        self.P = np.asarray(covariance, dtype = float)

        # motion model & jacobian
        self.f_model = motion_model
        self.F_jacobian = motion_jacobian

        # measurement model & jacobian
        self.h_model = measurement_model
        self.H_jacobian = measurement_jacobian

        # noise covariances
        self.Q = np.asarray(motion_noise_covariance, dtype = float)
        self.R = np.asarray(measurement_noise_covariance, dtype = float)

    def prediction(self,
                   control_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)

        # F jacobian about k-1 step
        F = np.asarray(self.F_jacobian(self.x, u, 0), dtype = float)

        # x_check, P_check about k step
        self.x = np.f_model(self.x, u, 0).reshape(-1)
        self.P = F @ self.P @ F.T + self.Q

    def correction(self,
                   measurement_vector: ArrayLike):
        
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)
        measurement_size = self.h_jacobian.shape[0]

        # H jacobian about k-1 step
        H = np.asarray(self.H_jacobian(self.x, 0), dtype = float)

        # kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        
        # x_hat, P_hat about K step
        x_check, P_check = self.x, self.P
        self.x = x_check + K @ (y - self.h_model(x_check, 0))
        self.P = (np.eye(measurement_size) - K @ H) @ P_check

    def extendedkalmanfilter(self,
                             control_vector: ArrayLike,
                             measurement_vector: ArrayLike):
        
        self.prediction(control_vector)
        self.correction(measurement_vector)
    
    @property
    def state(self):
        return self.x.copy()
    
    @property
    def covariance(self):
        return self.P.copy()





    