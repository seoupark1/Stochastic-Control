import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

class ExtendedKalmanFilter:
    
    def __init__(self,
                state: ArrayLike,
                covariance: ArrayLike,
                motion_model: Callable,
                motion_jacobian: Callable,
                motion_noise_jacobian: ArrayLike,
                measurement_model: Callable,
                measurement_jacobian: Callable,
                measurement_noise_jacobian: ArrayLike,
                motion_noise_covariance: ArrayLike,
                measurement_noise_covariance: ArrayLike):
        
        # state & covariance
        self.x = np.asarray(state, dtype = float).reshape(-1)
        self.P = np.asarray(covariance, dtype = float)

        # motion model & jacobian
        self.f_model = motion_model
        self.F_jacobian = motion_jacobian
        self.L_jacobian = np.asarray(motion_noise_jacobian, dtype = float)

        # measurement model & jacobian
        self.h_model = measurement_model
        self.H_jacobian = measurement_jacobian
        self.M_jacobian = np.asarray(measurement_noise_jacobian, dtype = float)

        # noise covariances
        self.Q = np.asarray(motion_noise_covariance, dtype = float)
        self.R = np.asarray(measurement_noise_covariance, dtype = float)

    def prediction(self,
                   control_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)

        # F jacobian about k-1 step
        F = np.asarray(self.F_jacobian(self.x, u, 0), dtype = float)

        # x_check, P_check about k step
        self.x = self.f_model(self.x, u, 0).reshape(-1)
        self.P = F @ self.P @ F.T + self.L_jacobian @ self.Q @ self.L_jacobian.T
        self.P = (self.P + self.P.T) / 2

    def correction(self,
                   measurement_vector: ArrayLike):
        
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)
        measurement_size = self.x.size

        # H jacobian @ h about k step
        H = np.asarray(self.H_jacobian(self.x, 0), dtype = float)
        h = np.asarray(self.h_model(self.x, 0), dtype = float).reshape(-1)

        # kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.M_jacobian @ self.R @ self.M_jacobian.T)
        
        # x_hat, P_hat about k step
        self.x = self.x + K @ (y - h)
        self.P = (np.eye(measurement_size) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2

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





    