import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

class UnscentedKalmanFilter:

    def __init__(self,
                 state_mean: ArrayLike,
                 covariance: ArrayLike,
                 motion_model: Callable,
                 motion_noise_covariance: ArrayLike,
                 measurement_model: Callable,
                 measurement_noise_covariance: ArrayLike):
        
        # state mean & covariance
        self.mean = np.asarray(state_mean, dtype = float).reshape(-1)
        self.P = np.asarray(covariance, dtype = float)

        # motion model & noise covariance
        self.f_model = motion_model
        self.Q = np.asarray(motion_noise_covariance, dtype = float)

        # measurement model & noise covariance
        self.h_model = measurement_model
        self.R = np.asarray(measurement_noise_covariance, dtype = float)

        # useful parameters
        self.propagated_sigma_points = None

    def get_propagated_sigma_points(self,
                                    control_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)
        n = self.mean.shape[0]
        kappa = 3 - n

        # cholesky decomposition
        L = np.linalg.cholesky(self.P)

        # compute sigma points
        sigma_points = np.vstack((self.mean,
                                  self.mean + np.sqrt(n + kappa) * L.T,
                                  self.mean - np.sqrt(n + kappa) * L.T))
        
        # propagate sigma points
        propagated_sigma_points = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            propagated_sigma_points[i, :] = self.f_model(sigma_points[i, :], u, 0)        
        
        return propagated_sigma_points
    
    def get_weights(self):
        n = self.mean.shape[0]
        kappa = 3 - n

        # weights
        a = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
        a[0] = kappa / (n + kappa)

        return a

    def prediction(self,
                   control_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)
        a = self.get_weights()
        n = self.mean.shape[0]
        
        # propagated sigma points
        propagated_sigma_points = self.get_propagated_sigma_points(u) 
        self.propagated_sigma_points = propagated_sigma_points     

        # compute x_check   
        x_check = a @ propagated_sigma_points
        
        # compute P_check
        P_check = np.zeros_like(self.P)
        for i in range(2 * n + 1):
            diff = propagated_sigma_points[i, :] - x_check
            P_check += a[i] * np.outer(diff, diff)

        self.mean = x_check
        self.P = P_check + self.Q
        self.P = (self.P + self.P.T) / 2


    def correction(self,
                   measurement_vector: ArrayLike):
        
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)
        n = self.mean.shape[0]
        m = y.size
        a = self.get_weights()
        propagated_sigma_points = self.get_propagated_sigma_points

        # predicted measurements from propagated sigma points
        predicted_measurements = np.zeros((2 * n + 1, len(y)))
        for i in range(2 * n + 1):
            predicted_measurements[i, :] = self.h_model(propagated_sigma_points[i, :], 0)

        # estimate mean of predicted measurements
        y_hat = a @ predicted_measurements

        # estimate covariance of predicted measurements
        P_y = np.zeros((m,m), dtype = float)
        for i in range(2 * n + 1):
            diff = predicted_measurements[i, :] - y_hat
            P_y += a[i] * np.outer(diff, diff)
        
        P_y += self.R
        P_y = (P_y + P_y.T) / 2

        # compute cross-covaraince
        P_xy = np.zeros((n, m))
        for i in range(2 * n + 1):
            diff_x = propagated_sigma_points[i, :] - self.mean
            diff_y = predicted_measurements[i, :] - y_hat
            P_xy += a[i] * np.outer(diff_x, diff_y)

        # kalman gain
        K = P_xy @ np.linalg.inv(P_y)

        # compute corrected mean & covariance
        self.mean += K @ (y - y_hat)
        self.P -= K @ P_y @ K.T
        self.P = (self.P + self.P.T) / 2
        self.propagated_sigma_points = None

    def unscentedkalmanfilter(self,
                              control_vector: ArrayLike,
                              measurement_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)
        self.prediction(u)
        self.correction(y)

    @property
    def state(self):
        return self.mean.copy()
    
    @property
    def covariance(self):
        return self.P.copy()

        




