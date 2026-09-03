import numpy as np
from numpy.typing import ArrayLike, NDArray
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

    def get_sigma_points(self,
                         mean: ArrayLike,
                         covariance: ArrayLike) -> NDArray[np.float64]:

        n = mean.shape[0]
        kappa = 3 - n

        # cholesky decomposition
        covariance = (covariance + covariance.T) / 2
        L = np.linalg.cholesky(covariance)

        # compute sigma points
        sigma_points = np.vstack((mean,
                                  mean + np.sqrt(n + kappa) * L.T,
                                  mean - np.sqrt(n + kappa) * L.T))

        return sigma_points
    
    def get_weights(self):
        n = self.mean.shape[0]
        kappa = 3 - n

        # weights
        weights = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
        weights[0] = kappa / (n + kappa)

        return weights

    def prediction(self,
                   control_vector: ArrayLike):
        
        u = np.asarray(control_vector, dtype = float).reshape(-1)
        a = self.get_weights()
        n = self.P.shape[0]
        
        # propagated sigma points
        sigma_points = self.get_sigma_points(self.mean, self.P)
        propagated_sigma_points = np.array([self.f_model(point, u) for point in sigma_points])

        # predicted mean   
        predicted_mean = a @ propagated_sigma_points
        
        # predicted covariance
        predicted_P = np.zeros((n, n))
        for weight, propagated_sigma_point in zip(a, propagated_sigma_points):
            diff = propagated_sigma_point - predicted_mean
            predicted_P += weight * np.outer(diff, diff)

        self.mean = predicted_mean
        self.P = predicted_P + self.Q
        self.P = (self.P + self.P.T) / 2

    def correction(self,
                   measurement_vector: ArrayLike):
        
        y = np.asarray(measurement_vector, dtype = float).reshape(-1)
        n = self.mean.shape[0]
        m = y.size
        a = self.get_weights()

        # propagated sigma points
        sigma_points = self.get_sigma_points(self.mean, self.P)
        predicted_measurements = np.array([self.h_model(points) for points in sigma_points])

        # mean of predicted measurements
        predicted_measurements_mean = a @ predicted_measurements

        # covariance of predicted measurements
        P_y = np.zeros((m, m))
        for weight, predicted_measurement in zip(a, predicted_measurements):
            diff = predicted_measurement - predicted_measurements_mean
            P_y += weight * np.outer(diff, diff)

        P_y += self.R
        P_y = (P_y + P_y.T) / 2

        # cross-covariance
        P_xy = np.zeros((n, m))
        for weight, sigma_point, predicted_measurement in zip(a, sigma_points, predicted_measurements):
            diff_x = sigma_point - self.mean
            diff_y = predicted_measurement - predicted_measurements_mean
            P_xy += weight * np.outer(diff_x, diff_y)

        # kalman gain
        A = P_y.T
        B = P_xy.T
        K = np.linalg.solve(A, B).T

        # corrected mean & covariance
        self.mean += K @ (y - predicted_measurements_mean)
        self.P -= K @ P_y @ K.T
        self.P = (self.P + self.P.T) / 2

    @property
    def state(self):
        return self.mean.copy()
    
    @property
    def covariance(self):
        return self.P.copy()

        




