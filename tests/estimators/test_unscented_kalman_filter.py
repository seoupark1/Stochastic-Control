import numpy as np

from stochastic_control.estimators.unscented_kalman_filter import UnscentedKalmanFilter

# 1D motion of a car which collects the angle difference between the motion direction and the top of the building
def test_prediction():

    # conditions
    dt = 0.1 # [s]
    u_0 = np.array([-2]) # [m/s^2]
    S = 20 # [m]
    D = 40 # [m]

    # state vector [position, velocity]
    x_0_hat = np.array([0, 5]).reshape(2,1)

    # initial state covariance
    P_0_hat = np.array([[0.01, 0],
                        [0, 1]])

    # motion model
    def f(state, control):
        F = np.array([[1, dt], [0, 1]])
        G = np.array([[0], [dt]])
        w = 0
        return F @ state + G @ control + w

    # measurement model
    def h(state):
        position = state[0]
        v = 0
        return np.arctan2(D - position, S) + v

    # noise covariances
    Q = 0.1 * np.eye(2)
    R = np.array([0.01])

    ukf = UnscentedKalmanFilter(state_mean = x_0_hat,
                                covariance = P_0_hat,
                                motion_model = f,
                                motion_noise_covariance = Q,
                                measurement_model = h,
                                measurement_noise_covariance = R)

    # expected values
    n = ukf.mean.shape[0]
    kappa = 3 - n

    weights = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights[0] = kappa / (n + kappa)
    a = weights

    size = ukf.P.shape[0]

    covariance = (ukf.P.copy() + ukf.P.copy().T) / 2
    L = np.linalg.cholesky(covariance)

    sigma_points = np.vstack((ukf.mean,
                              ukf.mean + np.sqrt(n + kappa) * L.T,
                              ukf.mean - np.sqrt(n + kappa) * L.T))

    propagated_sigma_points = np.array([ukf.f_model(point, u_0) for point in sigma_points])

    predicted_mean = a @ propagated_sigma_points

    predicted_P = np.zeros((size, size))
    for weight, propagated_sigma_point in zip(a, propagated_sigma_points):
        diff = propagated_sigma_point - predicted_mean
        predicted_P += weight * np.outer(diff, diff)

    expected_state = predicted_mean
    expected_covariance = predicted_P + ukf.Q
    expected_covariance = (expected_covariance + expected_covariance.T) / 2

    # predicted values
    ukf.prediction(u_0)
    x_check = ukf.mean
    P_check = ukf.P

    # test predicted state & covariance match with expected values
    np.testing.assert_allclose(expected_state, x_check)
    np.testing.assert_allclose(expected_covariance, P_check)

    # test covariance's symmetry
    np.testing.assert_allclose(P_check, P_check.T)

def test_correction():

    # conditions
    dt = 0.1 # [s]
    u_0 = np.array([-2]) # [m/s^2]
    y_1 = np.array([np.pi / 6]) # [rad]
    S = 20 # [m]
    D = 40 # [m]

    # state vector [position, velocity]
    x_0_hat = np.array([0, 5]).reshape(2,1)

    # initial state covariance
    P_0_hat = np.array([[0.01, 0],
                        [0, 1]])

    # motion model
    def f(state, control):
        F = np.array([[1, dt], [0, 1]])
        G = np.array([[0], [dt]])
        w = 0
        return F @ state + G @ control + w

    # measurement model
    def h(state):
        position = state[0]
        v = 0
        return np.arctan2(D - position, S) + v

    # noise covariances
    Q = 0.1 * np.eye(2)
    R = np.array([0.01])

    ukf = UnscentedKalmanFilter(state_mean = x_0_hat,
                                covariance = P_0_hat,
                                motion_model = f,
                                motion_noise_covariance = Q,
                                measurement_model = h,
                                measurement_noise_covariance = R)

    ukf.prediction(u_0)

    # expected values
    n = ukf.mean.shape[0]
    kappa = 3 - n

    weights = np.full(2 * n + 1, 1 / (2 * (n + kappa)))
    weights[0] = kappa / (n + kappa)
    a = weights

    m = y_1.size

    covariance = (ukf.P.copy() + ukf.P.copy().T) / 2
    L = np.linalg.cholesky(covariance)

    sigma_points = np.vstack((ukf.mean,
                              ukf.mean + np.sqrt(n + kappa) * L.T,
                              ukf.mean - np.sqrt(n + kappa) * L.T))
    
    predicted_measurements = np.array([ukf.h_model(points) for points in sigma_points])

    predicted_measurements_mean = a @ predicted_measurements

    P_y = np.zeros((m, m))
    for weight, predicted_measurement in zip(a, predicted_measurements):
        diff = predicted_measurement - predicted_measurements_mean
        P_y += weight * np.outer(diff, diff)

    P_y += ukf.R
    P_y = (P_y + P_y.T) / 2

    # cross-covariance
    P_xy = np.zeros((n, m))
    for weight, sigma_point, predicted_measurement in zip(a, sigma_points, predicted_measurements):
        diff_x = sigma_point - ukf.mean
        diff_y = predicted_measurement - predicted_measurements_mean
        P_xy += weight * np.outer(diff_x, diff_y)

    A = P_y.T
    B = P_xy.T
    K = np.linalg.solve(A, B).T

    expected_state = ukf.mean.copy() + K @ (y_1 - predicted_measurements_mean)
    expected_covariance = ukf.P.copy() - K @ P_y @ K.T
    expected_covariance = (expected_covariance + expected_covariance.T) / 2

    # corrected values
    ukf.correction(y_1)
    x_hat = ukf.mean
    P_hat = ukf.P

    # test predicted state & covariance match with expected values
    np.testing.assert_allclose(expected_state, x_hat)
    np.testing.assert_allclose(expected_covariance, P_hat)

    # test covariance's symmetry
    np.testing.assert_allclose(P_hat, P_hat.T)