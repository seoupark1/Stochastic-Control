import numpy as np

from stochastic_control.estimators.kalman_filter import KalmanFilter

# 1D motion of a car which collects position data using GPS
def test_prediction():

    # conditions
    dt = 0.1 # [s]
    u_0 = np.array([-2]) # [m/s^2]

    # state vector [position, velocity]
    x_0 = np.array([0, 5]).reshape(2,1)

    # initial state covariance
    P_0 = np.array([[0.01, 0],
                    [0, 1]])

    # motion & control & measurement jacobians
    F = np.array([[1, dt],
                  [0, 1]])
    G = np.array([0, dt]).reshape(2,1)
    H = np.array([1, 0]).reshape(2,1)

    # noise covariances
    Q = 0.1 * np.eye(2)
    R = np.array([0.01])

    kf = KalmanFilter(state = x_0,
                      covariance = P_0,
                      motion_jacobian = F,
                      control_jacobian = G,
                      measurement_jacobian = H,
                      motion_noise_covariance = Q,
                      measurement_noise_covariance = R)

    # expected values
    expected_state = kf.F @ kf.x + kf.G @ u_0
    expected_covariance = kf.F @ kf.P @ kf.F.T + kf.Q
    expected_covariance = (expected_covariance + expected_covariance.T) / 2

    # predicted values
    kf.prediction(u_0)
    x_check = kf.x
    P_check = kf.P

    # test predicted state & covariance match with expected values
    np.testing.assert_allclose(expected_state, x_check)
    np.testing.assert_allclose(expected_covariance, P_check)

    # test covariance's symmetry
    np.testing.assert_allclose(P_check, P_check.T)

def test_correction():

    # conditions
    dt = 0.1 # [s]
    u_0 = np.array([-2]) # [m/s^2]
    y_0 = np.array([1])

    # state vector [position, velocity]
    x_0 = np.array([0, 5]).reshape(2,1)

    # initial state covariance
    P_0 = np.array([[0.01, 0],
                    [0, 1]])

    # motion & control & measurement jacobians
    F = np.array([[1, dt],
                  [0, 1]])
    G = np.array([0, dt]).reshape(2,1)
    H = np.array([1, 0]).reshape(1,2)

    # noise covariances
    Q = 0.1 * np.eye(2)
    R = np.array([[0.01]])

    kf = KalmanFilter(state = x_0,
                      covariance = P_0,
                      motion_jacobian = F,
                      control_jacobian = G,
                      measurement_jacobian = H,
                      motion_noise_covariance = Q,
                      measurement_noise_covariance = R)

    # prediction
    kf.prediction(u_0)

    # expected corrected values
    n = len(kf.x)

    A = (kf.H @ kf.P @ kf.H.T + kf.R).T
    B = (kf.P @ kf.H.T).T
    K =  np.linalg.solve(A, B).T

    expected_state = kf.x + K @ (y_0 - kf.H @ kf.x)
    expected_covariance = (np.eye(n) - K @ kf.H) @ kf.P @ (np.eye(n) - K @ kf.H).T + K @ kf.R @ K.T
    expected_covariance = (expected_covariance + expected_covariance.T) / 2

    # correction
    kf.correction(y_0)
    x_hat = kf.x
    P_hat = kf.P

    # test predicted state & covariance match with expected values
    np.testing.assert_allclose(expected_state, x_hat)
    np.testing.assert_allclose(expected_covariance, P_hat)

    # test covariance's symmetry
    np.testing.assert_allclose(P_hat, P_hat.T)