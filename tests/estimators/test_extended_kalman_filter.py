import numpy as np

from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter

# 1D motion of a car which collects the angle difference between the motion direction and the top of the building
def test_prediction():

    # conditions
    dt = 0.1 # [s]
    u_0 = np.array([-2]) # [m/s^2]
    y_1 = np.pi / 6 # [rad]
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

    # motion jacobian
    def F(state, control):
        return np.array([[1, dt],
                         [0, 1]])

    # measurement model
    def h(state):
        position = state[0]
        v = 0
        return np.arctan2(D - position, S) + v

    # measurement jacobian
    def H(state):
        position = state[0]
        return np.array([S / ((D - position)**2 + S**2), 0]).reshape(1,2)

    # noise covariances
    Q = 0.1 * np.eye(2)
    R = np.array([0.01])

    # noise jacobians
    L = np.eye(2)
    M = np.array([1])

    ekf = ExtendedKalmanFilter(state = x_0_hat,
                               covariance = P_0_hat,
                               motion_model = f,
                               motion_jacobian = F,
                               motion_noise_jacobian = L,
                               measurement_model = h,
                               measurement_jacobian = H,
                               measurement_noise_jacobian = M,
                               motion_noise_covariance = Q,
                               measurement_noise_covariance = R)

    # current F jacobian
    expected_F = ekf.F_jacobian(ekf.x, u_0)

    # expected values
    expected_state = ekf.f_model(ekf.x, u_0)
    expected_covariance = expected_F @ ekf.P @ expected_F.T + ekf.L_jacobian @ ekf.Q @ ekf.L_jacobian.T
    expected_covariance = (expected_covariance + expected_covariance.T) / 2

    # predicted values
    ekf.prediction(u_0)
    x_check = ekf.x
    P_check = ekf.P

    # test predicted state & covariance match with expected values
    np.testing.assert_allclose(expected_state, x_check)
    np.testing.assert_allclose(expected_covariance, P_check)

    # test covariance's symmetry
    np.testing.assert_allclose(P_check, P_check.T)