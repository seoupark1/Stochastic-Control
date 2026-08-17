import numpy as np
import pytest

from stochastic_control.controllers.lqr.optimal_tracking import LinearQuadraticOptimalTrackingController
from stochastic_control.providers.reference_trajectory import TrajectoryReferenceProvider

@pytest.fixture
def reference_provider():

    def reference_x_function(t):
        position = (1/3) * t**3 + 4 * t**2 + 5 * t + 6
        velocity = t**2 + 8 * t + 5

        return np.array([position, velocity])

    def reference_u_function(t):
        u_d = 2 * t + 8

        return np.array([u_d])
    
    return TrajectoryReferenceProvider(reference_x_function, reference_u_function)

def test_get_Sxx_and_Sx_method(reference_provider):

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.diag([10, 1])
    tf = 50

    controller = LinearQuadraticOptimalTrackingController(A = A,
                                                          B = B,
                                                          Q = Q,
                                                          R = R,
                                                          Qf = Qf,
                                                          tf = tf,
                                                          reference_provider = reference_provider)

    # actual values at t = tf
    Sxx, Sx = controller.get_Sxx_and_Sx(tf)

    reference_tf = reference_provider.get_reference(tf)
    x_d_tf = reference_tf.reference_x

    # expected Sxx & Sx at t = tf
    expected_Sxx = Qf
    expected_Sx = - Qf @ x_d_tf

    # test actual values and expected values are equal at t = tf
    np.testing.assert_allclose(Sxx, expected_Sxx)
    np.testing.assert_allclose(Sx, expected_Sx)

@pytest.mark.parametrize('t', [5, 10, 15, 20, 25, 30, 35, 40])
def test_control_vector_method(reference_provider, t):

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.diag([10, 1])
    tf = 50

    # actual value
    controller = LinearQuadraticOptimalTrackingController(A = A,
                                                          B = B,
                                                          Q = Q,
                                                          R = R,
                                                          Qf = Qf,
                                                          tf = tf,
                                                          reference_provider = reference_provider)
    
    reference = reference_provider.get_reference(t)
    x_d = reference.reference_x
    u_d = reference.reference_u

    Sxx, Sx = controller.get_Sxx_and_Sx(t)
    control_vector = controller.control_vector(t, x_d)

    # expected value
    expected_u = u_d - np.linalg.inv(controller.R) @ controller.B.T @ (Sxx @ x_d + Sx)

    # test the actual u and the expected u is the same
    np.testing.assert_allclose(control_vector, expected_u)