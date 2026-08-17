import numpy as np
import pytest

from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.providers.reference_trajectory import TrajectoryReferenceProvider

# 2D pendulum rotation example
@pytest.fixture
def reference_provider():
    # conditions
    m = 3 # [kg]
    l = 5 # [m]
    g = 9.80665 # [m/s^2]

    def reference_x_function(t):
        theta = 0.3 * np.sin(t)
        omega = 0.3 * np.cos(t)

        return np.array([theta, omega])

    def reference_u_function(t):
        theta = 0.3 * np.sin(t)
        theta_ddot = -0.3 * np.sin(t)

        u_d = m * (l**2) * theta_ddot + m * g * l * np.sin(theta)

        return np.array([u_d])
    
    return TrajectoryReferenceProvider(reference_x_function, reference_u_function)

@pytest.fixture
def dynamics_function():
    # conditions
    m = 3 # [kg]
    l = 5 # [m]
    g = 9.80665 # [m/s^2]

    def dynamics(x, u):
        theta, omega = x

        return np.array([omega, -g * np.sin(theta) / l + u[0] / (m * l**2)])

    return dynamics

@pytest.mark.parametrize('t', [5, 10, 15, 20, 25, 30, 35, 40])
def test_expected_jacobians_and_the_result_are_the_same(reference_provider, dynamics_function, t):
    # conditions
    m = 3 # [kg]
    l = 5 # [m]
    g = 9.80665 # [m/s^2]

    Q = np.diag([5, 1])
    R = np.eye(1)
    Qf = np.diag([10, 1])
    tf = 50

    # actual values
    controller = LocalTrajectoryStabilizationLQRController(Q = Q,
                                                           R = R,
                                                           Qf = Qf,
                                                           tf = tf,
                                                           reference_provider = reference_provider,
                                                           dynamics_function = dynamics_function)
    A, B = controller.get_jacobians(t)

    # expected values
    reference = reference_provider.get_reference(t)
    x_d = reference.reference_x
    theta = x_d[0]

    expected_A = np.array([[0, 1],
                           [-g * np.cos(theta) / l, 0]])
    expected_B = np.array([[0],
                           [1 / (m * l**2)]])

    # test the actual jacobians and the expected jacobians are the same
    np.testing.assert_allclose(A, expected_A)
    np.testing.assert_allclose(B, expected_B)

def test_