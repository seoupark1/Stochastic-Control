import numpy as np
import pytest

from stochastic_control.controllers.lyapunov import LyapunovController
from stochastic_control.providers.state import MRPStateProvider
from stochastic_control.providers.attitude_reference import MRPReferenceProvider

@pytest.fixture
def inertia_tensor():
    return np.diag([100, 75, 80])

@pytest.fixture
def control_gain():
    return 5

@pytest.fixture
def damping_matrix():
    return 10 * np.eye(3)

@pytest.fixture
def integral_control_gain():
    return 0.005

'''@pytest.fixture
def unknown_disturbance():
    return np.array([0.5, -0.3, 0.2]) '''

@pytest.fixture
def fixed_reference_attitude_provider():
    return MRPReferenceProvider(sigma_function = lambda t: np.zeros(3), 
                                omega_function = lambda t: np.zeros(3), 
                                omega_dot_function = lambda t: np.zeros(3))

@pytest.fixture
def context_builder():
    return MRPStateProvider()

@pytest.fixture
def standard_controller(inertia_tensor,
                        control_gain,
                        damping_matrix,
                        fixed_reference_attitude_provider):
    
    return LyapunovController(inertia_tensor, control_gain, damping_matrix, fixed_reference_attitude_provider)

@pytest.fixture
def integral_controller(inertia_tensor,
                        control_gain,
                        damping_matrix,
                        fixed_reference_attitude_provider,
                        integral_control_gain):
    
    return LyapunovController(inertia_tensor, control_gain, damping_matrix, fixed_reference_attitude_provider, integral_control_gain)


def test_standard_tracking_error_is_0_at_equilibrium(standard_controller):

    estimated_rotational_state = np.zeros(6)
    controller = standard_controller

    dcm_BR, sigma_BR, omega_BR = controller.mrp_tracking_error(0, estimated_rotational_state)

    np.testing.assert_allclose(dcm_BR, np.eye(3))
    np.testing.assert_allclose(sigma_BR, np.zeros(3))
    np.testing.assert_allclose(omega_BR, np.zeros(3))

def test_standard_control_vector_is_0_at_equilibrium(standard_controller, context_builder):

    estimated_rotational_state = np.zeros(6)
    controller = standard_controller
    estiamted_context_builder = context_builder

    torque = controller.mrp_control_vector(0, estimated_rotational_state, estiamted_context_builder)

    np.testing.assert_allclose(torque, np.zeros(3))

def test_integral_tracking_error_is_0_at_equilibrium(integral_controller):

    estimated_rotational_state = np.zeros(6)
    controller = integral_controller

    dcm_BR, sigma_BR, omega_BR = controller.mrp_tracking_error(0, estimated_rotational_state)

    np.testing.assert_allclose(dcm_BR, np.eye(3))
    np.testing.assert_allclose(sigma_BR, np.zeros(3))
    np.testing.assert_allclose(omega_BR, np.zeros(3))

def test_integral_control_vector_is_0_at_equilibrium(integral_controller, context_builder):

    estimated_rotational_state = np.zeros(6)
    controller = integral_controller
    estiamted_context_builder = context_builder

    torque = controller.mrp_control_vector(0, estimated_rotational_state, estiamted_context_builder)

    np.testing.assert_allclose(torque, np.zeros(3))