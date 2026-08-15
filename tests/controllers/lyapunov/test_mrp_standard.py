import numpy as np
import pytest

from stochastic_control.controllers.lyapunov.standard import StandardLyapunovController
from stochastic_control.providers.body_state import MRPStateProvider
from stochastic_control.providers.reference_attitude import MRPReferenceProvider

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
    
    return StandardLyapunovController(inertia_tensor, control_gain, damping_matrix, fixed_reference_attitude_provider)

def test_tracking_error_is_0_at_equilibrium(standard_controller):

    estimated_rotational_state = np.zeros(6)
    controller = standard_controller

    dcm_BR, sigma_BR, omega_BR = controller.get_tracking_error(0, estimated_rotational_state)

    np.testing.assert_allclose(dcm_BR, np.eye(3))
    np.testing.assert_allclose(sigma_BR, np.zeros(3))
    np.testing.assert_allclose(omega_BR, np.zeros(3))

def test_control_vector_is_0_at_equilibrium(standard_controller, context_builder):

    estimated_rotational_state = np.zeros(6)
    controller = standard_controller
    estiamted_context_builder = context_builder

    control_vector = controller.control_vector(0, estimated_rotational_state, estiamted_context_builder)

    np.testing.assert_allclose(control_vector, np.zeros(3))