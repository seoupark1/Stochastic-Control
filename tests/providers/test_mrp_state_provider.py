import numpy as np
import pytest

from stochastic_control.providers.body_state import MRPStateProvider
from stochastic_control.attitude.mrp import mrp_b_matrix, mrp_to_dcm

def position_function(t):
    return np.array([t,
                     2 * t, 
                     3 * t])

def sigma_BN(t):
    return np.array([0.2 * np.sin(t), 
                    0.3 * np.cos(t), 
                    -0.3 * np.sin(t)])

def sigma_BN_dot(t):
    return np.array([0.2 * np.cos(t),
                    -0.3 * np.sin(t),
                    -0.3 * np.cos(t)])

def omega_BN_B(t):
    sigma = sigma_BN(t)
    sigma_dot = sigma_BN_dot(t)
    return np.linalg.solve(mrp_b_matrix(sigma), 4 * sigma_dot)


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_mrp_state_provider(t: float):

    provider = MRPStateProvider(position_function = position_function,
                                velocity_function = None)

    expected_sigma = sigma_BN(t)
    expected_omega = omega_BN_B(t)

    rotational_state = np.concatenate((expected_sigma, expected_omega))
    context = provider.build_context(t, rotational_state)

    assert context.position_N is not None
    assert context.velocity_N is None
    assert context.dcm_BN is not None
    assert context.angular_velocity_BN is not None

    np.testing.assert_allclose(context.position_N, position_function(t))
    np.testing.assert_allclose(context.dcm_BN, mrp_to_dcm(expected_sigma))
    np.testing.assert_allclose(context.angular_velocity_BN, expected_omega)