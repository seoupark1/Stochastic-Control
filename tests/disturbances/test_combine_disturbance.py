import numpy as np
import pytest

from stochastic_control.disturbances.combine_disturbances import CombineDisturbances
from stochastic_control.disturbances.constant_disturbance import ConstantDisturbance
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.providers.body_state import BodyStateContext
from stochastic_control.attitude.mrp import mrp_to_dcm

@pytest.fixture
def torque_1():
    return np.array([4, 23, 89])

@pytest.fixture
def torque_2():
    return np.array([546, 32, 97])

@pytest.fixture
def torque_3():
    inertia_tensor = np.diag([10, 20, 30])
    mu = 3.986e14

    return GravityGradient(inertia_tensor, mu)

@pytest.fixture
def body_position():
    mu, radius, altitude = 3.986e14, 6371e3, 500e3
    actual_radius = radius + altitude
    n = np.sqrt(mu / actual_radius**3)

    def position_N(t):
        return actual_radius * np.array([np.cos(n*t), np.sin(n*t), 0])

    return position_N

@pytest.fixture
def body_attitude():

    def dcm_BN(t):
        sigma_BN = np.array([0.2 * np.sin(t), 
                             0.3 * np.cos(t), 
                             -0.3 * np.sin(t)])
        
        return mrp_to_dcm(sigma_BN)
    
    return dcm_BN

@pytest.mark.parametrize("t", [10, 20, 30, 40, 50, 60, 70, 80, 90])
def test_combined_disturbance_is_correct(torque_1, torque_2, torque_3, body_attitude, body_position, t):

    # body state context
    dcm_BN = body_attitude(t)
    position_N = body_position(t)
    bodystate = BodyStateContext(position_N = position_N,
                                 velocity_N = None,
                                 dcm_BN = dcm_BN,
                                 angular_velocity_BN = None)

    # disturbances
    torque_1 = ConstantDisturbance(torque_1)
    torque_2 = ConstantDisturbance(torque_2)

    # combine torques
    models = [torque_1, torque_2, torque_3]
    combined_context = CombineDisturbances(models)
    combined_torque = combined_context.torque(t, bodystate)

    # assert
    mu, inertia_tensor = 3.986e14, np.diag([10, 20, 30])
    r_BN_B = dcm_BN @ position_N
    r = np.linalg.norm(position_N)
    expected_gravity_gradient = (3 * mu / r**5) * np.cross(r_BN_B, inertia_tensor @ r_BN_B)

    expected_torque = np.array([4, 23, 89]) + np.array([546, 32, 97]) + expected_gravity_gradient

    np.testing.assert_allclose(combined_torque, expected_torque)