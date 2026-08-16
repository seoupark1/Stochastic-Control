import numpy as np
import pytest

from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.providers.body_state import BodyStateContext
from stochastic_control.attitude.mrp import mrp_to_dcm

@pytest.fixture
def planet_information():
    mu = 3.986e14
    radius = 6371e3
    altitude = 500e3

    return mu, radius, altitude

@pytest.fixture
def circular_orbit_position(planet_information):

    mu, radius, altitude = planet_information
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
def test_torque_is_0_if_inertia_moments_are_equal(planet_information, circular_orbit_position, t):

    # gravity gradient
    mu, _, _ = planet_information
    inertia_tensor = np.diag([1, 1, 1])
    gravitygradient = GravityGradient(inertia_tensor, mu)

    # body state
    position_N = circular_orbit_position(t)
    dcm_BN = np.eye(3)
    bodystate = BodyStateContext(position_N = position_N,
                                 velocity_N = None,
                                 dcm_BN = dcm_BN,
                                 angular_velocity_BN = None)

    torque = gravitygradient.torque(t, bodystate)
    np.testing.assert_allclose(torque, np.zeros(3))

@pytest.mark.parametrize("t", [10, 20, 30, 40, 50, 60, 70, 80, 90])
def test_computed_torque_is_correct(planet_information, circular_orbit_position, body_attitude, t):

    # gravity gradient
    mu, _, _ = planet_information
    inertia_tensor = np.diag([10, 20, 30])
    gravitygradient = GravityGradient(inertia_tensor, mu)

    # body state
    position_N = circular_orbit_position(t)
    dcm_BN = body_attitude(t)
    bodystate = BodyStateContext(position_N = position_N,
                                 velocity_N = None,
                                 dcm_BN = dcm_BN,
                                 angular_velocity_BN = None)

    torque = gravitygradient.torque(t, bodystate)


    r_BN_B = dcm_BN @ position_N
    r = np.linalg.norm(position_N)
    expected_torque = (3 * mu / r**5) * np.cross(r_BN_B, inertia_tensor @ r_BN_B)

    np.testing.assert_allclose(torque, expected_torque)