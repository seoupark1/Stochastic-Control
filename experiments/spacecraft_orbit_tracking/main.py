import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative

from .desired_orbit import CircularOrbit
from .nadir_pointing import NadirPointingReference
from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.attitude.mrp import mrp_b_matrix, mrp_derivative, mrp_to_dcm
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.compensators.nonlinear_compensator.ekf_lqr import EKFLQRCompensator
from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter
from stochastic_control.noises.gaussian_noise import GaussianNoise
from stochastic_control.providers import TrajectoryReferenceProvider, MRPStateProvider, BodyStateContext

def simulation():

    dt = 0.1
    inertia_tensor = np.diag([0.2507, 0.2507, 0.0136])
    mu = 4.2828 * 10**13

    gravity_gradient = GravityGradient(inertia_tensor = inertia_tensor,
                                       gravitational_parameter = mu)
    
    spacecraft = RigidBody(inertia_tensor = inertia_tensor)

    orbit_provider = CircularOrbit(RAAN = np.deg2rad(20),
                                   inclination = np.deg2rad(30),
                                   initial_theta = np.pi / 6,
                                   radius = 3790 * 10**3,
                                   mu = mu)

    nadir_provider = NadirPointingReference(orbit_provider)

    def dynamics(t, state, control):

        # orbit properties
        r_N, v_N = orbit_provider.get_state(t)
        nadir = nadir_provider.nadir_pointing(t)
        sigma_BN = nadir[0:3]
        omega_BN_B = nadir[3:6]

        # gravity gradient disturbance
        body_state = BodyStateContext(position_N = r_N,
                                      velocity_N = v_N,
                                      dcm_BN = mrp_to_dcm(sigma_BN),
                                      angular_velocity_BN = omega_BN_B)

        disturbance = gravity_gradient.torque(t, body_state)

        # external torques
        total_torque = disturbance + control

        return spacecraft.mrp_derivatives(state, total_torque)

    def motion_model(t, state, control):
        return state + dt * dynamics(t, state, control)

    def motion_jacobian(t, state, control):

        A = approx_derivative(fun = lambda x: dynamics(t, x, control),
                              x0 = state,
                              method = '3-point')

        return np.eye(3) + dt * A

    def measurement_model(state):
        return state
    
    def measurement_jacobian(state):
        return np.eye(6)

    state = 
    covariance = 
    motion_noise_jacobian = np.eye(6)
    measurement_noise_jacobian = np.eye(6)
    motion_noise_covariance = 0.5 * np.eye(6)
    measurement_noise_covariance = 0.01 * np.eye(6)

    ekf = ExtendedKalmanFilter(state = state,
                               covariance = covariance,
                               motion_model = motion_model,
                               motion_jacobian = motion_jacobian,
                               motion_noise_jacobian = motion_noise_jacobian,
                               measurement_model = measurement_model,
                               measurement_jacobian = measurement_jacobian,
                               measurement_noise_jacobian = measurement_noise_jacobian,
                               motion_noise_covariance = motion_noise_covariance,
                               measurement_noise_covariance = measurement_noise_covariance)

    # reference trajectory (state, control)
    reference_x_function = nadir_provider.nadir_pointing
    reference_u_function = 
    reference_provider = TrajectoryReferenceProvider

    Q = 
    R = 
    Qf = 
    tf = 

    controller = LocalTrajectoryStabilizationLQRController(Q = ,
                                                           R = ,
                                                           Qf = ,
                                                           tf = ,
                                                           reference_provider = reference_provider,
                                                           dynamics_function = dynamics)


