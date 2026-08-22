import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative

from .desired_orbit import CircularOrbit
from .nadir_pointing import NadirPointingReference

from stochastic_control.math_tools import skew_symmetric
from stochastic_control.attitude.mrp import mrp_to_dcm
from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.noises.gaussian_noise import GaussianNoise
from stochastic_control.providers import TrajectoryReferenceProvider, MRPStateProvider, BodyStateContext
from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter
from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.compensators.nonlinear_compensator.ekf_lqr import EKFLQRCompensator

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

        # orbit property
        r_N, v_N = orbit_provider.get_state(t)

        sigma_BN = state[0:3]
        omega_BN_B = state[3:6]

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

    # reference trajectory (state, control)
    reference_x_function = nadir_provider.nadir_pointing

    def reference_u_function(t):

        r_N, v_N = orbit_provider.get_state(t)
        reference_state = nadir_provider.nadir_pointing(t)

        sigma_RN = reference_state[0:3]
        omega_RN_R = reference_state[3:6]
        omega_RN_R_dot = (-2) * np.dot(r_N, v_N) / np.dot(r_N, r_N) * omega_RN_R

        # gravity gradient disturbance
        body_state = BodyStateContext(position_N = r_N,
                                      velocity_N = v_N,
                                      dcm_BN = mrp_to_dcm(sigma_RN),
                                      angular_velocity_BN = omega_RN_R)

        disturbance = gravity_gradient.torque(t, body_state)

        control = inertia_tensor @ omega_RN_R_dot + skew_symmetric(omega_RN_R) @ inertia_tensor @ omega_RN_R - disturbance

        return control
    
    reference_provider = TrajectoryReferenceProvider(reference_x_function = reference_x_function, 
                                                     reference_u_function = reference_u_function)

    # ekf properties
    initial_state = reference_x_function(0) + np.array([-0.1, -0.2, -0.3, 0.1, 0.2, 0.3])
    initial_covariance = np.eye(6)
    motion_noise_jacobian = np.eye(6)
    measurement_noise_jacobian = np.eye(6)
    motion_noise_covariance = 0.5 * np.eye(6)
    measurement_noise_covariance = 0.01 * np.eye(6)

    ekf = ExtendedKalmanFilter(state = initial_state,
                               covariance = initial_covariance,
                               motion_model = motion_model,
                               motion_jacobian = motion_jacobian,
                               motion_noise_jacobian = motion_noise_jacobian,
                               measurement_model = measurement_model,
                               measurement_jacobian = measurement_jacobian,
                               measurement_noise_jacobian = measurement_noise_jacobian,
                               motion_noise_covariance = motion_noise_covariance,
                               measurement_noise_covariance = measurement_noise_covariance)

    Q = np.diag([50, 50, 50, 10, 10, 10])
    R = 5 * np.eye(3)
    Qf = 10 * Q
    tf = 120

    lqr = LocalTrajectoryStabilizationLQRController(Q = Q,
                                                    R = R,
                                                    Qf = Qf,
                                                    tf = tf,
                                                    reference_provider = reference_provider,
                                                    dynamics_function = dynamics)

    compensator = EKFLQRCompensator(ekf, lqr)

    # measurement noises
    rng = np.random.default_rng(seed = 2026)
    motion_noise = GaussianNoise(np.zeros(6), motion_noise_covariance)
    measurement_noise = GaussianNoise(np.zeros(6), measurement_noise_covariance)

    # run compensator
    total_step = int(tf / dt)
    time = np.zeros(total_step)
    true_state_history = np.zeros((6, total_step))
    reference_state_history = np.zeros((6, total_step))
    estimated_state_history = np.zeros((6, total_step))
    control_history = np.zeros((3, total_step))
    measurement_history = np.zeros((6, total_step))

    for k in range(total_step):

        # time
        t = k * dt
        time[k] = t

        # control
        u_cmd = compensator.control_vector(t)
        control_history[:, k] = u_cmd

        # true values
        x_true = motion_model(t, x_true, u_cmd) + motion_noise.get_sample(rng)
        reference_state_history[:, k] = reference_x_function(t)
        true_state_history[:, k] = x_true

        y = measurement_model(x_true) + measurement_noise.get_sample(rng)
        measurement_history[:, k] = y

        # estimate
        compensator.estimate(u_cmd, y)
        estimated_state_history[:, k] = ekf.x

    state_error_history = estimated_state_history - true_state_history
 
    # reference attitude vs true attitude vs estimated attitude
    plt.subplot(3, 1, 1)
    plt.plot(time, reference_state_history[0, :], label = 'reference')
    plt.plot(time, true_state_history[0, :], label = 'true')
    plt.plot(time, estimated_state_history[0, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_1')
    plt.title('Attitude(sigma_1) Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time, reference_state_history[1, :], label = 'reference')
    plt.plot(time, true_state_history[1, :], label = 'true')
    plt.plot(time, estimated_state_history[1, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_2')
    plt.title('Attitude(sigma_2) Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time, reference_state_history[2, :], label = 'reference')
    plt.plot(time, true_state_history[2, :], label = 'true')
    plt.plot(time, estimated_state_history[2, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_3')
    plt.title('Attitude(sigma_3) Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/reference_vs_true_vs_estimated_attitude.png')
    plt.close()

    # reference attitude vs true attitude vs estimated attitude
    plt.subplot(3, 1, 1)
    plt.plot(time, reference_state_history[3, :], label = 'reference')
    plt.plot(time, true_state_history[3, :], label = 'true')
    plt.plot(time, estimated_state_history[3, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_1')
    plt.title('Angular Velocity (omega_1) Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time, reference_state_history[4, :], label = 'reference')
    plt.plot(time, true_state_history[4, :], label = 'true')
    plt.plot(time, estimated_state_history[4, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_2')
    plt.title('Angular Velocity (omega_2) Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time, reference_state_history[5, :], label = 'reference')
    plt.plot(time, true_state_history[5, :], label = 'true')
    plt.plot(time, estimated_state_history[5, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_3')
    plt.title('Angular Velocity (omega_3) Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/reference_vs_true_vs_estimated_angular_velocity.png')
    plt.close()
'''
    # estimation error
    plt.subplot(2, 1, 1)
    plt.plot(time, state_error_history[0, :])
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.title('Position Error (estimated - true)')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, state_error_history[1, :])
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.title('Velocity Error (estimated - true)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/ekf_lqr_compensator/drag_acceleration/estimation_error.png')
    plt.close()

    # measurement y & control u
    plt.subplot(2, 1, 1)
    plt.plot(time, measurement_history)
    plt.xlabel('time [s]')
    plt.ylabel('measured angle [rad]')
    plt.title('Measurement History')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, control_history)
    plt.xlabel('time [s]')
    plt.ylabel('acceleration [m/s^2]')
    plt.title('Control History')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/ekf_lqr_compensator/drag_acceleration/measurement_and_control.png')
    plt.close()
'''
simulation()