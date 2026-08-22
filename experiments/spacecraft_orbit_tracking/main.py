import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative

from .desired_orbit import CircularOrbit
from .nadir_pointing import NadirPointingReference

from stochastic_control.math_tools import skew_symmetric
from stochastic_control.attitude.mrp import mrp_to_dcm, dcm_to_mrp
from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.noises.gaussian_noise import GaussianNoise
from stochastic_control.providers import TrajectoryReferenceProvider, BodyStateContext
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

        return np.eye(6) + dt * A

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
    motion_noise_covariance = np.diag([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6])
    measurement_noise_covariance = np.diag([1e-6, 1e-6, 1e-6, np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05)])

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
    tf = 50
    x_true = reference_x_function(0) + np.array([-0.1, -0.2, -0.3, 0.1, 0.2, 0.3])

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

    # total steps
    total_step = int(tf / dt)

    # histories 
    time = np.zeros(total_step + 1)
    true_state_history = np.zeros((6, total_step + 1))
    reference_state_history = np.zeros((6, total_step + 1))
    estimated_state_history = np.zeros((6, total_step + 1))
    control_history = np.zeros((3, total_step))
    measurement_history = np.zeros((6, total_step))
    attitude_error_angle_history = np.zeros(total_step + 1)
    attitude_estimation_error_angle_history = np.zeros(total_step + 1)

    # history at t = 0
    true_state_history[:, 0] = x_true
    reference_state_history[:, 0] = reference_x_function(0)
    estimated_state_history[:, 0] = ekf.x

    # run simulation
    for k in range(total_step):

        # time
        t = k * dt
        time[k] = t

        # control
        u_cmd = compensator.control_vector(t)
        control_history[:, k] = u_cmd

        # true value
        x_true = motion_model(t, x_true, u_cmd) + motion_noise.get_sample(rng)
        reference_state_history[:, k + 1] = reference_x_function(t)
        true_state_history[:, k + 1] = x_true

        # measure
        y = measurement_model(x_true) + measurement_noise.get_sample(rng)
        measurement_history[:, k] = y

        # estimate
        compensator.estimate(t, u_cmd, y)
        estimated_state_history[:, k + 1] = ekf.x

    # attitude errors
    for k in range(total_step + 1):

        sigma_BN = true_state_history[0:3, k]
        sigma_RN = reference_state_history[0:3, k]
        sigma_BbarN = estimated_state_history[0:3, k]

        # tracking error
        dcm_BR = mrp_to_dcm(sigma_BN) @ mrp_to_dcm(sigma_RN).T
        sigma_BR = dcm_to_mrp(dcm_BR)
        attitude_error_angle_history[k] = np.rad2deg(4 * np.arctan(np.linalg.norm(sigma_BR)))

        # estimation error
        dcm_BbarB =  mrp_to_dcm(sigma_BbarN) @ mrp_to_dcm(sigma_BN).T
        sigma_BbarB = dcm_to_mrp(dcm_BbarB)
        attitude_estimation_error_angle_history[k] = np.rad2deg(4 * np.arctan(np.linalg.norm(sigma_BbarB)))

    # angular velocity errors
    omega_error_history = true_state_history[3:6, :] - reference_state_history[3:6, :]
    omega_norm_error_history = np.linalg.norm(omega_error_history, axis = 0)
    omega_estimation_error_history = estimated_state_history[3:6, :] - true_state_history[3:6, :]
    omega_estimation_norm_error_history = np.linalg.norm(omega_estimation_error_history, axis = 0)
 
    # reference attitude vs true attitude vs estimated attitude
    plt.subplot(3, 1, 1)
    plt.plot(time, reference_state_history[0, :], label = 'reference')
    plt.plot(time, true_state_history[0, :], label = 'true')
    plt.plot(time, estimated_state_history[0, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_1')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time, reference_state_history[1, :], label = 'reference')
    plt.plot(time, true_state_history[1, :], label = 'true')
    plt.plot(time, estimated_state_history[1, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_2')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time, reference_state_history[2, :], label = 'reference')
    plt.plot(time, true_state_history[2, :], label = 'true')
    plt.plot(time, estimated_state_history[2, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('sigma_3')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Nadir Pointing Attitude Comparison')
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/reference_vs_true_vs_estimated_attitude.png')
    plt.close()

    # reference omega vs true omega vs estimated omega
    plt.subplot(3, 1, 1)
    plt.plot(time, reference_state_history[3, :], label = 'reference')
    plt.plot(time, true_state_history[3, :], label = 'true')
    plt.plot(time, estimated_state_history[3, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_1')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time, reference_state_history[4, :], label = 'reference')
    plt.plot(time, true_state_history[4, :], label = 'true')
    plt.plot(time, estimated_state_history[4, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_2')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time, reference_state_history[5, :], label = 'reference')
    plt.plot(time, true_state_history[5, :], label = 'true')
    plt.plot(time, estimated_state_history[5, :], label = 'estimated')
    plt.xlabel('time [s]')
    plt.ylabel('omega_3')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Nadir Pointing Angular Velocity Comparison')
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/reference_vs_true_vs_estimated_omega.png')
    plt.close()

    # tracking errors
    plt.subplot(2, 1, 1)
    plt.plot(time, attitude_error_angle_history, label = 'true - reference')
    plt.xlabel('time [s]')
    plt.ylabel('attitude error [deg]')
    plt.title('Nadir Pointing Attitude Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, omega_norm_error_history, label = 'true - reference')
    plt.xlabel('time [s]')
    plt.ylabel('angular velocity error [deg/s]')
    plt.title('Nadir Pointing Angular Velocity Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/tracking_error.png')
    plt.close()

    # attitude estimation error
    plt.subplot(2, 1, 1)
    plt.plot(time, attitude_estimation_error_angle_history, label = 'estimated - true')
    plt.xlabel('time [s]')
    plt.ylabel('attitude error [deg]')
    plt.title('Nadir Pointing Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, omega_estimation_norm_error_history, label = 'estimated - true')
    plt.xlabel('time [s]')
    plt.ylabel('angular velocity error [deg/s]')
    plt.title('Nadir Pointing Angular Velocity Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/spacecraft_orbit_tracking/results/estimation_error.png')
    plt.close()

    # control 
    plt.subplot(2, 1, 1)
    plt.plot(time[0:-1], control_history[0], label = 'u_1')
    plt.plot(time[0:-1], control_history[1], label = 'u_2')
    plt.plot(time[0:-1], control_history[2], label = 'u_3')
    plt.xlabel('time [s]')
    plt.ylabel('torque [Nm]')
    plt.title('Nadir Pointing Control')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time[0:-1], np.linalg.norm(control_history, axis = 0))
    plt.xlabel('time [s]')
    plt.ylabel('torque [Nm]')
    plt.title('Nadir Pointing Control Norm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/ekf_lqr_compensator/drag_acceleration/control.png')
    plt.close()

simulation()