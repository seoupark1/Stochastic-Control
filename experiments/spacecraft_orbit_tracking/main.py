import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative

from .desired_orbit import CircularOrbit
from .nadir_pointing import NadirPointingReference

from stochastic_control.math_tools import skew_symmetric
from stochastic_control.attitude.mrp import mrp_to_dcm, dcm_to_mrp, mrp_derivative
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

    # reference state and control
    def reference_x_function(t):
        return np.zeros(6)

    def reference_u_function(t):

        r_N, v_N = orbit_provider.get_state(t)
        reference_state = nadir_provider.nadir_pointing(t)

        sigma_RN = reference_state[0:3]
        omega_RN_R = reference_state[3:6]
        omega_RN_R_dot = nadir_provider.angular_acceleration(t)

        reference_body_state = BodyStateContext(position_N = r_N,
                                                velocity_N = v_N,
                                                dcm_BN = mrp_to_dcm(sigma_RN),
                                                angular_velocity_BN = omega_RN_R)
        
        disturbance = gravity_gradient.torque(t, reference_body_state)

        control = inertia_tensor @ omega_RN_R_dot + skew_symmetric(omega_RN_R) @ inertia_tensor @ omega_RN_R - disturbance

        return control
    
    reference_provider = TrajectoryReferenceProvider(reference_x_function = reference_x_function, 
                                                     reference_u_function = reference_u_function)

    # output : sigma_BR_dot, omega_BR_B_dot
    def dynamics(t, state, control):

        # tracking error
        sigma_BR = state[0:3]
        omega_BR_B = state[3:6]

        # reference attitude & omega
        reference_state = nadir_provider.nadir_pointing(t)
        sigma_RN = reference_state[0:3]
        omega_RN_R = reference_state[3:6]
        omega_RN_R_dot = nadir_provider.angular_acceleration(t)

        dcm_BR = mrp_to_dcm(sigma_BR)
        dcm_RN = mrp_to_dcm(sigma_RN)

        # true attitude & omega
        dcm_BN = dcm_BR @ dcm_RN
        omega_BN_B = omega_BR_B + dcm_BR @ omega_RN_R

        r_N, v_N = orbit_provider.get_state(t)

        body_state = BodyStateContext(position_N = r_N,
                                      velocity_N = v_N,
                                      dcm_BN = dcm_BN,
                                      angular_velocity_BN = omega_BN_B)

        total_torque = gravity_gradient.torque(t, body_state) + control

        omega_BN_B_dot = np.linalg.solve(inertia_tensor, total_torque - skew_symmetric(omega_BN_B) @ inertia_tensor @ omega_BN_B)

        # outputs
        sigma_BR_dot = mrp_derivative(sigma_BR, omega_BR_B)
        omega_BR_B_dot = omega_BN_B_dot + np.cross(omega_BR_B, dcm_BR @ omega_RN_R) - dcm_BR @ omega_RN_R_dot

        return np.concatenate((sigma_BR_dot, omega_BR_B_dot))

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

    # ekf properties
    initial_state = reference_x_function(0) + np.array([-0.1, -0.2, -0.3, np.deg2rad(10), np.deg2rad(-7), np.deg2rad(5)])
    initial_covariance = np.eye(6)
    motion_noise_jacobian = np.eye(6)
    measurement_noise_jacobian = np.eye(6)
    motion_noise_covariance = np.diag([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6])
    measurement_noise_covariance = np.diag([1e-6, 1e-6, 1e-6, np.deg2rad(0.05)**2, np.deg2rad(0.05)**2, np.deg2rad(0.05)**2])

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
    tf = 10
    x_true = reference_x_function(0) + np.array([-0.1, -0.2, -0.3, np.deg2rad(10), np.deg2rad(-7), np.deg2rad(5)])

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
    time = dt * np.arange(total_step + 1)

    # histories 
    true_state_history = np.zeros((6, total_step + 1))
    reference_state_history = np.zeros((6, total_step + 1))
    estimated_state_history = np.zeros((6, total_step + 1))
    control_history = np.zeros((3, total_step))
    measurement_history = np.zeros((6, total_step))

    attitude_norm_error_history = np.zeros(total_step + 1)
    attitude_estimation_error_angle_history = np.zeros(total_step + 1)
    omega_estimation_error_history = np.zeros((3, total_step + 1))

    # history at t = 0
    true_state_history[:, 0] = x_true
    reference_state_history[:, 0] = reference_x_function(0)
    estimated_state_history[:, 0] = ekf.x

    # run simulation
    for k in range(total_step):

        # time
        t = k * dt

        # control
        u_cmd = compensator.control_vector(t)
        control_history[:, k] = u_cmd

        # true value
        x_true = motion_model(t, x_true, u_cmd) + motion_noise.get_sample(rng)
        reference_state_history[:, k + 1] = reference_x_function(time[k + 1])
        true_state_history[:, k + 1] = x_true

        # measure
        y = measurement_model(x_true) + measurement_noise.get_sample(rng)
        measurement_history[:, k] = y

        # estimate
        compensator.estimate(t, u_cmd, y)
        estimated_state_history[:, k + 1] = ekf.x

    # compute tracking & estimation error
    for k in range(total_step + 1):

        sigma_BR = true_state_history[0:3, k]
        omega_BR_B = true_state_history[3:6, k]

        sigma_BhatR = estimated_state_history[0:3, k]
        omega_BhatR_B = estimated_state_history[3:6, k]

        # angle difference
        dcm_BR = mrp_to_dcm(sigma_BR)
        dcm_BhatR = mrp_to_dcm(sigma_BhatR)
        dcm_BhatB = dcm_BhatR @ dcm_BR.T
        sigma_BhatB = dcm_to_mrp(dcm_BhatB)

        angle_BR = 4 * np.arctan(np.linalg.norm(sigma_BR))
        angle_BhatB = 4 * np.arctan(np.linalg.norm(sigma_BhatB))

        # omega
        omega_Bhat_B = omega_BhatR_B - omega_BR_B

        # attitude tracking error
        attitude_norm_error_history[k] = angle_BR

        # estimation error
        attitude_estimation_error_angle_history[k] = angle_BhatB
        omega_estimation_error_history[:, k] = omega_Bhat_B

    # omega estimation norm error
    omega_estimation_norm_error_history = np.linalg.norm(omega_estimation_error_history, axis = 0)

    # omega tracking error
    omega_norm_error_history = np.linalg.norm(true_state_history[3:6, :], axis = 0)

    # tracking error
    plt.subplot(2, 1, 1)
    plt.plot(time, attitude_norm_error_history)
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude Error [rad]')
    plt.title('Nadir Pointing Attitude Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, omega_norm_error_history)
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity Error [deg/s]')
    plt.title('Nadir Pointing Angular Velocity Tracking Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/tracking_error.png')
    plt.close()

    # estimation error
    plt.subplot(2, 1, 1)
    plt.plot(time, attitude_estimation_error_angle_history)
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude Error [rad]')
    plt.title('Nadir Pointing Attitude Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, omega_estimation_norm_error_history)
    plt.xlabel('time [s]')
    plt.ylabel('Angular Velocity Error [rad/s]')
    plt.title('Nadir Pointing Omegs Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/estimation_error.png')
    plt.close()

    # control effort
    plt.subplot(2, 1, 1)
    plt.plot(time[0:-1], control_history[0], label = 'u_1')
    plt.plot(time[0:-1], control_history[1], label = 'u_2')
    plt.plot(time[0:-1], control_history[2], label = 'u_3')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Nadir Pointing Control Effort')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time[0:-1], np.linalg.norm(control_history, axis = 0))
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Nadir Pointing Control Effort Norm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/spacecraft_orbit_tracking/results/control_effort.png')
    plt.close()

simulation()