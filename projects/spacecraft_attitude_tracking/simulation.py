import numpy as np

from numpy.typing import ArrayLike
from scipy.optimize._numdiff import approx_derivative

from .references.circular_orbit import CircularOrbit
from .references.nadir_pointing import NadirPointingReference

from stochastic_control.math_tools import skew_symmetric
from stochastic_control.attitude.mrp import mrp_to_dcm, dcm_to_mrp, mrp_derivative, mrp_shadow_set
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.noises.gaussian_noise import GaussianNoise
from stochastic_control.providers import TrajectoryReferenceProvider, BodyStateContext
from stochastic_control.sensors import Gyroscope, StarTracker

from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter
from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.compensators.nonlinear_compensator.ekf_lqr import EKFLQRCompensator

def simulation(initial_x_true: ArrayLike,
               initial_x_hat : ArrayLike,
               initial_covariance : ArrayLike,
               motion_noise_covariance: ArrayLike,
               star_tracker_noise_covariance: ArrayLike,
               gyroscope_noise_covariance: ArrayLike,
               tf: float,
               star_tracker_sampling_rate: float,
               gyroscope_sampling_rate: float,
               control_limiter = None):

    # sampling rates
    dt = 0.01
    star_tracker_dt = 1 / star_tracker_sampling_rate
    gyroscope_dt = 1 / gyroscope_sampling_rate
    star_tracker_time = star_tracker_dt
    gyroscope_time = gyroscope_dt

    # spacecraft & planet properties
    inertia_tensor = np.diag([1448.3, 1346.2, 689.8])
    mu = 4.2828 * 10**13

    gravity_gradient = GravityGradient(inertia_tensor = inertia_tensor,
                                       gravitational_parameter = mu)

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
        reference_state = nadir_provider.get_state(t)

        sigma_RN = mrp_shadow_set(reference_state[0:3])
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

    def gravity_gradient_torque(t, state):
                
        # tracking error
        sigma_BR = mrp_shadow_set(state[0:3])
        omega_BR_B = state[3:6]

        # reference attitude & omega
        reference_state = nadir_provider.get_state(t)
        sigma_RN = mrp_shadow_set(reference_state[0:3])
        omega_RN_R = reference_state[3:6]

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

        return gravity_gradient.torque(t, body_state)

    # output : sigma_BR_dot, omega_BR_B_dot
    def dynamics(t, state, control):

        # tracking error
        sigma_BR = mrp_shadow_set(state[0:3])
        omega_BR_B = state[3:6]

        # reference attitude & omega
        reference_state = nadir_provider.get_state(t)
        omega_RN_R = reference_state[3:6]
        omega_RN_R_dot = nadir_provider.angular_acceleration(t)

        dcm_BR = mrp_to_dcm(sigma_BR)
        omega_BN_B = omega_BR_B + dcm_BR @ omega_RN_R

        total_torque = gravity_gradient_torque(t, state) + control

        omega_BN_B_dot = np.linalg.solve(inertia_tensor, total_torque - skew_symmetric(omega_BN_B) @ inertia_tensor @ omega_BN_B)

        # outputs
        sigma_BR_dot = mrp_derivative(sigma_BR, omega_BR_B)
        omega_BR_B_dot = omega_BN_B_dot + np.cross(omega_BR_B, dcm_BR @ omega_RN_R) - dcm_BR @ omega_RN_R_dot

        return np.concatenate((sigma_BR_dot, omega_BR_B_dot))

    def motion_model(t, state, control):

        # runge-kutta 4th order method
        k1 = dynamics(t, state, control)
        k2 = dynamics(t + dt / 2, state + dt * k1 / 2, control)
        k3 = dynamics(t + dt / 2, state + dt * k2 / 2, control)
        k4 = dynamics(t + dt, state + dt * k3, control)

        next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # shadow set transfer
        next_state[:3] = mrp_shadow_set(next_state[:3])

        return next_state

    def motion_jacobian(t, state, control):

        A = approx_derivative(fun = lambda x: dynamics(t, x, control),
                              x0 = state,
                              method = '3-point')

        return np.eye(6) + dt * A

    def measurement_model(t, state):

        sigma_BR = state[0:3]
        omega_BR_B = state[3:6]

        reference_state = nadir_provider.get_state(t)
        sigma_RN = reference_state[0:3]
        omega_RN_R = reference_state[3:6]

        dcm_BR = mrp_to_dcm(sigma_BR)
        dcm_RN = mrp_to_dcm(sigma_RN)
        dcm_BN = dcm_BR @ dcm_RN

        sigma_BN = mrp_shadow_set(dcm_to_mrp(dcm_BN))
        omega_BN_B = omega_BR_B + dcm_BR @ omega_RN_R

        return np.concatenate((sigma_BN, omega_BN_B))
    
    def measurement_jacobian(t, state):

        return approx_derivative(fun = lambda x: measurement_model(t, x),
                                 x0 = state,
                                 method = '3-point')

    # ekf properties
    motion_noise_jacobian = np.eye(6)
    measurement_noise_jacobian = np.eye(6)
    measurement_noise_covariance = np.block([[star_tracker_noise_covariance, np.zeros((3, 3))],
                                             [np.zeros((3, 3)), gyroscope_noise_covariance]])

    ekf = ExtendedKalmanFilter(state = initial_x_hat,
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
    x_true = initial_x_true

    lqr = LocalTrajectoryStabilizationLQRController(Q = Q,
                                                    R = R,
                                                    Qf = Qf,
                                                    tf = tf,
                                                    reference_provider = reference_provider,
                                                    dynamics_function = dynamics)

    compensator = EKFLQRCompensator(ekf, lqr)

    # noises & sensors
    rng = np.random.default_rng(seed = 2026)
    motion_noise_provider = GaussianNoise(np.zeros(6), motion_noise_covariance)
    star_tracker = StarTracker(np.zeros(3), star_tracker_noise_covariance)
    gyroscope = Gyroscope(np.zeros(3), gyroscope_noise_covariance)

    if star_tracker_sampling_rate > 1 / dt:
        raise ValueError(f'Star tracker sampling rate should be smaller than {float(1/dt)} Hz')

    if gyroscope_sampling_rate > 1 / dt:
        raise ValueError(f'Gyroscope sampling rate should be smaller than {float(1/dt)} Hz')

    # total steps
    total_step = int(tf / dt)
    time = dt * np.arange(total_step + 1)

    # state histories 
    true_state_history = np.zeros((6, total_step + 1))
    reference_state_history = np.zeros((6, total_step + 1))
    estimated_state_history = np.zeros((6, total_step + 1))
    true_gravity_gradient_history = np.zeros((3, total_step + 1))

    # state history at t = 0
    true_state_history[:, 0] = x_true
    reference_state_history[:, 0] = reference_x_function(0)
    estimated_state_history[:, 0] = ekf.x
    true_gravity_gradient_history[:, 0] = gravity_gradient_torque(0, x_true)

    # ekf covariance P history
    covariance_history = np.zeros((6, 6, total_step + 1))
    standard_deviation_history = np.zeros((6, total_step + 1))
    covariance_history[:, :, 0] = ekf.P

    # control & measurement histories
    cmd_control_history = np.zeros((3, total_step))
    actual_control_history = np.zeros((3, total_step))
    measurement_history = np.full((6, total_step), np.nan)
    star_tracker_correction_steps_history = np.zeros(total_step + 1)
    gyroscope_correction_steps_history = np.zeros(total_step + 1)

    # tracking error & estimation error histories
    attitude_tracking_error_norm_history = np.zeros(total_step + 1)
    attitude_estimation_error_angle_history = np.zeros(total_step + 1)
    omega_estimation_error_history = np.zeros((3, total_step + 1))

    # run simulation
    for k in range(total_step):

        t = k * dt
        next_t = time[k + 1]

        # control with saturation
        u_cmd = compensator.control_vector(t)

        if control_limiter is None:
            u_actual = u_cmd

        else: 
            u_actual = control_limiter.saturation(u_cmd)

        # true state propagation & mrp shadow set transfer
        x_true = motion_model(t, x_true, u_actual) + motion_noise_provider.get_sample(rng)
        x_true[0:3] = mrp_shadow_set(x_true[0:3])

        # prediction
        ekf.prediction(u_actual, t)

        # measure attitude & correction
        if next_t >= star_tracker_time:
        
            ideal_attitude = measurement_model(next_t, x_true)[0:3]
            y = star_tracker.measure(ideal_attitude, rng)
            measurement_history[0:3, k] = y

            # predicted model & jacobian
            predicted_star_tracker_measurement_model = measurement_model(next_t, ekf.x)[0:3]
            predicted_star_tracker_jacobian = measurement_jacobian(next_t, ekf.x)[0:3, :]

            ekf.correction(measurement_vector = y,
                           t = next_t,
                           measurement_model = predicted_star_tracker_measurement_model,
                           measurement_jacobian = predicted_star_tracker_jacobian,
                           measurement_noise_covariance = star_tracker_noise_covariance,
                           measurement_noise_jacobian = np.eye(3))
            
            ekf.x[0:3] = mrp_shadow_set(ekf.x[0:3])
            star_tracker_time += star_tracker_dt

            star_tracker_correction_steps_history[k + 1] = 1

        # measure omega & correction
        if next_t >= gyroscope_time:

            ideal_omega = measurement_model(next_t, x_true)[3:6]
            y = gyroscope.measure(ideal_omega, rng)
            measurement_history[3:6, k] = y

            # predicted model & jacobian
            predicted_gyroscope_measurement_model = measurement_model(next_t, ekf.x)[3:6]
            predicted_gyroscope_jacobian = measurement_jacobian(next_t, ekf.x)[3:6, :]

            ekf.correction(measurement_vector = y,
                           t = next_t,
                           measurement_model = predicted_gyroscope_measurement_model,
                           measurement_jacobian = predicted_gyroscope_jacobian,
                           measurement_noise_covariance = gyroscope_noise_covariance,
                           measurement_noise_jacobian = np.eye(3))

            ekf.x[0:3] = mrp_shadow_set(ekf.x[0:3])
            gyroscope_time += gyroscope_dt

            gyroscope_correction_steps_history[k + 1] = 1

        # update histories
        true_state_history[:, k + 1] = x_true
        reference_state_history[:, k + 1] = reference_x_function(next_t)
        estimated_state_history[:, k + 1] = ekf.x
        true_gravity_gradient_history[:, k + 1] = gravity_gradient_torque(next_t, x_true)
        covariance_history[:, :, k + 1] = ekf.P
        cmd_control_history[:, k] = u_cmd
        actual_control_history[:, k] = u_actual

    # compute tracking error & estimation error & standard deviation
    for k in range(total_step + 1):

        sigma_BR = true_state_history[0:3, k]
        omega_BR_B = true_state_history[3:6, k]

        sigma_BhatR = estimated_state_history[0:3, k]
        omega_BhatR_Bhat = estimated_state_history[3:6, k]

        # principal angle difference
        dcm_BR = mrp_to_dcm(sigma_BR)
        dcm_BhatR = mrp_to_dcm(sigma_BhatR)
        dcm_BhatB = dcm_BhatR @ dcm_BR.T
        sigma_BhatB = dcm_to_mrp(dcm_BhatB)

        angle_BR = 4 * np.arctan(np.linalg.norm(sigma_BR))
        angle_BhatB = 4 * np.arctan(np.linalg.norm(sigma_BhatB))

        # omega
        omega_BhatB_B = dcm_BhatB.T @ omega_BhatR_Bhat - omega_BR_B

        # attitude tracking error
        attitude_tracking_error_norm_history[k] = angle_BR

        # estimation error
        attitude_estimation_error_angle_history[k] = angle_BhatB
        omega_estimation_error_history[:, k] = omega_BhatB_B

        # standard deviation corresponding with covariance P
        standard_deviation_history[:, k] = np.sqrt(np.diag(covariance_history[:, :, k]))

    # omega estimation error norm
    omega_estimation_norm_error_history = np.linalg.norm(omega_estimation_error_history, axis = 0)

    # omega tracking error
    omega_tracking_error_norm_history = np.linalg.norm(true_state_history[3:6, :], axis = 0)

    # max commanded control
    max_u_cmd_abs = np.max(np.abs(cmd_control_history), axis = 1)       

    return {'time': time,
            'true_state': true_state_history,
            'reference_state': reference_state_history,
            'estimated_state': estimated_state_history,
            'true_gravity_gradient_torque' : true_gravity_gradient_history,
            'covariance_P' : covariance_history,
            'standard_deviation_P' : standard_deviation_history,
            'commanded_control': cmd_control_history,
            'max_abs_commanded_control': max_u_cmd_abs,
            'actual_control': actual_control_history,
            'measurement_y': measurement_history,
            'star_tracker_correction_steps': star_tracker_correction_steps_history,
            'gyroscope_correction_steps': gyroscope_correction_steps_history,
            'attitude_tracking_error': attitude_tracking_error_norm_history,
            'omega_tracking_error': omega_tracking_error_norm_history,
            'attitude_estimation_error': attitude_estimation_error_angle_history,
            'omega_estimation_error': omega_estimation_norm_error_history}