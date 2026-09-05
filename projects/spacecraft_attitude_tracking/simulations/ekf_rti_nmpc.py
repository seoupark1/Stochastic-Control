import numpy as np

from numpy.typing import ArrayLike
from scipy.optimize._numdiff import approx_derivative

from ..references.circular_orbit import CircularOrbit
from ..references.nadir_pointing import NadirPointingReference

from stochastic_control.math_tools import skew_symmetric
from stochastic_control.attitude.mrp import mrp_to_dcm, dcm_to_mrp, mrp_derivative, mrp_shadow_set, mrp_to_rotation_vector, mrp_b_matrix
from stochastic_control.disturbances.gravity_gradient import GravityGradient
from stochastic_control.noises.gaussian_noise import GaussianNoise
from stochastic_control.providers import BodyStateContext
from stochastic_control.sensors import Gyroscope, StarTracker

from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter
from stochastic_control.controllers.mpc.rti_nmpc import RealTimeNMPCController
####################################################################
from time import perf_counter
from collections import Counter
####################################################################
def simulation(initial_x_true: ArrayLike,
               initial_x_hat: ArrayLike,
               initial_covariance: ArrayLike,
               motion_noise_covariance: ArrayLike,
               star_tracker_noise_covariance: ArrayLike,
               gyroscope_noise_covariance: ArrayLike,
               simulation_tf: float,
               prediction_horizon: float,
               star_tracker_sampling_rate: float,
               gyroscope_sampling_rate: float,
               seed: int,
               control_bound: tuple,
               state_bound: tuple = None):

    # sampling rates
    dt = 0.01
    star_tracker_dt = 1 / star_tracker_sampling_rate
    gyroscope_dt = 1 / gyroscope_sampling_rate
    star_tracker_time = star_tracker_dt
    gyroscope_time = gyroscope_dt

    # spacecraft & planet properties
    inertia_tensor = np.diag([1448.3, 1346.2, 689.8])
    I_inv = np.linalg.inv(inertia_tensor)
    mu = 4.2828 * 10**13

    gravity_gradient = GravityGradient(inertia_tensor = inertia_tensor,
                                       gravitational_parameter = mu)

    orbit_provider = CircularOrbit(RAAN = np.deg2rad(20),
                                   inclination = np.deg2rad(30),
                                   initial_theta = np.pi / 6,
                                   radius = 3790 * 10**3,
                                   mu = mu)

    nadir_provider = NadirPointingReference(orbit_provider)

    # reference state & control
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

    def continuous_dynamics(t, state, control):

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

    def discrete_dynamics(t, state, control):

        # runge-kutta 4th order method
        k1 = continuous_dynamics(t, state, control)
        k2 = continuous_dynamics(t + dt / 2, state + dt * k1 / 2, control)
        k3 = continuous_dynamics(t + dt / 2, state + dt * k2 / 2, control)
        k4 = continuous_dynamics(t + dt, state + dt * k3, control)

        next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # shadow set transfer
        next_state[:3] = mrp_shadow_set(next_state[:3])

        return next_state

    def continuous_jacobians(t, state, control):

        # body state
        sigma_BR = state[0:3]
        omega_BR_B = state[3:6]

        reference_state = nadir_provider.get_state(t)

        # reference state
        sigma_RN = mrp_shadow_set(reference_state[0:3])
        omega_RN_R = reference_state[3:6]
        omega_RN_R_dot = nadir_provider.angular_acceleration(t)

        dcm_BR = mrp_to_dcm(sigma_BR)
        dcm_RN = mrp_to_dcm(sigma_RN)

        omega_RN_B = dcm_BR @ omega_RN_R
        omega_BN_B = omega_BR_B + omega_RN_B

        # gravity gradient properties
        r_N = orbit_provider.get_state(t)[0]

        r_R = dcm_RN @ r_N
        r_B = dcm_BR @ r_R

        r = np.linalg.norm(r_N)

        # A_11 (d_sigma_dot / d_sigma)
        A_11 = np.zeros((3, 3))

        for i in range(3):
            derivatived_sigma = np.eye(3)[:, i]
            derivatived_b_matrix = (-2 * sigma_BR[i] * np.eye(3) + 2 * skew_symmetric(derivatived_sigma) 
                                    + 2 * (sigma_BR @ derivatived_sigma.T + derivatived_sigma @ sigma_BR.T))

            A_11[:, i] = (1/4) * derivatived_b_matrix @ omega_BR_B

        # A_12 (d_sigma_dot / d_omega)
        A_12 = (1/4) * mrp_b_matrix(sigma_BR)

        # A_21 (d_omega_dot / d_sigma)
        A_21 = np.zeros((3, 3))

        for i in range(3):
            derivatived_sigma = np.eye(3)[:, i]

            derivatived_dcm_BR = - skew_symmetric(4 * np.linalg.inv(mrp_b_matrix(sigma_BR)) @ derivatived_sigma) @ dcm_BR

            derivatived_omega_RN_B = derivatived_dcm_BR @ omega_RN_R

            derivatived_omega_BN_B = derivatived_omega_RN_B

            derivatived_r_B = derivatived_dcm_BR @ r_R

            derivatived_gravity_gradient = ((3 * mu / r**5) * np.cross(derivatived_r_B, inertia_tensor @ r_B) 
                                            + np.cross(r_B, inertia_tensor @ derivatived_r_B))

            derivatived_gyroscopic_term = (np.cross(derivatived_omega_BN_B, inertia_tensor @ omega_BN_B) + 
                                           np.cross(omega_BN_B @ inertia_tensor @ derivatived_omega_BN_B))

            derivated_omega_RN_B_dot = - np.cross(omega_BR_B, derivatived_omega_RN_B) + derivatived_dcm_BR @ omega_RN_R_dot

            A_21[:, i] = I_inv @ (derivatived_gyroscopic_term + derivatived_gravity_gradient) - derivated_omega_RN_B_dot

        # A_22 (d_omega_dot / d_omega)
        A_22 = I_inv @ (skew_symmetric(inertia_tensor @ omega_BN_B) - skew_symmetric(omega_BN_B) @ inertia_tensor) - skew_symmetric(omega_RN_B)

        continuous_A = np.block([[A_11, A_12],
                                 [A_21, A_22]])

        continuous_B = np.vstack([np.zeros((3, 3)), I_inv])

        return continuous_A, continuous_B

    def discrete_jacobians(t, state, control):

        continuous_A, continuous_B = continuous_jacobians(t, state, control)

        # from continuous to discrete
        discrete_A = np.eye(6) + dt * continuous_A
        discrete_B = dt * continuous_B

        return discrete_A, discrete_B

    def motion_jacobian(t, state, control):

        return discrete_jacobians(t, state, control)[0]
    
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
                                 method = '2-point')

    # output: rotation vector
    def innovation_function(measured_mrp, predicted_mrp):
        # change mrp to dcm for (y - h) calculation
        measured_dcm = mrp_to_dcm(measured_mrp)
        predicted_dcm = mrp_to_dcm(predicted_mrp)

        innovation = measured_dcm @ predicted_dcm.T

        return mrp_to_rotation_vector(dcm_to_mrp(innovation))
    
    def innovation_jacobian(t, state):

        predicted_sigma = measurement_model(t, state)[0:3]

        jacobian = approx_derivative(fun = lambda nearby_state: innovation_function(measurement_model(t, nearby_state)[0:3], predicted_sigma),
                                     x0 = state,
                                     method = '3-point')

        return jacobian
    
    # ekf properties
    motion_noise_jacobian = np.eye(6)
    measurement_noise_jacobian = np.eye(6)
    measurement_noise_covariance = np.block([[star_tracker_noise_covariance, np.zeros((3, 3))],
                                             [np.zeros((3, 3)), gyroscope_noise_covariance]])

    ekf = ExtendedKalmanFilter(state = initial_x_hat,
                               covariance = initial_covariance,
                               motion_model = discrete_dynamics,
                               motion_jacobian = motion_jacobian,
                               motion_noise_jacobian = motion_noise_jacobian,
                               measurement_model = measurement_model,
                               measurement_jacobian = measurement_jacobian,
                               measurement_noise_jacobian = measurement_noise_jacobian,
                               motion_noise_covariance = motion_noise_covariance,
                               measurement_noise_covariance = measurement_noise_covariance)

    # ekf_lqr's lqr properties
    Q_lqr = np.diag([100, 100, 100, 500, 500, 500])
    R_lqr = 0.01 * np.eye(3)
    Qf_lqr = 2 * Q_lqr

    # mpc properties
    Q_mpc = dt * Q_lqr
    R_mpc = dt * R_lqr
    P_mpc = Qf_lqr
    N = int(round(prediction_horizon / dt))
    x_true = initial_x_true

    rti_nmpc = RealTimeNMPCController(Q = Q_mpc,
                                      R = R_mpc,
                                      P = P_mpc,
                                      N = N,
                                      dt = dt,
                                      reference_control_function = reference_u_function,
                                      discrete_nonlinear_dynamics = discrete_dynamics,
                                      discrete_jacobian_function = discrete_jacobians,
                                      control_bound = control_bound,
                                      state_bound = state_bound)

    # noises & sensors
    rng = np.random.default_rng(seed)
    motion_noise_provider = GaussianNoise(np.zeros(6), motion_noise_covariance)
    star_tracker = StarTracker(np.zeros(3), star_tracker_noise_covariance)
    gyroscope = Gyroscope(np.zeros(3), gyroscope_noise_covariance)

    if star_tracker_sampling_rate > 1 / dt:
        raise ValueError(f'Star tracker sampling rate should be smaller than {float(1/dt)} Hz')

    if gyroscope_sampling_rate > 1 / dt:
        raise ValueError(f'Gyroscope sampling rate should be smaller than {float(1/dt)} Hz')

    # total steps
    total_step = int(simulation_tf / dt)
    time = dt * np.arange(total_step + 1)

    # state histories 
    true_state_history = np.zeros((6, total_step + 1))
    reference_state_history = np.zeros((6, total_step + 1))
    estimated_state_history = np.zeros((6, total_step + 1))
    true_gravity_gradient_history = np.zeros((3, total_step + 1))

    # state history at t = 0
    true_state_history[:, 0] = x_true
    reference_state_history[:, 0] = reference_x_function(0)
    estimated_state_history[:, 0] = ekf.state
    true_gravity_gradient_history[:, 0] = gravity_gradient_torque(0, x_true)

    # ekf covariance P history
    covariance_history = np.zeros((6, 6, total_step + 1))
    covariance_history[:, :, 0] = ekf.P

    # control & measurement histories
    cmd_control_history = np.zeros((3, total_step))
    measurement_history = np.full((6, total_step + 1), np.nan)
    star_tracker_correction_steps_history = np.zeros(total_step + 1)
    gyroscope_correction_steps_history = np.zeros(total_step + 1)

    # tracking error & estimation error histories
    attitude_tracking_error_norm_history = np.zeros(total_step + 1)
    attitude_estimation_error_angle_history = np.zeros(total_step + 1)
    omega_estimation_error_history = np.zeros((3, total_step + 1))

    # qp history
    qp_status_history = []
    qp_iterations_history = np.zeros(total_step)

    # run simulation
    rti_nmpc.nominal_trajectory(0, ekf.state)
    rti_nmpc.preparation(0)
####################################################################
    qp_status_counter = Counter()
####################################################################
    for k in range(total_step):

        tk = k * dt
        next_t = time[k + 1]
####################################################################
        # current step
        if k % 10 == 0:
            print(
                f'\r[{k + 1:4d}/{total_step}] '
                f't = {tk:6.2f}s | solving QP...',
                end='',
                flush=True
            )

        # QP feedback
        start = perf_counter()
####################################################################
        u_cmd, histories = rti_nmpc.feedback(ekf.state)
####################################################################
        feedback_time = perf_counter() - start

        # QP status count
        qp_status_counter[histories['status']] += 1

        if k % 10 == 0:
            print(
                (
                    f'\r[{k + 1:4d}/{total_step}] '
                    f't = {tk:6.2f}s | '
                    f'QP = {feedback_time:.3f}s | '
                    f'status = {histories["status"]} | '
                    f'optimal = {qp_status_counter["optimal"]} | '
                    f'inaccurate = {qp_status_counter["optimal_inaccurate"]} | '
                    f'failed = {qp_status_counter["qp_failed"]}'
                ).ljust(140),
                end='',
                flush=True
            )
####################################################################
        # true state propagation & mrp shadow set transfer
        x_true = discrete_dynamics(tk, x_true, u_cmd) + motion_noise_provider.get_sample(rng)
        x_true[0:3] = mrp_shadow_set(x_true[0:3])

        # ekf prediction
        ekf.prediction(u_cmd, tk)

        # ekf measure attitude & correction
        if next_t >= star_tracker_time:
        
            ideal_attitude = measurement_model(next_t, x_true)[0:3]
            y = star_tracker.measure(ideal_attitude, rng)
            measurement_history[0:3, k + 1] = y

            # predicted model & jacobian
            predicted_star_tracker_measurement_model = measurement_model(next_t, ekf.state)[0:3]
            star_tracker_jacobian = innovation_jacobian(next_t, ekf.state)

            ekf.correction(measurement_vector = y,
                           t = next_t,
                           measurement_model = predicted_star_tracker_measurement_model,
                           measurement_noise_covariance = star_tracker_noise_covariance,
                           measurement_noise_jacobian = np.eye(3),
                           innovation_function = innovation_function,
                           innovation_jacobian = star_tracker_jacobian)

            ekf.x[0:3] = mrp_shadow_set(ekf.x[0:3])
            star_tracker_time += star_tracker_dt

            star_tracker_correction_steps_history[k + 1] = 1

        # ekf measure omega & correction
        if next_t >= gyroscope_time:

            ideal_omega = measurement_model(next_t, x_true)[3:6]
            y = gyroscope.measure(ideal_omega, rng)
            measurement_history[3:6, k + 1] = y

            # predicted model & jacobian
            predicted_gyroscope_measurement_model = measurement_model(next_t, ekf.state)[3:6]
            predicted_gyroscope_jacobian = measurement_jacobian(next_t, ekf.state)[3:6, :]

            ekf.correction(measurement_vector = y,
                           t = next_t,
                           measurement_model = predicted_gyroscope_measurement_model,
                           measurement_jacobian = predicted_gyroscope_jacobian,
                           measurement_noise_covariance = gyroscope_noise_covariance,
                           measurement_noise_jacobian = np.eye(3))

            ekf.x[0:3] = mrp_shadow_set(ekf.x[0:3])
            gyroscope_time += gyroscope_dt

            gyroscope_correction_steps_history[k + 1] = 1

        rti_nmpc.warm_start(next_t)
####################################################################
        if k % 10 == 0:
            print(
                (
                    f'\r[{k + 1:4d}/{total_step}] '
                    f't = {next_t:6.2f}s | preparing next step...'
                ).ljust(140),
                end='',
                flush=True
            )
####################################################################3
        rti_nmpc.preparation(next_t)

        # update state histories
        true_state_history[:, k + 1] = x_true
        reference_state_history[:, k + 1] = reference_x_function(next_t)
        estimated_state_history[:, k + 1] = ekf.state

        # qp status histories
        qp_status_history.append(histories['status'])
        qp_iterations_history[k] = histories['iterations']

        # update other histories
        true_gravity_gradient_history[:, k + 1] = gravity_gradient_torque(next_t, x_true)
        covariance_history[:, :, k + 1] = ekf.covariance
        cmd_control_history[:, k] = u_cmd

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

    # omega estimation error norm
    omega_estimation_norm_error_history = np.linalg.norm(omega_estimation_error_history, axis = 0)

    # omega tracking error
    omega_tracking_error_norm_history = np.linalg.norm(true_state_history[3:6, :], axis = 0)  

    return {'time': time,
            'true_state': true_state_history,
            'reference_state': reference_state_history,
            'estimated_state': estimated_state_history,
            'true_gravity_gradient_torque' : true_gravity_gradient_history,
            'covariance_P' : covariance_history,
            'commanded_control': cmd_control_history,
            'qp_status': qp_status_history,
            'qp_iterations': qp_iterations_history,
            'measurement_y': measurement_history,
            'star_tracker_correction_steps': star_tracker_correction_steps_history,
            'gyroscope_correction_steps': gyroscope_correction_steps_history,
            'attitude_tracking_error': attitude_tracking_error_norm_history,
            'omega_tracking_error': omega_tracking_error_norm_history,
            'attitude_estimation_error': attitude_estimation_error_angle_history,
            'omega_estimation_error': omega_estimation_norm_error_history}
