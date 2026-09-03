import numpy as np
import matplotlib.pyplot as plt

from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter
from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.providers.reference_trajectory import TrajectoryReferenceProvider
from stochastic_control.noises.gaussian_noise import GaussianNoise

def simulation():

    # estimator properties
    dt = 0.1
    c_drag = 0.05
    S = 200 # [m]
    D = 100 # [m]
    state = np.array([[0], [5]])
    covariance = np.array([[0.01, 0], [0, 1]])
    motion_noise_covariance = 0.1 * np.eye(2)
    measurement_noise_covariance = np.array([[0.01]])
    motion_noise_jacobian = np.eye(2)
    measurement_noise_jacobian = np.eye(1)

    def dynamics(state, control):
        velocity = state[1]
        position_dot = velocity
        velocity_dot = control[0] - c_drag * velocity * abs(velocity)

        return np.array([position_dot, velocity_dot])

    def f(state, control):
        return state + dt * dynamics(state, control)

    def F(state, control):
        velocity = state[1]
        return np.eye(2) + dt * np.array([[0, 1], [0, -2 * c_drag * abs(velocity)]])

    def h(state):
        position = state[0]
        return np.array([np.arctan2(D - position, S)])

    def H(state):
        position = state[0]
        return np.array([[S / ((D - position)**2 + S**2), 0]])
        

    estimator = ExtendedKalmanFilter(state = state,
                                     covariance = covariance,
                                     motion_model = f,
                                     motion_jacobian = F,
                                     motion_noise_jacobian = motion_noise_jacobian,
                                     measurement_model = h,
                                     measurement_jacobian = H,
                                     measurement_noise_jacobian = measurement_noise_jacobian,
                                     motion_noise_covariance = motion_noise_covariance,
                                     measurement_noise_covariance = measurement_noise_covariance)

    # controller properties
    Q = np.diag([5, 1])
    R = np.eye(1)
    Qf = np.diag([10, 1])
    tf = 10
    x_true = np.array([0, 1]) + np.array([0.6, -0.3])

    def reference_state_function(t):
        position = 2 * t**2 + np.sin(t)
        velocity = 4 * t + np.cos(t)
        return np.array([position, velocity])

    def reference_control_function(t):
        velocity = 4 * t + np.cos(t)
        acceleration = 4 - np.sin(t)
        return np.array([acceleration + c_drag * velocity * abs(velocity)])

    nominal_trajectory = TrajectoryReferenceProvider(reference_state_function, reference_control_function)

    # about nonlinear system
    controller = LocalTrajectoryStabilizationLQRController(Q = Q,
                                                           R = R,
                                                           Qf = Qf,
                                                           tf = tf,
                                                           reference_provider = nominal_trajectory,
                                                           dynamics_function = dynamics)

    # noises
    rng = np.random.default_rng(seed = 2026)
    motion_noise = GaussianNoise(np.zeros(2), motion_noise_covariance)
    measurement_noise = GaussianNoise(np.zeros(1), measurement_noise_covariance)

    # histories
    total_step = int(tf / dt)
    time = np.zeros(total_step)
    true_state_history = np.zeros((2, total_step))
    reference_state_history = np.zeros((2, total_step))
    estimated_state_history = np.zeros((2, total_step))
    control_history = np.zeros(total_step)
    measurement_history = np.zeros(total_step)

    # loop
    for k in range(total_step):

        # time
        t = k * dt
        time[k] = t

        # control
        u_cmd = controller.control_vector(t, estimator.state)
        control_history[k] = u_cmd[0]

        # true values
        x_true = f(x_true, u_cmd) + motion_noise.get_sample(rng)
        reference_state_history[:, k] = reference_state_function(t)
        true_state_history[:, k] = x_true

        y = h(x_true) + measurement_noise.get_sample(rng)
        measurement_history[k] = y[0]

        # estimate
        estimator.prediction(u_cmd)
        estimator.correction(y)
        estimated_state_history[:, k] = estimator.state

    state_error_history = estimated_state_history - true_state_history

    # true state vs estimated state
    plt.subplot(2, 1, 1)
    plt.plot(time, reference_state_history[0, :], label = 'reference position')
    plt.plot(time, true_state_history[0, :], label = 'true position')
    plt.plot(time, estimated_state_history[0, :], label = 'estimated position')
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.title('Position Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, reference_state_history[1, :], label = 'reference velocity')
    plt.plot(time, true_state_history[1, :], label = 'true velocity')
    plt.plot(time, estimated_state_history[1, :], label = 'estimated velocity')
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.title('Velocity Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/ekf_lqr_compensator/drag_acceleration/reference_vs_true_vs_estimated.png')
    plt.close()

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

simulation()