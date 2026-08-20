import numpy as np
import matplotlib.pyplot as plt

from stochastic_control.estimators.kalman_filter import KalmanFilter
from stochastic_control.controllers.lqr.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQRController
from stochastic_control.controllers.lqg.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQGController
from stochastic_control.noises.gaussian_noise import GaussianNoise

def simulation():

    dt = 0.1 # [s]

    # set estimator
    state = np.array([[0], 
                      [5]])
    covariance = np.array([[0.01, 0],
                           [0, 1]])
    A = np.array([[1, dt],
                  [0, 1]])
    B = np.array([[dt**2 / 2],
                  [dt]])
    H = np.array([[1, 0]])
    motion_noise_covariance = 0.1 * np.eye(2)
    measurement_noise_covariance = np.array([[0.01]])

    kalmanfilter = KalmanFilter(state = state,
                                covariance = covariance,
                                motion_jacobian = A,
                                control_jacobian = B,
                                measurement_jacobian = H,
                                motion_noise_covariance = motion_noise_covariance,
                                measurement_noise_covariance = measurement_noise_covariance)

    # set controller
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.diag([10, 1])
    N = 100

    lqr = DiscreteTimeFiniteHorizonLQRController(A = A,
                                                 B = B,
                                                 Q = Q,
                                                 R = R,
                                                 Qf = Qf,
                                                 N = N)
    
    lqg = DiscreteTimeFiniteHorizonLQGController(kalmanfilter, lqr)

    # noise models
    rng = np.random.default_rng(seed = 2026)
    motion_noise = GaussianNoise(np.zeros(2), motion_noise_covariance)
    measurement_noise = GaussianNoise(np.zeros(1), measurement_noise_covariance)

    # history
    time = np.zeros(N)
    true_state_history = np.zeros((2, N))
    estimated_state_history = np.zeros((2, N))
    measurement_history = np.zeros(N)
    control_history = np.zeros(N)

    # initial state
    x_true = np.array([3, 2])

    for k in range(N):

        time[k] = k * dt

        # create control vector
        u_cmd = lqg.control_vector(k)
        control_history[k] = u_cmd[0]

        # motion
        w = motion_noise.get_sample(rng)
        x_true = A @ x_true + B @ u_cmd + w
        true_state_history[:, k] = x_true

        # measurement
        v = measurement_noise.get_sample(rng)
        y = H @ x_true + v
        measurement_history[k] = y[0]

        # estimate current state
        lqg.estimate(u_cmd, y)
        estimated_state_history[:, k] = kalmanfilter.x

    # true vs estimated comparison
    plt.subplot(2, 1, 1)
    plt.plot(time, true_state_history[0, :], label = 'true position')
    plt.plot(time, estimated_state_history[0, :], label = 'estimated position')
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.title('Position Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, true_state_history[1, :], label = 'true velocity')
    plt.plot(time, estimated_state_history[1, :], label = 'estimated velocity')
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.title('Velocity Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/lqg/true_estimated_comparison.png')
    plt.close()

    # estimation error
    plt.subplot(2, 1, 1)
    plt.plot(time, estimated_state_history[0, :] - true_state_history[0, :])
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.title('Position Error (estimated - true)')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, estimated_state_history[1, :] - true_state_history[1, :])
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.title('Velocity Error (estimated - true)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/lqg/estimation_error.png')
    plt.close()

    # measurement y & control u
    plt.subplot(2, 1, 1)
    plt.plot(time, measurement_history)
    plt.xlabel('time [s]')
    plt.ylabel('measured position [m]')
    plt.title('Measurement History')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, control_history)
    plt.xlabel('time [s]')
    plt.ylabel('acceleration [m/s^2]')
    plt.title('Control History')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/lqg/measurement_and_control.png')
    plt.close()

simulation()