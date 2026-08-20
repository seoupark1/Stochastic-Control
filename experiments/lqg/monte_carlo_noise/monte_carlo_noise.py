import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from stochastic_control.estimators.kalman_filter import KalmanFilter
from stochastic_control.controllers.lqr.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQRController
from stochastic_control.controllers.lqg.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQGController
from stochastic_control.noises.gaussian_noise import GaussianNoise

# 1D acceleration lqg control
def simulation(dt: float,
               N: int,
               seed: int,
               x_true: ArrayLike,
               state: ArrayLike,
               covariance: ArrayLike,
               A: ArrayLike,
               B: ArrayLike,
               H: ArrayLike,
               motion_noise_covariance: ArrayLike,
               measurement_noise_covariance: ArrayLike,
               Q: ArrayLike,
               R: ArrayLike,
               Qf: ArrayLike):

    kalmanfilter = KalmanFilter(state = state,
                                covariance = covariance,
                                motion_jacobian = A,
                                control_jacobian = B,
                                measurement_jacobian = H,
                                motion_noise_covariance = motion_noise_covariance,
                                measurement_noise_covariance = measurement_noise_covariance)

    lqr = DiscreteTimeFiniteHorizonLQRController(A = A,
                                                 B = B,
                                                 Q = Q,
                                                 R = R,
                                                 Qf = Qf,
                                                 N = N)
    
    lqg = DiscreteTimeFiniteHorizonLQGController(kalmanfilter, lqr)

    # noise models
    rng = np.random.default_rng(seed)
    motion_noise = GaussianNoise(np.zeros(2), motion_noise_covariance)
    measurement_noise = GaussianNoise(np.zeros(1), measurement_noise_covariance)

    # state histories
    time = np.zeros(N)

    true_state_history = np.zeros((2, N + 1))
    estimated_state_history = np.zeros((2, N + 1))
    true_state_history[:, 0] = x_true
    estimated_state_history[:, 0] = kalmanfilter.x

    # additive histories
    measurement_history = np.zeros(N)
    control_history = np.zeros(N)

    for k in range(N):

        # update time
        time[k] = k * dt

        # create control vector
        u_cmd = lqg.control_vector(k)
        control_history[k] = u_cmd[0]

        # motion
        w = motion_noise.get_sample(rng)
        x_true = A @ x_true + B @ u_cmd + w
        true_state_history[:, k + 1] = x_true

        # measurement
        v = measurement_noise.get_sample(rng)
        y = H @ x_true + v
        measurement_history[k] = y[0]

        # estimate current state
        lqg.estimate(u_cmd, y)
        estimated_state_history[:, k + 1] = kalmanfilter.x

    # lqr's regulation evaluation about nominal state x = [0, 0]
    position_rmse = np.sqrt(np.mean(sum(p**2 for p in true_state_history[0])))
    velocity_rmse = np.sqrt(np.mean(sum(v**2 for v in true_state_history[1])))

    # kalman filter's estimation evaluation
    estimation_error_history = estimated_state_history - true_state_history
    position_estimation_rmse = np.sqrt(np.mean(sum(p**2 for p in estimation_error_history[0])))
    velocity_estimation_rmse = np.sqrt(np.mean(sum(v**2 for v in estimation_error_history[1])))

    # control evaluation
    control_rms = np.sqrt(np.mean(sum(u**2 for u in control_history)))

    return position_rmse, velocity_rmse, position_estimation_rmse, velocity_estimation_rmse, control_rms


# inputs
dt = 0.1
N = 100
x_true = np.array([3, 2])
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
Q = np.eye(2)
R = np.eye(1)
Qf = np.diag([10, 1])

# simulation conditions
num_simulation = 50
seed_list = np.random.randint(0, 1000, num_simulation)
for seed in seed_list:
    result = 