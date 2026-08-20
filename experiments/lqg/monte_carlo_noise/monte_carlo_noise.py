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
    position_regulation_rmse = np.sqrt(np.mean(true_state_history[0]**2))
    velocity_regulation_rmse = np.sqrt(np.mean(true_state_history[1]**2))

    # kalman filter's estimation evaluation
    estimation_error_history = estimated_state_history - true_state_history
    position_error_rmse = np.sqrt(np.mean(estimation_error_history[0]**2))
    velocity_error_rmse = np.sqrt(np.mean(estimation_error_history[1]**2))

    # control evaluation
    control_rms = np.sqrt(np.mean(control_history**2))

    return position_regulation_rmse, velocity_regulation_rmse, position_error_rmse, velocity_error_rmse, control_rms


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

# run simulation
num_simulation = 500

position_regulation_rmse_history = np.zeros(num_simulation)
velocity_regulation_rmse_history = np.zeros(num_simulation)
position_error_rmse_history = np.zeros(num_simulation)
velocity_error_rmse_history = np.zeros(num_simulation)
control_rms_history = np.zeros(num_simulation)

for trial in range(num_simulation):

    seed = np.random.randint(0, 1000)
    results = simulation(dt, 
                         N, 
                         seed, 
                         x_true, 
                         state, 
                         covariance, 
                         A,
                         B, 
                         H, 
                         motion_noise_covariance,
                         measurement_noise_covariance,
                         Q,
                         R,
                         Qf)

    position_regulation_rmse, velocity_regulation_rmse, position_error_rmse, velocity_error_rmse, control_rms = results

    position_regulation_rmse_history[trial] = position_regulation_rmse
    velocity_regulation_rmse_history[trial] = velocity_regulation_rmse
    position_error_rmse_history[trial] = position_error_rmse
    velocity_error_rmse_history[trial] = velocity_error_rmse
    control_rms_history[trial] = control_rms

# scatter results
plt.scatter(position_regulation_rmse_history, control_rms_history)
plt.xlabel('True Position RMSE')
plt.ylabel('Control RMS')
plt.title('The Relationship between Position Regulation and Control')
plt.grid(True)
plt.savefig('experiments/lqg/monte_carlo_noise/position_regulation_control_relationship.png')
plt.close()

plt.scatter(velocity_regulation_rmse_history, control_rms_history)
plt.xlabel('True Velocity RMSE')
plt.ylabel('Control RMS')
plt.title('The Relationship between Velocity Regulation and Control')
plt.grid(True)
plt.savefig('experiments/lqg/monte_carlo_noise/velocity_regulation_control_relationship.png')
plt.close()

plt.scatter(position_error_rmse_history, control_rms_history)
plt.xlabel('Position Error RMSE')
plt.ylabel('Control RMS')
plt.title('The Relationship between Position Estimation and Control')
plt.grid(True)
plt.savefig('experiments/lqg/monte_carlo_noise/position_estimation_control_relationship.png')
plt.close()

plt.scatter(velocity_error_rmse_history, control_rms_history)
plt.xlabel('Velocity Error RMSE')
plt.ylabel('Control RMS')
plt.title('TThe Relationship between Velocity Estimation and Control')
plt.grid(True)
plt.savefig('experiments/lqg/monte_carlo_noise/velocity_estimation_control_relationship.png')
plt.close()