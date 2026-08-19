import numpy as np
import matplotlib.pyplot as plt

from stochastic_control.estimators.kalman_filter import KalmanFilter
from stochastic_control.controllers.lqr.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQRController
from stochastic_control.controllers.lqg.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQGController
from stochastic_control.noises.gaussian_noise import GaussianNoise

def true_state_provider(t):

    position = (1/3) * t**3 + 4 * t**2 + 5 * t + 6
    velocity = t**2 + 8 * t + 5

    return np.array([position, velocity])

def simulation():

    dt = 0.1 # [s]

    # set estimator
    initial_state = np.array([[0], 
                              [5]])
    covariance = np.array([[0.01, 0],
                           [0, 1]])
    motion_jacobian = np.array([[1, dt],
                                [0, 1]])
    control_jacobian = np.array([[0],
                                 [1]])
    measurement_jacobian = np.array([[1, 0]])
    motion_noise_covariance = 5 * np.eye(2)
    measurement_noise_covariance = np.array([[10]])

    kalmanfilter = KalmanFilter(state = initial_state,
                                covariance = covariance,
                                motion_jacobian = motion_jacobian,
                                control_jacobian = control_jacobian,
                                measurement_jacobian = measurement_jacobian,
                                motion_noise_covariance = motion_noise_covariance,
                                measurement_noise_covariance = measurement_noise_covariance)

    # set controller
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.diag([10, 1])
    N = 100
    u_cmd = np.array([[-2]])

    lqr = DiscreteTimeFiniteHorizonLQRController(A = motion_jacobian,
                                                 B = control_jacobian,
                                                 Q = Q,
                                                 R = R,
                                                 Qf = Qf,
                                                 N = N)
    
    lqg = DiscreteTimeFiniteHorizonLQGController(kalmanfilter, lqr)

    # sensor property
    rng = np.random.default_rng(seed = 2026)
    mean = np.array([[0]])
    noise = GaussianNoise(mean, measurement_noise_covariance)

    # history
    time_step = np.zeros(N)
    measured_gps_history = np.zeros(N)
    nominal_position_history = np.zeros(N)
    estimated_state_history = np.zeros((2, N))
    control_vector_history = np.zeros(N)

    for k in range(N):

        t = interval * k
        time_step[k] = t

        # get measured sensor(gps) data
        nominal_position_history[k] = true_state_provider(t)[0]
        measured_position_vector = nominal_position_history[k] + noise.get_sample(rng)[0]
        measured_gps_history[k] = measured_position_vector

        # estimate current state
        lqg.estimate(u_cmd, measured_position_vector)
        estimated_state_history[0:2, k] = kalmanfilter.x

        # return control vector
        u_cmd = lqg.control_vector(k)
        control_vector_history[k] = u_cmd

    # plot sensor noise
    plt.plot(time_step, nominal_position_history, label = 'true position')
    plt.plot(time_step, measured_gps_history, label = 'gps data')
    plt.xlabel('time [s]')
    plt.ylabel('position [m]')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/lqg/sensor_noise.png')
    plt.close()

    # plot estimated state & control torque
    plt.subplot(3, 1, 1)
    plt.plot(time_step, estimated_state_history[0:3, ])
    plt.xlabel('time [s]')
    plt.ylabel('estimated position [m]')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time_step, estimated_state_history[3:6, ])
    plt.xlabel('time [s]')
    plt.ylabel('estimated velocity [m/s]')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time_step, control_vector_history, label = 'control vector')
    plt.xlabel('time [s]')
    plt.ylabel('control torque [Nm]')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/lqg/estimated_state_and_control_torque.png')
    plt.close()

simulation()