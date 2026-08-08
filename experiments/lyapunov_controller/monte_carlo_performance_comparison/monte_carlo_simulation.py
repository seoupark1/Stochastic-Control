import numpy as np
from scipy.integrate import solve_ivp

from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt

from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.controllers.lyapunov import LyapunovController
from stochastic_control.providers.attitude_reference import MRPReferenceProvider
from stochastic_control.providers.state import MRPStateProvider

from stochastic_control.attitude.mrp import mrp_b_matrix, mrp_derivative
from stochastic_control.attitude.math import skew_symmetric

def mrp_b_derivative(sigma, sigma_dot):
    return (-2 * np.dot(sigma, sigma_dot) * np.eye(3)
            + 2 * skew_symmetric(sigma_dot) 
            + 2 * np.outer(sigma_dot, sigma) 
            + 2 * np.outer(sigma, sigma_dot))

def reference_attitude(t, f):

    sigma = np.array([0.2 * np.sin(f * t), 
                      0.3 * np.cos(f * t), 
                      -0.3 * np.sin(f * t)])

    sigma_dot = f * np.array([0.2 * np.cos(f * t), 
                              -0.3 * np.sin(f * t), 
                              -0.3 * np.cos(f * t)])

    sigma_2dot = f**2 * np.array([-0.2 * np.sin(f * t), 
                                  -0.3 * np.cos(f * t), 
                                  0.3 * np.sin(f * t)])

    return sigma, sigma_dot, sigma_2dot

def reference_omega(t, f):
    sigma, sigma_dot, sigma_2dot = reference_attitude(t, f)
    omega = np.linalg.solve(mrp_b_matrix(sigma), 4 * sigma_dot)

    b_dot = mrp_b_derivative(sigma, sigma_dot)
    omega_dot = np.linalg.solve(mrp_b_matrix(sigma), 4 * sigma_2dot - b_dot @ omega)

    return omega, omega_dot

frequency = 0.05
reference_provider = MRPReferenceProvider(sigma_function = lambda t: reference_attitude(t, frequency)[0],
                                          omega_function = lambda t: reference_omega(t, frequency)[0], 
                                          omega_dot_function = lambda t: reference_omega(t, frequency)[1])

# simulation conditions
inertia_tensor = np.diag([100, 75, 80])
sigma_BN_B_0 = np.array([0.1, 0.2, -0.1])
omega_BN_B_0 = np.radians(np.array([3, 1, -2]))
initial_rotational_state = np.concatenate((sigma_BN_B_0, omega_BN_B_0))
initial_state_for_integral_controller = np.concatenate((initial_rotational_state, np.zeros(3)))

unknown_disturbance = np.array([0.5, -0.3, 0.2])
tspan = (0, 300)
teval = np.linspace(0, 300, 3001)

rigid_body = RigidBody(inertia_tensor)
state_provider = MRPStateProvider()

def get_random_gains():

    K = np.random.uniform(0.0, 50.0)
    KI = 10 ** np.random.uniform(-5.0, 1.0)
    P = make_spd_matrix(3)

    return K, KI, P

def simulate_standard_controller(K, P):

    control_gain = float(K)
    damping_matrix = np.asarray(P, dtype = float).reshape(3,3)

    standard_controller = LyapunovController(inertia_tensor, 
                                             control_gain, 
                                             damping_matrix, 
                                             reference_provider,
                                             estimated_disturbance_model = None)

    def performance_of_standard_controller(t,
                                       estimated_rotational_state):

        control_vector = standard_controller.mrp_control_vector(t,
                                                                estimated_rotational_state,
                                                                state_provider)

        total_torque = control_vector + unknown_disturbance

        return rigid_body.mrp_derivatives(estimated_rotational_state,
                                          total_torque)

    standard_sol = solve_ivp(performance_of_standard_controller,
                             t_span = tspan,
                             y0 = initial_rotational_state,
                             method = 'RK45',
                             t_eval = teval)

    return standard_controller, standard_sol

def simulate_integral_controller(K, P, KI):

    control_gain = float(K)
    integral_control_gain = float(KI)
    damping_matrix = np.asarray(P, dtype = float).reshape(3,3)

    integral_controller = LyapunovController(inertia_tensor, 
                                             control_gain, 
                                             damping_matrix, 
                                             reference_provider,
                                             integral_control_gain,
                                             estimated_disturbance_model = None)

    _, _, omega_BR_B_0 = integral_controller.mrp_tracking_error(0, initial_rotational_state)

    def performance_of_integral_controller(t,
                                           estimated_state):
        
        estimated_rotational_state = estimated_state[0:6]
        sigma_BN_B = estimated_state[0:3]
        omega_BN_B = estimated_state[3:6]
        integral_state = estimated_state[6:9]
        
        control_vector, eta_dot = integral_controller.mrp_integral_control_vector(t,
                                                                                  estimated_rotational_state,
                                                                                  integral_state,
                                                                                  omega_BR_B_0,
                                                                                  state_provider)

        total_torque = control_vector + unknown_disturbance

        sigma_dot = mrp_derivative(sigma_BN_B, omega_BN_B)
        omega_dot = rigid_body.angular_acceleration(omega_BN_B, total_torque)

        return np.concatenate((sigma_dot, omega_dot, eta_dot))

    integral_sol = solve_ivp(performance_of_integral_controller,
                             t_span = tspan,
                             y0 = initial_state_for_integral_controller,
                             method = 'RK45',
                             t_eval = teval)

    return integral_controller, integral_sol


# monte carlo simulation
num_simulations = 500
standard_sigma_rms_history = np.zeros(num_simulations)
standard_omega_rms_history = np.zeros(num_simulations)
integral_sigma_rms_history = np.zeros(num_simulations)
integral_omega_rms_history = np.zeros(num_simulations)

for trial in range(num_simulations):

        K, KI, P = get_random_gains()
        standard_controller, standard_sol = simulate_standard_controller(K, P)
        integral_controller, integral_sol = simulate_integral_controller(K, P, KI)

        # standard controller rms history
        standard_sigma_norm_history = np.zeros(len(standard_sol.t))
        standard_omega_norm_history = np.zeros(len(standard_sol.t))

        for i, t in enumerate(standard_sol.t):
            standard_state = standard_sol.y[:, i]
            _, sigma_BR_B, omega_BR_B = standard_controller.mrp_tracking_error(t, standard_state)

            standard_sigma_norm_history[i] = np.linalg.norm(sigma_BR_B)
            standard_omega_norm_history[i] = np.linalg.norm(omega_BR_B)

        standard_sigma_rms_history[trial] = np.sqrt(np.mean(standard_sigma_norm_history ** 2))
        standard_omega_rms_history[trial] = np.sqrt(np.mean(standard_omega_norm_history ** 2))

        # integral controller rms history
        integral_sigma_norm_history = np.zeros(len(integral_sol.t))
        integral_omega_norm_history = np.zeros(len(integral_sol.t))

        for j, t in enumerate(integral_sol.t):
            integral_state = integral_sol.y[0:6, i]
            _, sigma_BR_B, omega_BR_B = integral_controller.mrp_tracking_error(t, integral_state)

            integral_sigma_norm_history[i] = np.linalg.norm(sigma_BR_B)
            integral_omega_norm_history[i] = np.linalg.norm(omega_BR_B)

        integral_sigma_rms_history[trial] = np.sqrt(np.mean(integral_sigma_norm_history ** 2))
        integral_omega_rms_history[trial] = np.sqrt(np.mean(integral_omega_norm_history ** 2))

# get scatter
plt.scatter(standard_sigma_rms_history, integral_sigma_rms_history, s = 10)
plt.xlabel('Standard Lyapunov Controller RMS')
plt.ylabel('Integral Lyapunov Controller RMS')
plt.title('Attitude Tracking RMS Comparison')
plt.savefig('experiments/lyapunov_controller/monte_carlo_performance_comparison/attitude_rms_comparison.png')
plt.close()
plt.scatter(standard_omega_rms_history, integral_omega_rms_history, s = 10)
plt.xlabel('Standard Lyapunov Controller RMS')
plt.ylabel('Integral Lyapunov Controller RMS')
plt.title('Angular Velocity Tracking RMS Comparison')
plt.savefig('experiments/lyapunov_controller/monte_carlo_performance_comparison/angular_velocity_rms_comparison.png')
plt.close()