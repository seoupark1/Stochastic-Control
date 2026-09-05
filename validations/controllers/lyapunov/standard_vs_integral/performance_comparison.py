import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.controllers.lyapunov.standard import StandardLyapunovController
from stochastic_control.controllers.lyapunov.integral import IntegralLyapunovController
from stochastic_control.providers.reference_attitude import MRPReferenceProvider
from stochastic_control.providers.body_state import MRPStateProvider

from stochastic_control.attitude.mrp import mrp_b_matrix, mrp_derivative
from stochastic_control.math_tools import skew_symmetric

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

inertia_tensor = np.diag([100, 75, 80])
sigma_BN_B_0 = np.array([0.1, 0.2, -0.1])
omega_BN_B_0 = np.radians(np.array([3, 1, -2]))
initial_rotational_state = np.concatenate((sigma_BN_B_0, omega_BN_B_0))
initial_state_for_integral_controller = np.concatenate((sigma_BN_B_0, omega_BN_B_0, np.zeros(3)))
control_gain = 5
damping_matrix = 10 * np.eye(3)
integral_control_gain = 0.005
unknown_disturbance = np.array([0.5, -0.3, 0.2])
tspan = (0, 300)
teval = np.linspace(0, 300, 3001)

standard_controller = StandardLyapunovController(inertia_tensor, 
                                                 control_gain, 
                                                 damping_matrix, 
                                                 reference_provider,
                                                 estimated_disturbance_model = None)

_, _, omega_BR_B_0 = standard_controller.get_tracking_error(0, initial_rotational_state)

integral_controller = IntegralLyapunovController(inertia_tensor, 
                                                 control_gain, 
                                                 damping_matrix, 
                                                 reference_provider,
                                                 integral_control_gain,
                                                 estimated_disturbance_model = None)

rigid_body = RigidBody(inertia_tensor)
state_provider = MRPStateProvider()

def performance_of_standard_controller(t,
                                       estimated_rotational_state):

    control_vector = standard_controller.control_vector(t,
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

def performance_of_integral_controller(t,
                                       estimated_state):
    
    estimated_rotational_state = estimated_state[0:6]
    sigma_BN_B = estimated_state[0:3]
    omega_BN_B = estimated_state[3:6]
    integral_state = estimated_state[6:9]
    
    control_vector, eta_dot = integral_controller.control_vector(t,
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
    
def get_graph(time,
              history,
              xlabel: str,
              ylabel: str,
              labels: list[str],
              graph_title: str,
              png_title: str):
    
    for i in range(3):
        plt.plot(time, history[i], label = labels[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graph_title)
    plt.legend()
    plt.grid(True)
    plt.savefig('validations/controllers/lyapunov/standard_vs_integral/' + png_title)
    plt.close()

def get_tracking_error_history(sol,
                               sigma_BR_history,
                               omega_BR_history):

    for i, t in enumerate(sol.t):
        rotational_state = sol.y[0:6, i]

        dcm_BR_B, sigma_BR_B, omega_BR_B = standard_controller.get_tracking_error(t, rotational_state)
        sigma_BR_history[:, i] = sigma_BR_B
        omega_BR_history[:, i] = omega_BR_B

    return sigma_BR_history, omega_BR_history

# standard controller body attitude
sigma_history = standard_sol.y[0:3]
omega_history = standard_sol.y[3:6]

get_graph(standard_sol.t, 
          sigma_history, 
          'time', 
          'body attitude', 
          ['sigma_1', 'sigma_2', 'sigma_3'],
          'Standard Controller', 
          'standard_controller_body_attitude.png')

get_graph(standard_sol.t, 
          omega_history, 
          'time', 
          'body angular velocity', 
          ['omega_1', 'omega_2', 'omega_3'],
          'Standard Controller', 
          'standard_controller_body_angular_velocity.png')

# standard controller tracking error
sigma_BR_history_1 = np.zeros((3, len(standard_sol.t)))
omega_BR_history_1 = np.zeros((3, len(standard_sol.t)))

sigma_BR_history_1, omega_BR_history_1 = get_tracking_error_history(standard_sol,
                                                                    sigma_BR_history_1,
                                                                    omega_BR_history_1)

get_graph(standard_sol.t, 
          sigma_BR_history_1, 
          'time', 
          'attitude error', 
          ['sigma_BR_1', 'sigma_BR_2', 'sigma_BR_3'],
          'Standard Controller', 
          'standard_controller_attitude_error.png')

get_graph(standard_sol.t, 
          omega_BR_history_1, 
          'time', 
          'angular velocity error', 
          ['omega_BR_1', 'omega_BR_2', 'omega_BR_3'],
          'Standard Controller', 
          'standard_controller_angular_velocity_error.png')

# integral controller body attitude
sigma_history = integral_sol.y[0:3]
omega_history = integral_sol.y[3:6]

get_graph(integral_sol.t, 
          sigma_history, 
          'time', 
          'body attitude', 
          ['sigma_1', 'sigma_2', 'sigma_3'],
          'Integral Controller', 
          'integral_controller_body_attitude.png')

get_graph(integral_sol.t, 
          omega_history, 
          'time', 
          'body angular velocity', 
          ['omega_1', 'omega_2', 'omega_3'],
          'Integral Controller', 
          'integral_controller_body_angular_velocity.png')

# integral controller tracking error
sigma_BR_history_2 = np.zeros((3, len(integral_sol.t)))
omega_BR_history_2 = np.zeros((3, len(integral_sol.t)))

sigma_BR_history_2, omega_BR_history_2 = get_tracking_error_history(integral_sol,
                                                                    sigma_BR_history_2,
                                                                    omega_BR_history_2)

get_graph(integral_sol.t, 
          sigma_BR_history_2, 
          'time', 
          'attitude error', 
          ['sigma_BR_1', 'sigma_BR_2', 'sigma_BR_3'],
          'Integral Controller', 
          'integral_controller_attitude_error.png')

get_graph(integral_sol.t, 
          omega_BR_history_2, 
          'time', 
          'angular velocity error', 
          ['omega_BR_1', 'omega_BR_2', 'omega_BR_3'],
          'Integral Controller', 
          'integral_controller_angular_velocity_error.png')

# compare performance in single graph
sigma_BR_history_1_norm = np.zeros((len(standard_sol.t)))
omega_BR_history_1_norm = np.zeros((len(standard_sol.t)))
sigma_BR_history_2_norm = np.zeros((len(integral_sol.t)))
omega_BR_history_2_norm = np.zeros((len(integral_sol.t)))

for i, t in enumerate(standard_sol.t):
    sigma_BR_history_1_norm[i] = np.linalg.norm(sigma_BR_history_1[:, i])
    omega_BR_history_1_norm[i] = np.linalg.norm(omega_BR_history_1[:, i])

for i, t in enumerate(integral_sol.t):
    sigma_BR_history_2_norm[i] = np.linalg.norm(sigma_BR_history_2[0:6, i])
    omega_BR_history_2_norm[i] = np.linalg.norm(omega_BR_history_2[0:6, i])

plt.subplot(2,1,1)
plt.plot(standard_sol.t, sigma_BR_history_1_norm, 'b', label = 'standard controller')
plt.plot(standard_sol.t, sigma_BR_history_2_norm, 'r', label = 'integral controller')
plt.xlabel('time [s]')
plt.ylabel('attitude error (mrp)')
plt.legend()
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(standard_sol.t, omega_BR_history_1_norm, 'b', label = 'standard controller')
plt.plot(standard_sol.t, omega_BR_history_2_norm, 'r', label = 'integral controller')
plt.xlabel('time [s]')
plt.ylabel('angular velocity error [rad/s]')
plt.legend()
plt.grid(True)
plt.savefig('validations/controllers/lyapunov/standard_vs_integral/final_result.png')
plt.close()