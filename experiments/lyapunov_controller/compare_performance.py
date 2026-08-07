import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.controllers.lyapunov import LyapunovController
from stochastic_control.providers.attitude_reference import MRPReferenceProvider
from stochastic_control.providers.state import MRPStateProvider

from stochastic_control.attitude.mrp import mrp_b_matrix
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

inertia_tensor = np.diag([100, 75, 80])
sigma_BN_B_0 = np.array([0.1, 0.2, -0.1])
omega_BN_B_0 = np.radians(np.array([3, 1, -2]))
initial_rotational_state = np.concatenate((sigma_BN_B_0, omega_BN_B_0))
control_gain = 5
damping_matrix = 10 * np.eye(3)
integral_control_gain = 0.005
unknown_disturbance = np.array([0.5, -0.3, 0.2])
tspan = (0, 300)
teval = np.linspace(0, 300, 3001)

standard_controller = LyapunovController(inertia_tensor, 
                                         control_gain, 
                                         damping_matrix, 
                                         reference_provider,
                                         estimated_disturbance_model = None)

integral_controller = LyapunovController(inertia_tensor, 
                                         control_gain, 
                                         damping_matrix, 
                                         reference_provider,
                                         integral_control_gain,
                                         estimated_disturbance_model = None)

rigid_body = RigidBody(inertia_tensor)
state_provider = MRPStateProvider()

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
    plt.savefig('experiments/lyapunov_controller/' + png_title)
    plt.close()

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
sigma_BR_history = np.zeros((3, len(standard_sol.t)))
omega_BR_history = np.zeros((3, len(standard_sol.t)))

for i, t in enumerate(standard_sol.t):
    rotational_state = standard_sol.y[:, i]

    dcm_BR_B, sigma_BR_B, omega_BR_B = standard_controller.mrp_tracking_error(t, rotational_state)
    sigma_BR_history[:, i] = sigma_BR_B
    omega_BR_history[:, i] = omega_BR_B

get_graph(standard_sol.t, 
          sigma_BR_history, 
          'time', 
          'attitude error', 
          ['sigma_BR_1', 'sigma_BR_2', 'sigma_BR_3'],
          'Standard Controller', 
          'standard_controller_attitude_error.png')

get_graph(standard_sol.t, 
          omega_BR_history, 
          'time', 
          'angular velocity error', 
          ['omega_BR_1', 'omega_BR_2', 'omega_BR_3'],
          'Standard Controller', 
          'standard_controller_angular_velocity_error.png')