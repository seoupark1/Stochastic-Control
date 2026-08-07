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

controller = LyapunovController(inertia_tensor, 
                                control_gain, 
                                damping_matrix, 
                                reference_provider,
                                estimated_disturbance_model = None)


rigid_body = RigidBody(inertia_tensor)
state_provider = MRPStateProvider()

def performance_of_standard_controller(t,
                                       estimated_rotational_state):

    control_vector = controller.mrp_control_vector(t,
                                                   estimated_rotational_state,
                                                   state_provider)

    total_torque = control_vector + unknown_disturbance

    return rigid_body.mrp_derivatives(estimated_rotational_state,
                                      total_torque)

sol = solve_ivp(performance_of_standard_controller,
                t_span = tspan,
                y0 = initial_rotational_state,
                method = 'RK45')

sigma_history = sol.y[0:3]
omega_history = sol.y[3:6]
plt.plot(sol.t, sigma_history[0], label = 'sigma_1')
plt.plot(sol.t, sigma_history[1], label = 'sigma_2')
plt.plot(sol.t, sigma_history[2], label = 'sigma_3')
plt.xlabel('time')
plt.ylabel('body attitude')
plt.title('Standard Controller')
plt.savefig('experiments/lyapunov_controller/standard_controller_body_attitude.png')
plt.close()