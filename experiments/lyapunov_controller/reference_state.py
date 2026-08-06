import numpy as np

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