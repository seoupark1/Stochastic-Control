import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from src.stochastic_control.attitude.tools import skew_symmetric
from src.stochastic_control.attitude.mrp import mrp_derivative

def control_vector(state, I, P, K):
    sigma = state[0:3]
    omega = state[3:6]
    u = - K * sigma - P @ omega + skew_symmetric(omega) @ I @ omega
    return u

def state_derivative(t, state, I, P, K):
    sigma = state[0:3]
    omega = state[3:6]
    u = control_vector(state, I, P, K)
    sigma_dot = mrp_derivative(sigma, omega)
    omega_dot = inv(I) @ (- skew_symmetric(omega) @ I @ omega + u)

    return np.concatenate((sigma_dot, omega_dot))

sigma_0 = np.array([0.1, 0.2, -0.1])
omega_0 = np.deg2rad(np.array([30, 10, -20]))
state_0 = np.concatenate((sigma_0, omega_0))
I = np.diag([100, 75, 80])
P = np.diag([np.sqrt(500), np.sqrt(375), np.sqrt(400)])
print('Gain P that makes damping ratio 1: ')
print(P)
K = 5
tspan = (0, 100)

states = solve_ivp(fun = state_derivative,
                   t_span = tspan, 
                   y0 = state_0,
                   t_eval = [30],
                   args = (I, P, K))
attitude_error_30 = states.y[0:3, 0]
print('Tracking error MRP norm at 30 seconds: ', np.linalg.norm(attitude_error_30))
