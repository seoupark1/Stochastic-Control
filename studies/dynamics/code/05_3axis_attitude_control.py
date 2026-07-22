import numpy as np
from src.stochastic_control.kinematics.tools import skew_symmetric

def B_matrix(sigma):
    return ((1-np.vdot(sigma, sigma)) * np.eye(3) + 2 * skew_symmetric(sigma) + 2 * np.outer(sigma, sigma)) / 4

def control_vector(sigma, omega, I, K, P):
    return -K * sigma - P @ omega - I @ omega + skew_symmetric(omega) @ I @ omega

def state_dot(state, I, K, P):
    sigma = state[0:3]
    omega = state[3:6]

    sigma_dot = B_matrix(sigma) @ omega
    u = control_vector(sigma, omega, I, K, P)
    omega_dot = np.linalg.inv(I) @ ((-skew_symmetric(omega) @ I @ omega) + u)

    return np.concatenate((sigma_dot, omega_dot))

def rk4(state_dot, state_current, I, K, P, dt):

    k1 = state_dot(state_current, I, K, P)
    k2 = state_dot(state_current + 0.5 * dt * k1, I, K, P)
    k3 = state_dot(state_current + 0.5 * dt * k2, I, K, P)
    k4 = state_dot(state_current + dt * k3, I, K, P)

    state_next = state_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # shadow set
    sigma = state_next[0:3]
    if np.vdot(sigma, sigma) > 1:
        state_next[0:3] = - sigma / np.vdot(sigma, sigma)
        
    return state_next

# initial conditions
sigma_initial = np.array([0.1, 0.2, -0.1])
omega_initial = np.radians(np.array([30, 10, -20]))
state_current = np.concatenate((sigma_initial, omega_initial))
I = np.array([[100, 0, 0],
              [0, 75, 0],
              [0, 0, 80]])
t_span = 120
dt = 0.1
total_step = t_span / dt
K = 5
P = 10 * np.eye(3)
sigma_hist = np.zeros((total_step, 3))


for i in range(int(total_step)):
    sigma_hist[i, :] = state_current[0:3]
    state_next = rk4(state_dot, state_current, I, K, P, dt)
    state_current = state_next
    
print(sigma_hist[int(30/dt)])
