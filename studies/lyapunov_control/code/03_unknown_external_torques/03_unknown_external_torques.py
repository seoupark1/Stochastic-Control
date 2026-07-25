import numpy as np
from src.stochastic_control.kinematics.mrp import mrp_derivative, mrp_shadow_set

def state_derivative(state_current, I, P, K, external_torque):
    sigma = state_current[0:3]
    omega = state_current[3:6]

    sigma_dot = mrp_derivative(sigma, omega)
    omega_dot = np.linalg.inv(I) @ (external_torque - P @ omega - K * sigma)

    return np.concatenate((sigma_dot, omega_dot))

def rk4(state_derivative, state_current, I, P, K, external_torque, dt):
    k1 = state_derivative(state_current, I, P, K, external_torque)
    k2 = state_derivative(state_current + 0.5 * dt * k1, I, P, K, external_torque)
    k3 = state_derivative(state_current + 0.5 * dt * k2, I, P, K, external_torque)
    k4 = state_derivative(state_current + dt * k3, I, P, K, external_torque)

    state_next = state_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    sigma_next = state_next[0:3]
    if np.linalg.norm(sigma_next) > 1:
        shadow_set = mrp_shadow_set(sigma_next)
        state_next[0:3] = shadow_set

    return state_next

# initial conditions
sigma_BR = np.array([0.1, 0.2, -0.1])
omega_BR = np.radians(np.array([30, 10, -20]))
state_current = np.concatenate((sigma_BR, omega_BR))
K = 5
P = 10 * np.eye(3)
I = np.array([[100, 0, 0],
              [0, 75, 0],
              [0, 0, 80]])
external_torque = np.array([0.5, -0.3, 0.2])
dt, tspan = 0.1, 200

# history
state_hist = np.zeros((int(tspan/dt), 6))
state_derivative_hist = np.zeros((int(tspan/dt), 6))

for i in range(0, int(tspan/dt)):
    state_hist[i, :] = state_current
    state_next = rk4(state_derivative, state_current, I, P, K, external_torque, dt)
    state_current = state_next

print('predicted steady state attitude error through rk4: {}'.format(state_hist[-1, 0:3]))
# result = [0.10005713 -0.05994843 0.03999278]

# using formula
sigma_ss = external_torque / K
print('predicted steady state attitude error through formula: {}'.format(sigma_ss))
# result = [0.1 -0.06 0.04]