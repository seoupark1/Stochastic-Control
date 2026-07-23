import numpy as np
from src.stochastic_control.kinematics.tools import skew_symmetric
from src.stochastic_control.kinematics.mrp import mrp_to_dcm, dcm_to_mrp

def sigma_RN(t, f):
    return np.array([0.2 * np.sin(f*t), 0.3 * np.cos(f*t), -0.3 * np.sin(f*t)])

def sigma_RN_dot(t, f):
    return f * np.array([0.2 * np.cos(f*t), -0.3 * np.sin(f*t), -0.3 * np.cos(f*t)])

def control_vector(omega_BR, sigma_BR, K, P):
    return -K * sigma_BR - P @ omega_BR

def state_dot(state_current, state_reference, I, K, P):
    sigma_BN = state_current[0:3]
    omega_BN = state_current[3:6]
    sigma_RN = state_reference[0:3]
    omega_RN = state_reference[3:6]
    sigma_BR = dcm_to_mrp(mrp_to_dcm(sigma_BN) @ np.linalg.inv(mrp_to_dcm(sigma_RN)))
    omega_BR = dcm_to_mrp(mrp_to_dcm(omega_BN) @ np.linalg.inv(mrp_to_dcm(omega_RN)))

    omega_dot = np.linalg.inv(I) @ (-skew_symmetric(omega_BN) @ I @ omega_BN + control_vector(omega_BR, sigma_BR, K, P))
    B_matrix = ((1-np.vdot(sigma_BN, sigma_BN)) * np.eye(3) + 2 * skew_symmetric(sigma_BN) + 2 * np.outer(sigma_BN, sigma_BN)) / 4
    sigma_dot = B_matrix @ omega_BN
    state_dot = np.concatenate((sigma_dot, omega_dot))

    return state_dot

def rk4(state_dot, state_current, state_reference, I, K, P, dt):

    k1 = state_dot(state_current, state_reference,I, K, P)
    k2 = state_dot(state_current + 0.5 * dt * k1, state_reference,I, K, P)
    k3 = state_dot(state_current + 0.5 * dt * k2, state_reference,I, K, P)
    k4 = state_dot(state_current + dt * k3, state_reference,I, K, P)

    state_next = state_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # shadow set
    sigma = state_next[0:3]
    if np.vdot(sigma, sigma) > 1:
        state_next[0:3] = - sigma / np.vdot(sigma, sigma)
        
    return state_next


# initial values
sigma_BN = np.array([0.1, 0.2, -0.1])
omega_BN = np.radians(np.array([30, 10, -20]))
state_current = np.concatenate((sigma_BN, omega_BN))
f = 0.05
K = 5
P = 10 * np.eye(3)
I = np.array([[100, 0, 0],
              [0, 75, 0],
              [0, 0, 80]])
dt = 0.1
total_step = 300
state_reference_hist = np.zeros((total_step, 6))
state_hist = np.zeros((total_step, 6))

# reference state vectors
for i in range(total_step):
    t = i * dt
    sigma = sigma_RN(t, f)
    sigma_dot = sigma_RN_dot(t, f)
    numerator = 4 * ((1 - np.vdot(sigma, sigma)) * np.eye(3) - 2 * skew_symmetric(sigma) + 2 * np.outer(sigma, sigma)) @ sigma_dot
    denominator = (1 + np.vdot(sigma, sigma))**2

    # reference sigma, omega
    state_reference_hist[i, 0:3] = sigma
    state_reference_hist[i, 3:6] = numerator / denominator

for i in range(total_step):
    state_hist[i, :] = state_current
    state_next = rk4(state_dot, state_current, state_reference_hist[i, :], I, K, P, dt)
    state_current = state_next

sigma_BN_20 = state_hist[int(20/dt), 0:3]
sigma_RN_20 = state_reference_hist[int(20/dt), 0:3]
dcm_BR_20 = mrp_to_dcm(sigma_BN_20) @ np.linalg.inv(mrp_to_dcm(sigma_RN_20))
sigma_BR_20 = dcm_to_mrp(dcm_BR_20)
print(print(np.linalg.norm(sigma_BR_20)))

