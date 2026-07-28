import numpy as np
from numpy.linalg import inv
from stochastic_control.attitude.math import skew_symmetric
from src.stochastic_control.attitude.mrp import mrp_derivative, mrp_b_matrix, mrp_to_dcm, dcm_to_mrp, mrp_shadow_set

def mrp_b_dot(sigma, sigma_dot):
    return (-2 * np.dot(sigma, sigma_dot) * np.eye(3)
            + 2 * skew_symmetric(sigma_dot) 
            + 2 * np.outer(sigma_dot, sigma) 
            + 2 * np.outer(sigma, sigma_dot))

def reference_attitude(t, f):
    sigma = np.array([0.2 * np.sin(f*t), 
                      0.3 * np.cos(f*t), 
                      -0.3 * np.sin(f*t)])

    sigma_dot = f * np.array([0.2 * np.cos(f*t), 
                              -0.3 * np.sin(f*t), 
                              -0.3 * np.cos(f*t)])

    sigma_2dot = f**2 * np.array([-0.2 * np.sin(f*t), 
                                  -0.3 * np.cos(f*t), 
                                  0.3 * np.sin(f*t)])

    return sigma, sigma_dot, sigma_2dot

def reference_angular_velocity(t, f):
    sigma, sigma_dot, sigma_2dot = reference_attitude(t, f)
    omega = 4 * inv(mrp_b_matrix(sigma)) @ sigma_dot

    b_dot = mrp_b_dot(sigma, sigma_dot)
    omega_dot = np.linalg.solve(mrp_b_matrix(sigma), 4 * sigma_2dot - b_dot @ omega)

    return omega, omega_dot

def attitude_error(sigma_BN, sigma_RN):
    dcm_BN = mrp_to_dcm(sigma_BN)
    dcm_RN = mrp_to_dcm(sigma_RN)

    sigma_BR = dcm_to_mrp(dcm_BN @ dcm_RN.T)

    if np.linalg.norm(sigma_BR) > 1:
        sigma_BR = mrp_shadow_set(sigma_BR)
    
    return sigma_BR, dcm_BN @ dcm_RN.T

def state_derivative(state_current, K, P, I, f, KI, del_L, t, omega_BR_0):
    sigma_BN = state_current[0:3]
    omega_BN = state_current[3:6]
    eta = state_current[6:9]

    # reference
    sigma_RN, sigma_RN_dot, sigma_RN_2dot = reference_attitude(t, f)
    omega_RN_R, omega_RN_R_dot = reference_angular_velocity(t, f)

    # errors
    sigma_BR, dcm_BR = attitude_error(sigma_BN, sigma_RN)
    omega_RN_B, omega_RN_B_dot = dcm_BR @ omega_RN_R, dcm_BR @ omega_RN_R_dot
    omega_BR = omega_BN - omega_RN_B

    # compute z
    z = K * eta + I @ (omega_BR - omega_BR_0)

    sigma_BN_dot = mrp_derivative(sigma_BN, omega_BN)

    control_vector = (- K * sigma_BR - P @ omega_BR + I @ (omega_RN_B_dot - np.cross(omega_BN, omega_RN_B)) + 
                      skew_symmetric(omega_BN) @ I @ omega_BN - P @ (KI * z))
    omega_BN_dot = inv(I) @ (-skew_symmetric(omega_BN) @ I @ omega_BN + control_vector + del_L)
    eta_dot = sigma_BR

    return np.concatenate((sigma_BN_dot, omega_BN_dot, eta_dot))

def rk4(state_derivative, state_current, K, P, I, f, KI, del_L, dt, t, omega_BR_0):
    k1 = state_derivative(state_current, K, P, I, f, KI, del_L, t, omega_BR_0)
    k2 = state_derivative(state_current + 0.5 * dt * k1, K, P, I, f, KI, del_L, t + 0.5 * dt, omega_BR_0)
    k3 = state_derivative(state_current + 0.5 * dt * k2, K, P, I, f, KI, del_L, t + 0.5 * dt, omega_BR_0)
    k4 = state_derivative(state_current + dt * k3, K, P, I, f, KI, del_L, t+ 0.5 * dt, omega_BR_0)

    state_next = state_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    sigma_next = state_next[0:3]
    if np.linalg.norm(sigma_next) > 1:
        shadow_set = mrp_shadow_set(sigma_next)
        state_next[0:3] = shadow_set

    return state_next

# initial conditions
sigma_BN_0 = np.array([0.1, 0.2, -0.1])
omega_BN_0 = np.radians(np.array([3, 1, -2]))
eta = np.zeros(3)
state_current = np.concatenate((sigma_BN_0, omega_BN_0, eta))
K = 5
P = 10 * np.eye(3)
I = np.array([[100, 0, 0],
              [0, 75, 0],
              [0, 0, 80]])
f = 0.05
KI = 0.005
del_L = np.array([0.5, -0.3, 0.2])
dt, tspan = 0.1, 240
sigma_RN_0 = reference_attitude(0, f)[0]
omega_RN_0 = reference_angular_velocity(0, f)[0]
sigma_BR_0, dcm_BR_0 = attitude_error(sigma_BN_0, sigma_RN_0)
omega_RN_B_0 = dcm_BR_0 @ omega_RN_0
omega_BR_0 = omega_BN_0 - omega_RN_B_0

# history
state_hist = np.zeros((int(tspan/dt), 9))

for i in range(0, int(tspan/dt)):
    state_hist[i, :] = state_current
    state_next = rk4(state_derivative, state_current, K, P, I, f, KI, del_L, dt, i*dt, omega_BR_0)
    state_current = state_next

sigma_BN_45 = state_hist[int(45/dt), 0:3]
sigma_RN_45 = reference_attitude(45, f)[0]
sigma_BR_45 = attitude_error(sigma_BN_45, sigma_RN_45)[0]
print('Tracking error MRP norm at 45 seconds:', np.linalg.norm(sigma_BR_45))

print('Tracking erorr MRP norm at steady state:', np.linalg.norm(KI*inv(P)@del_L))

