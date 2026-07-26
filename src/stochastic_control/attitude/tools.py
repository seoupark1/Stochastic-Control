import numpy as np
from numpy.typing import ArrayLike, NDArray

# tilde operator
def skew_symmetric(v: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(v, dtype = float).flatten()
    result = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return result

# numerical integrator (runge-kutta 4th order method)
def runge_kutta_4th(func, state_current, input_constant, dt):
    # weights
    k1 = func(state_current, input_constant)
    k2 = func(state_current + 0.5 * dt * k1, input_constant)
    k3 = func(state_current + 0.5 * dt * k2, input_constant)
    k4 = func(state_current + dt * k3, input_constant)

    state_next = state_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return state_next

# get principal inertias (descending order)
def get_principal_inertias(Ic_B):
    # get eigenvalues & eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(Ic_B)

    # change eigenvalue's index
    I_min, I_med, I_max = eig_vals[0], eig_vals[1], eig_vals[2]

    principal_inertia_tensor = np.array([[I_max, 0, 0],
                                         [0, I_med, 0],
                                         [0, 0, I_min]])

    return principal_inertia_tensor
