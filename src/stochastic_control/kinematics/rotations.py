import numpy as np
from numpy.typing import ArrayLike, NDArray

# tilde operator
def skew_symmetric(v):
    v = np.array(v).flatten()
    result = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return result

# Get DCM time derivative from body angular velocity
def dcm_derivative(dcm_bn: ArrayLike, angular_velocity_bn: ArrayLike) -> NDArray[np.float64]:
    c_bn = np.asarray(dcm_bn, dtype = float).reshape(3,3)
    omega = np.asarray(angular_velocity_bn, dtype = float).reshape(3,1)

    dcm_dot = - skew_symmetric(omega) @ c_bn

    return dcm_dot

# get Principal Inertias (descending order)
def get_principal_inertias(Ic_B):
    # get eigenvalues & eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(Ic_B)

    # change eigenvalue's index
    I_min, I_med, I_max = eig_vals[0], eig_vals[1], eig_vals[2]

    principal_inertia_tensor = np.array([[I_max, 0, 0],
                                         [0, I_med, 0],
                                         [0, 0, I_min]])

    return principal_inertia_tensor
