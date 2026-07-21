import numpy as np
from numpy.typing import ArrayLike, NDArray
from .tools import skew_symmetric

# directional cosine matrix to classical rodrigues parameters
def dcm_to_crp(dcm):
    zeta_square = 1 + np.trace(dcm)

     # check singularity
    if zeta_square < 1e-10:
        raise ValueError('Error: CRP Singularity')
    
    q = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]) / zeta_square

    return q

# classical rodrigues parameters to directional cosine matrix
def crp_to_dcm(q: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(q, dtype = float).reshape(3)
    dcm = ((1 - np.vdot(q, q)) * np.eye(3) + 2 * np.outer(q, q) - 2 * skew_symmetric(q)) / (1 + np.vdot(q, q))

    return dcm

# get crps time derivative from body angular velocity
def crp_derivative(crp: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(crp, dtype = float).reshape(3,1)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    q_dot = (1/2) * (np.eye(3) + skew_symmetric(q) + q @ q.T) @ omega
    
    return q_dot.flatten()