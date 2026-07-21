import numpy as np
from numpy.typing import ArrayLike, NDArray
from .tools import skew_symmetric

# switch mrp to its shadow_set to avoid singularity
def mrp_shadow_set(mrp: ArrayLike) -> NDArray[np.float64]:
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    norm_sigma = np.vdot(sigma, sigma)

    if norm_sigma > 1:
        sigma = - sigma / norm_sigma
    
    return sigma.flatten()
    
# directional cosine matrix to modified rodrigues parameters
def dcm_to_mrp(dcm):
    zeta = np.sqrt(1 + np.trace(dcm))

    # check singularity
    if abs(zeta) < 1e-10:
        raise ValueError('Error: MRP Singularity')
    
    sigma = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]) / (zeta*(zeta + 2))

    return sigma

# modified rodrigues parameters to directional cosine matrix
def mrp_to_dcm(sigma: ArrayLike) -> NDArray[np.float64]:
    sigma = np.asarray(sigma, dtype = float).reshape(3)
    dcm = (np.eye(3) + (8 * skew_symmetric(sigma) @ skew_symmetric(sigma) - 4 * (1 - np.vdot(sigma,sigma)) * skew_symmetric(sigma)) / (1 + np.vdot(sigma,sigma))**2)

    return dcm

# get mrps time derivative from body angular velocity
def mrp_derivative(mrp: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    sigma_dot = (1/4) * ((1 - np.vdot(sigma, sigma)) * np.eye(3) + 2 * skew_symmetric(sigma) + 2 * sigma @ sigma.T) @ omega

    return sigma_dot.flatten()
