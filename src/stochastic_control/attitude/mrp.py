import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..math_tools import skew_symmetric
from .quaternion import dcm_to_quaternion, normalize_quaternion

# switch mrp to its shadow_set to avoid singularity
def mrp_shadow_set(mrp: ArrayLike) -> NDArray[np.float64]:
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    norm_sigma = np.vdot(sigma, sigma)

    if norm_sigma > 1:
        sigma = - sigma / norm_sigma
    
    return sigma.flatten()

# quaternion to modified rodrigues parameters to avoid singularity
def quaternion_to_mrp(quaternion: ArrayLike) -> NDArray[np.float64]:
    quaternion = np.asarray(quaternion, dtype = float).reshape(4)
    b = normalize_quaternion(quaternion)

    # shortest path
    if b[0] < 0:
        b = - b

    b0, b1, b2, b3 = b
    sigma = np.array([b1/(1 + b0) , b2/(1 + b0), b3/(1 + b0)])

    return sigma
    
# directional cosine matrix to modified rodrigues parameters
def dcm_to_mrp(dcm: ArrayLike):
    dcm = np.asarray(dcm, dtype = float).reshape(3,3)
    quaternion = dcm_to_quaternion(dcm)
    sigma = quaternion_to_mrp(quaternion)

    return sigma.flatten()

# modified rodrigues parameters to directional cosine matrix
def mrp_to_dcm(mrp: ArrayLike):
    sigma = np.asarray(mrp, dtype = float).reshape(3)
    dcm = (np.eye(3) + (8 * skew_symmetric(sigma) @ skew_symmetric(sigma) - 4 * (1 - np.vdot(sigma,sigma)) * skew_symmetric(sigma)) / (1 + np.vdot(sigma,sigma))**2)

    return dcm

# get mrps time derivative from body angular velocity
def mrp_derivative(mrp: ArrayLike, angular_velocity_b: ArrayLike):
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    sigma_dot = (1/4) * ((1 - np.vdot(sigma, sigma)) * np.eye(3) + 2 * skew_symmetric(sigma) + 2 * sigma @ sigma.T) @ omega

    return sigma_dot.flatten()

# get B matrix used in mrp_derivative
def mrp_b_matrix(mrp: ArrayLike):
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    B_matrix = (1 - np.vdot(sigma, sigma)) * np.eye(3) + 2 * skew_symmetric(sigma) + 2 * sigma @ sigma.T

    return B_matrix

def rotation_vector_to_mrp(rotation_vector: ArrayLike):

    rotation_vector = np.asarray(rotation_vector, dtype = float).reshape(3)
    angle = np.linalg.norm(rotation_vector)

    rotation_axis = rotation_vector / angle
    sigma = np.tan(angle / 4) * rotation_axis

    return mrp_shadow_set(sigma)

def mrp_to_rotation_vector(mrp: ArrayLike):

    sigma = np.asarray(mrp, dtype = float).reshape(3)

    sigma = mrp_shadow_set(sigma)
    angle = 4 * np.arctan(np.linalg.norm(sigma))
    rotation_axis = sigma / np.linalg.norm(sigma)

    return angle * rotation_axis

def mrp_addition(mrp_1: ArrayLike,
                 mrp_2: ArrayLike):

    sigma_1 = np.asarray(mrp_1, dtype = float).reshape(3)
    sigma_2 = np.asarray(mrp_2, dtype = float).reshape(3)

    dcm_1 = mrp_to_dcm(sigma_1)
    dcm_2 = mrp_to_dcm(sigma_2)

    return mrp_shadow_set(dcm_to_mrp(dcm_1 @ dcm_2))