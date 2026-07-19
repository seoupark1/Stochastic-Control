import numpy as np
from numpy.typing import ArrayLike, NDArray

# tilde operator
def skew_symmetric(v):
    v = np.array(v).flatten()
    tilde = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return tilde

''' Euler Angles '''

# directional cosine matrix to (3-2-1) euler angles
def dcm_to_ea321(dcm):
    # allocate parameters
    theta1 = np.arctan2(dcm[0,1], dcm[0,0])
    theta2 = -np.arcsin(dcm[0,2])
    theta3 = np.arctan2(dcm[1,2], dcm[2,2])

    euler_angles_rad = np.array([theta1, theta2, theta3])

    return euler_angles_rad

# (3-2-1) euler angles to directional cosine matrix
def ea321_to_dcm(euler_angles_rad):
    # allocate parameters
    theta1, theta2, theta3 = euler_angles_rad.flatten()

    # refactoring
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)

    dcm = np.array([[c2*c1, c2*s1, -s2],
                    [s3*s2*c1 - c3*s1, s3*s2*s1 + c3*c1, s3*c2],
                    [c3*s2*c1 + s3*s1, c3*s2*s1 - s3*c1, c3*c2]])

    return dcm

# get (3-2-1) euler anlges time derivative from body angular velocity
def ea321_derivative(euler_angles_rad: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    euler_angles = np.asarray(euler_angles_rad, dtype = float).reshape(3)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    # allocate parameters
    psi, theta, phi = euler_angles

    # catch gimbal-lock
    if abs(np.cos(theta)) < 1e-6:
        raise ValueError('Error: Gimbal-lock in (3-2-1) Euler Angles')

    euler_angles_dot = (1 / np.cos(theta)) * np.array([[0, np.sin(phi), np.cos(phi)],
                                                       [0, np.cos(phi)*np.cos(theta), -np.sin(phi)*np.cos(theta)],
                                                       [np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta)]]) @ omega
    
    return euler_angles_dot.flatten()

# directional dosine matrix to (3-1-3) euler angles
def dcm_to_ea313(dcm):
    # allocate parameters
    theta1 = np.arctan2(dcm[2,0], -dcm[2,1])
    theta2 = np.arccos(dcm[2,2])
    theta3 = np.arctan2(dcm[0,2], dcm[1,2])

    euler_angles_rad = np.array([theta1, theta2, theta3])

    return euler_angles_rad

# (3-1-3) euler angles to directional cosine matrix
def ea313_to_dcm(euler_angles_rad):
    # allocate parameters
    theta1, theta2, theta3 = euler_angles_rad.flatten()

    # refactoring
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)

    dcm = np.array([[c3*c1 - s3*c2*s1, c3*s1 + s3*c2*c1, s3*s2],
                    [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
                    [s2*s1, -s2*c1, c2]])

    return dcm

# get (3-1-3) euler anlges time derivative from body angular velocity
def ea313_derivative(euler_angles_rad: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    euler_angles = np.asarray(euler_angles_rad, dtype = float).reshape(3)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    # allocate parameters
    theta1, theta2, theta3 = euler_angles

    # catch gimbal-lock
    if abs(np.sin(theta2)) < 1e-6:
        raise ValueError('Error: Gimbal-lock in (3-1-3) Euler Angles-lock')

    euler_angles_dot = (1 / np.sin(theta2)) * np.array([[np.sin(theta3), np.cos(theta3), 0],
                                                       [np.cos(theta3)*np.sin(theta2), -np.sin(theta3)*np.sin(theta2), 0],
                                                       [-np.sin(theta3)*np.cos(theta2), -np.cos(theta3)*np.cos(theta2), np.sin(theta2)]]) @ omega
    
    return euler_angles_dot.flatten()

''' Quaternions '''
    
# directional cosine matrix to quaternions (sheppard's method)
def dcm_to_quaternion(dcm):
    # allocate parameters
    b0_square = (1 + np.trace(dcm)) / 4
    b1_square = (1 + 2 * dcm[0,0] - np.trace(dcm)) / 4
    b2_square = (1 + 2 * dcm[1,1] - np.trace(dcm)) / 4
    b3_square = (1 + 2 * dcm[2,2] - np.trace(dcm)) / 4

    # find max_index of the largest b_square
    b_square = np.array([b0_square, b1_square, b2_square, b3_square])
    max_index = np.argmax(b_square)

    if max_index == 0:
        b0 = np.sqrt(max(0, b0_square))
        b1 = (dcm[1,2] - dcm[2,1]) / (4 * b0)
        b2 = (dcm[2,0] - dcm[0,2]) / (4 * b0)
        b3 = (dcm[0,1] - dcm[1,0]) / (4 * b0)

    elif max_index == 1:
        b1 = np.sqrt(max(0, b1_square))
        b0 = (dcm[1,2] - dcm[2,1]) / (4 * b1)
        b2 = (dcm[0,1] + dcm[1,0]) / (4 * b1)
        b3 = (dcm[2,0] + dcm[0,2]) / (4 * b1)

    elif max_index == 2:
        b2 = np.sqrt(max(0, b2_square))
        b0 = (dcm[2,0] - dcm[0,2]) / (4 * b2)
        b1 = (dcm[0,1] + dcm[1,0]) / (4 * b2)
        b3 = (dcm[1,2] + dcm[2,1]) / (4 * b2)

    elif max_index == 3:
        b3 = np.sqrt(max(0, b3_square))
        b0 = (dcm[0,1] - dcm[1,0]) / (4 * b3)
        b1 = (dcm[2,0] + dcm[0,2]) / (4 * b3)
        b2 = (dcm[1,2] + dcm[2,1]) / (4 * b3)

    if b0 < 0:
        b0, b1, b2, b3 = -b0, -b1, -b2, -b3
        
    quaternions = np.array([b0, b1, b2, b3])

    return quaternions

# quaternions to directional cosine matrix
def quaternion_to_dcm(quaternions):
    # allocate parameters (b0 is a scalar part)
    b0, b1, b2, b3 = quaternions.flatten()

    dcm = np.array([[b0**2 + b1**2 - b2**2 - b3**2,  2*(b1*b2 + b0*b3), 2*(b1*b3 - b0*b2)],
                    [2*(b1*b2 - b0*b3), b0**2 - b1**2 + b2**2 - b3**2,  2*(b2*b3 + b0*b1)],
                    [2*(b1*b3 + b0*b2), 2*(b2*b3 - b0*b1), b0**2 - b1**2 - b2**2 + b3**2]])
        
    return dcm

# get quaternions time derivative from body angular velocity
def quaternion_derivative(quaternions: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    euler_parameters = np.asarray(quaternions, dtype = float).reshape(4)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3)

    # allocate parameters
    b0, b1, b2, b3 = euler_parameters

    quaternion_dot = (1/2) * np.array([[b0, -b1, -b2, -b3],
                                       [b1, b0, -b3, b2],
                                       [b2, b3, b0, -b1],
                                       [b3, -b2, b1, b0]]) @ np.array([0, omega[0], omega[1], omega[2]]).reshape(4,1)
    
    return quaternion_dot.flatten()

''' Classical Rodrigues Parameters'''

# directional cosine matrix to classical rodrigues parameters
def dcm_to_crp(dcm):
    zeta_square = 1 + np.trace(dcm)

     # check singularity
    if zeta_square < 1e-6:
        raise ValueError('Error: CRP Singularity')
    
    q = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]) / zeta_square

    return q

# classical rodrigues parameters to directional cosine matrix
def crp_to_dcm(q):
    dcm = ((1 - np.vdot(q, q)) * np.eye(3) + 2 * np.outer(q, q) - 2 * skew_symmetric(q)) / (1 + np.vdot(q, q))

    return dcm

# get crps time derivative from body angular velocity
def crp_derivative(crp: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(crp, dtype = float).reshape(3,1)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    q_dot = (1/2) * (np.eye(3) + skew_symmetric(q) + q @ q.T) @ omega
    
    return q_dot.flatten()

'''Modified Rodrigues Parameters'''

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
    if abs(zeta) < 1e-6:
        raise ValueError('Error: MRP Singularity')
    
    sigma = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]) / (zeta*(zeta + 2))

    return sigma

# modified rodrigues parameters to directional cosine matrix
def mrp_to_dcm(sigma):
    sigma = mrp_shadow_set(sigma)
    dcm = (np.eye(3) + (8 * skew_symmetric(sigma) @ skew_symmetric(sigma) - 4 * (1 - np.vdot(sigma,sigma)) * skew_symmetric(sigma)) / (1 + np.vdot(sigma,sigma))**2)

    return dcm

# get mrps time derivative from body angular velocity
def mrp_derivative(mrp: ArrayLike, angular_velocity_b: ArrayLike) -> NDArray[np.float64]:
    sigma = np.asarray(mrp, dtype = float).reshape(3,1)
    omega = np.asarray(angular_velocity_b, dtype = float).reshape(3,1)

    sigma_dot = (1/4) * ((1 - np.vdot(sigma, sigma)) * np.eye(3) + 2 * skew_symmetric(sigma) + 2 * sigma @ sigma.T) @ omega

    return sigma_dot.flatten()

