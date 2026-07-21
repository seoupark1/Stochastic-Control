import numpy as np
from numpy.typing import ArrayLike, NDArray

# directional cosine matrix to (3-2-1) euler angles
def dcm_to_ea321(dcm):
    # allocate parameters
    theta1 = np.arctan2(dcm[0,1], dcm[0,0])
    theta2 = -np.arcsin(dcm[0,2])
    theta3 = np.arctan2(dcm[1,2], dcm[2,2])

    euler_angles_rad = np.array([theta1, theta2, theta3])

    return euler_angles_rad

# (3-2-1) euler angles to directional cosine matrix
def ea321_to_dcm(euler_angles_rad: NDArray) -> NDArray[np.float64]:
    # allocate parameters
    theta1, theta2, theta3 = np.asarray(euler_angles_rad, dtype = float).reshape(3)

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
def ea313_to_dcm(euler_angles_rad: ArrayLike) -> NDArray[np.float64]:
    # allocate parameters
    theta1, theta2, theta3 = np.asarray(euler_angles_rad, dtype = float).reshape(3)

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
