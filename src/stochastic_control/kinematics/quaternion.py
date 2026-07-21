import numpy as np
from numpy.typing import ArrayLike, NDArray

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
def quaternion_to_dcm(quaternions: ArrayLike) -> NDArray[np.float64]:
    # allocate parameters (b0 is a scalar part)
    quaternion = np.asarray(quaternions, dtype = float).reshape(4)
    quaternion /= np.linalg.norm(quaternion)
    b0 ,b1, b2, b3 = quaternion

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