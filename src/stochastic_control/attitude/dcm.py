import numpy as np
from numpy.typing import ArrayLike, NDArray
from .tools import skew_symmetric

# get dcm time derivative from body angular velocity
def dcm_derivative(dcm_bn: ArrayLike, angular_velocity_bn_b: ArrayLike) -> NDArray[np.float64]:
    c_bn = np.asarray(dcm_bn, dtype = float).reshape(3,3)
    omega = np.asarray(angular_velocity_bn_b, dtype = float).reshape(3,1)

    dcm_dot = - skew_symmetric(omega) @ c_bn

    return dcm_dot