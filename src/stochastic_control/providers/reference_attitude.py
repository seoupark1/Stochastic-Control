import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

class MRPReferenceState:

    def __init__(self,
                 sigma_RN: ArrayLike,
                 omega_RN_R: ArrayLike,
                 omega_RN_dot_R: ArrayLike):
        
        self.sigma_RN = sigma_RN
        self.omega_RN_R = omega_RN_R
        self.omega_RN_dot_R = omega_RN_dot_R

class MRPReferenceProvider:

    def __init__(self,
                 sigma_function: Callable,
                 omega_function: Callable,
                 omega_dot_function: Callable):

        self.sigma_function = sigma_function
        self.omega_function = omega_function
        self.omega_dot_function = omega_dot_function

    def get_reference(self,
                      t: float):

        sigma_RN = np.asarray(self.sigma_function(t), dtype = float).reshape(3)
        omega_RN = np.asarray(self.omega_function(t), dtype = float).reshape(3)
        omega_RN_dot = np.asarray(self.omega_dot_function(t), dtype = float).reshape(3)

        return MRPReferenceState(sigma_RN, omega_RN, omega_RN_dot)


class QuaternionReferenceState:

    def __init__(self,
                 quaternion_RN: ArrayLike,
                 omega_RN_R: ArrayLike,
                 omega_RN_dot_R: ArrayLike):
    
        self.quaternion_RN = quaternion_RN
        self.omega_RN_R = omega_RN_R
        self.omega_RN_dot_R = omega_RN_dot_R

class QuaternionReferenceProvider:

    def __init__(self,
                 quaternion_function: Callable,
                 omega_function: Callable,
                 omega_dot_function: Callable):

        self.quaternion_function = quaternion_function
        self.omega_function = omega_function
        self.omega_dot_function = omega_dot_function

    def get_reference(self,
                      t: float):

        quaternion_RN = np.asarray(self.quaternion_function(t), dtype = float).reshape(4)
        omega_RN = np.asarray(self.omega_function(t), dtype = float).reshape(3)
        omega_RN_dot = np.asarray(self.omega_dot_function(t), dtype = float).reshape(3)

        return QuaternionReferenceState(quaternion_RN, omega_RN, omega_RN_dot)