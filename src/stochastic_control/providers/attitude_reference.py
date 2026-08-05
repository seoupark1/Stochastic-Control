import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from numpy.typing import NDArray

@dataclass(frozen = True)
class MRPReferenceState:
    sigma_RN : NDArray[np.float64]
    omega_RN_R : NDArray[np.float64]
    omega_RN_dot_R : NDArray[np.float64]

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

@dataclass(frozen = True)
class QuaternionReferenceState:
    
    quaternion_RN : NDArray[np.float64]
    omega_RN_R : NDArray[np.float64]
    omega_RN_dot_R : NDArray[np.float64]

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