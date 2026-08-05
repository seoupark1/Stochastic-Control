import numpy as np
from collections.abc import Callable

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

        return sigma_RN, omega_RN, omega_RN_dot

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

        return quaternion_RN, omega_RN, omega_RN_dot