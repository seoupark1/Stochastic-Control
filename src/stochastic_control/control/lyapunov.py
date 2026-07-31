import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable
from ..attitude.math import skew_symmetric
from ..attitude.mrp import mrp_to_dcm, dcm_to_mrp

class Lyapunov:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 K: float,
                 P: ArrayLike,
                 reference_provider: Callable,
                 disturbance: Callable):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.K = float(K)
        self.P = float(P)
        self.reference_provider = reference_provider
        self.disturbance = disturbance

    def lyapunov_function_mrp(self,
                              t: float,
                              sigma_BN: ArrayLike,
                              omega_BN_B: ArrayLike):

        # body state
        sigma_BN = np.asarray(sigma_BN, dtype = float).reshape(3)
        omega_BN_B = np.asarray(omega_BN_B, dtype = float).reshape(3)

        # reference data
        sigma_RN_R, omega_RN_R, omega_RN_dot_R = self.reference_provider.get_reference(t)

        # difference between body and reference
        dcm_BN = mrp_to_dcm(sigma_BN)
        dcm_RN = mrp_to_dcm(sigma_RN_R)
        dcm_BR = dcm_BN @ np.linalg.inv(dcm_RN)

        sigma_BR = dcm_to_mrp(dcm_BR)
        omega_BR = omega_BN_B - dcm_BR @ omega_RN_R

        # compute lyapunov function
        detumbling = (1/2) * omega_BR.T @ self.inertia_tensor @ omega_BR
        tracking = 2 * self.K * np.log(1 + sigma_BR.T @ sigma_BR)
        lyapunov_function = detumbling + tracking

        return lyapunov_function

    def control_vector_mrp(self,
                           t: float,
                           sigma_BN: ArrayLike,
                           omega_BN_B: ArrayLike):
        
        # body state
        sigma_BN = np.asarray(sigma_BN, dtype = float).reshape(3)
        omega_BN_B = np.asarray(omega_BN_B, dtype = float).reshape(3)

        # reference data
        sigma_RN_R, omega_RN_R, omega_RN_dot_R = self.reference_provider.get_reference(t)

        # difference between body and reference
        dcm_BN = mrp_to_dcm(sigma_BN)
        dcm_RN = mrp_to_dcm(sigma_RN_R)
        dcm_BR = dcm_BN @ np.linalg.inv(dcm_RN)

        sigma_BR = dcm_to_mrp(dcm_BR)
        omega_BR = omega_BN_B - dcm_BR @ omega_RN_R
        omega_RN_B = dcm_BR @ omega_RN_R
        omega_RN_dot_B = dcm_BR @ omega_RN_dot_R

        # compute control vector
        feedforward = self.inertia_tensor @ (omega_RN_dot_B - skew_symmetric(omega_BN_B) @ omega_RN_B) - self.disturbance.torque(t, state)
