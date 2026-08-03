import numpy as np
from numpy.typing import ArrayLike

from ..attitude.math import skew_symmetric
from ..attitude.mrp import mrp_to_dcm, dcm_to_mrp

class LyapunovControl:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 K: float,
                 P: ArrayLike,
                 reference_provider,
                 disturbance):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.K = float(K)
        self.P = np.asarray(P, dtype = float).reshape(3,3)
        self.reference_provider = reference_provider
        self.disturbance = disturbance

    def mrp_lyapunov_function(self,
                              t: float,
                              state: ArrayLike):

        current_state = np.asarray(state, dtype = float).reshape(6)
        
        # body state
        sigma_BN = current_state[0:3]
        omega_BN_B = current_state[3:6]

        # reference data
        sigma_RN_R, omega_RN_R, omega_RN_dot_R = self.reference_provider.get_reference(t)

        # difference between body and reference
        dcm_BN = mrp_to_dcm(sigma_BN)
        dcm_RN = mrp_to_dcm(sigma_RN_R)
        dcm_BR = dcm_BN @ dcm_RN.T

        sigma_BR = dcm_to_mrp(dcm_BR)
        omega_BR = omega_BN_B - dcm_BR @ omega_RN_R

        # compute lyapunov function
        detumbling = (1/2) * omega_BR.T @ self.inertia_tensor @ omega_BR
        tracking = 2 * self.K * np.log(1 + sigma_BR.T @ sigma_BR)

        return detumbling + tracking

    def mrp_control_vector(self,
                           t: float,
                           state: ArrayLike):

        current_state = np.asarray(state, dtype = float).reshape(6)
        
        # body state
        sigma_BN = current_state[0:3]
        omega_BN_B = current_state[3:6]

        # reference data
        sigma_RN, omega_RN_R, omega_RN_dot_R = self.reference_provider.get_reference(t)

        # difference between body and reference
        dcm_BN = mrp_to_dcm(sigma_BN)
        dcm_RN = mrp_to_dcm(sigma_RN)
        dcm_BR = dcm_BN @ dcm_RN.T

        sigma_BR = dcm_to_mrp(dcm_BR)
        omega_BR_B = omega_BN_B - dcm_BR @ omega_RN_R
        omega_RN_B = dcm_BR @ omega_RN_R
        omega_RN_dot_B = dcm_BR @ omega_RN_dot_R

        # compute control vector
        feedforward = self.inertia_tensor @ (omega_RN_dot_B - skew_symmetric(omega_BN_B) @ omega_RN_B) - self.disturbance.torque(t, state)
        control_vector = -self.K * sigma_BR - self.P @ omega_BR_B + feedforward + skew_symmetric(omega_RN_B) @ self.inertia_tensor @ omega_RN_B
        
        return control_vector
