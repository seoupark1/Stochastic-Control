import numpy as np
from numpy.typing import ArrayLike

from ..attitude.math import skew_symmetric
from ..attitude.mrp import mrp_to_dcm, dcm_to_mrp

class LyapunovController:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 K: float,
                 P: ArrayLike,
                 reference_provider,
                 KI = None,
                 estimated_disturbance_model = None):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.K = float(K)
        self.KI = np.asarray(KI, dtype = float).reshape(3,3)
        self.P = np.asarray(P, dtype = float).reshape(3,3)
        self.reference_provider = reference_provider
        self.disturbance_model = estimated_disturbance_model

    def mrp_lyapunov_function(self,
                              t: float,
                              estimated_rotational_state: ArrayLike):

        x_hat = np.asarray(estimated_rotational_state, dtype = float).reshape(6)
        
        # body state
        sigma_BN = x_hat[0:3]
        omega_BN_B = x_hat[3:6]

        # reference state
        reference = self.reference_provider.get_reference(t)
        sigma_RN_R = reference.sigma_RN
        omega_RN_R = reference.omega_RN_R

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
                           estimated_rotational_state: ArrayLike,
                           estimated_context_builder):

        x_hat = np.asarray(estimated_rotational_state, dtype = float).reshape(6)
        estimated_context = estimated_context_builder.build_context(t, x_hat)
        estimated_disturbance = np.zeros(3)

        if self.disturbance_model is not None:
            estimated_disturbance = self.disturbance_model.torque(t, estimated_context)
        
        # body state
        sigma_BN = x_hat[0:3]
        omega_BN_B = x_hat[3:6]

        # reference data
        reference = self.reference_provider.get_reference(t)
        sigma_RN = reference.sigma_RN
        omega_RN_R = reference.omega_RN_R
        omega_RN_dot_R = reference.omega_RN_dot_R

        # difference between body and reference
        dcm_BN = mrp_to_dcm(sigma_BN)
        dcm_RN = mrp_to_dcm(sigma_RN)
        dcm_BR = dcm_BN @ dcm_RN.T

        sigma_BR = dcm_to_mrp(dcm_BR)
        omega_BR_B = omega_BN_B - dcm_BR @ omega_RN_R
        omega_RN_B = dcm_BR @ omega_RN_R
        omega_RN_dot_B = dcm_BR @ omega_RN_dot_R

        # compute control vector
        feedforward = self.inertia_tensor @ (omega_RN_dot_B - skew_symmetric(omega_BN_B) @ omega_RN_B) - estimated_disturbance
        control_vector = -self.K * sigma_BR - self.P @ omega_BR_B + feedforward + skew_symmetric(omega_RN_B) @ self.inertia_tensor @ omega_RN_B
        
        return control_vector
