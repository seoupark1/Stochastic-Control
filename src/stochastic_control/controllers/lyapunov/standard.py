import numpy as np
from numpy.typing import ArrayLike

from ...math_tools import skew_symmetric
from ...attitude.mrp import mrp_to_dcm, dcm_to_mrp

class StandardLyapunovController:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 K: float,
                 P: ArrayLike,
                 reference_provider,
                 estimated_disturbance_model = None):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.K = float(K)
        self.P = np.asarray(P, dtype = float).reshape(3,3)

        self.reference_provider = reference_provider
        self.disturbance_model = estimated_disturbance_model

    def get_tracking_error(self,
                           t: float,
                           estimated_rotational_state: ArrayLike):

        # check parameter
        x_hat = np.asarray(estimated_rotational_state, dtype = float).reshape(6)

        # body state
        sigma_BN_B = x_hat[0:3]
        omega_BN_B = x_hat[3:6]

        # reference state
        reference = self.reference_provider.get_reference(t)
        sigma_RN_R = reference.sigma_RN
        omega_RN_R = reference.omega_RN_R

        dcm_BN = mrp_to_dcm(sigma_BN_B)
        dcm_RN = mrp_to_dcm(sigma_RN_R)
        dcm_BR = dcm_BN @ dcm_RN.T

        sigma_BR_B = dcm_to_mrp(dcm_BR)
        omega_BR_B = omega_BN_B - dcm_BR @ omega_RN_R

        return dcm_BR, sigma_BR_B, omega_BR_B
    
    def lyapunov_function(self,
                          t: float,
                          estimated_rotational_state: ArrayLike):

        # tracking error
        _, sigma_BR_B, omega_BR_B = self.mrp_tracking_error(t, estimated_rotational_state)

        return (1/2) * omega_BR_B.T @ self.inertia_tensor @ omega_BR_B + 2 * self.K * np.log(1 + sigma_BR_B.T @ sigma_BR_B)

    def control_vector(self,
                       t: float,
                       estimated_rotational_state: ArrayLike,
                       estimated_context_builder):

        # check parameter
        x_hat = np.asarray(estimated_rotational_state, dtype = float).reshape(6)

        # body state
        omega_BN_B = x_hat[3:6]

        # estimated disturbance
        estimated_context = estimated_context_builder.build_context(t, x_hat)
        estimated_disturbance = np.zeros(3)

        if self.disturbance_model is not None:
            estimated_disturbance = self.disturbance_model.torque(t, estimated_context)

        # tracking error
        dcm_BR, sigma_BR_B, omega_BR_B = self.get_tracking_error(t, estimated_rotational_state)

        # reference
        reference = self.reference_provider.get_reference(t)
        omega_RN_B = dcm_BR @ reference.omega_RN_R
        omega_RN_dot_B = dcm_BR @ reference.omega_RN_dot_R

        # control vector terms
        attitude_feedback = -self.K * sigma_BR_B
        omega_feedback = - self.P @ omega_BR_B
        feedforward_term = self.inertia_tensor @ (omega_RN_dot_B - skew_symmetric(omega_BN_B) @ omega_RN_B)
        gyroscopic_term = skew_symmetric(omega_BN_B) @ self.inertia_tensor @ omega_BN_B
        control_vector = attitude_feedback + omega_feedback + feedforward_term + gyroscopic_term - estimated_disturbance
        
        return control_vector