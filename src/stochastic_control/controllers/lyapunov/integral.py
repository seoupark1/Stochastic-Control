import numpy as np
from numpy.typing import ArrayLike

from ...math_tools import skew_symmetric
from ...attitude.mrp import mrp_to_dcm, dcm_to_mrp

class IntegralLyapunovController:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 K: float,
                 P: ArrayLike,
                 reference_provider,
                 KI: ArrayLike,
                 estimated_disturbance_model = None):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.K = float(K)
        self.KI = None if KI is None else float(KI)
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
    
    def control_vector(self,
                       t: float,
                       estimated_rotational_state: ArrayLike,
                       integral_state: ArrayLike,
                       initial_angular_velocity_error: ArrayLike,
                       estimated_context_builder):

        # check parameters
        x_hat = np.asarray(estimated_rotational_state, dtype = float).reshape(6)
        eta = np.asarray(integral_state, dtype = float).reshape(3)
        omega_BR_B_0 = np.asarray(initial_angular_velocity_error, dtype = float).reshape(3)

        # estimated disturbance
        estimated_context = estimated_context_builder.build_context(t, x_hat)
        estimated_disturbance = np.zeros(3)

        if self.disturbance_model is not None:
            estimated_disturbance = self.disturbance_model.torque(t, estimated_context)

        if self.KI is None:
            raise ValueError('KI is required to compute integral control vector')

        # body state
        omega_BN_B = x_hat[3:6]

        # tracking error
        reference = self.reference_provider.get_reference(t)
        dcm_BR, sigma_BR_B, omega_BR_B = self.get_tracking_error(t, x_hat)
        omega_RN_B = dcm_BR @ reference.omega_RN_R
        omega_RN_dot_B = dcm_BR @ reference.omega_RN_dot_R

        # compute z
        z = self.K * eta + self.inertia_tensor @ (omega_BR_B - omega_BR_B_0)

        # compute control vector & pass eta_dot to solve_ivp integrator and get next step's eta
        attitude_feedback = - self.K * sigma_BR_B
        omega_feedback = - self.P @ omega_BR_B
        integral_feedback = - self.P @ (self.KI * z)
        feedforward_term = self.inertia_tensor @ (omega_RN_dot_B - skew_symmetric(omega_BN_B) @ omega_RN_B)
        gyroscopic_term = skew_symmetric(omega_BN_B) @ self.inertia_tensor @ omega_BN_B

        control_vector = attitude_feedback + omega_feedback + integral_feedback + feedforward_term + gyroscopic_term - estimated_disturbance
        eta_dot = sigma_BR_B

        return control_vector, eta_dot