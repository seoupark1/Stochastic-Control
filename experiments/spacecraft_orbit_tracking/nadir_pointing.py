import numpy as np
from collections.abc import Callable

from stochastic_control.attitude.mrp import dcm_to_mrp

class NadirPointingReference:

    def __init__(self,
                 orbit_provider: Callable):

        self.orbit_provider = orbit_provider

    def nadir_pointing(self,
                       t: float):

        r_N, v_N = self.orbit_provider.get_state(t)

        # planet frame
        p_1 = r_N / np.linalg.norm(r_N)
        p_2 = np.cross(r_N, v_N) / np.linalg.norm(np.cross(r_N, v_N))
        p_3 = np.cross(p_1, p_2)
        dcm_NP = np.column_stack((p_1, p_2, p_3))

        # spacecraft reference frame
        dcm_RP = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
        dcm_RN = dcm_RP @ dcm_NP.T

        # spacecraft reference mrp
        sigma_RN = dcm_to_mrp(dcm_RN)

        # spacecraft reference angular velocity
        omega_RN_N = np.cross(r_N, v_N) / np.vdot(r_N, r_N)
        omega_RN_R = dcm_RN @ omega_RN_N

        return np.concatenate((sigma_RN, omega_RN_R))

    def angular_acceleration(self,
                             t: float):
        
        r_N, v_N = self.orbit_provider.get_state(t)
        omega_RN_R = self.nadir_pointing(t)[3:6]
        
        omega_RN_R_dot = (-2) * np.vdot(r_N, v_N) / np.vdot(r_N, r_N) * omega_RN_R

        return omega_RN_R_dot