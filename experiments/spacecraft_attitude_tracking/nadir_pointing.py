import numpy as np
from collections.abc import Callable

from stochastic_control.attitude.mrp import dcm_to_mrp

class NadirPointingReference:

    def __init__(self,
                 orbit_provider: Callable):

        # orbit provider must have single variable t and return r_N, v_N 
        self.orbit_provider = orbit_provider

    def nadir_pointing(self,
                       t):

        r_N, v_N = self.orbit_provider(t)

        # mars frame
        m_1 = r_N / np.linalg.norm(r_N)
        m_2 = np.cross(r_N, v_N) / np.linalg.norm(np.cross(r_N, v_N))
        m_3 = np.cross(m_1, m_2)
        dcm_NM = np.column_stack((m_1, m_2, m_3))

        # reference frame
        dcm_RM = np.array([-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1])
        dcm_RN = dcm_RM @ dcm_NM.T

        # reference mrp
        sigma_RN = dcm_to_mrp(dcm_RN)

        # reference omega
        omega_RN_N = np.cross(r_N, v_N) / np.vdot(r_N, r_N)
        omega_RN_R = dcm_RN @ omega_RN_N

        return np.concatenate((sigma_RN, omega_RN_R)), dcm_RN

