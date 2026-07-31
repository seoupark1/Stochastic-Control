import numpy as np
from numpy.typing import ArrayLike
from ..states.context import StateContext
from ..attitude.mrp import mrp_to_dcm
from ..attitude.quaternion import quaternion_to_dcm

class MRPStateProvider:

    def __init__(self,
                 position_function = None,
                 velocity_function = None):

        self.position_function = position_function
        self.velocity_function = velocity_function

    def build_context(self,
                      t: float,
                      state: ArrayLike):
        
        sigma_BN = state[0:3]
        omega_BN_B = state[3:6]

        current_state = StateContext(position_N = self.position_function(t),
                                     velocity_N = self.velocity_function(t),
                                     attitude_BN = mrp_to_dcm(sigma_BN),
                                     angular_velocity_BN = omega_BN_B)

        return current_state

class QuaternionStateProvider:

    def __init__(self,
                 position_function = None,
                 velocity_function = None):

        self.position_function = position_function
        self.velocity_function = velocity_function

    def build_context(self,
                      t: float,
                      state: ArrayLike):
        
        quaternion_BN = state[0:4]
        omega_BN_B = state[4:7]

        current_state = StateContext(position_N = self.position_function(t),
                                     velocity_N = self.velocity_function(t),
                                     attitude_BN = quaternion_to_dcm(quaternion_BN),
                                     angular_velocity_BN = omega_BN_B)

        return current_state