import numpy as np
from numpy.typing import ArrayLike

from ..attitude.mrp import mrp_to_dcm
from ..attitude.quaternion import quaternion_to_dcm

class BodyStateContext:

    def __init__(self,
                 position_N: ArrayLike | None = None,
                 velocity_N: ArrayLike | None = None,
                 dcm_BN: ArrayLike | None = None,
                 angular_velocity_BN: ArrayLike | None = None):

        self.position_N = np.asarray(position_N, dtype = float).reshape(3)
        self.velocity_N = np.asarray(velocity_N, dtype = float).reshape(3)
        self.dcm_BN = np.asarray(dcm_BN, dtype = float).reshape(3, 3)
        self.angular_velocity_BN = np.asarray(angular_velocity_BN, dtype = float).reshape(3)

class MRPStateProvider:

    def __init__(self,
                 position_function = None,
                 velocity_function = None):

        self.position_function = position_function
        self.velocity_function = velocity_function

    def build_context(self,
                      t: float,
                      rotational_state: ArrayLike) -> BodyStateContext:
        
        # rotational body state
        rotational_state = np.asarray(rotational_state, dtype = float).reshape(6)
        sigma_BN = rotational_state[0:3]
        omega_BN_B = rotational_state[3:6]

        # translational body state
        position_N = None if self.position_function is None else np.asarray(self.position_function(t), dtype = float).reshape(3)
        velocity_N = None if self.velocity_function is None else np.asarray(self.velocity_function(t), dtype = float).reshape(3)

        current_state = BodyStateContext(position_N = position_N,
                                         velocity_N = velocity_N,
                                         dcm_BN = mrp_to_dcm(sigma_BN),
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
                      rotational_state: ArrayLike) -> BodyStateContext:
        
        # rotational body state
        rotational_state = np.asarray(rotational_state, dtype = float).reshape(7)
        quaternion_BN = rotational_state[0:4]
        omega_BN_B = rotational_state[4:7]

        # translational body state
        position_N = None if self.position_function is None else np.asarray(self.position_function(t), dtype = float).reshape(3)
        velocity_N = None if self.velocity_function is None else np.asarray(self.velocity_function(t), dtype = float).reshape(3)

        current_state = BodyStateContext(position_N = position_N,
                                         velocity_N = velocity_N,
                                         dcm_BN = quaternion_to_dcm(quaternion_BN),
                                         angular_velocity_BN = omega_BN_B)

        return current_state