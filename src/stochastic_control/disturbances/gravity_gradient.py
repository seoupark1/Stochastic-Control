import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..states.context import StateContext

class GravityGradient:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 gravitational_parameter: float):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.mu = float(gravitational_parameter)

    def torque(self,
               t: float,
               context: StateContext) -> NDArray[np.float64]:

        if context.position_N is None:
            raise ValueError('position_N is required to compute GravityGradient')

        if context.dcm_BN is None:
            raise ValueError('dcm_BN is required to compute GravityGradient')

        r_BN_N = context.position_N
        C_BN = context.dcm_BN

        # displacement(r_BN) measured in the body frame
        r_BN_B = C_BN @ r_BN_N

        # distance between the Planet and the body
        r = np.linalg.norm(r_BN_N)

        torque = (3 * self.mu / r**5) * np.cross(r_BN_B, self.inertia_tensor @ r_BN_B)

        return torque