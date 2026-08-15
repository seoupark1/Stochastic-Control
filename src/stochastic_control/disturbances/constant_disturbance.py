import numpy as np
from numpy.typing import ArrayLike
from ..providers.body_state import BodyStateContext

class ConstantDisturbance:

    def __init__(self,
                 constant_disturbance_B: ArrayLike):

        self.constant_disturbance = np.asarray(constant_disturbance_B, dtype = float).reshape(3)

    def torque(self,
               t: float,
               context: BodyStateContext):

        return self.constant_disturbance