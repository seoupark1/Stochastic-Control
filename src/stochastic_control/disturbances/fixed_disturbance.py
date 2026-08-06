import numpy as np
from numpy.typing import ArrayLike
from ..states.context import StateContext

class FixedDisturbance:

    def __init__(self,
                 fixed_disturbance_B: ArrayLike):

        self.fixed_disturbance = fixed_disturbance_B

    def torque(self,
               t: float,
               context: StateContext):

        return self.fixed_disturbance