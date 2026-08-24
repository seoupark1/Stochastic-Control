import numpy as np
from numpy.typing import ArrayLike

class ReactionWheel:

    def __init__(self,
                 max_torque: ArrayLike):

        self.max_torque = np.asarray(max_torque, dtype = float).reshape(-1)

    def saturation(self,
                   )