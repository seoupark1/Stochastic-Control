import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

class TrajectoryReferenceState:

    def __init__(self,
                 reference_x: ArrayLike,
                 reference_u: ArrayLike):
        
        self.reference_x = reference_x
        self.reference_u = reference_u

class TrajectoryReferenceProvider:

    def __init__(self,
                 reference_x_function: Callable,
                 reference_u_function: Callable):

        self.reference_x_function = reference_x_function
        self.reference_u_function = reference_u_function

    def get_reference(self,
                      t: float):

        reference_x = np.asarray(self.reference_x_function(t), dtype = float).reshape(-1)
        reference_u = np.asarray(self.reference_u_function(t), dtype = float).reshape(-1)

        return TrajectoryReferenceState(reference_x, reference_u)