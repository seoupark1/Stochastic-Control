import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

class TrajectoryReferenceState:

    def __init__(self,
                 reference_x: ArrayLike,
                 reference_u: ArrayLike):
        
        reference_x = np.asarray(reference_x, dtype = float)
        reference_u = np.asarray(reference_u, dtype = float)

class TrajectoryReferenceProvider:

    def __init__(self,
                 reference_x_function: Callable,
                 reference_u_function: Callable):

        self.reference_x_function = reference_x_function
        self.reference_u_function = reference_u_function

    def get_reference(self,
                      t: float):

        reference_x = np.asarray(self.reference_x_function(t), dtype = float)
        reference_u = np.asarray(self.reference_u_function(t), dtype = float)

        return TrajectoryReferenceState(reference_x, reference_u)