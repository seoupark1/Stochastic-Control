import numpy as np
from numpy.typing import ArrayLike, NDArray
from stochastic_control.attitude.math import skew_symmetric

class Inertia:

    def __init__(self,
                 inertia_tensor: ArrayLike,
                 mass: float = None):

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float)
        self.mass = mass

    # get principal inertia tensor & principal axes
    def principal_values(self):
        # get eigenvalues & eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(self.inertia_tensor)

        if np.linalg.det(eig_vecs) < 0:
            eig_vecs[:, 1] *= -1

        return eig_vals, eig_vecs

    def parallel_axis_shift(self,
                            displacement: ArrayLike) -> NDArray[np.float64]:

        if self.mass is None:
            raise ValueError('Mass is required for parallel axis shift')
        
        r = np.asarray(displacement, dtype = float).reshape(3)

        return self.inertia_tensor + self.mass * skew_symmetric(r) @ skew_symmetric(r).T




