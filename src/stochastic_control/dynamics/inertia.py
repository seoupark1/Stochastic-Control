import numpy as np
from src.stochastic_control.attitude.tools import skew_symmetric

class Inertia:

    def __init__(self,
                 inertia_tensor,
                 mass = None):

        self.inertia_tensor = inertia_tensor
        self.mass = mass

    # get principal inertia tensor & principal axes
    def principal_values(self):
        # get eigenvalues & eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(self.inertia_tensor)

        return eig_vals, eig_vecs

    def parallel_axis_shift(self,
                            displacement):
        r = displacement
        shifted = self.inertia_tensor + self.mass * skew_symmetric(r) @ skew_symmetric(r).T

        return shifted




