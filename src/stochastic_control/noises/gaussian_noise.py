import numpy as np
from numpy.typing import ArrayLike

from stochastic_control.math_tools import is_PSD

class GaussianNoise:

    def __init__(self,
                 mean: ArrayLike,
                 covariance: ArrayLike):

        # check symmetric positive semi-definite
        if not np.allclose(covariance, covariance.T):
            raise ValueError('Covariance must be symmetric')

        if not is_PSD(covariance):
            raise ValueError('Covariance must be positive semi-definite')

        self.mean = np.asarray(mean, dtype = float).reshape(-1)
        self.covariance = np.asarray(covariance, dtype = float)

    def get_sample(self, rng):
        return rng.multivariate_normal(self.mean, self.covariance)