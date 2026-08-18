import numpy as np
from numpy.typing import ArrayLike

from stochastic_control.math_tools import is_PSD

class GaussianNoise:

    def __init__(self,
                 mean: ArrayLike,
                 covariance: ArrayLike):

        if not np.allclose(covariance, covariance.T):
            raise ValueError('Covariance must be symmetric')

        if not is_PSD(covariance):
            raise ValueError('Covariance must be positive semi-definite')

        self.mean = mean
        self.covariance = covariance

    def get_sample(self):

        return np.random.multivariate_normal(self.mean, self.covariance)