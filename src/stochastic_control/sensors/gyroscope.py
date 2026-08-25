import numpy as np
from numpy.typing import ArrayLike

class Gyroscope:

    def __init__(self,
                 mean: ArrayLike,
                 noise_covariance: ArrayLike,
                 sampling_rate: float):

        self.mean = np.asarray(mean, dtype = float)
        self.noise_covariance = np.asarray(noise_covariance, dtype = float)
        self.sampling_rate = float(sampling_rate)

    def measure(self,
                ideal_omega_BN_B: ArrayLike,
                rng):

        measured_omega_BN_B = ideal_omega_BN_B + rng.multivariate_normal(self.mean, self.noise_covariance)

        return measured_omega_BN_B
    