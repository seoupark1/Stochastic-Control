import numpy as np
from numpy.typing import ArrayLike

from stochastic_control.attitude.mrp import mrp_shadow_set, mrp_to_dcm, dcm_to_mrp, rotation_vector_to_mrp

class StarTracker:

    def __init__(self,
                 mean: ArrayLike,
                 noise_covariance: ArrayLike):

        self.mean = np.asarray(mean, dtype = float)
        self.noise_covariance = np.asarray(noise_covariance, dtype = float)

    def measure(self,
                ideal_sigma_BN: ArrayLike,
                rng):

        ideal_sigma_BN = np.asarray(ideal_sigma_BN, dtype = float).reshape(-1)
        ideal_dcm_BN = mrp_to_dcm(ideal_sigma_BN)

        # change noise to small rotation
        noise_rotation_vector = rng.multivariate_normal(self.mean, self.noise_covariance)
        noise_sigma = rotation_vector_to_mrp(noise_rotation_vector)
        noise_dcm = mrp_to_dcm(noise_sigma)

        measured_mrp = dcm_to_mrp(noise_dcm @ ideal_dcm_BN)

        return mrp_shadow_set(measured_mrp)
    