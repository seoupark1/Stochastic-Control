import numpy as np
from numpy.typing import ArrayLike

from stochastic_control.estimators.kalman_filter import KalmanFilter
from stochastic_control.controllers.lqr.discrete_time_finite_horizon import DiscreteTimeFiniteHorizonLQRController

class DiscreteTimeFiniteHorizonLQGController:

    def __init__(self,
                 estimator: KalmanFilter,
                 controller: DiscreteTimeFiniteHorizonLQRController):

        self.estimator = estimator
        self.controller = controller

    def estimate(self,
                 control_vector: ArrayLike,
                 measurement_vector: ArrayLike):

        self.estimator.prediction(control_vector)
        self.estimator.correction(measurement_vector)

    def control_vector(self,
                       k_step: int):

        return self.controller.control_vector(k_step, self.estimator.x)