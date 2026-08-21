import numpy as np

from stochastic_control.controllers.lqr.local_trajectory_stabilization import LocalTrajectoryStabilizationLQRController
from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter

class NonlinearCompensator:

    def __init__(self,
                 estimator: ExtendedKalmanFilter,
                 controller: LocalTrajectoryStabilizationLQRController):

        self.estimator = estimator
        self.controller = controller

    def estimate(self,
                 control_vector,
                 measurement_vector):

        self.estimator.prediction(control_vector)
        self.estimator.correction(measurement_vector)

    def control_vector(self,
                       t: float):
        
        return self.controller.control_vector(t, self.estimator.x)