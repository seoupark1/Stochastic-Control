import numpy as np

from stochastic_control.controllers.mpc.nmpc import NMPCController
from stochastic_control.estimators.extended_kalman_filter import ExtendedKalmanFilter

class EKFNMPCCompensator:

    def __init__(self,
                 estimator: ExtendedKalmanFilter,
                 controller: NMPCController):

        self.estimator = estimator
        self.controller = controller

    def estimate(self,
                 t,
                 control_vector,
                 measurement_vector):

        self.estimator.prediction(control_vector, t)
        self.estimator.correction(measurement_vector, t)

    def control_vector(self,
                       t,
                       X_bar,
                       U_bar,
                       max_iteration,
                       alpha,
                       del_z_tolerance):
        
        optimal_u, X_bar, U_bar = self.controller.control_vector(t,
                                                                 self.estimator.x,
                                                                 X_bar,
                                                                 U_bar,
                                                                 max_iteration,
                                                                 alpha,
                                                                 del_z_tolerance)
        return optimal_u