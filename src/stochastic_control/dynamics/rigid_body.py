import numpy as np
from numpy.linalg import inv
from src.stochastic_control.attitude.tools import skew_symmetric

class RigidBody:

    def __init__(self,
                 inertia_tensor,
                 disturbance = None):

        self.inertia_tensor = inertia_tensor
        self.disturbance = disturbance

    def angular_velocity_derivative(self,
                                    omega):
        I = self.inertia_tensor
        L = self.disturbance
        gryoscopic = - skew_symmetric(omega) @ self.inertia_tensor @ omega
        omega_dot = inv(I) @ (gryoscopic + L)

        return omega_dot

    def rotational_enery(self,
                         omega):
        I = self.inertia_tensor

        return (1/2) * omega.T @ I  @ omega