import numpy as np
from numpy.typing import ArrayLike

from ..attitude.math import skew_symmetric
from ..attitude.mrp import mrp_derivative
from ..attitude.quaternion import quaternion_derivative

class RigidBody:

    def __init__(self,
                 inertia_tensor):
        
        self.I = np.asarray(inertia_tensor, dtype = float).reshape(3, 3)
        
    def rotational_energy(self,
                         angular_velocity: ArrayLike):
        
        omega = np.asarray(angular_velocity, dtype = float).reshape(3)
        return (1/2) * omega.T @ self.I @ omega

    def angular_momentum(self,
                         angular_velocity: ArrayLike):
        
        omega = np.asarray(angular_velocity, dtype = float).reshape(3)
        return self.I @ omega

    def angular_acceleration(self,
                             angular_velocity: ArrayLike,
                             total_torque : ArrayLike):
        
        L = np.asarray(total_torque, dtype = float)
        omega = np.asarray(angular_velocity, dtype = float).reshape(3)
        gyroscopic = - skew_symmetric(omega) @ self.I @ omega

        return np.linalg.solve(self.I, gyroscopic + L)

    def mrp_derivatives(self,
                        rotational_state: ArrayLike,
                        total_torque: ArrayLike):

        rotational_state = np.asarray(rotational_state, dtype = float).reshape(6)
        total_torque = np.asarray(total_torque, dtype = float).reshape(3)

        # rotational state vectors
        sigma = rotational_state[0:3]
        omega = rotational_state[3:6]

        # derivatives
        sigma_dot = mrp_derivative(sigma, omega)
        omega_dot = self.angular_acceleration(omega, total_torque)

        return np.concatenate((sigma_dot, omega_dot))

    def quaternion_derivatives(self,
                               rotational_state: ArrayLike,
                               total_torque: ArrayLike):

        rotational_state = np.asarray(rotational_state, dtype = float).reshape(7)
        total_torque = np.asarray(total_torque, dtype = float).reshape(3)

        # rotational state vectors
        quaternion = rotational_state[0:4]
        omega = rotational_state[4:7]

        # derivatives
        quaternion_dot = quaternion_derivative(quaternion, omega)
        omega_dot = self.angular_acceleration(omega, total_torque)

        return np.concatenate((quaternion_dot, omega_dot))