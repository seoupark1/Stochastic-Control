import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import inv
from src.stochastic_control.attitude.math import skew_symmetric
from src.stochastic_control.attitude.mrp import mrp_derivative
from src.stochastic_control.attitude.quaternion import quaternion_derivative

class RigidBody:

    def __init__(self,
                 inertia_tensor):
        
        self.I = inertia_tensor

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
                             torque : ArrayLike):
        
        L = np.asarray(torque, dtype = float)
        omega = np.asarray(angular_velocity, dtype = float).reshape(3)
        gyroscopic = - skew_symmetric(omega) @ self.I @ omega

        return inv(self.I) @ (gyroscopic + L)

    def mrp_state_derivatives(self,
                              t: float,
                              state: ArrayLike,
                              disturbance, # disturbance는 작용하는 모든 외란을 torque 매서드로 합친 객체
                              control_torque_func) -> NDArray[np.float64]:

        state = np.asarray(state, dtype = float).reshape(6)
        sigma = state[0:3]
        omega = state[3:6]
        control_torque = control_torque_func(t, state)
        disturbance_torque = disturbance.torque(t, state)
        total_torque = control_torque + disturbance_torque

        # derivatives
        sigma_dot = mrp_derivative(sigma, omega)
        omega_dot = self.angular_acceleration(omega, total_torque)

        return np.concatenate((sigma_dot, omega_dot))

    def quaternion_state_derivative(self,
                                    t: float,
                                    state: ArrayLike,
                                    disturbance, # disturbance는 작용하는 모든 외란을 torque 매서드로 합친 객체
                                    control_torque_func) -> NDArray[np.float64]:

        state = np.asarray(state, dtype = float).reshape(7)
        quaternion = state[0:4]
        omega = state[4:7]
        control_torque = control_torque_func(t, state)
        disturbance_torque = disturbance.torque(t, state)
        total_torque = control_torque + disturbance_torque

        # derivatives
        quaternion_dot = quaternion_derivative(quaternion, omega)
        omega_dot = self.angular_acceleration(omega, total_torque)

        return np.concatenate((quaternion_dot, omega_dot))