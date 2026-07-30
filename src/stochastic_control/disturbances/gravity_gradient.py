import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable

class GravityGradient:

    def __init__(self,
                 body_position_function: Callable,
                 body_attitude_function: Callable,
                 inertia_tensor: ArrayLike,
                 mass: float,
                 gravitational_parameter: float):
        
        # position of the body measured in the inertial frame
        self.body_position_function = body_position_function

        # attitude(dcm) of the body measured in the inertial frame
        self.body_attitude_function = body_attitude_function

        self.inertia_tensor = np.asarray(inertia_tensor, dtype = float).reshape(3,3)
        self.mass = float(mass)
        self.mu = float(gravitational_parameter)

    def torque(self,
               t: float,
               state: ArrayLike) -> NDArray[np.float64]:

        state = np.asarray(state, dtype = float).reshape(-1)

        r_BN_N = np.asarray(self.body_position_function(t, state), dtype = float).reshape(3)
        C_BN = np.asarray(self.body_attitude_function(t, state), dtype = float).reshape(3,3)

        # displacement(r_BN) measured in the body frame
        r_BN_B = C_BN @ r_BN_N

        # distance between the Planet and the body
        r = np.linalg.norm(r_BN_N)

        torque = (3 * self.mu / r**5) * np.cross(r_BN_B, self.inertia_tensor @ r_BN_B)

        return torque

    def force(self,
              t: float,
              state: ArrayLike) -> NDArray[np.float64]:

        state = np.asarray(state, dtype = float).reshape(-1)

        r_BN_N = np.asarray(self.body_position_function(t, state), dtype = float).reshape(3)
        C_BN = np.asarray(self.body_attitude_function(t, state), dtype = float).reshape(3,3)

        # displacement(r_BN) measured in the body frame
        r_BN_B = C_BN @ r_BN_N

        # distance between the Planet and the body
        r = np.linalg.norm(r_BN_N)

        r_hat_B = r_BN_B / r
        middle_term = np.trace(self.inertia_tensor) - 5 * (r_hat_B.T @ self.inertia_tensor @ r_hat_B)
        force = (-self.mu * self.mass / r**3) * (1 + (3/(self.mass * r**2)) * (self.inertia_tensor + (1/2) * middle_term * np.eye(3))) @ r_BN_B

        return force