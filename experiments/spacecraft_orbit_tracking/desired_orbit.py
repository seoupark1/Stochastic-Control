import numpy as np

from stochastic_control.attitude.euler import ea313_to_dcm

class CircularOrbit:

    def __init__(self,
                 RAAN: float,
                 inclination: float,
                 initial_theta: float,
                 radius: float,
                 mu):

        self.RAAN = float(RAAN) # [rad]
        self.inclination = float(inclination) # [rad]
        self.initial_theta = float(initial_theta) # [rad]

        self.r = float(radius) # [m]
        self.mu = float(mu) # [m^3 / s^2]

        self.theta_dot = np.sqrt(self.mu / (self.r)**3) # [rad/s]
        self.dcm = ea313_to_dcm([self.RAAN, self.inclination, 0])

    def get_state(self,
                  t: float):

        # create 2D circular orbit
        theta = self.initial_theta + self.theta_dot * t
        r_N_2d = self.r * np.array([np.cos(theta), np.sin(theta), 0])
        v_N_2d = self.r * self.theta_dot * np.array([-np.sin(theta), np.cos(theta), 0])

        # 2D -> 3D
        r_N = self.dcm @ r_N_2d
        v_N = self.dcm @ v_N_2d

        return r_N.reshape(-1), v_N.reshape(-1)