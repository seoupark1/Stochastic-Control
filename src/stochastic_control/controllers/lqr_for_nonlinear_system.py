import numpy as np
from scipy.optimize._numdiff import approx_derivative

from numpy.typing import ArrayLike
from collections.abc import Callable

from scipy.linalg import inv
from scipy.integrate import solve_ivp

from ..math_tools import is_SPD


class TimeVaryingLQRController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 Qf: ArrayLike,
                 tf: float,
                 dynamics_function: Callable,
                 reference_provider):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)

        self.tf = float(tf)
        self.dynamics_function = dynamics_function
        self.reference_provider = reference_provider

        # sizes
        self.x_size = self.Q.shape[0]
        self.u_size = self.R.shape[0]

        # check Q, Qf, R
        if not is_SPD(self.Q):
            raise ValueError('Q is not symmetric positive definite matrix')

        if not is_SPD(self.Qf):
            raise ValueError('Qf is not symmetric positive definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

        # initial(t = tf) trajectory
        tf_reference = self.reference_provider.get_reference(self.tf)
        x_d_tf = tf_reference.reference_x

        # initial Sxx, Sx
        self.initial_Sxx = self.Qf
        self.initial_Sx = -self.Qf @ x_d_tf
        self.initial_Ss = np.concatenate((self.initial_Sxx.reshape(-1), self.initial_Sx))

    def get_jacobians(self,
                      t: float):

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        # get A, B jacobians
        A = approx_derivative(fun = self.dynamics_function,
                              x0 = x_d,
                              method = '3-point',
                              args = u_d)
        
        B = approx_derivative(fun = lambda u: self.dynamics_function(x_d, u),
                              x0 = u_d,
                              method = '3-point')

        return A, B

    def ricatti_ode(self,
                    t: float,
                    Sxx_and_Sx: ArrayLike):

        # check parameters
        t = float(t)
        Sxx_and_Sx = np.asarray(Sxx_and_Sx, dtype = float)

        # current Sxx and Sx
        n = self.x_size
        Sxx = Sxx_and_Sx[0:n*n].reshape(n, n)
        Sx = Sxx_and_Sx[n*n:]

        # check floating point error
        Sxx = (Sxx + Sxx.T) / 2
        
        # get current A, B
        A, B = self.get_jacobians(t)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        Sxx_dot = - (self.Q - Sxx @ B @ inv(self.R) @ B.T @ Sxx + Sxx @ A + A.T @ Sxx)
        Sx_dot = - ( -self.Q @ x_d + (A.T - Sxx @ B @ inv(self.R) @ B.T) @ Sx + Sxx @ B @ u_d)

        return np.concatenate((Sxx_dot.reshape(-1), Sx_dot))

    def get_Sxx_and_Sx(self,
                       t: float):
        
        # check parameter
        t = float(t)

        # conditions
        t_span = (self.tf, t)
        t_eval = [t]
        n = self.x_size

        # integrate
        sol = solve_ivp(func = self.ricatti_ode, 
                        t_span = t_span, 
                        y0 = self.initial_Ss,
                        method = 'RK45', 
                        t_eval = t_eval)

        # results
        Sxx = sol.y[0:n*n, 0].reshape(n, n)
        Sxx = (Sxx + Sxx.T) / 2
        Sx = sol.y[n*n:, 0]

        return Sxx, Sx

    def control_vector(self,
                       t: float,
                       estimated_state: ArrayLike):

        # check parameters
        t = float(t)
        x_hat = np.asarray(estimated_state, dtype = float)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        u_d = reference.reference_u

        # get current A, B
        _, B = self.get_jacobians(t)

        # Sxx, Sx
        Sxx, Sx = self.get_Sxx_and_Sx(t)

        control_vector = u_d - inv(self.R) @ B.T @ (Sxx @ x_hat + Sx)

        return control_vector