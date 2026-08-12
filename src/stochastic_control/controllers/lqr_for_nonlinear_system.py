import numpy as np
import sympy as sp

from numpy.typing import ArrayLike
from collections.abc import Callable

from scipy.linalg import inv
from scipy.integrate import solve_ivp

from ..math_tools import is_SPD, is_PSD


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

        # check Q, Qf, R
        if not is_SPD(self.Q):
            raise ValueError('Q is not symmetric positive definite matrix')

        if not is_SPD(self.Qf):
            raise ValueError('Qf is not symmetric positive definite matrix')

        if not is_PSD(self.R):
            raise ValueError('R is not symmetric positive semi-definite matrix')

        # get time-varying A, B function
        x, u = sp.symbols('x u')
        f = self.dynamics_function

        dfdx = sp.diff(f, x)
        dfdu = sp.diff(f, u)

        self.A_function = sp.lambdify((x, u), dfdx, 'numpy')
        self.B_function = sp.lambdify((x, u), dfdu, 'numpy')

        # initial(t = tf) trajectory
        tf_reference = self.reference_provider.get_reference(self.tf)
        x_d_tf = tf_reference.reference_x

        # initial Sxx, Sx, S0
        self.initial_Sxx = self.Qf
        self.initial_Sx = -self.Qf @ x_d_tf
        self.initial_S0 = x_d_tf.T @ self.Qf @ x_d_tf

    def get_jacobians(self,
                      t: float):

        # check parameter
        t = float(t)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        # get current A, B jacobians
        A = np.asarray(self.A_function(x_d, u_d), dtype = float)
        B = np.asarray(self.B_function(x_d, u_d), dtype = float)

        return A, B

    def ricatti_quadratic_term(self,
                               t: float,
                               current_Sxx: ArrayLike):

        # check parameters
        t = float(t)
        Sxx = np.asarray(current_Sxx, dtype = float)

        # get current A, B
        A, B = self.get_jacobians(t)

        Sxx_dot = - (self.Q - Sxx @ B @ inv(self.R) @ B.T @ Sxx + Sxx @ A + A.T @ Sxx)

        return Sxx_dot.reshape(-1)

    def ricatti_linear_term(self,
                            t: float,
                            current_Sxx: ArrayLike,
                            current_Sx: ArrayLike):

        # check parameters
        t = float(t)
        Sxx = np.asarray(current_Sxx, dtype = float)
        Sx = np.asarray(current_Sx, dtype = float)

        # get current A, B
        A, B = self.get_jacobians(t)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        Sx_dot = - ( -self.Q @ x_d + (A.T - Sxx @ B @ inv(self.R) @ B.T) @ Sx + Sxx @ B @ u_d)

        return Sx_dot.reshape(-1)

    def ricatti_constant_term(self,
                              t: float,
                              current_Sx: ArrayLike):

        # check parameters
        t = float(t)
        Sx = np.asarray(current_Sx, dtype = float)

        # get current A, B
        _, B = self.get_jacobians(t)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        S0_dot = - (x_d.T @ self.Q @ x_d - Sx.T @ B @ inv(self.R) @ B.T @ Sx + 2 * Sx.T @ B @ u_d)

        return S0_dot.reshape(-1)
        
    def get_all_S(self,
                  t: float):
        
        # check parameter
        t = float(t)

        # conditions
        t_span = (self.tf, t)
        t_eval = [t]

        # sizes
        size_Sxx = self.initial_Sxx.shape[0]
        size_Sx = self.initial_Sx.shape[0]

        # integrate
        Sxx_sol = solve_ivp(func = self.ricatti_quadratic_term, 
                            t_span = t_span, 
                            y0 = self.initial_Sxx,
                            method = 'RK45', 
                            t_eval = t_eval)

        Sxx = Sxx_sol.y[:, 1].reshape(size_Sxx, size_Sxx)
        Sxx = (Sxx + Sxx.T) / 2

        # ricatti_linear_term 함수에서 Sxx에 특정값을 대입해 t와 Sx의 함수로 만들고 싶음
        updated_ricatti_linear_term = self.ricatti_linear_term(t, Sxx, )

        Sx_sol = solve_ivp(func = updated_ricatti_linear_term, 
                           t_span = t_span, 
                           y0 = self.initial_Sx, 
                           method = 'RK45', 
                           t_eval = t_eval)

        Sx = Sx_sol.y[:, 1].reshape(size_Sx, size_Sx)
        Sx = (Sx + Sx.T) / 2

        S0 = self.ricatti_constant_term(t, Sx) * (self.tf - t)
        S0 = (S0 + S0.T) / 2

        return Sxx, Sx, S0

    def control_vector(self,
                       t: float,
                       estimated_state: ArrayLike):

        # check parameters
        t = float(t)
        x_hat = np.asarray(estimated_state, dtype = float)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        # get current A, B
        _, B = self.get_jacobians(t)

        # Sxx, Sx, S0
        Sxx, Sx, _ = self.get_all_S(t)

        control_vector = u_d - inv(self.R) @ B.T @ (Sxx @ x_hat + Sx)

        return control_vector