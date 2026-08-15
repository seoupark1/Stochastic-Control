import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import inv
from scipy.integrate import solve_ivp

from ...math_tools import is_SPD, is_PSD

# for linear system
class LinearQuadraticOptimalTrackingController:

    def __init__(self,
                 A: ArrayLike,
                 B: ArrayLike,
                 Q: ArrayLike,
                 R: ArrayLike,
                 Qf: ArrayLike,
                 tf: float,
                 reference_provider):

        self.A = np.asarray(A, dtype = float)
        self.B = np.asarray(B, dtype = float)
        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)

        self.tf = float(tf)
        self.reference_provider = reference_provider

        # sizes
        self.x_size = self.Q.shape[0]
        self.u_size = self.R.shape[0]

        # check Q, Qf, R
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_PSD(self.Qf):
            raise ValueError('Qf is not symmetric positive semi-definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

        # initial(t = tf) trajectory
        tf_reference = self.reference_provider.get_reference(self.tf)
        x_d_tf = tf_reference.reference_x

        # initial Sxx, Sx
        self.initial_Sxx = self.Qf
        self.initial_Sx = -self.Qf @ x_d_tf
        self.initial_Ss = np.concatenate((self.initial_Sxx.reshape(-1), self.initial_Sx))

    def riccati_ode(self,
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

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        Sxx_dot = - (self.Q - Sxx @ self.B @ inv(self.R) @ self.B.T @ Sxx + Sxx @ self.A + self.A.T @ Sxx)
        Sx_dot = - ( -self.Q @ x_d + (self.A.T - Sxx @ self.B @ inv(self.R) @ self.B.T) @ Sx + Sxx @ self.B @ u_d)

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
        sol = solve_ivp(func = self.riccati_ode, 
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

        # Sxx, Sx
        Sxx, Sx = self.get_Sxx_and_Sx(t)

        control_vector = u_d - inv(self.R) @ self.B.T @ (Sxx @ x_hat + Sx)

        return control_vector