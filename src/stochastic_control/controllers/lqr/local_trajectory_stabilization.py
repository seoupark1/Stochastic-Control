import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import inv
from scipy.integrate import solve_ivp
from scipy.optimize._numdiff import approx_derivative

from ...math_tools import is_PSD, is_SPD

# for both time variant linear & non-linear system
class LocalTrajectoryStabilizationLQRController:

    def __init__(self, 
                 Q: ArrayLike, 
                 R: ArrayLike, 
                 Qf: ArrayLike, 
                 tf: float,
                 reference_provider,
                 continuous_dynamics_function = None,
                 continuous_jacobian_function = None,
                 A = None,
                 B = None):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)

        self.tf = float(tf)
        self.reference_provider = reference_provider

        self.continuous_dynamics_function = continuous_dynamics_function
        self.continuous_jacobian_function = continuous_jacobian_function
        self.A = None if A is None else np.asarray(A, dtype = float)
        self.B = None if B is None else np.asarray(B, dtype = float)

        # check Q, Qf, R
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_PSD(self.Qf):
            raise ValueError('Qf is not symmetric positive semi-definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

        self.riccati_solution = None
        self.R_inv = inv(self.R)

    def get_jacobians(self, t: float):

        # linear system
        if self.A is not None and self.B is not None and self.continuous_dynamics_function is None:
            return self.A, self.B

        # non-linear system
        elif self.A is None and self.B is None and self.continuous_dynamics_function is not None: 

            # reference trajectory
            reference = self.reference_provider.get_reference(t)
            x_d = reference.reference_x
            u_d = reference.reference_u

            if self.continuous_jacobian_function is None:

                A = approx_derivative(fun = lambda x: self.continuous_dynamics_function(t, x, u_d),
                                    x0 = x_d,
                                    method = '3-point')
                
                B = approx_derivative(fun = lambda u: self.continuous_dynamics_function(t, x_d, u),
                                    x0 = u_d,
                                    method = '3-point')

                return A, B

            else:
                return self.continuous_jacobian_function(t, x_d, u_d)

        # errors
        else:
            raise ValueError('Input Data Combination (A, B, dynamics function) Should Be Changed')

    def riccati_ode(self, 
                    t: float, 
                    S: ArrayLike):

        # check parameter
        n = self.Qf.shape[0]
        S = np.asarray(S, dtype = float).reshape(n, n)

        # solve floating point error
        S = (S + S.T) / 2

        # time varying A, B jacobians
        A, B = self.get_jacobians(t)

        # get S derivative
        S_dot = - (S @ A + A.T @ S - S @ B @ self.R_inv @ B.T @ S + self.Q)

        return S_dot.reshape(-1)

    def solve_riccati_ode(self):

        if self.riccati_solution is not None:
            return

        t_span = (self.tf, 0.0)

        self.riccati_solution = solve_ivp(fun = self.riccati_ode,
                                          t_span = t_span,
                                          y0 = self.Qf.reshape(-1),
                                          method = 'RK45',
                                          dense_output = True)

    def get_S(self, t: float):

        # t = tf case
        if t == self.tf:
            return self.Qf

        # integrate
        self.solve_riccati_ode()
    
        n = self.Qf.shape[0]

        S = self.riccati_solution.sol(t).reshape(n, n)

        return (S + S.T) / 2

    # local trajectory stabilization
    def control_vector(self, 
                       t: float,
                       estimated_state: ArrayLike):

        # check parameter
        x_hat = np.asarray(estimated_state, dtype = float).reshape(-1)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        x_d = reference.reference_x
        u_d = reference.reference_u

        # get current B & S
        _, B = self.get_jacobians(t)
        S = self.get_S(t)

        # get control vector
        control_vector = u_d - self.R_inv @ B.T @ S @ (x_hat - x_d)

        return control_vector.reshape(-1)