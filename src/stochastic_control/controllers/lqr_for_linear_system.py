import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import solve_continuous_are, inv
from scipy.integrate import solve_ivp

# check symmetric positive definite
def is_SPD(matrix):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric positive definite
    if matrix == matrix.T and (np.linalg.cholesky(matrix) is True):
        return True

    else:
        return False

# check symmetric positive semi-definite
def is_PSD(matrix):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric positive semi-definite
    eigenvalues = np.linalg.eigvals(matrix)
    if matrix == matrix.T and (np.all(eigenvalues > 0) is True):
        return True

    else:
        return False

class InfiniteHorizonLQRController:

    def __init__(self, A, B, Q, R):

        # check Q, R
        if not is_SPD(Q):
            raise ValueError('Q is not symmetric positive definite matrix')

        if not is_PSD(R):
            raise ValueError('R is not symmetric positive semi-definite matrix')

        S = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ S)

    def control_vector(self,
                       error_state):

        return -self.K @ error_state

class FiniteHorizonLQRController:

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

        # check Q, Qf, R
        if not is_SPD(self.Q):
            raise ValueError('Q is not symmetric positive definite matrix')

        if not is_SPD(self.Qf):
                    raise ValueError('Qf is not symmetric positive definite matrix')

        if not is_PSD(self.R):
            raise ValueError('R is not symmetric positive semi-definite matrix')

    def differential_ricatti_equation(self,
                                      current_S: ArrayLike):

        # check parameter & solve floating point errors
        S = np.asarray(current_S, dtype = float)
        S = (S + S.T) / 2

        # get S derivative
        S_dot = - (S @ self.A + self.A.T @ S - S @ self.B @ inv(self.R) @ self.B.T @ S + self.Q)

        return S_dot.reshape(-1)

    def get_S(self,
              current_t: float):

        # check parameter
        current_t = float(current_t)

        # conditions
        t_span = (self.tf, current_t)
        t_eval = [current_t]
        initial_S = self.Qf
        size_S = self.Q.shape[0]

        # integrate
        sol = solve_ivp(func = self.differential_ricatti_equation, 
                        t_span = t_span, 
                        y0 = initial_S, 
                        method = 'RK45', 
                        t_eval = t_eval)

        # reshape & solve floating point errors
        S = sol.y[:, 1].reshape(size_S, size_S)
        S = (S + S.T) / 2

        return S

    # local trajectory stabilization
    def control_vector(self, 
                       t: float,
                       estimated_state: ArrayLike):

        # check parameters
        t = float(t)
        x_hat = np.asarray(estimated_state, dtype = float)

        # reference trajectory
        reference = self.reference_provider.get_reference(t)
        reference_x = reference.reference_x
        reference_u = reference.reference_u

        # get control vector
        S = self.get_S(t)
        control_vector = reference_u - inv(self.R) @ self.B.T @ S @ (x_hat - reference_x)

        return control_vector