import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import solve_continuous_are, inv
from scipy.integrate import solve_ivp

def is_SPD(matrix):


class InfiniteHorizonLQRController:

    def __init__(self, A, B, Q, R):

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
                 tf: float):

        self.A = np.asarray(A, dtype = float)
        self.B = np.asarray(B, dtype = float)
        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)

        self.tf = float(tf)

        # SPD 검증 코드
        #
        #
        #


    def differential_ricatti_equation(self,
                                      current_S: ArrayLike):

        S = np.asarray(current_S, dtype = float)
        S = (S + S.T) / 2

        S_dot = - (S @ self.A + self.A.T @ S - S @ self.B @ inv(self.R) @ self.B.T @ S + self.Q)

        return S_dot.reshape(-1)

    def get_S(self,
              current_t: float):

        # time conditions
        current_t = float(current_t)
        t_span = (self.tf, current_t)
        t_eval = [current_t]

        # S conditions
        initial_S = self.Qf
        size_S = self.Q.shape[0]

        sol = solve_ivp(func = self.differential_ricatti_equation, 
                        t_span = t_span, 
                        y0 = initial_S, 
                        method = 'RK45', 
                        t_eval = t_eval)

        current_S = sol.y[:, 1].reshape(size_S, size_S)
        current_S = (current_S + current_S.T) / 2

        return current_S

    def control_vector(self, 
                       t: float,
                       estimated_state: ArrayLike):

        # check parameters
        t = float(t)
        x_hat = np.asarray(estimated_state, dtype = float)

        # get control vector
        S = self.get_S(t)
        control_vector = -inv(self.R) @ self.B.T @ S @ x_hat

        return control_vector