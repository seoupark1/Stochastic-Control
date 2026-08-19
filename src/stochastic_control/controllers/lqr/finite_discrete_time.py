import numpy as np

from numpy.typing import ArrayLike
from numpy.linalg import inv

from stochastic_control.math_tools import is_PSD

class DiscreteTimeFiniteHorizonLQRController:

    def __init__(self,
                 A: ArrayLike,
                 B: ArrayLike,
                 Q: ArrayLike,
                 R: ArrayLike,
                 Qf: ArrayLike,
                 N: float):

        self.A = np.asarray(A, dtype = float)
        self.B = np.asarray(B, dtype = float)

        matrix_list = [Q, R, Qf]

        # check symmetric positive semi-definite
        for matrix in matrix_list:
            if not np.allclose(matrix, matrix.T):
                raise ValueError(f'{matrix} must be symmetric')
            
            if not is_PSD(matrix):
                raise ValueError(f'{matrix} must be positive semi-definite')

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)
        self.N = float(N)

        # make S & K list
        self.S = np.zeros(self.N + 1)
        self.K = np.zeros(self.N)
        self.S[N] = self.Qf

        for k in range(N-1, -1, -1):
            # discrete time riccati equation
            term1 = self.Q
            term2 = self.A.T @ self.S[k+1] @ self.A
            term3 = self.A.T @ self.S[k+1] @ self.B @ inv(self.R + self.B.T @ self.S[k+1] @ self.B) @ self.B.T @ self.S[k+1] @ self.A

            self.S[k] = term1 + term2 - term3
            self.K[k] = inv(self.R + self.B.T @ self.S[k+1] @ self.B) @ self.B.T @ self.S[k+1] @ self.A

    def control_vector(self,
                       k_step: float,
                       estimated_state: ArrayLike):

        x_hat = np.asarray(estimated_state, dtype = float)

        return -self.K[k_step] @ x_hat