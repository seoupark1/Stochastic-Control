import numpy as np

from numpy.typing import ArrayLike
from numpy.linalg import inv

from stochastic_control.math_tools import is_PSD, is_SPD

class DiscreteTimeFiniteHorizonLQRController:

    def __init__(self,
                 A: ArrayLike,
                 B: ArrayLike,
                 Q: ArrayLike,
                 R: ArrayLike,
                 Qf: ArrayLike,
                 N: int):

        self.A = np.asarray(A, dtype = float)
        self.B = np.asarray(B, dtype = float)
        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.Qf = np.asarray(Qf, dtype = float)
        self.N = int(N)

        # check symmetric positive (semi) definite
        for name, matrix in (('Q', self.Q), ('Qf', self.Qf), ('R', self.R)):
            # symmetry
            if not np.allclose(matrix, matrix.T):
                raise ValueError(f'{name} must be symmetric')

            # positive (semi) definite
            if name == 'R':
                if not is_SPD(R):
                    raise ValueError('R must be positive semi-definite')

            else: 
                if not is_PSD(matrix):
                    raise ValueError(f'{name} must be positive semi-definite')
                
        # make S & K list
        n = self.A.shape[0]
        m = self.B.shape[1]

        self.S = np.zeros((self.N + 1, n, n))
        self.K = np.zeros((self.N, m, n))
        self.S[N] = self.Qf

        for k in range(N-1, -1, -1):
            # discrete time riccati equation
            term1 = self.Q
            term2 = self.A.T @ self.S[k+1] @ self.A
            term3 = self.A.T @ self.S[k+1] @ self.B @ inv(self.R + self.B.T @ self.S[k+1] @ self.B) @ self.B.T @ self.S[k+1] @ self.A

            self.S[k] = term1 + term2 - term3
            self.K[k] = inv(self.R + self.B.T @ self.S[k+1] @ self.B) @ self.B.T @ self.S[k+1] @ self.A

    def control_vector(self,
                       k_step: int,
                       estimated_state: ArrayLike):

        x_hat = np.asarray(estimated_state, dtype = float).reshape(-1)

        return -self.K[k_step] @ x_hat