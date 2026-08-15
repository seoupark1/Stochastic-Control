import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import solve_continuous_are, solve

from ..math_tools import is_PSD, is_SPD

class InfiniteHorizonLQRController:

    def __init__(self,
                 A: ArrayLike, 
                 B: ArrayLike, 
                 Q: ArrayLike, 
                 R: ArrayLike):

        self.A = np.asarray(A, dtype = float)
        self.B = np.asarray(B, dtype = float)
        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)

        # check Q, R
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

    def control_vector(self,
                       estimated_state: ArrayLike):

        # check parameter
        estimated_state = np.asarray(estimated_state, dtype = float).reshape(-1)

        S = solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = solve(self.R, self.B.T @ S)

        return -K @ estimated_state