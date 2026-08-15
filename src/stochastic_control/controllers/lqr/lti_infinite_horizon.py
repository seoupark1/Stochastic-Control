import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import solve_continuous_are, solve

from ...math_tools import is_PSD, is_SPD

# for linear time-invariant system
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
        if not is_PSD(Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_SPD(R):
            raise ValueError('R is not symmetric positive definite matrix')

        S = solve_continuous_are(A, B, Q, R)
        self.K = solve(R, B.T @ S)

    def control_vector(self,
                       estimated_state: ArrayLike):

        # check parameter
        estimated_state = np.asarray(estimated_state, dtype = float).reshape(-1)

        return -self.K @ estimated_state