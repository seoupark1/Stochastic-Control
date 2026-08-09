import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:

    def __init__(self, A, B, Q, R):

        S = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ S)

    def control_vector(self,
                       error_state):

        return -self.K @ error_state