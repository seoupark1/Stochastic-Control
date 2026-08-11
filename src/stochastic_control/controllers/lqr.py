import numpy as np
from scipy.linalg import solve_continuous_are

class InfiniteHorizonLQRController:

    def __init__(self, A, B, Q, R):

        S = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ S)

    def control_vector(self,
                       error_state):

        return -self.K @ error_state

class FiniteHorizonLQRController:

    def __init__(self, A, B, Q, R, Qf):

        self.Q = Q
        if self.Q != self.Q.T:
            raise ValueError('Q matrix should be ')
        
        self.Qf = Qf


'''A, B, Q, R, Qf 행렬이 만족해야 하는 symmetric positive definite 혹은 semi-definite 처음에 확인, differential ricatti eqn은 따로 구해야함'''