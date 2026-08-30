import numpy as np

from numpy.typing import ArrayLike
from collections.abc import Callable

from scipy.optimize._numdiff import approx_derivative

class NMPCController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 P: ArrayLike,
                 N: int,
                 dt: float,
                 reference_provider,
                 nonlinear_dynamics_function: Callable):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.P = np.asarray(P, dtype = float)

        self.N = int(N)
        self.dt = float(dt)
        
        self.reference_provider = reference_provider
        self.f = nonlinear_dynamics_function

        # size
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        if self.Q.shape != self.P.shape:
            raise ValueError('Q and P should be the same size matrix')

    def get_jacobians(self,
                      t: float,
                      state: ArrayLike,
                      control: ArrayLike):

        # inputs
        t = float(t)
        state = np.asarray(state, dtype = float).reshape(-1)
        control = np.asarray(control, dtype = float).reshape(-1)

        A = approx_derivative(fun = lambda x: self.f(t, x, control),
                              x0 = state,
                              method = '3-point')
        
        B = approx_derivative(fun = lambda u: self.f(t, state, u),
                              x0 = control,
                              method = '3-point')

        return A, B

    def get_defect(self,
                   t: float,
                   state: ArrayLike,
                   control: ArrayLike,
                   next_state: ArrayLike):
        
        # inputs
        t = float(t)
        state = np.asarray(state, dtype = float).reshape(-1)
        control = np.asarray(control, dtype = float).reshape(-1)
        next_state = np.asarray(next_state, dtype = float).reshape(-1)

        dk = next_state - self.f(t, state, control)

        return dk

    def get_dynamics_constraints(self,
                                 t: float,
                                 current_state: ArrayLike,
                                 X_bar: ArrayLike,
                                 U_bar: ArrayLike):

        # inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1)
        X_bar = np.asarray(X_bar, dtype = float) # x_(1) ~ x_(N)
        U_bar = np.asarray(U_bar, dtype = float) # u_(0) ~ u_(N-1)

        # check size
        if x0.shape != (self.n):
            raise ValueError('x0 should be (n)')
        
        if X_bar.shape != (self.N, self.n):
            raise ValueError('X_bar should be (N, n)')

        if U_bar.shape != (self.N, self.m):
            raise ValueError('U_bar should be (N, m)')

        # zeros for dynamics_A, dynamics_B
        dynamics_A = np.zeros((self.N * self.n, self.N * self.n + self.N * self.m))
        dynamics_B = np.zeros(self.N * self.n)

        # get B0, d0
        _, B0 = self.get_jacobians(t, x0, U_bar[0, :])
        d0 = - (X_bar[0, :] - B0 @ U_bar[0, :])
        dynamics_A[0 : self.n, 0 : self.n] = np.eye(self.n)
        dynamics_A[0 : self.n, self.N : self.N + self.n] = B0
        dynamics_B[0 : self.n] = -d0

        for k in range(self.N):

            if k == 0:
                continue

            tk = t + self.dt * k
            xk_bar = X_bar[k-1, :]
            x_k_next_bar = X_bar[k, :]
            uk_bar = U_bar[k, :]

            Ak, Bk = self.get_jacobians(tk, xk_bar, uk_bar)
            dk = self.get_defect(tk, xk_bar, uk_bar, x_k_next_bar)

            dynamics_A[k * self.n : (k+1) * self.n, (k-1) * self.n : k * self.n] = Ak
            dynamics_A[k * self.n : (k+1) * self.n, k * self.n : (k+1) * self.n] = np.eye(self.n)
            dynamics_A[k * self.n : (k+1) * self.n, k * self.n + self.N * self.n : (k+1) * self.n + self.N * self.n] = Bk
            dynamics_B[k * self.N : (k+1) * self.N] = -dk

        return dynamics_A, dynamics_B




    

