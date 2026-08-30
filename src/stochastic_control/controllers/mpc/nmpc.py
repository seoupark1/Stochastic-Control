import numpy as np
import scipy as sp
import cvxpy as cp

from scipy.linalg import block_diag
from numpy.typing import ArrayLike
from collections.abc import Callable

class NMPCController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 P: ArrayLike,
                 N: int,
                 dt: float,
                 reference_provider,
                 continuous_nonlinear_dynamics: Callable):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.P = np.asarray(P, dtype = float)

        self.N = int(N)
        self.dt = float(dt)
        
        self.reference_provider = reference_provider
        self.f = continuous_nonlinear_dynamics

        # size
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

    def discrete_nonlinear_dynamics(self,
                                    t: float,
                                    currennt_state: ArrayLike,
                                    control: ArrayLike):

        # inputs 
        t = float(t)
        x_current = np.asarray(currennt_state, dtype = float)
        u = np.asarray(control, dtype = float)

        # runge-kutta 4th order method
        k1 = self.f(t, x_current, u)
        k2 = self.f(t + self.dt / 2, x_current + self.dt * k1 / 2, u)
        k3 = self.f(t + self.dt / 2, x_current + self.dt * k2 / 2, u)
        k4 = self.f(t + self.dt, x_current + self.dt * k3, u)

        return x_current + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_gradient(self,
                     X_bar: ArrayLike,
                     U_bar: ArrayLike):

        # inputs
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        gradient_x = np.zeros(self.N, self.n)
        gradient_u = np.zeros(self.N, self.m)

        for k in range(self.N):

            gradient_u[k, :] = 2 * self.R * U_bar[k, :]

            if k == (self.N - 1):
                # terminal x
                gradient_x[k, :] = 2 * self.P * X_bar[k, :]
            else:
                gradient_x[k, :] = 2 * self.Q * X_bar[k, :]

        return np.block([gradient_x.reshape(-1), gradient_u.reshape(-1)])

    def get_hessian(self):

        hessian = []

        for k in range(2 * self.N):
            if k < (self.N - 1):
                hessian.append(self.Q)

            if k == (self.N - 1):
                hessian.append(self.P)

            if k > (self.N - 1):
                hessian.append(self.R)

        return block_diag([hessian])
        
    def objective_function(self,
                           current_state: ArrayLike,
                           X_bar: ArrayLike,
                           U_bar: ArrayLike):

        # inputs
        x0 = np.asarray(current_state, dtype = float).reshape(-1)
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        x_terminal = X_bar[self.N - 1, :].T
        u_initial = U_bar[0, :].T

        # initial & terminal cost
        terminal_cost = cp.quad_form(x_terminal, self.P)
        initial_cost = cp.quad_form(x0, self.Q) + cp.quad_form(u_initial, self.R)
        cost = initial_cost + terminal_cost

        # total cost
        for k in range(self.N - 1):

            # k step state & control
            x = X_bar[k, :].T
            u = U_bar[k + 1, :].T

            cost += cp.quad_form(x, self.Q) + cp.quad_form(u, self.R)

        return cp.Minimize(cost)


    

