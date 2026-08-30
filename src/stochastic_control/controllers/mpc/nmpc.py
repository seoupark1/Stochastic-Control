import numpy as np
import scipy as sp
import cvxpy as cp

from numpy.typing import ArrayLike
from collections.abc import Callable

from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import block_diag

from stochastic_control.math_tools import is_PSD

class NMPCController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 P: ArrayLike,
                 N: int,
                 dt: float,
                 continuous_nonlinear_dynamics: Callable,
                 control_bound = None,
                 state_constraint_function = None):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.P = np.asarray(P, dtype = float)

        self.N = int(N)
        self.dt = float(dt)
        self.f = continuous_nonlinear_dynamics

        self.state_constraint_function = state_constraint_function
        self.control_bound = control_bound

        # size
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        # check symmetry & positive (semi) definite
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_PSD(self.P):
            raise ValueError('P is not symmetric positive semi-definite matrix')

        if not is_PSD(self.R):
            raise ValueError('R is not symmetric positive semi-definite matrix')

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

        gradient_x = np.zeros((self.N, self.n))
        gradient_u = np.zeros((self.N, self.m))

        for k in range(self.N):

            gradient_u[k, :] = 2 * self.R @ U_bar[k, :]

            if k == (self.N - 1):
                # terminal x
                gradient_x[k, :] = 2 * self.P @ X_bar[k, :]
            else:
                gradient_x[k, :] = 2 * self.Q @ X_bar[k, :]

        return np.concatenate([gradient_x.reshape(-1), gradient_u.reshape(-1)])

    def get_hessian(self):

        hessian = []

        for k in range(2 * self.N):
            if k < (self.N - 1):
                hessian.append(2 * self.Q)

            elif k == (self.N - 1):
                hessian.append(2 * self.P)

            elif k > (self.N - 1):
                hessian.append(2 * self.R)

        return block_diag(*hessian)
        
    def objective(self,
                  X_bar: ArrayLike,
                  U_bar: ArrayLike):

        # inputs
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        # gradient & hessian
        gradient = self.get_gradient(X_bar, U_bar)
        hessian = self.get_hessian()

        # correction
        del_z = cp.Variable(hessian.shape[0])

        # objective
        objective = cp.Minimize((1/2) * cp.quad_form(del_z, hessian) + gradient.T @ del_z)

        return del_z, objective

    def dynamics_constraint(self,
                            del_z,
                            t: float,
                            current_state: ArrayLike,
                            X_bar: ArrayLike,
                            U_bar: ArrayLike):
        
        # inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1) # x(0)
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        constraints = []

        for k in range(self.N):

            # time
            tk = t + self.dt * k

            # k state properties
            if k == 0:
                x_k_bar = x0
                del_x_k = np.zeros(self.n)
            
            else:
                x_k_bar = X_bar[k-1, :]
                del_x_k = del_z[(k-1) * self.n : k * self.n]

            # k control properties
            u_k_bar = U_bar[k, :]
            del_u_k = del_z[self.n * self.N + k * self.m : self.n * self.N + (k+1) * self.m]

            # k+1 state properties
            x_next_bar = X_bar[k, :]
            del_x_next = del_z[k * self.n : (k+1) * self.n]

            # A, B jacobians & defect
            A = approx_derivative(fun = lambda x: self.discrete_nonlinear_dynamics(tk, x, u_k_bar),
                                  x0 = x_k_bar,
                                  method = '3-point')
            
            B = approx_derivative(fun = lambda u: self.discrete_nonlinear_dynamics(tk, x_k_bar, u),
                                  x0 = u_k_bar,
                                  method = '3-point')

            defect = x_next_bar - self.discrete_nonlinear_dynamics(tk, x_k_bar, u_k_bar)

            constraints.append(defect == A @ del_x_k + B @ del_u_k - del_x_next)

        return constraints
    
    def control_constraint(self,
                           del_u,
                           U_bar: ArrayLike):

        if self.control_bound is None:
            return []
        
        lb, ub = self.control_bound
        constraints = []

        for k in range(self.N):

            u_k_bar = U_bar[k, :]

            if lb is not None:
                constraints.append(del_u >= lb - u_k_bar)
            
            if ub is not None:
                constraints.append(del_u <= ub - u_k_bar)

        return constraints

    def state_constraint(self,
                         del_x,
                         t: float,
                         current_state: ArrayLike,
                         X_bar: ArrayLike):
        
        if self.state_constraint_function is None:
            return []

        # inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1)
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n)

        constraints = []

        for k in range(self.N):

            x_k_bar = X_bar[k, :]
            del_x_k = del_x[k * self.n : (k+1) * self.n]

            gradient_g = approx_derivative(fun = lambda x: self.state_constraint_function(x),
                                           x0 = x_k_bar,
                                           method = '3-point')

            constraints.append(self.state_constraint_function(x_k_bar) + gradient_g.T @ del_x_k <= 0)

        return constraints

    def solve_nmpc(self,
                   initial_guess,
                   )