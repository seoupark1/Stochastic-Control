import numpy as np
import cvxpy as cp

from numpy.typing import ArrayLike
from collections.abc import Callable

from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import block_diag

from stochastic_control.math_tools import is_PSD, is_SPD

class NMPCController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 P: ArrayLike,
                 N: int,
                 dt: float,
                 continuous_nonlinear_dynamics: Callable,
                 control_bound = None,
                 state_bound = None):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.P = np.asarray(P, dtype = float)

        self.N = int(N)
        self.dt = float(dt)
        self.f = continuous_nonlinear_dynamics

        self.state_bound= state_bound
        self.control_bound = control_bound

        # size
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        # check symmetry & positive (semi) definite
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_PSD(self.P):
            raise ValueError('P is not symmetric positive semi-definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

        # backup for warm start
        self.control_sequence = None

    def discrete_nonlinear_dynamics(self,
                                    t: float,
                                    current_state: ArrayLike,
                                    control: ArrayLike):

        # inputs 
        t = float(t)
        x = np.asarray(current_state, dtype = float).reshape(-1)
        u = np.asarray(control, dtype = float).reshape(-1)

        # runge-kutta 4th order method
        k1 = self.f(t, x, u)
        k2 = self.f(t + self.dt / 2, x + self.dt * k1 / 2, u)
        k3 = self.f(t + self.dt / 2, x + self.dt * k2 / 2, u)
        k4 = self.f(t + self.dt, x + self.dt * k3, u)

        return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

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

        return constraints, defect
    
    def control_constraint(self,
                           del_u,
                           U_bar: ArrayLike):

        if self.control_bound is None:
            return []
        
        lb, ub = self.control_bound
        constraints = []

        for k in range(self.N):

            u_k_bar = U_bar[k, :]
            del_u_k = del_u[k * self.m : (k+1) * self.m]

            if lb is not None:
                constraints.append(del_u_k >= lb - u_k_bar)
            
            if ub is not None:
                constraints.append(del_u_k <= ub - u_k_bar)

        return constraints

    def state_constraint(self,
                         del_x,
                         X_bar: ArrayLike):
        
        if self.state_bound is None:
            return []
        
        lb, ub = self.state_bound
        constraints = []

        for k in range(self.N):

            x_k_bar = X_bar[k, :]
            del_x_k = del_x[k * self.n : (k+1) * self.n]

            if lb is not None:
                constraints.append(del_x_k >= lb - x_k_bar)
            
            if ub is not None:
                constraints.append(del_x_k <= ub - x_k_bar)

        return constraints

    def initial_X_bar(self,
                      t: float,
                      current_state: ArrayLike,
                      U_bar: ArrayLike):
    
        # inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m)
    
        X_bar = np.zeros((self.N, self.n)) # x1 ~ xN

        # proper x_k which satisfies the dynamics (defect = 0)
        for k in range(self.N):
            tk = t + self.dt * k

            u_k_bar = U_bar[k, :]
            x_k_bar = self.discrete_nonlinear_dynamics(tk, x0, u_k_bar)

            X_bar[k, :] = x_k_bar

        return X_bar

    def control_vector(self,
                   t: float,
                   current_state: ArrayLike,
                   X_bar: ArrayLike,
                   U_bar: ArrayLike,
                   max_iteration: int,
                   alpha: float,
                   tolerance: float):

        # inputs
        t = float(t)
        alpha = float(alpha)
        tolerance = float(tolerance)
        max_iteration = int(max_iteration)

        x0 = np.asarray(current_state, dtype = float).reshape(-1)

        if self.control_sequence is None:
            U_bar = np.zeros((self.N, self.m))

        else: 
            U_bar = self.control_sequence

        X_bar = self.initial_X_bar(t, x0, U_bar)

        # check alpha
        if not 0 <= alpha <= 1:
            raise ValueError('Alpha must be between 0 and 1')

        for j in range(max_iteration):

            del_z, objective = self.objective(X_bar, U_bar)

            del_x = del_z[0 : self.N * self.n]
            del_u = del_z[self.N * self.n : ]

            # constraints
            dynamics, defect = self.dynamics_constraint(del_z, 
                                                        t, 
                                                        x0, 
                                                        X_bar, 
                                                        U_bar)
            
            control = self.control_constraint(del_u,
                                              U_bar)
            
            state = self.state_constraint(del_x,
                                          X_bar)

            constraints = dynamics + control + state

            # solve QP
            qp = cp.Problem(objective, constraints)
            qp.solve()

            if qp.status != cp.OPTIMAL:
                raise RuntimeError(f'QP failed because of {qp.status}')

            # optimal correction
            optimal_del_z = del_z.value

            optimal_del_x = optimal_del_z[0 : self.N * self.n].reshape(self.N, self.n)
            optimal_del_u = optimal_del_z[self.N * self.n : ].reshape(self.N, self.m)

            X_bar += alpha * optimal_del_x
            U_bar += alpha * optimal_del_u

            # when correction is small enough
            if np.linalg.norm(optimal_del_z) < tolerance and np.linalg.norm(defect) < tolerance:
                break

        optimal_u = U_bar[0, :].reshape(-1)
        self.control_sequence = np.concatenate((U_bar[1 :, : ], U_bar[-1 :, :]), axis = 0)

        return optimal_u, X_bar, U_bar

