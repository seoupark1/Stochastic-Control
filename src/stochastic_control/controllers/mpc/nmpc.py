import numpy as np
import cvxpy as cp
from numpy.typing import ArrayLike

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
                 reference_control_function = None,
                 discrete_nonlienar_dynamics = None,
                 continuous_nonlinear_dynamics = None,
                 control_bound = None,
                 state_bound = None):

        self.Q = np.asarray(Q, dtype = float)
        self.R = np.asarray(R, dtype = float)
        self.P = np.asarray(P, dtype = float)

        # check symmetry & positive (semi) definite
        if not is_PSD(self.Q):
            raise ValueError('Q is not symmetric positive semi-definite matrix')

        if not is_PSD(self.P):
            raise ValueError('P is not symmetric positive semi-definite matrix')

        if not is_SPD(self.R):
            raise ValueError('R is not symmetric positive definite matrix')

        self.N = int(N)
        self.dt = float(dt)

        self.reference_control_function = reference_control_function
        self.discrete_f = discrete_nonlienar_dynamics
        self.continuous_f = continuous_nonlinear_dynamics

        if self.discrete_f is None and self.continuous_f is None:
            raise ValueError('At least one dynamics must be provided')

        if self.discrete_f is not None and self.continuous_f is not None:
            raise ValueError('Only one dynamics is required')

        self.state_bound = state_bound
        self.control_bound = control_bound
    
        self.hessian = self.get_hessian()

        # size
        self.n = self.Q.shape[0]
        self.m = self.R.shape[0]

        # warm start
        self.warm_start_u = None

        self.build_qp()

    def discrete_nonlinear_dynamics(self,
                                    t: float,
                                    current_state: ArrayLike,
                                    control: ArrayLike):

        # inputs 
        t = float(t)
        x = np.asarray(current_state, dtype = float).reshape(-1)
        u = np.asarray(control, dtype = float).reshape(-1)

        if self.discrete_f is not None:
            return self.discrete_f(t, current_state, control)

        else: 
            # runge-kutta 4th order method
            k1 = self.continuous_f(t, x, u)
            k2 = self.continuous_f(t + self.dt / 2, x + self.dt * k1 / 2, u)
            k3 = self.continuous_f(t + self.dt / 2, x + self.dt * k2 / 2, u)
            k4 = self.continuous_f(t + self.dt, x + self.dt * k3, u)

            return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_gradient(self,
                     t: float,
                     X_bar: ArrayLike,
                     U_bar: ArrayLike):

        # inputs
        t = float(t)
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        gradient_x = np.zeros((self.N, self.n))
        gradient_u = np.zeros((self.N, self.m))

        for k in range(self.N):

            tk = t + k * self.dt

            if self.reference_control_function is None:
                reference_u = np.zeros(self.m)
            else:
                reference_u = np.asarray(self.reference_control_function(tk), dtype = float).reshape(-1)

            gradient_u[k, :] = 2 * self.R @ (U_bar[k, :] - reference_u)

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

    def get_jacobians(self,
                      tk: float,
                      x_k_bar: ArrayLike,
                      u_k_bar: ArrayLike):
        
        # inputs
        tk = float(tk)
        x_k_bar = np.asarray(x_k_bar, dtype = float).reshape(-1)
        u_k_bar = np.asarray(u_k_bar, dtype = float).reshape(-1)

        # A, B jacobians
        A = approx_derivative(fun = lambda x: self.discrete_nonlinear_dynamics(tk, x, u_k_bar),
                              x0 = x_k_bar,
                              method = '2-point')
            
        B = approx_derivative(fun = lambda u: self.discrete_nonlinear_dynamics(tk, x_k_bar, u),
                              x0 = u_k_bar,
                              method = '2-point')

        return A, B

    def get_defects(self,
                    t: float,
                    current_state: ArrayLike,
                    X_bar: ArrayLike,
                    U_bar: ArrayLike):

        # inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1) # x(0)
        X_bar = np.asarray(X_bar, dtype = float).reshape(self.N, self.n) # x(1) ~ x(N)
        U_bar = np.asarray(U_bar, dtype = float).reshape(self.N, self.m) # u(0) ~ u(N-1)

        defects = []

        for k in range(self.N):

            tk = t + self.dt * k
            u_k_bar = U_bar[k, :]

            if k == 0:
                x_k_bar = x0
            
            else:
                x_k_bar = X_bar[k-1, :]

            defect = X_bar[k, :] - self.discrete_nonlinear_dynamics(tk, x_k_bar, u_k_bar)
            defects.append(defect)

        return np.concatenate(defects)

    def build_qp(self):

        # variable (del_z)
        self.del_z = cp.Variable(self.N * (self.n + self.m))
        del_x = self.del_z[0 : self.N * self.n]  # del_x(1) ~ del_x(N)
        del_u = self.del_z[self.N * self.n : ] # del_u(0) ~ del_u(N-1)

        # parameters (gradient, X_bar, U_bar, A, B, defect, constraints)
        self.gradient_parameter = cp.Parameter(self.N * (self.n + self.m))
        self.X_bar_parameter = cp.Parameter((self.N, self.n))
        self.U_bar_parameter = cp.Parameter((self.N, self.m))

        self.A_parameters = []
        self.B_parameters = []
        self.defect_parameters = []
        constraints = []
 
        # dynamics constraint
        for k in range(self.N):

            A_k = cp.Parameter((self.n, self.n))
            B_k = cp.Parameter((self.n, self.m))
            defect_k = cp.Parameter(self.n)

            self.A_parameters.append(A_k)
            self.B_parameters.append(B_k)
            self.defect_parameters.append(defect_k)

            if k == 0:
                del_x_k = np.zeros(self.n)

            else:
                del_x_k = del_x[(k - 1) * self.n : k * self.n]

            del_x_next = del_x[k * self.n : (k + 1) * self.n]
            del_u_k = del_u[k * self.m : (k + 1) * self.m]

            constraints.append(defect_k == A_k @ del_x_k + B_k @ del_u_k - del_x_next)

        # control constraint
        if self.control_bound is not None:
            
            lb, ub = self.control_bound

            for k in range(self.N):

                del_u_k = del_u[k * self.m : (k + 1) * self.m]
                u_k_bar = self.U_bar_parameter[k, :]

                if lb is not None:
                    constraints.append(del_u_k >= lb - u_k_bar)
                
                if ub is not None:
                    constraints.append(del_u_k <= ub - u_k_bar)

        # state constraint
        if self.state_bound is not None:
        
            lb, ub = self.state_bound

            for k in range(self.N):

                del_x_k = del_x[k * self.n : (k + 1) * self.n]
                x_k_bar = self.X_bar_parameter[k, :]

                if lb is not None:
                    constraints.append(del_x_k >= lb - x_k_bar)
                
                if ub is not None:
                    constraints.append(del_x_k <= ub - x_k_bar)

        # build objective & problem
        objective = cp.Minimize((1/2) * cp.quad_form(self.del_z, self.hessian) + self.gradient_parameter @ self.del_z)
        self.qp = cp.Problem(objective, constraints)

    def update_qp(self,
                  t: float,
                  current_state: ArrayLike,
                  X_bar: ArrayLike,
                  U_bar: ArrayLike):

        self.X_bar_parameter.value = X_bar
        self.U_bar_parameter.value = U_bar
        self.gradient_parameter.value = self.get_gradient(t, X_bar, U_bar)

        # update A, B, defect, constraints
        for k in range(self.N):

            tk = t + k * self.dt

            if k == 0:
                x_k_bar = current_state
            
            else:
                x_k_bar = X_bar[k - 1, :]

            x_next_bar = X_bar[k, :]
            u_k_bar = U_bar[k, :]

            A_k, B_k = self.get_jacobians(tk, x_k_bar, u_k_bar)
            defect_k = x_next_bar - self.discrete_nonlinear_dynamics(tk, x_k_bar, u_k_bar)

            self.A_parameters[k].value = A_k
            self.B_parameters[k].value = B_k
            self.defect_parameters[k].value = defect_k

    def corresponding_X_bar(self,
                            t: float,
                            current_state: ArrayLike,
                            U_bar: ArrayLike):

        x_k_bar = np.asarray(current_state, dtype = float).reshape(-1) # x0
        X_bar = np.zeros((self.N, self.n)) # x1 ~ xN

        for k in range(self.N):

            tk = t + k * self.dt

            u_k_bar = U_bar[k, :]
            x_k_bar = self.discrete_nonlinear_dynamics(tk, x_k_bar, u_k_bar)

            X_bar[k, :] = x_k_bar

        return X_bar

    def control_vector(self,
                       t: float,
                       current_state: ArrayLike,
                       max_iteration: int,
                       alpha: float,
                       del_z_tolerance: float,
                       defect_tolerance: float):

        # inputs
        t = float(t)
        alpha = float(alpha)
        del_z_tolerance = float(del_z_tolerance)
        defect_tolerance = float(defect_tolerance)
        max_iteration = int(max_iteration)

        x0 = np.asarray(current_state, dtype = float).reshape(-1)

        # warm start
        if self.warm_start_u is None:
            U_bar = np.zeros((self.N, self.m))

        else:
            U_bar = self.warm_start_u.copy()

        X_bar = self.corresponding_X_bar(t, x0, U_bar)

        # save previous optimal u for qp failure
        backup_U_bar = U_bar.copy()
        backup_X_bar = X_bar.copy()

        # check alpha
        if not 0 < alpha <= 1:
            raise ValueError('Alpha must be between 0 and 1')

        status = None
        iterations = 0

        for j in range(max_iteration):

            self.update_qp(t, x0, X_bar, U_bar)

            self.qp.solve(solver = cp.OSQP,
                          warm_start = True)

            if self.qp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):

                status = 'qp failed'

                # use previous optimal u
                U_bar = backup_U_bar
                X_bar = backup_X_bar
                break

            iterations += 1

            # optimal correction
            optimal_del_z = self.del_z.value
            optimal_del_x = optimal_del_z[0 : self.N * self.n].reshape(self.N, self.n)
            optimal_del_u = optimal_del_z[self.N * self.n : ].reshape(self.N, self.m)

            # update correction
            X_bar += alpha * optimal_del_x
            U_bar += alpha * optimal_del_u

            # when correction is small enough
            if np.linalg.norm(optimal_del_z) < del_z_tolerance:
                
                defects = self.get_defects(t, x0, X_bar, U_bar)

                # when defect is small enough
                if np.linalg.norm(defects) < defect_tolerance:

                    status = 'converged'
                    break

            # status update at final iteration
            if j == max_iteration - 1:

                status = 'max iteration'
        
        optimal_u = U_bar[0, :].reshape(-1)

        histories = {'status': status,
                     'iterations': iterations}

        if status != 'qp failed':
            self.warm_start_u = np.concatenate((U_bar[1 :, :], U_bar[-1 :, :]), axis = 0)

        return optimal_u, histories