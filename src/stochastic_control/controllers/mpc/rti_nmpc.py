import numpy as np
import cvxpy as cp
from numpy.typing import ArrayLike

from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import block_diag

from stochastic_control.math_tools import is_PSD, is_SPD

class RealTimeNMPCController:

    def __init__(self,
                 Q: ArrayLike,
                 R: ArrayLike,
                 P: ArrayLike,
                 N: int,
                 dt: float,
                 reference_control_function = None,
                 discrete_nonlinear_dynamics = None,
                 continuous_nonlinear_dynamics = None,
                 discrete_jacobian_function = None,
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
        self.discrete_f = discrete_nonlinear_dynamics
        self.continuous_f = continuous_nonlinear_dynamics
        self.discrete_jacobian_function = discrete_jacobian_function

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

        # for real time iteration
        self.X_bar = None
        self.U_bar = None

        self.build_qp()

    def discrete_dynamics(self,
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

    def nominal_trajectory(self,
                           t: float,
                           current_state: ArrayLike):

        # check inputs
        t = float(t)
        x0 = np.asarray(current_state, dtype = float).reshape(-1)

        self.X_bar = np.zeros((self.N + 1, self.n)) # x(0) ~ x(N)
        self.U_bar = np.zeros((self.N, self.m)) # u(0) ~ u(N-1)

        self.X_bar[0, :] = x0

        for k in range(self.N):

            tk = t + k * self.dt
            x_k_bar = self.X_bar[k, :]

            if self.reference_control_function is None:
                u_k_bar = np.zeros(self.m)

            else:
                u_k_bar = np.asarray(self.reference_control_function(tk), dtype = float).reshape(-1)

            self.U_bar[k, :] = u_k_bar
            self.X_bar[k + 1, :] = self.discrete_dynamics(tk, x_k_bar, u_k_bar)

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

        if self.discrete_jacobian_function is not None:
            return self.discrete_jacobian_function(tk, x_k_bar, u_k_bar)

        else:
            discrete_A = approx_derivative(fun = lambda x: self.discrete_dynamics(tk, x, u_k_bar),
                                           x0 = x_k_bar,
                                           method = '3-point')
                
            discrete_B = approx_derivative(fun = lambda u: self.discrete_dynamics(tk, x_k_bar, u),
                                           x0 = u_k_bar,
                                           method = '3-point')

            return discrete_A, discrete_B

    def build_qp(self):

        # variable (del_z)
        self.del_z = cp.Variable(self.N * (self.n + self.m))
        del_x = self.del_z[0 : self.N * self.n]  # del_x(1) ~ del_x(N)
        del_u = self.del_z[self.N * self.n : ] # del_u(0) ~ del_u(N-1)

        # parameters (A @ del_x0, gradient, X_bar, U_bar, A, B, defect, constraints)
        self.A_del_x0_parameter = cp.Parameter(self.n)
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
        
            del_x_next = del_x[k * self.n : (k + 1) * self.n]
            del_u_k = del_u[k * self.m : (k + 1) * self.m]

            if k == 0:
                A_del_x_k = self.A_del_x0_parameter
                constraints.append(defect_k == A_del_x_k + B_k @ del_u_k - del_x_next)

            else:
                del_x_k = del_x[(k - 1) * self.n : k * self.n]
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

    def preparation(self,
                    t: float):

        # check input
        t = float(t)

        self.X_bar_parameter.value = self.X_bar[1:, :]
        self.U_bar_parameter.value = self.U_bar
        self.gradient_parameter.value = self.get_gradient(t, self.X_bar[1:, :], self.U_bar)

        # update A, B, defect, constraints
        for k in range(self.N):

            tk = t + k * self.dt

            x_k_bar = self.X_bar[k, :]
            x_next_bar = self.X_bar[k + 1, :]
            u_k_bar = self.U_bar[k, :]

            A_k, B_k = self.get_jacobians(tk, x_k_bar, u_k_bar)
            defect_k = x_next_bar - self.discrete_dynamics(tk, x_k_bar, u_k_bar)

            self.A_parameters[k].value = A_k
            self.B_parameters[k].value = B_k
            self.defect_parameters[k].value = defect_k

    def feedback(self,
                 estimated_state: ArrayLike):

        # check input
        x0_hat = np.asarray(estimated_state, dtype = float).reshape(-1)

        A_0 = self.A_parameters[0].value
        del_x0 = x0_hat - self.X_bar[0, :]
        self.A_del_x0_parameter.value = A_0 @ del_x0

        # save previous optimal u for qp failure
        backup_U_bar = self.U_bar.copy()
        backup_X_bar = self.X_bar.copy()

        try:
            self.qp.solve(solver = cp.OSQP,
                          warm_start = True,
                          max_iter = 20000,
                          adaptive_rho = True)

        except cp.error.Solvererror:

            self.U_bar = backup_U_bar
            self.X_bar = backup_X_bar

            histories = {'status': 'qp_failed',
                         'iterations': 0}

            return backup_U_bar[0, :], histories

        if self.qp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.del_z.value is None:

            # use previous optimal u
            self.U_bar = backup_U_bar
            self.X_bar = backup_X_bar

            histories = {'status': 'qp_failed',
                         'iterations': 0}

            return backup_U_bar[0, :], histories

        # optimal correction
        optimal_del_z = self.del_z.value
        optimal_del_x = optimal_del_z[0 : self.N * self.n].reshape(self.N, self.n)
        optimal_del_u = optimal_del_z[self.N * self.n : ].reshape(self.N, self.m)

        # update correction
        self.X_bar[0, :] = x0_hat
        self.X_bar[1:, :] += optimal_del_x
        self.U_bar += optimal_del_u

        optimal_u = self.U_bar[0, :].reshape(-1)

        histories = {'status': self.qp.status,
                     'iterations': self.qp.solver_stats.num_iters}
        
        return optimal_u, histories

    def warm_start(self,
                   next_t: float):

        # check input
        next_t = float(next_t)

        # next step u: u(0) ~ u(N-1)
        self.U_bar[:-1, :] = self.U_bar[1:, :]
        self.U_bar[-1, :] = self.U_bar[-2, :]

        # next step x: x(1) ~ x(N)
        self.X_bar[:-1, :] = self.X_bar[1:, :]
        self.X_bar[-1, :] = self.discrete_dynamics(next_t + (self.N - 1) * self.dt, self.X_bar[-2, :], self.U_bar[-1, :])
