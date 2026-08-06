import numpy as np
from scipy.integrate import solve_ivp

from stochastic_control.dynamics.rigid_body import RigidBody
from stochastic_control.controllers.lyapunov import LyapunovController
from stochastic_control.providers.attitude_reference import MRPReferenceProvider
from stochastic_control.providers.state import MRPStateProvider
from experiments.lyapunov_controller.reference_state import reference_attitude, reference_omega

reference_provider = MRPReferenceProvider(sigma_function = reference_attitude[0],
                                          omega_function = reference_omega[0], 
                                          omega_dot_function = reference_omega[1])

inertia_tensor = np.diag([100, 75, 80])
sigma_BN_B_0 = np.array([0.1, 0.2, -0.1])
omega_BN_B_0 = np.radians(np.array([3, 1, -2]))
initial_rotational_state = np.concatenate((sigma_BN_B_0, omega_BN_B_0))
control_gain = 5
damping_matrix = 10 * np.eye(3)
integral_control_gain = 0.005
frequency = 0.05
unknown_disturbance = np.array([0.5, -0.3, 0.2])
tspan = (0, 300)

controller = LyapunovController(inertia_tensor, 
                                control_gain, 
                                damping_matrix, 
                                reference_provider,
                                estimated_disturbance_model = None)


rigid_body = RigidBody(inertia_tensor)
state_provider = MRPStateProvider()

def standard_controller(t,
                        estimated_rotational_state):

    control_vector = controller.mrp_control_vector(t,
                                                   estimated_rotational_state,
                                                   state_provider)

    total_torque = control_vector + unknown_disturbance

    return rigid_body.mrp_derivatives(estimated_rotational_state,
                                      total_torque)

result = solve_ivp(standard_controller,
                   t_span = tspan,
                   y0 = initial_rotational_state,
                   str = 'RK45')


'''sigma, omega가 안정화되는 모습 그래프로 그리고, sigma_BR, omega_BR이 0으로 수렴하는 모습도 그래프로 그리기 (tracking error를 return 함수 만들어야함)'''