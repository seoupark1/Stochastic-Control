import numpy as np
import matplotlib.pyplot as plt

from ..simulation import simulation
from stochastic_control.actuators.reaction_wheel import ReactionWheel

# star tracker performance
star_tracker_sampling_rate = 10 # [Hz]
star_tracker_accuracy = 5 # [arcsec]
star_tracker_std = np.deg2rad(star_tracker_accuracy / 3600)
star_tracker_noise_covariance = star_tracker_std**2 * np.eye(3)

# gyroscope performance
gyroscope_sampling_rate = 100 # [Hz]
gyroscope_std = 4.3633 * 1e-4 # [rad/s]
gyroscope_noise_covariance = gyroscope_std**2 * np.eye(3)

# mrp & omega uncertainty
mrp_std = np.tan(np.deg2rad(9) / 4) / np.sqrt(3)
omega_std = np.deg2rad(3) / np.sqrt(3)

# extreme case
initial_x_true = np.array([0.09, -0.09, -0.03, np.deg2rad(-7.5), np.deg2rad(-6), np.deg2rad(3)])
initial_x_hat = initial_x_true + np.array([-0.015, 0.006, -0.03, np.deg2rad(1.5), np.deg2rad(-0.9), np.deg2rad(0.6)])
initial_covariance = np.diag([mrp_std**2, mrp_std**2, mrp_std**2, omega_std**2, omega_std**2, omega_std**2])
motion_noise_covariance = np.diag([1e-9, 1e-9, 1e-9, 1e-7, 1e-7, 1e-7])

# control limiter
max_u_cmd_abs = [38.3882, 49.6486, 12.2470]
actuator = ReactionWheel((1/3) * max_u_cmd_abs)

extreme_result = simulation(initial_x_true = initial_x_true,
                            initial_x_hat = initial_x_hat,
                            initial_covariance = initial_covariance,
                            motion_noise_covariance = motion_noise_covariance,
                            star_tracker_noise_covariance = star_tracker_noise_covariance,
                            gyroscope_noise_covariance = gyroscope_noise_covariance,
                            simulation_tf = 60,
                            controller_tf = 120,
                            star_tracker_sampling_rate = star_tracker_sampling_rate,
                            gyroscope_sampling_rate = gyroscope_sampling_rate,
                            control_limiter = actuator)

time = extreme_result['time']

# tracking error
plt.subplot(2, 1, 1)
plt.plot(time, extreme_result['attitude_tracking_error'])
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [rad]')
plt.title('Nadir Pointing Attitude Tracking Error')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, extreme_result['omega_tracking_error'])
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity Error [rad/s]')
plt.title('Nadir Pointing Angular Velocity Tracking Error')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/tracking_error.png')
plt.close()

# estimation error
estimation_tf = 5
time_range = time <= estimation_tf
plt.subplot(2, 1, 1)
plt.plot(time[time_range], extreme_result['attitude_estimation_error'][time_range])
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [rad]')
plt.title('Nadir Pointing Attitude Estimation Error')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time[time_range], extreme_result['omega_estimation_error'][time_range])
plt.xlabel('time [s]')
plt.ylabel('Angular Velocity Error [rad/s]')
plt.title('Nadir Pointing Omega Estimation Error')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/estimation_error.png')
plt.close()

# saturated control (actual control)
plt.subplot(2, 1, 1)
plt.plot(time[0:-1], extreme_result['actual_control'][0], label = 'u_1')
plt.axhline(y = max_u_cmd_abs[0], linestyle = '-')
plt.axhline(y = - max_u_cmd_abs[0], linestyle = '-')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.subplot(2, 1, 2)
plt.plot(time[0:-1], extreme_result['actual_control'][1], label = 'u_2')
plt.axhline(y = max_u_cmd_abs[1], linestyle = '-')
plt.axhline(y = - max_u_cmd_abs[1], linestyle = '-')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.subplot(2, 1, 3)
plt.plot(time[0:-1], extreme_result['actual_control'][2], label = 'u_3')
plt.axhline(y = max_u_cmd_abs[2], linestyle = '-')
plt.axhline(y = - max_u_cmd_abs[2], linestyle = '-')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.suptitle('Nadir Pointing Actual Control')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/saturated_control.png')
plt.close()

# saturated control norm
plt.plot(time[0:-1], np.linalg.norm(extreme_result['actual_control'], axis = 0))
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('Nadir Pointing Actual Control Norm')
plt.grid(True)
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/saturated_control_norm.png')
plt.close()

# true gravity gradient
plt.subplot(2, 1, 1)
plt.plot(time, extreme_result['true_gravity_gradient_torque'][0], label = 'gg_1')
plt.plot(time, extreme_result['true_gravity_gradient_torque'][1], label = 'gg_2')
plt.plot(time, extreme_result['true_gravity_gradient_torque'][2], label = 'gg_3')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('True Gravity Gradient Torque of Spacecraft')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, np.linalg.norm(extreme_result['true_gravity_gradient_torque'], axis = 0))
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('True Gravity Gradient Torque Norm of Spacecraft')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/true_gravity_gradient.png')
plt.close()