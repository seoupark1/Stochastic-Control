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
max_u_cmd_abs = np.array([38.3882, 49.6486, 12.2470])
u_max = max_u_cmd_abs / 3
actuator = ReactionWheel(u_max)

unsaturated_result = simulation(initial_x_true = initial_x_true,
                                initial_x_hat = initial_x_hat,
                                initial_covariance = initial_covariance,
                                motion_noise_covariance = motion_noise_covariance,
                                star_tracker_noise_covariance = star_tracker_noise_covariance,
                                gyroscope_noise_covariance = gyroscope_noise_covariance,
                                simulation_tf = 60,
                                controller_tf = 120,
                                star_tracker_sampling_rate = star_tracker_sampling_rate,
                                gyroscope_sampling_rate = gyroscope_sampling_rate)

saturated_result = simulation(initial_x_true = initial_x_true,
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

time = saturated_result['time']
u_cmd = saturated_result['commanded_control']
u_actual = saturated_result['actual_control']

# tracking error
plt.subplot(2, 1, 1)
plt.plot(time, saturated_result['attitude_tracking_error'], label = 'saturated')
plt.plot(time, unsaturated_result['attitude_tracking_error'], label = 'unsaturated')
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [rad]')
plt.title('Nadir Pointing Attitude Tracking Error')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, saturated_result['omega_tracking_error'], label = 'saturated')
plt.plot(time, unsaturated_result['omega_tracking_error'], label = 'unsaturated')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity Error [rad/s]')
plt.title('Nadir Pointing Angular Velocity Tracking Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/tracking_error.png')
plt.close()

# estimation error
estimation_tf = 5
time_range = time <= estimation_tf
plt.subplot(2, 1, 1)
plt.plot(time[time_range], saturated_result['attitude_estimation_error'][time_range], label = 'saturated')
plt.plot(time[time_range], unsaturated_result['attitude_estimation_error'][time_range], label = 'unsaturated')
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [rad]')
plt.title('Nadir Pointing Attitude Estimation Error')
plt.grid(True)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time[time_range], saturated_result['omega_estimation_error'][time_range], label = 'saturated')
plt.plot(time[time_range], unsaturated_result['omega_estimation_error'][time_range], label = 'unsaturated')
plt.xlabel('time [s]')
plt.ylabel('Angular Velocity Error [rad/s]')
plt.title('Nadir Pointing Omega Estimation Error')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/estimation_error.png')
plt.close()

# control
fig, axes = plt.subplots(3, 1, figsize = (6.4, 11), sharex = True)

for i in range(3):
    axes[i].plot(time[0:-1], u_actual[i], label = 'u_actual')
    axes[i].plot(time[0:-1], u_cmd[i], label = 'u_cmd')
    axes[i].axhline(u_max[i], linestyle = '--')
    axes[i].axhline(-u_max[i], linestyle = '--')
    axes[i].set_ylabel(f'u_{i + 1} Torque [Nm]')
    axes[i].legend()
    axes[i].grid(True)

axes[2].set_xlabel('Time [s]')
fig.suptitle('Actual Control vs Commanded Control')
fig.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/saturated_control.png')
plt.close()

# true gravity gradient
plt.subplot(2, 1, 1)
plt.plot(time, saturated_result['true_gravity_gradient_torque'][0], label = 'gg_1')
plt.plot(time, saturated_result['true_gravity_gradient_torque'][1], label = 'gg_2')
plt.plot(time, saturated_result['true_gravity_gradient_torque'][2], label = 'gg_3')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('True Gravity Gradient Torque of Spacecraft')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, np.linalg.norm(saturated_result['true_gravity_gradient_torque'], axis = 0))
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('True Gravity Gradient Torque Norm of Spacecraft')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/actuator_saturation/true_gravity_gradient.png')
plt.close()