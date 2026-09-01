import numpy as np
import matplotlib.pyplot as plt
from ..simulations.ekf_nmpc import simulation

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

# Extreme case
initial_x_true = np.array([0.09, -0.09, -0.03, np.deg2rad(-7.5), np.deg2rad(-6), np.deg2rad(3)])
initial_x_hat = initial_x_true + np.array([-0.015, 0.006, -0.03, np.deg2rad(1.5), np.deg2rad(-0.9), np.deg2rad(0.6)])
initial_covariance = np.diag([mrp_std**2, mrp_std**2, mrp_std**2, omega_std**2, omega_std**2, omega_std**2])
motion_noise_covariance = np.diag([1e-9, 1e-9, 1e-9, 1e-7, 1e-7, 1e-7])

# nmpc properties
prediction_horizon = 1 # [s]
max_iteration = 5 # [step]
alpha = 0.5,
del_z_tolerance = 1e-4
defect_tolerance = 1e-5
u_max = (20, 20, 20)

nmpc_result = simulation(initial_x_true = initial_x_true,
                         initial_x_hat = initial_x_hat,
                         initial_covariance = initial_covariance,
                         motion_noise_covariance = motion_noise_covariance,
                         star_tracker_noise_covariance = star_tracker_noise_covariance,
                         gyroscope_noise_covariance = gyroscope_noise_covariance,
                         simulation_tf = 60,
                         prediction_horizon = prediction_horizon,
                         star_tracker_sampling_rate = star_tracker_sampling_rate,
                         gyroscope_sampling_rate = gyroscope_sampling_rate,
                         seed = 2005,
                         max_iteration = max_iteration,
                         alpha = alpha,
                         del_z_tolerance = del_z_tolerance,
                         defect_tolerance = defect_tolerance,
                         control_bound = (-u_max, u_max))

time = nmpc_result['time']
control = nmpc_result['commanded_control']

# commanded control
plt.subplot(2, 1, 1)
plt.plot(time[0:-1], control[0], label = 'u_1')
plt.plot(time[0:-1], control[1], label = 'u_2')
plt.plot(time[0:-1], control[2], label = 'u_3')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('NMPC Nadir Pointing Commanded Control')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time[0:-1], np.linalg.norm(control, axis = 0))
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('NMPC Nadir Pointing Commanded Control Norm')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/ekf_nmpc/commanded_control.png')
plt.close()