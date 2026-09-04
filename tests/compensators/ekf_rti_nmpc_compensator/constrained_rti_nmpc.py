import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from projects.spacecraft_attitude_tracking.simulations.ekf_rti_nmpc import simulation

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
prediction_horizon = 2 # [s]
u_max = np.asarray([20, 20, 20]) # [Nm]

result = simulation(initial_x_true = initial_x_true,
                    initial_x_hat = initial_x_hat,
                    initial_covariance = initial_covariance,
                    motion_noise_covariance = motion_noise_covariance,
                    star_tracker_noise_covariance = star_tracker_noise_covariance,
                    gyroscope_noise_covariance = gyroscope_noise_covariance,
                    simulation_tf = 20,
                    prediction_horizon = prediction_horizon,
                    star_tracker_sampling_rate = star_tracker_sampling_rate,
                    gyroscope_sampling_rate = gyroscope_sampling_rate,
                    seed = 2026,
                    control_bound = (-u_max, u_max))

time = result['time']

# tracking error
plt.subplot(2, 1, 1)
plt.plot(time, np.rad2deg(result['attitude_tracking_error']))
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [deg]')
plt.title('NMPC Nadir Pointing Attitude Tracking Error')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(result['omega_tracking_error']))
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity Error [deg/s]')
plt.title('NMPC Nadir Pointing Angular Velocity Tracking Error')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'tests/compensators/ekf_rti_nmpc_compensator/tracking_error.png')
plt.close()

# qp status
print('QP status:', Counter(result['qp_status']))
print('max iterations:', np.max(result['qp_iterations']))
print('mean iterations:', np.mean(result['qp_iterations']))

# commanded constrained control
fig, axes = plt.subplots(3, 1, figsize = (6.4, 7.2))

for i in range(3):
    axes[i].plot(time[0:-1], result['commanded_control'][i])
    axes[i].axhline(u_max[i], linestyle = '--')
    axes[i].axhline(-u_max[i], linestyle = '--')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel(f'u_{i + 1} Torque [Nm]')
    axes[i].grid(True)

fig.suptitle('NMPC Nadir Pointing Commanded Control')
fig.tight_layout()
plt.savefig(f'tests/compensators/ekf_rti_nmpc_compensator/commanded_control.png')
plt.close()