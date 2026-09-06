import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from ..simulations import ekf_rti_nmpc, ekf_lqr
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

# Extreme case
initial_x_true = np.array([0.09, -0.09, -0.03, np.deg2rad(-7.5), np.deg2rad(-6), np.deg2rad(3)])
initial_x_hat = initial_x_true + np.array([-0.015, 0.006, -0.03, np.deg2rad(1.5), np.deg2rad(-0.9), np.deg2rad(0.6)])
initial_covariance = np.diag([mrp_std**2, mrp_std**2, mrp_std**2, omega_std**2, omega_std**2, omega_std**2])
motion_noise_covariance = np.diag([1e-9, 1e-9, 1e-9, 1e-7, 1e-7, 1e-7])

# u limit
u_max = np.asarray([20, 20, 20]) # [Nm]

lqr_result = ekf_lqr.simulation(initial_x_true = initial_x_true,
                                initial_x_hat = initial_x_hat,
                                initial_covariance = initial_covariance,
                                motion_noise_covariance = motion_noise_covariance,
                                star_tracker_noise_covariance = star_tracker_noise_covariance,
                                gyroscope_noise_covariance = gyroscope_noise_covariance,
                                simulation_tf = 150,
                                controller_tf = 200,
                                star_tracker_sampling_rate = star_tracker_sampling_rate,
                                gyroscope_sampling_rate = gyroscope_sampling_rate,
                                seed = 2026,
                                control_limiter = ReactionWheel(u_max))

prediction_horizon = 2 # [s]
mpc_result = ekf_rti_nmpc.simulation(initial_x_true = initial_x_true,
                                     initial_x_hat = initial_x_hat,
                                     initial_covariance = initial_covariance,
                                     motion_noise_covariance = motion_noise_covariance,
                                     star_tracker_noise_covariance = star_tracker_noise_covariance,
                                     gyroscope_noise_covariance = gyroscope_noise_covariance,
                                     simulation_tf = 150,
                                     prediction_horizon = prediction_horizon,
                                     star_tracker_sampling_rate = star_tracker_sampling_rate,
                                     gyroscope_sampling_rate = gyroscope_sampling_rate,
                                     seed = 2026,
                                     control_bound = (-u_max, u_max))

time = lqr_result['time']

# raw data
np.savez_compressed('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/raw_data.npz',

    time = time,
    lqr_true_state = lqr_result['true_state'],
    lqr_reference_state = lqr_result['reference_state'],
    lqr_estimated_state = lqr_result['estimated_state'],
    lqr_true_gravity_gradient = lqr_result['true_gravity_gradient_torque'],
    lqr_covariance_P = lqr_result['covariance_P'],
    lqr_commanded_control = lqr_result['commanded_control'],
    lqr_max_abs_commanded_control = lqr_result['max_abs_commanded_control'],
    lqr_actual_control = lqr_result['actual_control'],
    lqr_measurement_y = lqr_result['measurement_y'],
    lqr_star_tracker_correction_steps = lqr_result['star_tracker_correction_steps'],
    lqr_gyroscope_correction_steps = lqr_result['gyroscope_correction_steps'],
    lqr_attitude_tracking_error = lqr_result['attitude_tracking_error'],
    lqr_omega_tracking_error = lqr_result['omega_tracking_error'],
    lqr_attitude_estimation_error = lqr_result['attitude_estimation_error'],
    lqr_omega_estimation_error = lqr_result['omega_estimation_error'],

    mpc_true_state = mpc_result['true_state'],
    mpc_reference_state = mpc_result['reference_state'],
    mpc_estimated_state = mpc_result['estimated_state'],
    mpc_true_gravity_gradient = mpc_result['true_gravity_gradient_torque'],
    mpc_covariance_P = mpc_result['covariance_P'],
    mpc_commanded_control = mpc_result['commanded_control'],
    mpc_qp_iterations = mpc_result['qp_iterations'],
    mpc_qp_status = np.asarray(mpc_result['qp_status'], dtype = 'U20'),
    mpc_measurement_y = mpc_result['measurement_y'],
    mpc_star_tracker_correction_steps = mpc_result['star_tracker_correction_steps'],
    mpc_gyroscope_correction_steps = mpc_result['gyroscope_correction_steps'],
    mpc_attitude_tracking_error = mpc_result['attitude_tracking_error'],
    mpc_omega_tracking_error = mpc_result['omega_tracking_error'],
    mpc_attitude_estimation_error = mpc_result['attitude_estimation_error'],
    mpc_omega_estimation_error = mpc_result['omega_estimation_error'])

# tracking error
plt.subplot(2, 1, 1)
plt.plot(time, np.rad2deg(lqr_result['attitude_tracking_error']), label = 'EKF + LQR')
plt.plot(time, np.rad2deg(mpc_result['attitude_tracking_error']), label = 'EKF + RTI-NMPC')
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [deg]')
plt.title('Nadir Pointing Attitude Tracking Error')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(lqr_result['omega_tracking_error']), label = 'EKF + LQR')
plt.plot(time, np.rad2deg(mpc_result['omega_tracking_error']), label = 'EKF + RTI-NMPC')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity Error [deg/s]')
plt.title('Nadir Pointing Angular Velocity Tracking Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png')
plt.close()

# final tracking error
lqr_final_attitude_error = np.rad2deg(lqr_result['attitude_tracking_error'][-1])
lqr_final_omega_error = np.rad2deg(lqr_result['omega_tracking_error'][-1])
mpc_final_attitude_error = np.rad2deg(mpc_result['attitude_tracking_error'][-1])
mpc_final_omega_error = np.rad2deg(mpc_result['omega_tracking_error'][-1])

context = ('Final Tracking Error\n'
          f'LQR attitude Error [deg]       = {lqr_final_attitude_error: .4f}\n'
          f'RTI-NMPC attitude Error [deg]  = {mpc_final_attitude_error: .4f}\n'
          f'LQR omega Error [deg/s]        = {lqr_final_omega_error: .4f}\n'
          f'RTI-NMPC omega Error [deg/s]   = {mpc_final_omega_error: .4f}\n')

with open('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/final_tracking_error.txt', 'w') as file:
    file.write(context)

# tracking error RMSE
simulation_tf = 150
dt = 0.01
N = int(simulation_tf / dt)
lqr_attitude_error_rmse = np.rad2deg(np.sqrt(np.mean(np.square(lqr_result['attitude_tracking_error']))))
lqr_omega_error_rmse = np.rad2deg(np.sqrt(np.mean(np.square(lqr_result['omega_tracking_error']))))
mpc_attitude_error_rmse = np.rad2deg(np.sqrt(np.mean(np.square(mpc_result['attitude_tracking_error']))))
mpc_omega_error_rmse = np.rad2deg(np.sqrt(np.mean(np.square(mpc_result['omega_tracking_error']))))

context = ('Tracking Error RMSE\n'
          f'LQR attitude RMSE [deg]       = {lqr_attitude_error_rmse: .4f}\n'
          f'RTI-NMPC attitude RMSE [deg]  = {mpc_attitude_error_rmse: .4f}\n'
          f'LQR omega RMSE [deg/s]        = {lqr_omega_error_rmse: .4f}\n'
          f'RTI-NMPC omega RMSE [deg/s]   = {mpc_omega_error_rmse: .4f}\n')

with open('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error_rmse.txt', 'w') as file:
    file.write(context)

# control
fig, axes = plt.subplots(3, 1, figsize = (6.4, 7.2))

for i in range(3):
    axes[i].plot(time[0:-1], lqr_result['actual_control'][i], label = 'EKF + LQR (u_actual)')
    axes[i].plot(time[0:-1], lqr_result['commanded_control'][i], label = 'EKF + LQR (u_cmd)')
    axes[i].plot(time[0:-1], mpc_result['commanded_control'][i], label = 'EKF + RTI-NMPC (u_cmd)')
    axes[i].axhline(u_max[i], linestyle = '--')
    axes[i].axhline(-u_max[i], linestyle = '--')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel(f'u_{i + 1} Torque [Nm]')
    axes[i].grid(True)

axes[0].legend()
fig.suptitle('Nadir Pointing Control')
fig.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png')
plt.close()

# control before 30s
fig, axes = plt.subplots(3, 1, figsize = (6.4, 7.2))

for i in range(3):
    axes[i].plot(time[0:-1], lqr_result['actual_control'][i], label = 'EKF + LQR (u_actual)')
    axes[i].plot(time[0:-1], lqr_result['commanded_control'][i], label = 'EKF + LQR (u_cmd)')
    axes[i].plot(time[0:-1], mpc_result['commanded_control'][i], label = 'EKF + RTI-NMPC (u_cmd)')
    axes[i].axhline(u_max[i], linestyle = '--')
    axes[i].axhline(-u_max[i], linestyle = '--')
    axes[i].set_xlim(0, 30)
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel(f'u_{i + 1} Torque [Nm]')
    axes[i].grid(True)

axes[0].legend()
fig.suptitle('Nadir Pointing Initial 30-Second Control')
fig.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png')
plt.close()

# control limit violation
lqr_violation = 0
mpc_violation = 0

for time_step in range(N):

    for i in range(3):

        if np.abs(lqr_result['commanded_control'][i, time_step]) > u_max[i]:
            lqr_violation += 1

    for j in range(3):

        if np.abs(mpc_result['commanded_control'][j, time_step]) > u_max[j]:
            mpc_violation += 1

lqr_violation_rate = (lqr_violation / (3 * N)) * 100 # [%]
mpc_violation_rate = (mpc_violation / (3 * N)) * 100 # [%]

context = ('Control Limit Violation\n'
          f'LQR violation rate [%]        = {lqr_violation_rate: .4f}\n'
          f'RTI-NMPC violation rate [%]   = {mpc_violation_rate: .4f}\n')

with open('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_limit_violation.txt', 'w') as file:
    file.write(context)

# control effort
lqr_control_effort = dt * np.sum(np.square(lqr_result['actual_control']))
mpc_control_effort = dt * np.sum(np.square(mpc_result['commanded_control']))
control_effort_ratio = (mpc_control_effort / lqr_control_effort) * 100

context = ('Control Effort\n'
          f'LQR control effort [Nm^2 s]   = {lqr_control_effort: .4f}\n'
          f'RTI-NMPC effort [Nm^2 s]      = {mpc_control_effort: .4f}\n'
          f'RTI-NMPC used {control_effort_ratio: .4f}% of the control effort of LQR')

with open('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_effort.txt', 'w') as file:
    file.write(context)

# qp status
status_count = Counter(mpc_result['qp_status'])
optimal_count = status_count['optimal']
optimal_inaccurate_count = status_count['optimal_inaccurate']
failed_count = status_count['qp_failed']

context = ('RTI-NMPC QP Status\n'
          f'optimal                       = {optimal_count}\n'
          f'optimal_inaccurate            = {optimal_inaccurate_count}\n'
          f'qp_failed                     = {failed_count}\n'
          f'max iterations                = {np.max(mpc_result['qp_iterations'])}\n'
          f'mean iterations               = {np.mean(mpc_result['qp_iterations'])}')

with open('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_status.txt', 'w') as file:
    file.write(context)

# qp iterations
plt.plot(time[0:-1], mpc_result['qp_iterations'])
plt.xlabel('Time [s]')
plt.ylabel('QP Solver Iterations')
plt.title('RTI-NMPC QP Solver Iterations')
plt.grid(True)
plt.savefig('projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png')
plt.close()

# estimation error
plt.subplot(2, 1, 1)
plt.plot(time, np.rad2deg(lqr_result['attitude_estimation_error']), label = 'EKF + LQR')
plt.plot(time, np.rad2deg(mpc_result['attitude_estimation_error']), label = 'EKF + RTI-NMPC')
plt.xlabel('Time [s]')
plt.ylabel('Attitude Error [deg]')
plt.title('Nadir Pointing Attitude Estimation Error')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(lqr_result['omega_estimation_error']), label = 'EKF + LQR')
plt.plot(time, np.rad2deg(mpc_result['omega_estimation_error']), label = 'EKF + RTI-NMPC')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity Error [deg/s]')
plt.title('Nadir Pointing Angular Velocity Estimation Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png')
plt.close()