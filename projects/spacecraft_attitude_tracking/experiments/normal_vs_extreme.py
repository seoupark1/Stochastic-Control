import numpy as np
import matplotlib.pyplot as plt

from ..simulation import simulation

def get_max_abs_u_cmd(results):
    print(results['max_abs_commanded_control'])

def plot_graphs(results,
                path: str):
    
    # tracking error
    plt.subplot(2, 1, 1)
    plt.plot(results['time'], results['attitude_tracking_error'])
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude Error [rad]')
    plt.title('Nadir Pointing Attitude Tracking Error')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(results['time'], results['omega_tracking_error'])
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity Error [rad/s]')
    plt.title('Nadir Pointing Angular Velocity Tracking Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'projects/spacecraft_attitude_tracking/results/normal_vs_extreme/{path}/tracking_error.png')
    plt.close()

    # estimation error
    plt.subplot(2, 1, 1)
    plt.plot(results['time'], results['attitude_estimation_error'])
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude Error [rad]')
    plt.title('Nadir Pointing Attitude Estimation Error')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(results['time'], results['omega_estimation_error'])
    plt.xlabel('time [s]')
    plt.ylabel('Angular Velocity Error [rad/s]')
    plt.title('Nadir Pointing Omega Estimation Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'projects/spacecraft_attitude_tracking/results/normal_vs_extreme/{path}/estimation_error.png')
    plt.close()

    # commanded control
    plt.subplot(2, 1, 1)
    plt.plot(results['time'][0:-1], results['commanded_control'][0], label = 'u_1')
    plt.plot(results['time'][0:-1], results['commanded_control'][1], label = 'u_2')
    plt.plot(results['time'][0:-1], results['commanded_control'][2], label = 'u_3')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Nadir Pointing Commanded Control')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(results['time'][0:-1], np.linalg.norm(results['commanded_control'], axis = 0))
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Nadir Pointing Commanded Control Norm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'projects/spacecraft_attitude_tracking/results/normal_vs_extreme/{path}/commanded_control.png')
    plt.close()

    # true gravity gradient
    plt.subplot(2, 1, 1)
    plt.plot(results['time'], results['true_gravity_gradient_torque'][0], label = 'gg_1')
    plt.plot(results['time'], results['true_gravity_gradient_torque'][1], label = 'gg_2')
    plt.plot(results['time'], results['true_gravity_gradient_torque'][2], label = 'gg_3')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('True Gravity Gradient Torque of Spacecraft')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(results['time'], np.linalg.norm(results['true_gravity_gradient_torque'], axis = 0))
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('True Gravity Gradient Torque Norm of Spacecraft')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'projects/spacecraft_attitude_tracking/results/normal_vs_extreme/{path}/true_gravity_gradient.png')
    plt.close()

# star tracker performance
star_tracker_sampling_rate = 10 # [Hz]
star_tracker_accuracy = 5 # [arcsec]
star_tracker_std = np.deg2rad(star_tracker_accuracy / 3600)
star_tracker_noise_covariance = star_tracker_std**2 * np.eye(3)

# gyroscope performance
gyroscope_sampling_rate = 100 # [Hz]
gyroscope_std = 4.3633 * 1e-4 # [rad/s]
gyroscope_noise_covariance = gyroscope_std**2 * np.eye(3)

# Normal case
x_true = np.array([0.03, -0.03, -0.01, np.deg2rad(-5), np.deg2rad(-4), np.deg2rad(3)])
initial_x_hat = x_true + np.array([-0.02, 0.02, -0.01, np.deg2rad(3), np.deg2rad(-2), np.deg2rad(-1.5)])
initial_covariance = np.block([[star_tracker_noise_covariance, np.eye(3)],
                               [np.eye(3), gyroscope_noise_covariance]])
motion_noise_covariance = np.diag([1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8])

normal_result = simulation(initial_x_true = x_true,
                           initial_x_hat = initial_x_hat,
                           initial_covariance = initial_covariance,
                           motion_noise_covariance = motion_noise_covariance,
                           star_tracker_noise_covariance = star_tracker_noise_covariance,
                           gyroscope_noise_covariance = gyroscope_noise_covariance,
                           simulation_tf = 100,
                           controller_tf = 120,
                           star_tracker_sampling_rate = star_tracker_sampling_rate,
                           gyroscope_sampling_rate = gyroscope_sampling_rate)

get_max_abs_u_cmd(normal_result)
plot_graphs(normal_result, 'normal_case')


'''
# Extreme case (aboout 3 times tracking error and estimation error of the normal case)
x_true = np.array([0.09, -0.09, -0.03, np.deg2rad(-15), np.deg2rad(-12), np.deg2rad(9)])
initial_x_hat = x_true + np.array([-0.06, 0.06, -0.03, np.deg2rad(9), np.deg2rad(-6), np.deg2rad(-4.5)])
mrp_std = np.tan(np.deg2rad(30) / 4) / np.sqrt(3)
omega_std = np.deg2rad(25) / np.sqrt(3)
initial_covariance = np.diag([mrp_std**2, mrp_std**2, mrp_std**2, omega_std**2, omega_std**2, omega_std**2])
motion_noise_covariance = np.diag([1e-9, 1e-9, 1e-9, 1e-7, 1e-7, 1e-7])
star_tracker_noise_covariance = np.diag([5.88 * 1e-10, 5.88 * 1e-10, 5.88 * 1e-10])
gyroscope_noise_covariance = np.diag([1.5 * 1e-4, 1.5 * 1e-4, 1.5 * 1e-4])
'''