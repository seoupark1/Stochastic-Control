# Stochastic Spacecraft Attitude Tracking under Sensor Noise and Actuator Constraints

<p align="center">
  <a href="assets/gifs/ekf_lqr_mars_orbiting.gif">
    <img src="assets/gifs/ekf_lqr_mars_orbiting.gif"
         width="49%"
         alt="EKF + LQR">
  </a>
  <a href="assets/gifs/ekf_rti_nmpc_mars_orbiting.gif">
    <img src="assets/gifs/ekf_rti_nmpc_mars_orbiting.gif"
         width="49%"
         alt="EKF + RTI-NMPC">
  </a>
</p>

> **Summary:** Under the same control environment, LQR generated large torques at the beginning to reduce the attitude error quickly, which causing actuator torque saturation. In contrast, Real-Time NMPC satisfied the torque constraints and used less control effort, but the attitude error converged more slowly.

## Overview
This project analyzes the nadir-pointing attitude tracking performance of a spacecraft under sensor noise, gravity-gradient disturbance, and actuator torque constraints. The spacecraft attitude and angular velocity are estimated using an Extended Kalman Filter (EKF) based on measurements from a Star Tracker and a Gyroscope operating at different sampling rates.

Using the estimated state, the performances of LQR and RTI-NMPC are compared. LQR first computes the commanded control with no constraint, which is then clipped according to the torque limit. In contrast, RTI-NMPC includes the torque constraints directly in its optimization problem and solves a Quadratic Program (QP) to obtain the control input.

## Settings
|  | Models / Methods |
|---|---|
| Orbit | Mars Nadir-pointing Circular Orbit |
| State | MRP attitude error + body angular velocity error |
| Estimator | Extended Kalman Filter |
| Controllers | LQR / RTI-NMPC |
| Actuator | Reaction Wheel |
| Sensors | Star Tracker + Gyroscope |
| Disturbance | Gravity-gradient torque + Gaussian noise |
| Sensor Noise | Gaussian |

## Contents
[1. LQR vs RTI-NMPC](#experiment-1---lqr-vs-rti-nmpc)

[2. Actuator Saturation](#experiment-2---actuator-saturation)

[3. Normal case vs Extreme case](#experiment-3---normal-case-vs-extreme-case)

## Experiment 1 - LQR vs RTI-NMPC

### Question

### Setup

### Results

#### Summary

|  | EKF + LQR | EKF + RTI-NMPC |
|---|---:|---:|
| Attitude Tracking RMSE [deg] | 14.3449 | 42.9950 |
| Angular Velocity Tracking RMSE [deg/s] | 1.7224 | 2.0530 |
| Final Attitude Error [deg] | 1.8173 | 8.5154 |
| Final Angular Velocity Error [deg/s] | 0.1850 | 0.1896 |
| Control Effort [Nm^2 s] | 8699.2887 | 3996.7246 |
| Control Limit Violation [%] | 3.3667 | 0.0000 |

#### 1) Tracking Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png"
      width="75%"
      alt="Tracking Error">
  </a>
</p>

The attitude tracking RMSE was 14.3449 [deg] for EKF + LQR and 42.9950 [deg] for EKF + RTI-NMPC. At the end of the 150 s simulation, the attitude tracking error were 1.8173 [deg] and 8.5154 deg respectively.

#### 2) Initial 30-second Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png"
      width="75%"
      alt="Initial 30-second Control Input">
  </a>
</p>

The LQR's commanded control exceeded the 20 [Nm] torque limit during the initial steps, while the RTI-NMPC's commanded control remained within the torque limit. The control limit violation rates over the full simulation were 3.3667% and 0% respectively.

#### 3) Full Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png"
      width="75%"
      alt="Full Control Input">
  </a>
</p>

The total control effort was 8699.2887 [Nm^2s] for LQR and 3996.7246 [Nm^2s] for RTI-NMPC. Therefore, RTI-NMPC used 45.94% of the LQR control effort in this simulation.

#### 4) RTI-NMPC QP Solver Iterations

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png"
      width="75%"
      alt="RTI-NMPC QP Solver Iterations">
  </a>
</p>

Among 15,000 total QP solves, 14,863 were `optimal`, 136 were `optimal_inaccurate`, and one solved failed. The mean of the OSQP iterations was 3111.7.

#### 5) State Estimation Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png"
      width="75%"
      alt="State Estimation Error">
  </a>
</p>

### Interpretation

## Experiment 2 - Actuator Saturation

### Question

### Setup

### Results

#### 1) Commanded vs Actual Control

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/saturated_control.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/saturated_control.png"
      width="75%"
      alt="Commanded vs Actual Control under Actuator Saturation">
  </a>
</p>

#### 2) Tracking Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/tracking_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/tracking_error.png"
      width="75%"
      alt="Saturated vs Unsaturated Tracking Error">
  </a>
</p>


#### 3) State Estimation Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png"
      width="75%"
      alt="State Estimation Error under Actuator Saturation">
  </a>
</p>

#### 4) Gravity-Gradient Disturbance

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png"
      width="75%"
      alt="Gravity Gradient Distubance">
  </a>
</p>

### Interpretation

## Experiment 3 - Normal Case vs Extreme Case

### Question

### Setup

### Results

#### Peak abs Commanded Control

|  | Axis 1 [Nm] | Axis 2 [Nm] | Axis 3 [Nm] |
|---|---:|---:|---:|
| Normal | 12.8076 | 16.2495 | 4.8823 |
| Extreme | 38.3882 | 49.6486 | 12.2470 |

#### 1) Tracking Error

<table>
  <tr>
    <th align="center">Normal Case</th>
    <th align="center">Extreme Case</th>
  </tr>
  <tr>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/tracking_error.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/tracking_error.png"
          width="100%"
          alt="Normal Case Tracking Error">
      </a>
    </td>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/tracking_error.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/tracking_error.png"
          width="100%"
          alt="Extreme Case Tracking Error">
      </a>
    </td>
  </tr>
</table>


#### 2) Commanded Control

<table>
  <tr>
    <th align="center">Normal Case</th>
    <th align="center">Extreme Case</th>
  </tr>
  <tr>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/commanded_control.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/commanded_control.png"
          width="100%"
          alt="Normal Case Commanded Control">
      </a>
    </td>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/commanded_control.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/commanded_control.png"
          width="100%"
          alt="Extreme Case Commanded Control">
      </a>
    </td>
  </tr>
</table>

#### 3) State Estimation Error

<table>
  <tr>
    <th align="center">Normal Case</th>
    <th align="center">Extreme Case</th>
  </tr>
  <tr>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/estimation_error.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/estimation_error.png"
          width="100%"
          alt="Normal Case State Estimation Error">
      </a>
    </td>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/estimation_error.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/estimation_error.png"
          width="100%"
          alt="Extreme Case State Estimation Error">
      </a>
    </td>
  </tr>
</table>


#### 4) Gravity-Gradient Disturbance

<table>
  <tr>
    <th align="center">Normal Case</th>
    <th align="center">Extreme Case</th>
  </tr>
  <tr>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/true_gravity_gradient.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/normal_case/true_gravity_gradient.png"
          width="100%"
          alt="Normal Case Gravity Gradient Torque">
      </a>
    </td>
    <td align="center">
      <a href="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/true_gravity_gradient.png">
        <img
          src="projects/spacecraft_attitude_tracking/results/normal_vs_extreme/extreme_case/true_gravity_gradient.png"
          width="100%"
          alt="Extreme Case Gravity Gradient Torque">
      </a>
    </td>
  </tr>
</table>

### Interpretation
