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

<p align="center">
  GIFs were created with AI assistance, using the raw data from experiment 3
</p>

> **Summary:** Under the same control environment, LQR generated large commanded control at the beginning to reduce the attitude error quickly, which caused actuator saturation. In contrast, Real-Time Iteration NMPC (RTI-NMPC) satisfied the torque constraint and used less control effort, but the attitude error converged more slowly.

## Overview

This project analyzes the nadir-pointing attitude tracking performance of a spacecraft under sensor noise, gravity-gradient disturbance, and actuator torque constraint. The spacecraft attitude and angular velocity are estimated using an Extended Kalman Filter (EKF) based on measurements from a Star Tracker and a Gyroscope operating at different sampling rates.

Using the estimated state, the performance of LQR and RTI-NMPC is compared. LQR first computes the commanded control without considering the torque constraint, which is then clipped according to the torque limit. In contrast, RTI-NMPC includes the torque constraint directly in its optimization and solves a Quadratic Program (QP) to obtain the control input.

## Basic Settings

| Setting | Model / Method |
|:---|:---:|
| Orbit | Mars Nadir-Pointing Circular Orbit |
| Attitude Representation | Modified Rodrigues Parameters (MRPs) |
| State | Attitude Error + Body Angular Velocity Error |
| Estimator | Extended Kalman Filter |
| Controllers | LQR / RTI-NMPC |
| Actuator | Reaction Wheel |
| Sensors | Star Tracker + Gyroscope |
| Disturbance | Gravity-Gradient Torque + Gaussian Noise |
| Sensor Noise | Gaussian |

## Contents

- [1. Normal Case vs Extreme Case](#experiment-1---normal-case-vs-extreme-case)
- [2. Actuator Saturation](#experiment-2---actuator-saturation)
- [3. LQR vs RTI-NMPC](#experiment-3---lqr-vs-rti-nmpc) -> main simulation

## Experiment 1 - Normal Case vs Extreme Case

### Question

How does the same EKF-LQR compensator behave when the initial tracking error and the initial estimation uncertainty become much larger?

### Setup

Two initial conditions were compared using the same EKF-LQR simulation.

The Extreme case was created by increasing the initial tracking error, estimation uncertainty, and process noise.

| Parameter | Normal Case | Extreme Case |
|:---|:---:|:---:|
| Initial MRP | [0.03, -0.03, -0.01] | [0.09, -0.09, -0.03] |
| Initial Angular Velocity [deg/s] | [-2.5, -2, 1] | [-7.5, -6, 3] |
| Attitude Uncertainty [deg] | 3 | 9 |
| Angular Velocity Uncertainty [deg/s] | 1 | 3 |
| Simulation Time [s] | 60 | 60 |

| Cost Weight | Setting |
|:---|:---:|
| State Weight | `Q = diag(100, 100, 100, 500, 500, 500)` |
| Control Weight | `R = 0.01 * I3` |
| Terminal Weight | `Qf = diag(200, 200, 200, 1000, 1000, 1000)` |

### Results

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

#### 3) Peak Absolute Commanded Torque

| Case | Axis 1 [Nm] | Axis 2 [Nm] | Axis 3 [Nm] |
|:---|:---:|:---:|:---:|
| Normal Case | 12.8076 | 16.2495 | 4.8823 |
| Extreme Case | 38.3882 | 49.6486 | 12.2470 |

<details>
<summary><b>State Estimation Error</b></summary>

<br>

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
</details>

<details>
<summary><b>Gravity-Gradient Disturbance</b></summary>

<br>

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
</details>

### Interpretation

The tracking error was larger in the Extreme case as expected.

A larger difference was observed in the required control torque. The peak commanded torque increased from [12.81, 16.25, 4.88] [Nm] in the normal case to [38.39, 49.65, 12.25] [Nm] in the extreme case.

In the real world, the commanded control cannot always be applied in full because of the torque limit.

## Experiment 2 - Actuator Saturation

### Question

How much does actuator saturation affect the tracking performance when LQR requires a large control torque?

### Setup

The Extreme case defined in Experiment 1 was used for this test.

First, I ran the EKF-LQR simulation without an actuator limit and measured the maximum absolute commanded torque of each axis. Then, the torque limit was set to half of each maximum value to intentionally produce actuator saturation.

|  | Axis 1 [Nm] | Axis 2 [Nm] | Axis 3 [Nm] |
|:---:|:---:|:---:|:---:|
| Torque Limit | 19.19 | 24.82 | 6.12 |

The saturated and unsaturated cases used the same initial state, sensor noise, process noise, and random seed.

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

<details>
<summary><b>State Estimation Error</b></summary>

<br>

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png"
      width="75%"
      alt="State Estimation Error under Actuator Saturation">
  </a>
</p>
</details>

<details>
<summary><b>Gravity-Gradient Disturbance</b></summary>

<br>

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png"
      width="75%"
      alt="Gravity Gradient Disturbance">
  </a>
</p>
</details>

### Interpretation

The commanded control and actual control were different when the LQR command exceeded the actuator limit. The LQR itself returned the unconstrained control, while the reaction-wheel clipped the torque before it was applied to the spacecraft.

Because less torque was available during the initial phase, the saturated case reduced the tracking error more slowly than the unsaturated case.

LQR is not aware of the actuator limit. Therefore, the actual applied control is different from the unconstrained commanded control calculated by LQR. I wanted the controller to consider the torque constraint while calculating the control itself.

## Experiment 3 - LQR vs RTI-NMPC

### Question

How do LQR and RTI-NMPC behave differently under the same control environment when the actuator has a fixed torque limit?

### Setup

The same dynamics, EKF, initial state, sensor models, process noise, and random seed were used for both controllers.

| Parameter | Setting |
|:---|:---:|
| Simulation Time [s] | 150 |
| Simulation Step [s] | 0.01 |
| Star Tracker Sampling Rate [Hz] | 10 |
| Gyroscope Sampling Rate [Hz] | 100 |
| Torque Limit [Nm] | -20 ~ 20 |
| RTI-NMPC Prediction Horizon [s] | 2 |

| Cost Weight | LQR | RTI-NMPC |
|:---|:---:|:---:|
| State Weight | `Q = diag(100, 100, 100, 500, 500, 500)` | `Q = diag(1, 1, 1, 5, 5, 5)` |
| Control Weight | `R = 0.01 * I3` | `R = 0.0001 * I3` |
| Terminal Weight | `Qf = diag(200, 200, 200, 1000, 1000, 1000)` | `P = diag(200, 200, 200, 1000, 1000, 1000)` |

For RTI-NMPC, the stage weights Q and R were multiplied by the simulation step `dt = 0.01` to match the scaling of the continuous cost used by LQR, while the terminal state weight P was set equal to the LQR terminal state weight Qf.

### Results

#### 1) Scalar Metrics

| Metric | EKF + LQR | EKF + RTI-NMPC |
|:---|:---:|:---:|
| Attitude Tracking RMSE [deg] | 14.3449 | 42.9950 |
| Angular Velocity Tracking RMSE [deg/s] | 1.7224 | 2.0530 |
| Final Attitude Error [deg] | 1.8173 | 8.5154 |
| Final Angular Velocity Error [deg/s] | 0.1850 | 0.1896 |
| Control Effort [(Nm)^2 s] | 8699.2887 | 3996.7246 |
| Control Limit Violation [%] | 3.3667 | 0.0000 |

#### 2) Tracking Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png"
      width="75%"
      alt="Tracking Error">
  </a>
</p>

The attitude tracking RMSE was 14.3449 [deg] for EKF + LQR and 42.9950 [deg] for EKF + RTI-NMPC. At the end of the 150 [s] simulation, the attitude tracking errors were 1.8173 [deg] and 8.5154 [deg], respectively.

#### 3) Initial 30-second Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png"
      width="75%"
      alt="Initial 30-second Control Input">
  </a>
</p>

The LQR's commanded control exceeded the 20 [Nm] torque limit during the initial phase, while the RTI-NMPC's commanded control stayed inside the torque limit. The control limit violation rates over the full simulation were 3.3667% and 0%, respectively.

#### 4) Full Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png"
      width="75%"
      alt="Full Control Input">
  </a>
</p>

The total control effort was 8699.2887 [(Nm)^2 s] for LQR and 3996.7246 [(Nm)^2 s] for RTI-NMPC. Therefore, RTI-NMPC used 45.94% of the LQR control effort in this simulation.

<details>
<summary><b>RTI-NMPC QP Solver Iterations</b></summary>

<br>

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png"
      width="75%"
      alt="RTI-NMPC QP Solver Iterations">
  </a>
</p>

Among 15,000 total QP solves, 14,863 solves were `optimal`, 136 solves were `optimal_inaccurate`, and one solve failed. The mean number of the OSQP iterations was 3111.7.
</details>

<details>
<summary><b>State Estimation Error</b></summary>

<br>

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png"
      width="75%"
      alt="State Estimation Error">
  </a>
</p>
</details>

### Interpretation

The most noticeable difference was the initial control behavior. LQR tried to reduce the attitude tracking error quickly and generated large commanded control. Since the LQR was not aware of the actuator torque limit, part of the commanded control was clipped by the reaction-wheel torque saturation.

RTI-NMPC behaved more conservatively. Its control input stayed inside the torque constraint and the total control effort was much smaller than that of LQR. However, the attitude tracking error decreased more slowly.

In this simulation, the angular velocity weights were set larger than the attitude weights. I think this tuning and the actuator constraint made RTI-NMPC avoid using large torque only to reduce the attitude tracking error quickly.

Therefore, I could not conclude this result as RTI-NMPC having better tracking performance than LQR. LQR showed much faster attitude tracking in this case. The advantage I observed from RTI-NMPC was that the actuator constraint was considered before the control was applied instead of clipping an already calculated commanded control. As a result, RTI-NMPC did not violate the actuator constraint during the simulation.

The QP result also showed a computational limitation. Most QPs were solved successfully, but the mean OSQP iteration count was relatively large and one QP failed during the simulation.
