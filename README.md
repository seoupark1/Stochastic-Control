# Spacecraft Attitude Tracking under Stochastic Environment and Control Constraints

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

> **Summary:** Under the same control environment, LQR generated strong torque at the beginning to reduce the attitude error quickly, which led reaction wheel to saturation. In contrast, Real-Time NMPC satisfied the torque constraints and used relatively less control effort, but the attitude error converged more slowly.

## Overview
This Project analyzes the nadir-pointing attitude tracking performance of a spacecraft under sensor noise, gravity-gradient disturbance, and actuator torque constraints. The spacecraft attitude and angular velocity are esimated using an Extended Kalman Filter (EKF) based on measurements from a Star Tracker and a Gyroscope operating at different sampling rates.

Using the estimated state, the performances of LQR and RTI-NMPC are compared. LQR first computes the commanded control with no constraint, and then clip it according to the torque limit. In contrast, RTI-NMPC includes the torque constraints directly in its optimization problem and solves a Quadratic Program (QP) to obtain the optimal control.

## Settings
| Components | Models / Methods |
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

#### Tracking Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/tracking_error.png"
      width="75%"
      alt="Tracking Error">
  </a>
</p>


#### Initial 30-second Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control_initial_30s.png"
      width="75%"
      alt="Initial 30-second Control Input">
  </a>
</p>


#### Full Control Input

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/control.png"
      width="75%"
      alt="Full Control Input">
  </a>
</p>


#### State Estimation Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/estimation_error.png"
      width="75%"
      alt="State Estimation Error">
  </a>
</p>


#### RTI-NMPC QP Solver Iterations

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/ekf_lqr_vs_ekf_rti_nmpc/rti_nmpc_qp_iterations.png"
      width="75%"
      alt="RTI-NMPC QP Solver Iterations">
  </a>
</p>

### Interpretation

## Experiment 2 - Actuator Saturation

### Question

### Setup

### Results

#### Tracking Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/tracking_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/tracking_error.png"
      width="75%"
      alt="Saturated vs Unsaturated Tracking Error">
  </a>
</p>


#### Commanded vs Actual Control

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/saturated_control.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/saturated_control.png"
      width="75%"
      alt="Commanded vs Actual Control under Actuator Saturation">
  </a>
</p>


#### State Estimation Error

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/estimation_error.png"
      width="75%"
      alt="State Estimation Error under Actuator Saturation">
  </a>
</p>


#### Gravity-Gradient Disturbance

<p align="center">
  <a href="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png">
    <img
      src="projects/spacecraft_attitude_tracking/results/actuator_saturation/half_of_max/true_gravity_gradient.png"
      width="75%"
      alt="Gravity Gradient Distubance">
  </a>
</p>

normal_case_u_max_abs : [12.8076, 16.2495, 4.8823]

extreme_case_u_max_abs : [38.3882, 49.6486, 12.2470]

### Interpretation

## Experiment 3 - Normal Case vs Extreme Case

### Question

### Setup

### Results

### Interpretation
