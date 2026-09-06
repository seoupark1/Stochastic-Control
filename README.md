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
[1. Normal Case vs Extreme Case](##1-normal-case-vs-extreme-case-(ekf-+-lqr))

[2. Actuator Torque Saturation](##2-actuator-torque-saturation-(ekf-+-lqr))

[3. Performance Comparison](##3-performance-comparison-(ekf-+-lqr-vs-ekf-+-rti-nmpc)) -> main

## 1. Normal Case vs Extreme Case (EKF + LQR)

### Overview

### Conditions & Assumptions

### Results

### Analysis

### 후속연구주제

## 2. Actuator Torque Saturation (EKF + LQR)

### Overview

### Conditions & Assumptions

### Results

normal_case_u_max_abs : [12.8076, 16.2495, 4.8823]

extreme_case_u_max_abs : [38.3882, 49.6486, 12.2470]

### Analysis

## 3. Performance Comparison (EKF + LQR vs EKF + RTI-NMPC)

### Overview

### Conditions & Assumptions

### Results
