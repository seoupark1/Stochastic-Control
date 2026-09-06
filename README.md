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
| Components | Models |
|---|---|
| Trajectory | Mars Nadir-pointing Circular Orbit |
| State | MRP attitude error + body angular velocity error |
| Estimator | Extended Kalman Filter |
| Controllers | LQR + clipping / RTI-NMPC + control constraints |
| Actuator | Reaction Wheel |
| Sensors | Star Tracker + Gyroscope |
| Disturbance | Gravity-gradient torque |
| Motion uncertainty | Gaussian disturbance noise |
| Measurement uncertainty | Gaussian sensor noise |

## Experiments
1. [EKF + LQR] : Normal Case vs Extreme Case
2. [EKF + LQR] : Actuator Torque Saturation
3. [EKF + RTI-NMPC] : Performance comparison -> main experiment

## 1. EKF + LQR vs EKF + Real-Time NMPC

### simulation 설명

### Conditions & Assumptions

### Results

### Analysis

### 후속연구주제

## 2. EKF + LQR : Normal Case vs Extreme Case

### simulation 설명

### Conditions & Assumptions

### Results

normal_case_u_max_abs : [12.8076, 16.2495, 4.8823]

extreme_case_u_max_abs : [38.3882, 49.6486, 12.2470]

### Analysis

## 3. EKF + LQR : Actuator Torque Saturation
