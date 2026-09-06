# Spacecraft Attitude Tracking under Gaussian Noise and Control Constraints

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
  <sub>Click the GIFs to view the full-size animation.</sub><br>
  <sub>These GIFs were generated using raw data from the simulation, assisted by AI.</sub>
</p>

## Project Preview
동일한 spacecraft와 sensing environment에서 LQR은 빠른 attitude correction을 위해 공격적인 torque를 사용하며 actuator saturation이 발생하였다. RTI-NMPC는 동일한 torque bound를 직접 만족하면서 더 작은 control effort를 사용했지만, attitude tracking error의 수렴은 상대적으로 느렸다.

## Core Question
How do the tracking performances of LQR and Real-Time NMPC differ when a large tracking error occurs in a spacecraft attitude control environment characterized by sensor noise, disturbances, and control constraints?

## Goal
우주에서 미션을 수행하는 spacecraft에 예상치 못한 자세오차가 발생했을 때, 이 error를 해소하지 않는다면 그 시간대의 데이터를 놓치거나 미션에 실패하는 등의 결과로 이어질 수 있다. 우주선의 자세를 제어하는 Actuator는 구조적 안정성으로 인해 최대로 낼 수 있는 회전수와 토크에 제약이 있다. 따라서 좋은 controller는 이러한 constraints를 고려해 u를 반환해야 한다. 이 프로젝트에서 우주선의 현재 자세를 측정하는 estimator는 Extended Kalman Filter를 사용했다. Star Tracker와 Gyroscope 센서에서 각각 다른 sampling rate로 measurement를 실시하고 estimated attitude, angular velocity를 controller에 전달한다. Attitude를 표현하는 방식으로 Modified Rodrigues Parameter(MRP)를 사용했다. MRP는 3개의 파라미터로 구성되어 singularity를 가지지만 shadow set 치환을 통해 이를 해결할 수 있다. 매순간 normalize 해야하는 quaternion에 비해 연산에서 이점을 가진다. Controller로 LQR과 Real-Time NMPC를 두었다. LQR은 control constraint을 고려하지 못해 u_max를 초과하는 control을 command할 시, u_max로 clipped 된 control이 실제로 우주선에 작용한다. Real-Time NMPC는 내부에서 control constraint을 고려할 수 있으므로 반환되는 모든 optimal control이 u_max를 넘지 않는다. 따라서 최종 목표는 Gaussian Sensor Noise, Gravity Gradient External Torque, unexpected small torque noise가 존재하는 제어환경에서 control constraint가 있을 때 controllers가 어떻게 반응해 어느 정도의 tracking performance를 보이는지 알아보는 것이다.

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
