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

> **Summary:** 동일한 제어환경에서 LQR은 초반에 빠르게 attitude error를 해소하기 위해 actuator saturation을 유발하면서 강한 torque를 생성했다. 반면 RTI-NMPC는 torque constraint를 직접 만족하면서 상대적으로 더 작은 control effort를 사용했지만 그만큼 attitude error 해소 시점이 늦게 나타났다.

## Overview
이 프로젝트는 sensor noise, gravity gradient disturbance, 그리고 actuator torque constraint이 존재하는 환경에서 spacecraft의 nadir-pointing attitude tracking 성능을 분석한다. Spacecraft의 자세와 각속도는 서로 다른 sampling rate로 작동하는 Star Tracker와 Gyroscope의 measurement를 이용해 Extended Kalman Filter (EKF)로 추정한다. 자세는 Modified Rodrigues Parameters (MRP)로 표현한다. Estimated state를 기반으로 LQR과 Real-Time Iteration NMPC (RTI-NMPC)를 비교한다. LQR은 commanded control을 계산한 뒤 실제 actuator 입력에서 torque limit을 적용해 clip한다. 반면 RTI-NMPC는 해당 torque constraint를 input으로 받아 내부에서 Quadratic Problem을 풀어 optimal control을 반환한다. 큰 attitude tracking error가 발생했을 때, 동일한 제어환경과 control constraint에서 LQR과 RTI-NMPC는 tracking performance와 control usage에서 어떠한 차이를 보이는가?


## Settings




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
