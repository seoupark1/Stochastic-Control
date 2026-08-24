# Stochastic-Control

## Project
Spacecraft Attitude Control Under Stochastic Uncertainty

# Experiment 1
LQG output-feedback baseline

linear system에 대해서만 성립. 가장 쉽게 접근 가능하고 공부할 수 있는 LQG로 시작하는 것이 loop의 전체 구조를 잡는데 유리하다. Gaussian Noise가 고려된 x를 plant가 받고 x_hat 내놓는 것으로 시작.

# Experiment 2
Nonlinear MRP control under Noise and Saturation

LQR 말고 lyapunov controller를 활용. K, P, KI에 따라 standard가 좋을 수도 있고 integral이 좋을 수도 있음. 

# < EKF & LQR Conditions >
[Normal Case: Small initial tracking error + trustworthy EKF]

Initial Attitude Tracking Error = around 7 degrees
Initial Angular Velocity Tracking Error = around 4 degrees

[Extreme Case: Large initial tracking error + uncertain EKF]

Initial Attitude Tracking Error = around 35 degrees
Initial Angular Velocity Tracking Error = around 30 degrees
