# Stochastic-Control

## Project
< Risk-aware Stochastic Spacecraft Nadir Pointing Attitude Tracking Under Actuator Contraints >
Nadir Pointing은 행성 궤도를 공전하는 물체의 방향이 항상 행성 쪽을 바라보도록 
우주선의 자세표현법으로 Modified Rodrigues Parameter를 사용함. 

# Project Result Simulation GIF
gif

# Experiment 1
EKF + TVLQR 환경에서 normal case와 extreme case 그래프(tracking error, estimation error,  추출


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
