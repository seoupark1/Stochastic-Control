# Stochastic-Control

## Project
< Risk-aware Stochastic Spacecraft Nadir Pointing Attitude Tracking Under Actuator Contraints >
Nadir Pointing은 행성 궤도를 공전하는 물체의 방향이 항상 행성 쪽을 바라보도록 
우주선의 자세표현법으로 Modified Rodrigues Parameter를 사용함. 

# < Simulation Conditions >
[Normal Case: Small initial tracking error + trustworthy EKF]

Initial Attitude Tracking Error = around 7 degrees
Initial Angular Velocity Tracking Error = around 4 degrees

[Extreme Case: Large initial tracking error + uncertain EKF]

Initial Attitude Tracking Error = around 35 degrees
Initial Angular Velocity Tracking Error = around 30 degrees

# Project Result Simulation GIF
gif

# Experiment 1 (LQR normal case vs extreme case)
EKF + TVLQR 환경에서 normal case와 extreme case 그래프(tracking error, estimation error, commanded control, true gravity gradient) 추출

u abs max cmd
normal : [12.8076, 16.2495, 4.8823]
extreme : [38.3882, 49.6486, 12.2470]


# Experiment 2 (LQR actuator saturation)

# Experiment 3 (EKF + LQR vs EKF + RTI NMPC)

# Experiment 4 (RTI NMPC vs chance constrained RTI NMPC)
