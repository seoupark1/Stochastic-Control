# Stochastic-Control

## Project
< Stochastic Spacecraft Nadir-Pointing Attitude Tracking Under Control Contraints >

# < Simulation Conditions >
[Normal Case: Small initial tracking error + trustworthy EKF]

Initial Attitude Tracking Error = around 7 degrees
Initial Angular Velocity Tracking Error = around 4 degrees

[Extreme Case: Large initial tracking error + uncertain EKF]

Initial Attitude Tracking Error = around 35 degrees
Initial Angular Velocity Tracking Error = around 30 degrees

# Results
gif

# Experiment 1 (LQR normal case vs extreme case)
EKF + TVLQR 환경에서 normal case와 extreme case 그래프(tracking error, estimation error, commanded control, true gravity gradient) 추출

u abs max cmd

normal : [12.8076, 16.2495, 4.8823]

extreme : [38.3882, 49.6486, 12.2470]


# Experiment 2 (LQR actuator saturation)

# Experiment 3 (EKF + LQR vs EKF + RTI NMPC)
