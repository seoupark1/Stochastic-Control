clear; clc; close all;

% crp differential kinematic equations 정의
function q_dot = crp_diff(t, q)
    % omega
    omega = [sin(0.1*t); 0.01; cos(0.1*t)] * 3 * (pi/180);

    % tilde 연산자로 행렬 cross product 계산
    q_tilde = [0, -q(3), q(2);
               q(3), 0, -q(1);
               -q(2), q(1), 0];

    q_dot = 1/2 * (eye(3) + q_tilde + q*transpose(q)) * omega;

end


% time, initial value
tspan = [0, 42];
q0 = [0.4; 0.2; -0.1];

% 함수 실행
[t, q] = ode45(@crp_diff, tspan, q0);

% t=42s에서의 norm 계산
q_norm = norm(q(end, :));
disp(q_norm);
