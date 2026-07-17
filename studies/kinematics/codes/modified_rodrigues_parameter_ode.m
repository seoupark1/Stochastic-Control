clear; clc; clear all;

% tilde operator
function tilde = tilde(v)
    tilde = [0, -v(3), v(2); 
            v(3), 0, -v(1); 
            -v(2), v(1), 0];

end

% MRP kinematic differential equation
function sigma_dot = mrp_ode(sigma, omega)
    B = (1 - dot(sigma, sigma)) * eye(3) + 2 * tilde(sigma) + 2 * sigma * transpose(sigma);
    sigma_dot = 1/4 * B * omega;

end

% 수치적분기 (following Runge-Kutta 4th order method)
function sigma_next = rk4(func, sigma_current, omega, dt)
    % 가중치 부여
    k1 = func(sigma_current, omega);
    k2 = func(sigma_current + 0.5 * dt * k1, omega);
    k3 = func(sigma_current + 0.5 * dt * k2, omega);
    k4 = func(sigma_current + dt * k3, omega);

    sigma_next = sigma_current + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4);

end

% 초기 설정
dt = 0.01;
t_final = 42;
total_step = t_final / dt;
sigma_current = [0.4; 0.2; -0.1];

% next step 계산을 위한 history 기록
time_history = [0:total_step-1] * dt;
sigma_history = zeros(3, total_step);

for k = 1:total_step

    t = time_history(k);

    omega = [sin(0.1*t); 0.01; cos(0.1*t)] * 20 * (pi/180);

    sigma_history(:, k) = sigma_current;

    sigma_next = rk4(@mrp_ode, sigma_current, omega, dt);
    
    % shadow set 치환
    if norm(sigma_next) > 1
        sigma_next = -sigma_next / norm(sigma_next)^2;

    end

    sigma_current = sigma_next;

end

result = norm(sigma_history(:, total_step));
disp(result);
