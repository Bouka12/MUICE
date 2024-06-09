clear;
figure(1);

% Parameters
m = 1;                  % Mass (kg)
g = [0; -9.81];         % Gravity (m/s^2)
h = 0.01;               % Time step (s)
steps = 100;            % Number of time steps

% Initial conditions
r0 = [0; 0];
v0 = [1; 4];

% Variables for storing results
x_num = [];             % Numerical x positions
y_num = [];             % Numerical y positions
x_ana = [];             % Analytical x positions
y_ana = [];             % Analytical y positions

% Initialize positions and velocities
r = r0;
v = v0;

for t = 0:h:(steps*h)
    % Numerical integration using Midpoint Method
    if t > 0  % Skip the first step to avoid duplication with initial condition
        v_mid = v + (h/2)*(g);
        r_mid = r + (h/2)*v;
        v = v + h*(g);
        r = r + h*v_mid;
    end

    % Store numerical results
    x_num = [x_num; r(1)];
    y_num = [y_num; r(2)];

    % Analytical calculation
    r_ana = r0 + v0*t + 0.5*g*t^2;
    
    % Store analytical results
    x_ana = [x_ana; r_ana(1)];
    y_ana = [y_ana; r_ana(2)];

    % Plot both trajectories
    plot(x_num, y_num, 'b-', x_ana, y_ana, 'r--');
    legend('Numerical', 'Analytical');
    title('Comparison of Numerical and Analytical Solutions');
    axis([0 2 -1 1]);  % Modified axis ranges to accommodate longer flight
    xlabel('Distance (m)');
    ylabel('Height (m)');
    pause(0.01);
end

figure(2);
plot(x_num, y_num, 'b-', x_ana, y_ana, 'r--');
legend('Numerical', 'Analytical');
title('Comparison of Numerical and Analytical Solutions');
xlabel('Distance (m)');
ylabel('Height (m)');
axis([0 2 -1 1]);  % Modified axis ranges to accommodate longer flight

