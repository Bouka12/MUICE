clear;
figure(1);

% Variables
r = [];
F = [];
x = [];
y = [];

% Parameters
m = 1;  % Mass of the projectile
g = [0; -9.81];  % Gravitational acceleration (m/s^2)
h = 0.01;  % Time step

% Initial conditions
r0 = [0; 0];  % Initial position
v0 = [0.1; 4];  % Initial velocity

% External force
F = m * g;

% Initialize position
r = r0;

% Simulation loop for 100 time steps
for step = 1:100
    plot(r(1), r(2), 'ob');  % Plot the current position
    title(['Paso: ' num2str(step)]);
    axis([-0.2 0.2 -1 1]);  % Set the axis limits
    set(gca, 'dataAspectRatio', [1 1 1]);
    pause(0.01);

    x = [x; r(1)];  % Append current x to array
    y = [y; r(2)];  % Append current y to array

    % Numerical integration to update position
    t = step*h;
    r = r0 + v0 * t + 0.5 * g * t^2;  % Update position
    %v = v + g * h;  % Update velocity
end

% Plot the final trajectory
figure(2);
plot(x, y, 'b');
title('Complete Trajectory of Parabolic Throw (Analytical)');
xlabel('Distance (m)');
ylabel('Height (m)');
axis([0 0.2 -1 1]);  % Set the axis limits

