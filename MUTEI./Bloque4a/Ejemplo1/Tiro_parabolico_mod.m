clear;
figure(1);

% Variables
r = [];
v = [];
F = [];
x = [];
y = [];

% Modified Parameters
m = 2;  % Increased mass
g = [0; -9.81];  % Gravity remains constant
h = 0.005;  % Smaller time step for more accuracy
n_steps = 400;  % Increased number of iterations for a longer simulation

% Initial conditions modified
r0 = [0; 0.5];  % Slightly elevated initial position
v0 = [2; 8];  % Higher initial velocity

% External force remains the same
F = m*g;

% Initialize variables
r = r0;
v = v0;

% Simulation loop with increased number of time steps
for step = 1:n_steps
    plot(r(1), r(2), 'ob');
    title(['Paso:' num2str(step)]);
    axis([0 6 -4 4]);  % Modified axis ranges to accommodate longer flight
    set(gca, 'dataAspectRatio',[1 1 1]);
    pause(0.01);

    x = [x; r(1)];
    y = [y; r(2)];

    % Previous values (for clarity)
    ra = r;
    va = v;

    % Midpoint Method integration step
    v_mid = va + (h/2)*(F/m);
    r_mid = ra + (h/2)*va;

    v = va + h*(F/m);
    r = ra + h*v_mid;
end

figure(2);
plot(x, y, 'r');  % Changed color to red for visibility
axis([0 6 -4 4]);
set(gca,'dataAspectRatio',[1 1 1]);
title('Modified Trajectory of Parabolic Throw');
xlabel('Distance (m)');
ylabel('Height (m)');
