clear;
figure(1);

% Variables
r = [];
v = [];
F = [];
x = [];
y = [];

% Parameters
m = 1;
g = [0; -9.81];  % Gravity
h = 0.01;        % Time step

% Initial conditions
r0 = [0; 0];
v0 = [1; 4];    % Initial velocity

% External variable
F = m*g;

% Initialize variables
r = r0;
v = v0;

% Simulation loop for 100 time steps
for step = 1:100
    plot(r(1), r(2), 'ob');
    title(['Paso:' num2str(step)]);
    axis([0 2 -1 1]);
    set(gca, 'dataAspectRatio',[1 1 1]);
    pause(0.01);

    x = [x; r(1)];
    y = [y; r(2)];

    % Guardar valor anterior
    ra = r;
    va = v;

    % Paso integracion : Midpoint Method
    v_mid = va + (h/2)*(F/m); % Velocity at the midpoint
    r_mid = ra + (h/2)*va;     % Position at the midpoint

    v = va + h*(F/m);         % Update velocity
    r = ra + h*v_mid;         % Update position
end

figure(2);
plot(x, y, 'g');
axis([0 2 -1 1]);
set(gca,'dataAspectRatio',[1 1 1]);
title('Complete Trajectory of Parabolic Throw (Initial)');
xlabel('Distance (m)');
ylabel('Height (m)');
