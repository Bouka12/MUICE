clear;
figure(1);

% Variables for numerical solution
theta_graf = [];
theta = [];
w = [];
pos = []; % Position

% Variables for analytical solution
theta2 = [];
theta_graf2 = [];
pos2 = []; % Analytical position

% Parameters
m = 1;
g = 9.81;
L = 1; % Length of the pendulum
C = 0; % No damping
h = 0.01;

% Initial conditions
theta_0 = 5*(pi/180); % set el angulo t 5 degree
w_0 = 0; % No initial angular velocity

% Initialization for both solutions
theta = theta_0;
w = w_0;
pos = [L*sin(theta); -L*cos(theta)];
alpha = -(L*w*C + m*g*sin(theta))/(L*m);

% Initialize analytical solution
theta2 = theta_0;
pos2 = [L*sin(theta2); -L*cos(theta2)];

for step = 1:1000
    % Numerical update
    hold off;
    plot(pos(1), pos(2), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    plot([0 pos(1)], [0 pos(2)], 'b');
    
    % Analytical update
    plot(pos2(1), pos2(2), 'o', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
    plot([0 pos2(1)], [0 pos2(2)], 'r');
    
    title(['Step: ' num2str(step)]);
    axis([-L L -L 0]);
    set(gca, 'dataAspectRatio', [1 1 1]);
    pause(0.001);
    
    % Integration step for numerical solution
    theta_a = theta;
    wa = w;
    wpm = wa + (h/2)*alpha;
    theta_pm = theta_a + (h/2)*wa;
    alpha_pm = -(L*wpm*C + m*g*sin(theta_pm))/(L*m);

    w = wa + h*alpha_pm;
    theta = theta_a + h*wpm;
    pos = [L*sin(theta); -L*cos(theta)];
    alpha = -(L*w*C + m*g*sin(theta))/(L*m);
    theta_graf = [theta_graf, theta];

    % Update analytical solution
    theta2 = theta_0 * cos(sqrt(g/L)*step*h);
    pos2 = [L*sin(theta2); -L*cos(theta2)];
    theta_graf2 = [theta_graf2, theta2];
end

figure(2);
hold on;
plot(theta_graf, 'b');
plot(theta_graf2, 'r');
legend('Numerical', 'Analytical');
