clear;
figure(1);

% Variables
theta_graf = [];

% Variables
theta = [];
w = [];
pos = []; %position

% Parameters
m = 1;
g = 9.81;
L = 1; % Length of the pendulum
C = 0; % No damping
h = 0.01;

% Initial conditions
theta_0 = 60*(pi/180); % 60 degrees in radians
w_0 = 0; % No initial angular velocity

% Movement initialization
theta = theta_0;
w = w_0;
pos = [L*sin(theta); -L*cos(theta)];
alpha = -(L*w*C + m*g*sin(theta))/(L*m);

for step = 1:1000
    hold off;
    plot(pos(1), pos(2), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    plot([0 pos(1)], [0 pos(2)]);
    title(['Step: ' num2str(step)]);
    axis([-L L -L 0]);
    set(gca, 'dataAspectRatio', [1 1 1]);
    pause(0.001);
    
    theta_a = theta;
    wa = w;

    % Integration step
    wpm = wa + (h/2)*alpha;
    theta_pm = theta_a + (h/2)*wa;
    alpha_pm = -(L*wpm*C + m*g*sin(theta_pm))/(L*m);

    w = wa + h*alpha_pm;
    theta = theta_a + h*wpm;
    pos = [L*sin(theta); -L*cos(theta)];
    alpha = -(L*w*C + m*g*sin(theta))/(L*m);
    theta_graf = [theta_graf theta];

end

figure(2);
plot(theta_graf, 'b');
