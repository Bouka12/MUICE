clear;
figure(1);

% Variables
theta_graf = [];

% Variables
theta = [];
w = [];
pos = []; %posicion

% Parámetros
m = 1;
g = 9.81;
L = 1; % Longitud del péndulo
C = 0.75; % Constante de fricción viscosa
h = 0.01;

% Condiciones iniciales
theta_0 = 30*(pi/180); % Sentiré ~pi/2 y pi/2 en radianes
w_0 = 0/L;

% Inicialización movimiento
theta = theta_0;
w = w_0;
pos = [L*sin(theta); -L*cos(theta)];
alpha = -(L*w*C + m*g*sin(theta))/(L*m);

for step = 1:1000
    hold off;
    plot(pos(1), pos(2), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    plot([0 pos(1)], [0 pos(2)]);
    title(['Paso: ' num2str(step)]);
    axis([-L L -L 0]);
    set(gca, 'dataAspectRatio', [1 1 1]);
    pause(0.001);
    
    theta_a = theta;
    wa = w;

    % Paso de integración
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
