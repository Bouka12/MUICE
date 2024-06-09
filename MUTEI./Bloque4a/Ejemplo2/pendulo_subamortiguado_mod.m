clear;
figure(1);

% Variables
theta_graf = [];

% Parámetros modificados
m = 2;  % Masa aumentada
g = 9.81;
L = 1.5;  % Longitud aumentada
C = 0.6;  % Constante de fricción ajustada para sub-amortiguación
h = 0.01;

% Condiciones iniciales modificadas
theta_0 = 45*(pi/180); % Ángulo inicial aumentado
w_0 = 0.2/L;  % Velocidad angular inicial aumentada

% Inicialización movimiento
theta = theta_0;
w = w_0;
pos = [L*sin(theta); -L*cos(theta)];
alpha = -(L*w*C + m*g*sin(theta))/(L*m);

% Simulación con más iteraciones
for step = 1:1500
    hold off;
    plot(pos(1), pos(2), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    plot([0 pos(1)], [0 pos(2)]);
    title(['Paso: ' num2str(step)]);
    axis([-1.5*L 1.5*L -1.5*L 0]);
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
