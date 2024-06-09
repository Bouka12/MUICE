clear;
figure(1);
theta_graf=[];

% Variables
theta = [];
w = [];
pos = []; % posición

% Parámetros
m = 1;
g = 9.81;
L = 1; % longitud del péndulo
h = 0.01;

% Condiciones iniciales
theta_0 = 30*(pi/180); % Convertir de grados a radianes

pos = [L*sin(theta_0); -L*cos(theta_0)]; % posición inicial

% Inicialización de movimiento
for step = 1:1000
    hold off;
    plot(pos(1), pos(2), 'o', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    plot([0 pos(1)], [0 pos(2)]);
    title(['Paso: ' num2str(step)]);
    axis([-L L -L 0]);
    set(gca, 'dataAspectRatio', [1 1 1]);
    pause(0.001);

    % Cálculo del siguiente paso
    t = step*0.001;
    theta = theta_0*sin((sqrt(g/L)*t+pi/2)); % Actualizar ángulo
    pos = [L*sin(theta); -L*cos(theta)]; % Actualizar posición
    theta_graf = [theta_graf, theta]; % Guardar historia del ángulo
end

figure(2);
plot(theta_graf, 'b'); % Graficar la historia del ángulo
