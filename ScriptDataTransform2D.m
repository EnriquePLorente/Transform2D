disp(" ")

obs = load('dataTransform2D.txt');

%Nivel de confianza para establecer la bondad del ajuste
nivel_confianza = 0.95;

%Limites de parada 
pa0 = 7.8448*10^-7; 
pa1 = 9.6216*10-8;
pa2 = 0.0001;


%número de incognitas
n = 4;
%Devuelve las observaciones
c = size(obs,1);

%Matrices de trabajo. Las hacemos de 0 o 1 según nos convenga el rellenarla
A = ones(2*c,n);
B = zeros(2*c,2*2*c); %Para cada punto hay 2 coord
K = zeros(2*c,1); %Vector término independiente. En este caso la z

%Monta la matriz de observaciones
A(1:2:end,1) = obs(1:end,1);
A(2:2:end,1) = obs(1:end,2);
A(1:2:end,2) = obs(1:end,2);
A(2:2:end,2) = -obs(1:end,1);
A(2:2:end,3) = 0;
A(1:2:end,4) = 0;

%Monta vector independiente
K(1:2:end) = obs(:,5);
K(2:2:end) = obs(:,6);

%Obtiene los valores iniciales de los parámetros por el método de ec. de
%observación
%Deshace el cambio de variable
x = inv(A'*A)*A'*K; 
rotacion = x(2)/x(1);
escala = x(2)/sin(rotacion);
disp('Parametros iniciales [α, L, Tx, Ty]:');
rotacion = atan2(x(2),x(1));
rotacion_deg = rad2deg(rotacion);
escala = x(2)/sin(rotacion);
fprintf("Rotación: %.4f\n", rotacion_deg)
fprintf("Escala: %.4f\n", escala)
fprintf("Desplazamiento en x: %.4f\n", x(3))
fprintf("Desplazamiento en y: %.4f\n", x(4))

%Matriz de pesos
dto = obs(:,[3,4,7,8])'; %Desviaciones
Q = diag(dto(:).^2); %dto(:) te alinea en 1 columna
P = inv(Q);

% %Observaciones de trabajo
obst = obs(:,[1,2,5,6]);
%
% Residuos con respecto a las incógnintas iniciales. 
vprimas = A*x-K;
v = zeros(2*2*c,1);
v(1:4:end) = vprimas(1:2:end);
v(2:4:end) = vprimas(2:2:end);
%v = zeros(2*2*c,1) %Si inicializas a 0 debes dejar que itere más de 2


% %Vueltas del bucle
vueltas = -1;

while 1
    %Contador de vueltas
    vueltas = vueltas + 1;
   
    %Jacobiana
    A(1:2:end,1) = obst(1:end,1);
    A(2:2:end,1) = obst(1:end,2);
    A(1:2:end,2) = obst(1:end,2);
    A(2:2:end,2) = -obst(1:end,1);
    A(2:2:end,3) = 0;
    A(1:2:end,4) = 0;
    
    
    contador_columnas = 0;
    for i = 1:2:2*c %Coloca los valores de las observaciones
        contador_columnas = contador_columnas + 1;
        pos = 4*contador_columnas-3; 
        B(i,pos:pos+3) = [x(1) x(2) -1 0]; %Al cuadrar los elementos se machacan
        B(i+1,pos:pos+3) = [-x(2) x(1) 0 -1];
    end

    %Términos independientes
    funcion1 = obst(:,3) - x(1)*obst(:,1) - x(2)*obst(:,2) - x(3);
    funcion2 = obst(:,4) - x(1)*obst(:,2) + x(2)*obst(:,1) - x(4);
    K(1:2:end) = funcion1;
    K(2:2:end) = funcion2;
    Kaux = B*v;
    K = K + Kaux;

    %Matriz de pesos equivalentes
    Pe = inv(B*Q*B');
    Ne = A'*Pe*A;
    de = A'*Pe*K;
    %Solución
    Qxx = inv(Ne);
    dx = Qxx*de;
    x = x + dx;

    %Residuos
    ve = K-A*dx;
    v = Q*B'*Pe*ve;

    %Actualiza las observaciones
    obst = obs(:,[1,2,5,6])+ reshape(v,4,c)'; %Reorganizamos el vector a matriz

    if (vueltas > 2) && (abs(dx(1))<pa0)&&(abs(dx(2))<pa1) && (abs(dx(3))<pa2) && (abs(dx(4))<pa2)
        break;
    end
end

P = inv(Q);

%Varianza del observable a posteriori
vtpv = v'*P*v;
s02 = vtpv/(2*c-n);

%Matriz de varianzas-covarianzas de las incógnitas
Exx = s02*Qxx;
sx = sqrt(diag(Exx));

%Resultados
alpha = 1 - nivel_confianza;
chi2 = (2*c-n)*s02;
chi2_0 = chi2inv(alpha/2, 2*c-n);
chi2_1 = chi2inv(1-alpha/2, 2*c-n);


disp("-------------------------------------------")

%Condicional por el que se hace el contraste de hipótesis
%H0: varianza a priori = varianza a posteriori
%H1: varianza a priori != varianza a posteriori
if (chi2 > chi2_0)&&(chi2 < chi2_1)
    disp("Se acepta H0. Se da por bueno el modelo")
else
    disp("Se rechaza H0. Se deberán añadir más observaciones o utilizar un método robusto")
end

disp(" ")

fprintf("nº de iteraciones: %d\n", vueltas)
disp(" ")

rotacion = atan2(x(2),x(1));
rotacion_deg = rad2deg(rotacion);
escala = x(2)/sin(rotacion);
disp('Parametros del modelo ajustados [α, L, Tx, Ty]:');
fprintf("Rotación: %.4f\n", rotacion_deg)
fprintf("Escala: %.4f\n", escala)
fprintf("Desplazamiento en x: %.4f\n", x(3))
fprintf("Desplazamiento en y: %.4f\n", x(4))

disp(" ")

disp("Matriz de varianzas y covarianzas:")
disp(double(Exx))