clc;
clear;
close all;

load twotankdata

%% Parameters
Ts = 1;
lambda = 1e-4;      % sparsification threshold
polyOrder = 3;

%% Make column vectors
u = u(:);
y = y(:);

%% Estimate derivative
y_s = smoothdata(y,"sgolay",31);
dy = gradient(y,Ts);

%% Build library Theta(y,u)

Theta = ones(length(y),1);

% Linear
Theta = [Theta ...
         y ...
         u];

% Quadratic
if polyOrder >= 2
    Theta = [Theta ...
             y.^2 ...
             y.*u ...
             u.^2];
end

% Cubic
if polyOrder >= 3
    Theta = [Theta ...
             y.^3 ...
             y.^2.*u ...
             y.*u.^2 ...
             u.^3];
end

%% Initial least-squares estimate

Xi = Theta \ dy;

%% Sequential Thresholded Least Squares (STLS)

for k = 1:15

    small = abs(Xi) < lambda;

    Xi(small) = 0;

    big = ~small;

    Xi(big) = Theta(:,big)\dy;

end

%% Display coefficients

disp('Identified coefficients:')
disp(Xi)

%% Predicted derivative

dy_hat = Theta*Xi;

%% Integrate identified model

y_hat = zeros(size(y));
y_hat(1) = y(1);

for k = 1:length(y)-1
    y_hat(k+1) = y_hat(k) + Ts*dy_hat(k);
end

%% Plots

figure

subplot(2,1,1)
plot(y,'b')
hold on
plot(y_hat,'r--')
legend('Measured','CINDy')
title('Output Comparison')
grid on

subplot(2,1,2)
plot(dy,'b')
hold on
plot(dy_hat,'r--')
legend('Measured derivative','Estimated derivative')
title('Derivative Comparison')
grid on