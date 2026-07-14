clc; clear; close all;
load("data.mat");
load("features.mat");

max_lag = max([significant_input_lags(:); top_output_lags(:)]);
N = size(u,1);

X = zeros(N-max_lag,(max_lag-dead_time+1) * 2);
Y = zeros(N-max_lag,1);

for k = max_lag+1:N
    row = k-max_lag;
    X(row,:) = [u(row:k-dead_time)' y(row:k-dead_time)'];
    Y(row) = y(k);
end
