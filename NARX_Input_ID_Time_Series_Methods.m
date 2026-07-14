clc; close all;

%load twotankdata;   %Load the data
load("data.mat");

%Find state lag 
max_lag = 150;

%Compute and plot the partial autocorrelation function of output of system
figure;
[acf_vals, lags_pacf] = parcorr(y, max_lag);
parcorr(y, max_lag);
title('Autocorrelation of Output State');
xlabel('Lags');
ylabel('Sample Autocorrelation');

%Clip the autocorrelation at the point x = 0 as its always 1
acf_vals(lags_pacf == 0) = [];
lags_pacf(lags_pacf == 0) = [];

[~, sort_idx] = sort(abs(acf_vals), 'descend'); %Sort the correlations in descending manner
top_output_lags = lags_pacf(sort_idx(1:min(50, length(sort_idx))));    %Take the 5 max points

%Finding Input Lag

%Differentiate the signals
u_diff = diff(u);
y_diff = diff(y);


max_lag = 150;  % Check cross-correlation up to 100 



%Compute and plot the cross correlation between input and output
[cc_vals, lags_cc] = crosscorr(u_diff, y_diff, max_lag);
figure;
crosscorr(u_diff, y_diff, max_lag);
title('Cross-Correlation between Input and Output (Differentiated)');
xlim([0, max_lag]);

%Clip the negative part of data
positive_lags_idx = lags_cc > 0;
cc_vals_pos = cc_vals(positive_lags_idx);
lags_cc_pos = lags_cc(positive_lags_idx);

%Create confidence interval
N = length(u_diff);
confidence_threshold = 0.05; 



% Find the first positive lag where correlation breaks above the threshold
significant_input_lags = lags_cc_pos(abs(cc_vals_pos) > confidence_threshold);

if ~isempty(significant_input_lags)
    dead_time = significant_input_lags(1); % First significant lag = Dead Time
else
    dead_time = 0; % Default to 0 if no lag breaks the confidence threshold
end
save('features.mat', 'top_output_lags', 'significant_input_lags', 'dead_time');
