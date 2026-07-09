clc; clear; close all;

load twotankdata;

max_lag = 25;

figure;
parcorr(y, max_lag);
title('Autocorrelation of Output State');
xlabel('Lags');
ylabel('Sample Autocorrelation');


u_diff = diff(u);
y_diff = diff(y);
figure;
max_lag = 100;
% Check cross-correlation up to 100 lags
crosscorr(u_diff, y_diff, max_lag); 
title('Cross-Correlation between Input and Output (Differentiated)');
xlim([0, max_lag]);

figure;
crosscorr(u, y, max_lag); 
title('Cross-Correlation between Input and Output (Differentiated)');
xlim([0, max_lag]);
