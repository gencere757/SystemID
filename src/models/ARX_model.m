z = iddata(y,u,1);

nk = dead_time;               % Your identified dead time (e.g., 2)
nb = length(significant_input_lags); % Total number of input terms (e.g., 78)
na = max(top_output_lags);   % Max past output dependence (e.g., 5)

% Define the structural matrix
orders = [na, nb, nk];

%% Split data for validation (first 70% for training, last 30% for testing)
N_split = round(0.7 * z.N);
z_train = z(1:N_split);
z_val = z(N_split+1:end);

%% Train the ARX model
sys_arx = arx(z_train, orders);

%% Validation
figure;
compare(z_val, sys_arx);

[~, fit_info] = compare(z_val, sys_arx);
fprintf('Validation Data Fit Percentage: %.2f%%\n', fit_info);
