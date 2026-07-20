clc; clear; close all;

%% ---- Step 1: Select and parse a single raw data file ----
[fileName, filePath] = uigetfile('ELEVATION_SQUARE_WAVE_25HZ_10URAD.mat', 'Select a raw data file to parse');
if isequal(fileName, 0)
    error('No file selected.');
end
fullPath = fullfile(filePath, fileName);

S = load("Test Raw Data\ELEVATION_SQUARE_WAVE_25HZ_10URAD.mat");
if ~isfield(S, 'data')
    error('"%s" does not contain a variable named "data" — check the file format.', fileName);
end
data = S.data;

try
    outputData = data{1};
    ts = outputData.Values;
    T = ts.Time(2) - ts.Time(1);
    y = outputData.Values.Data;

    inputData = data{2};
    u = squeeze(inputData.Values.Data);
catch ME
    error('Failed to parse "%s": %s', fileName, ME.message);
end

%% Trim transient (adjust as needed per dataset — check visually first if unsure)
transientEnd = 2500;
if transientEnd >= length(y)
    warning('transientEnd (%d) >= signal length (%d) — skipping trim.', transientEnd, length(y));
else
    y = y(transientEnd+1:end);
    u = u(transientEnd+1:end);
end

fprintf('Loaded and parsed "%s": %d samples, T=%.6g\n', fileName, length(y), T);

%% ---- Step 2: Package u for Simulink and set up the run ----
timeVector = (0:length(u)-1)' * T;
u_matrix = [timeVector, u(:)];   % From Workspace expects [time, data]

modelName = "SimulinkModel";     % <-- change to your actual model name (no .slx extension)

% Push u_matrix (and anything else your model's From Workspace blocks need)
% into the base workspace so the model can see it when it runs
assignin('base', 'u_matrix', u_matrix);
assignin('base', 'T', T);

%% Load the model if it isn't already open
if ~bdIsLoaded(modelName)
    load_system(modelName);
end

%% Set stop time to match this dataset's duration
simStopTime = (length(u) - 1) * T;
set_param(modelName, 'StopTime', num2str(simStopTime));

%% ---- Step 3: Run the simulation and pull the logged signal back ----
simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');

if ~isprop(simOut, 'logsout') && ~isfield(simOut, 'logsout')
    error('No "logsout" found in simulation output. Check that signal logging is enabled.');
end

logsout = simOut.logsout;

if isa(logsout, 'timeseries')
    % Single logged signal — logsout IS the timeseries directly
    y_sim_ts = logsout;
elseif isa(logsout, 'Simulink.SimulationData.Dataset')
    % Multiple logged signals — need to find 'y_sim' by name
    y_sim_element = [];
    for i = 1:logsout.numElements
        if strcmp(logsout{i}.Name, 'y_sim')
            y_sim_element = logsout{i};
            break;
        end
    end
    if isempty(y_sim_element)
        error('Signal "y_sim" not found among logged signals.');
    end
    y_sim_ts = y_sim_element.Values;
else
    error('Unexpected logsout type: %s', class(logsout));
end

y_sim = y_sim_ts.Data;
fprintf('Pulled "y_sim" from Simulink: %d samples\n', length(y_sim));

%% ---- Step 4: Align and compare ----
maxLag = 0; % <-- set this to your actual maxLag if you want to trim warm-up samples
if maxLag > 0
    y_sim_trimmed = y_sim(maxLag+1:end);
    y_real_trimmed = y(maxLag+1:end);
else
    y_sim_trimmed = y_sim;
    y_real_trimmed = y;
end

n = min(length(y_sim_trimmed), length(y_real_trimmed));
if length(y_sim_trimmed) ~= length(y_real_trimmed)
    warning('Sample count mismatch (sim=%d, real=%d) — truncating to shorter length (%d) for comparison.', ...
        length(y_sim_trimmed), length(y_real_trimmed), n);
end
y_sim_trimmed = y_sim_trimmed(1:n);
y_real_trimmed = y_real_trimmed(1:n);

%% Metrics
rmse_sim = sqrt(mean((y_real_trimmed - y_sim_trimmed).^2));
mae_sim = mean(abs(y_real_trimmed - y_sim_trimmed));
fit_sim = 100 * (1 - norm(y_real_trimmed - y_sim_trimmed) / norm(y_real_trimmed - mean(y_real_trimmed)));

fprintf('Closed-loop Simulink RMSE: %.4f\n', rmse_sim);
fprintf('Closed-loop Simulink MAE:  %.4f\n', mae_sim);
fprintf('Closed-loop Simulink Fit:  %.2f%%\n', fit_sim);

%% Plot
figure;
plot(y_real_trimmed, 'b', 'LineWidth', 1.2); hold on;
plot(y_sim_trimmed, 'r', 'LineWidth', 1.2);
legend('Real (recorded) y', 'Closed-loop Simulink prediction');
xlabel('Sample'); ylabel('Output');
title(sprintf('%s vs Simulink digital twin (RMSE=%.3f, Fit=%.1f%%)', fileName, rmse_sim, fit_sim), 'Interpreter', 'none');
grid on;
