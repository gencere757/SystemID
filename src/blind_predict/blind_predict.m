%% run_lno_simulink_and_compare.m
%   Author: Arda Gencer
%   Date: 29/07/2026
%   Runs the LNO_simulink model against a USER-SELECTED input dataset.
%   If the selected file contains 'y' as well as 'u', compares the
%   predicted output against it (RMSE/MAE/Fit + overlay plot). If the
%   file only contains 'u', just runs and plots the closed-loop
%   prediction on its own -- no comparison possible without ground truth.
clc; clear; close all;
modelName = 'LNO_simulink';

%% Load SimulinkParams.mat directly into this workspace (flat, not into a
% struct) -- the LTI System blocks need C_optd/notchD as plain base-
% workspace variables. Doing this explicitly here, rather than relying
% solely on the model's InitFcn, avoids ordering issues after a `clear`.
if ~isfile('SimulinkParams.mat')
    error('SimulinkParams.mat not found. Run generate_simulink_params_lno.m first.');
end
load('SimulinkParams.mat');   % brings in: net, max_lag, dead_time, L, muX, sigmaX, muY, sigmaY, T, C_optd, notchD
load('lno_fast_params.mat');

%% Let the user pick the input dataset ------------------------------------
% The selected file must contain 'u'. 'y' is optional -- if present,
% you get a full comparison; if absent, you just get the prediction.
[dataFile, dataPath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
    'Select input dataset (must contain u)', ...
    fullfile('data'));   % opens starting in the data/ folder if it exists

if isequal(dataFile, 0)
    error('No file selected. Aborting.');
end

cleanedDataFile = fullfile(dataPath, dataFile);
fprintf('Using dataset: %s\n', cleanedDataFile);

D = load(cleanedDataFile);
if ~isfield(D, 'u')
    error('"%s" must contain variable "u".', cleanedDataFile);
end
u_raw = D.u(:);

hasGroundTruth = isfield(D, 'y');
if hasGroundTruth
    y_real = D.y(:);
    if length(u_raw) ~= length(y_real)
        error(['"%s" has mismatched lengths: u has %d samples, y has %d samples. ' ...
            'They must be the same length (paired input/output).'], ...
            cleanedDataFile, length(u_raw), length(y_real));
    end
else
    fprintf('No "y" found in this file -- running prediction only, no comparison.\n');
end

%% Build u_timeseries for the "From Workspace" block from the selected file
T_sample = T;   % from SimulinkParams.mat, loaded above
timeVectorReal = (0:length(u_raw)-1)' * T_sample;
u_timeseries = timeseries(u_raw, timeVectorReal);

%% Match the model's stop time to however long this dataset actually is,
% so you don't need to hand-edit the model's solver config per file.
stopTime = timeVectorReal(end);
set_param(modelName, 'StopTime', num2str(stopTime));

%% Run the Simulink model
fprintf('Running %s (StopTime = %.4f s, %d samples)...\n', ...
    modelName, stopTime, length(u_raw));
set_param(modelName, 'SimulationMode', 'accelerator');
simOut = sim(modelName);

%% Extract predicted output
% "To Workspace" block ("logsout") logs the Unit Delay output (= the
% model's denormalized prediction, fed back as y_hat_feedback).
if isprop(simOut, 'logsout') || isfield(simOut, 'logsout')
    predTS = simOut.logsout;   % timeseries, per ToWorkspace SaveFormat="Timeseries"
else
    error(['Could not find "logsout" in the simulation output. Check that the ' ...
'"To Workspace" block''s VariableName is still "logsout" and that ' ...
'SaveFormat is set to "Timeseries".']);
end
timePred = predTS.Time;
yPred = predTS.Data(:);

%% Warm-up window: the persistent buffers inside laplaceFastPredictor
%% start at zero, so predictions during the first samples aren't
%% meaningful. Using 2*L (rather than L) also accounts for closed-loop
%% feedback contamination -- see prior discussion on why raw L alone
%% can still look off.
warmupSamples = 2*L + dead_time;
warmupTime = warmupSamples * T_sample;
validIdx = timePred > warmupTime;

if hasGroundTruth
    %% Align predicted vs real by time (interpolate real y onto the
    %% simulation's output time vector, in case sample times don't align
    %% exactly due to solver step sizes)
    yRealAligned = interp1(timeVectorReal, y_real, timePred, 'linear', 'extrap');

    yPredValid = yPred(validIdx);
    yRealValid = yRealAligned(validIdx);

    if isempty(yPredValid)
        error(['No samples remain after excluding the warm-up period (%.4f s). ' ...
            'The selected dataset is too short for this model''s window length.'], ...
            warmupTime);
    end

    %% Metrics
    rmse = sqrt(mean((yRealValid - yPredValid).^2));
    mae  = mean(abs(yRealValid - yPredValid));
    fit  = 100 * (1 - norm(yRealValid - yPredValid) / norm(yRealValid - mean(yRealValid)));
    fprintf('\nLNO Simulink Closed-Loop Prediction vs Real:\n');
    fprintf('  Dataset: %s\n', dataFile);
    fprintf('  RMSE: %.4f\n', rmse);
    fprintf('  MAE:  %.4f\n', mae);
    fprintf('  Fit:  %.2f%%\n', fit);
    fprintf('  (warm-up period of %d samples / %.4f s excluded from metrics)\n', ...
        warmupSamples, warmupTime);

    %% Plot: real vs predicted
    resultsFig = figure('Name', 'LNO_Simulink_RealVsPredicted');
    plot(timePred, yRealAligned, 'b', 'LineWidth', 1.2); hold on;
    plot(timePred, yPred, 'r', 'LineWidth', 1.2);
    xline(warmupTime, 'k--', 'Warm-up end');
    legend('Real y', 'Predicted y (closed-loop)', 'Location', 'best');
    xlabel('Time (s)'); ylabel('Output');
    title(sprintf('LNO Simulink: %s (RMSE=%.3f, MAE=%.3f, Fit=%.1f%%)', ...
        dataFile, rmse, mae, fit), 'Interpreter', 'none');
    grid on;
else
    %% No ground truth -- just show the prediction on its own
    rmse = NaN; mae = NaN; fit = NaN;
    fprintf('\nLNO Simulink Closed-Loop Prediction (no ground truth available):\n');
    fprintf('  Dataset: %s\n', dataFile);
    fprintf('  (warm-up period of %d samples / %.4f s marked but not excluded from plot)\n', ...
        warmupSamples, warmupTime);

    resultsFig = figure('Name', 'LNO_Simulink_Prediction');
    plot(timePred, yPred, 'r', 'LineWidth', 1.2);
    xline(warmupTime, 'k--', 'Warm-up end');
    legend('Predicted y (closed-loop)', 'Location', 'best');
    xlabel('Time (s)'); ylabel('Output');
    title(sprintf('LNO Simulink Prediction: %s', dataFile), 'Interpreter', 'none');
    grid on;
end

%% Log results ------------------------------------------------------------
% Creates data\results\<timestamp>_LNO_network\ containing the plot, and
% appends a summary row to data\results\results_log.csv.
resultsRoot = fullfile('data', 'results');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

timestampStr = datestr(now, 'yyyy-mm-dd_HHMMSS');
runFolderName = sprintf('%s_LNO_network', timestampStr);
runFolderPath = fullfile(resultsRoot, runFolderName);
mkdir(runFolderPath);

% Save the plot (both .fig for re-editing and .png for quick viewing)
savefig(resultsFig, fullfile(runFolderPath, 'prediction_plot.fig'));
saveas(resultsFig, fullfile(runFolderPath, 'prediction_plot.png'));

% Save the underlying data for this run too, in case it's needed later
if hasGroundTruth
    save(fullfile(runFolderPath, 'run_data.mat'), ...
        'timePred', 'yPred', 'yRealAligned', 'rmse', 'mae', 'fit', ...
        'dataFile', 'cleanedDataFile', 'warmupSamples', 'warmupTime');
else
    save(fullfile(runFolderPath, 'run_data.mat'), ...
        'timePred', 'yPred', ...
        'dataFile', 'cleanedDataFile', 'warmupSamples', 'warmupTime');
end

% Append to the shared CSV log, writing the header only if the file is new
csvPath = fullfile(resultsRoot, 'results_log.csv');
csvIsNew = ~isfile(csvPath);
fid = fopen(csvPath, 'a');
if fid == -1
    warning('Could not open "%s" for logging results.', csvPath);
else
    if csvIsNew
        fprintf(fid, 'Timestamp,Script,Dataset,RMSE,MAE,Fit,RunFolder\n');
    end
    fprintf(fid, '%s,%s,%s,%.6f,%.6f,%.4f,%s\n', ...
        timestampStr, mfilename, dataFile, rmse, mae, fit, runFolderPath);
    fclose(fid);
end

fprintf('\nLogged results to: %s\n', runFolderPath);
fprintf('Summary row appended to: %s\n', csvPath);
