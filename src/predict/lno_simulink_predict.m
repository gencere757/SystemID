%% run_lno_simulink_and_compare.m
%   Author: Arda Gencer
%   Date: 29/07/2026
%   Runs the LNO_simulink model, extracts the predicted output (logged
%   via the "Unit Delay" -> "To Workspace" path as 'logsout'), and
%   compares it against the real measured y from the same dataset used
%   to build u_timeseries in SimulinkParams.mat.


clc; clear; close all;

modelName = 'LNO_simulink';

%% Load SimulinkParams.mat directly into this workspace (flat, not into a
% struct) -- the LTI System blocks need C_optd/notchD as plain base-
% workspace variables. Doing this explicitly here, rather than relying
% solely on the model's InitFcn, avoids ordering issues after a `clear`.
if ~isfile('SimulinkParams.mat')
    error('SimulinkParams.mat not found. Run generate_simulink_params_lno.m first.');
end
load('SimulinkParams.mat');   % brings in: net, max_lag, dead_time, L, muX, sigmaX, muY, sigmaY, u, u_timeseries, T, C_optd, notchD

load('lno_fast_params.mat'); 

%% Load the real y for comparison — must be the SAME file that
%% generate_simulink_params_lno.m used to build u_timeseries.
dataDir = fullfile("data", "test");
matFiles = dir(fullfile(dataDir, "*.mat"));

if isempty(matFiles)
    error("No .mat file found in %s", dataDir);
end

cleanedDataFile = fullfile(matFiles(1).folder, matFiles(1).name);
D = load(cleanedDataFile);
if ~isfield(D,'y')
    error('"%s" must contain variable "y".', cleanedDataFile);
end
y_real = D.y(:);
T_sample = T;   % from SimulinkParams.mat, loaded above
timeVectorReal = (0:length(y_real)-1)' * T_sample;

%% Run the Simulink model
fprintf('Running %s...\n', modelName);
set_param(modelName, 'SimulationMode', 'accelerator');
simTime = u_timeseries.Time(end);
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

%% Align predicted vs real by time (interpolate real y onto the
%% simulation's output time vector, in case sample times don't align
%% exactly due to solver step sizes)
yRealAligned = interp1(timeVectorReal, y_real, timePred, 'linear', 'extrap');

%% Exclude the warm-up period: the persistent u/y buffers inside
%% "Regressor Reconstruction" start at zero, so predictions during the
%% first (L + dead_time) samples are not meaningful.
warmupSamples = L + dead_time;
warmupTime = warmupSamples * T_sample;
validIdx = timePred > warmupTime;

yPredValid = yPred(validIdx);
yRealValid = yRealAligned(validIdx);
timeValid = timePred(validIdx);

%% Metrics
rmse = sqrt(mean((yRealValid - yPredValid).^2));
mae  = mean(abs(yRealValid - yPredValid));
fit  = 100 * (1 - norm(yRealValid - yPredValid) / norm(yRealValid - mean(yRealValid)));

fprintf('\nLNO Simulink Closed-Loop Prediction vs Real:\n');
fprintf('  RMSE: %.4f\n', rmse);
fprintf('  MAE:  %.4f\n', mae);
fprintf('  Fit:  %.2f%%\n', fit);
fprintf('  (warm-up period of %d samples / %.4f s excluded from metrics)\n', ...
    warmupSamples, warmupTime);

%% Plot
figure('Name', 'LNO_Simulink_RealVsPredicted');
plot(timePred, yRealAligned, 'b', 'LineWidth', 1.2); hold on;
plot(timePred, yPred, 'r', 'LineWidth', 1.2);
xline(warmupTime, 'k--', 'Warm-up end');
legend('Real y', 'Predicted y (closed-loop)', 'Location', 'best');
xlabel('Time (s)'); ylabel('Output');
title(sprintf('LNO Simulink: Real vs Predicted (RMSE=%.3f, MAE=%.3f, Fit=%.1f%%)', ...
    rmse, mae, fit));
grid on;    
