clc; clear; close all;

%% Load model + normalization stats
M = load("Models/ResNet18_Spectrogram_trained.mat");   % must contain: trainedNet, targetMean, targetStd, horizon
S = load("Image Training Data Spectrogram/normalization_stats.mat");   % must contain: mu_u, sigma_u, mu_y, sigma_y

trainedNet = M.trainedNet;
targetMean = M.targetMean;
targetStd = M.targetStd;
horizon = M.horizon;

mu_u = S.mu_u;
sigma_u = S.sigma_u;
mu_y = S.mu_y;
sigma_y = S.sigma_y;

%% Config
evalFolder = "Training Data";   % change to a separate "Test Data" folder if you have one
window_size = 200;
winLength = 64;
noverlap = round(0.9 * winLength);
nfft = 128;
nStd = 3;
evalStride = winLength - noverlap;   % matches the deployed Simulink block's update cadence

cmap = parula(256);
hammingWin = hamming(winLength);
numRows = floor(nfft/2) + 1;
numCols = floor((window_size - noverlap) / (winLength - noverlap));

fileList = dir(fullfile(evalFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', evalFolder);
end

%% Results table
results = table('Size', [numel(fileList), 4], ...
    'VariableTypes', {'string','double','double','double'}, ...
    'VariableNames', {'Dataset','RMSE','MAE','Fit'});

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    Sd = load(fpath);
    if ~isfield(Sd,'y') || ~isfield(Sd,'u') || ~isfield(Sd,'T')
        warning('Skipping "%s": missing y, u, or T.', fileList(i).name);
        continue;
    end
    y = Sd.y(:);
    u = Sd.u(:);
    Fs = 1/Sd.T;

    N = length(y);
    endIdxList = window_size:evalStride:(N-horizon);
    if isempty(endIdxList)
        warning('Skipping "%s": too short for window_size=%d, horizon=%d.', fileList(i).name, window_size, horizon);
        continue;
    end

    numWindows = numel(endIdxList);
    images = zeros(2*numRows, numCols, 3, numWindows);
    targets = zeros(numWindows,1);

    for w = 1:numWindows
        endIdx = endIdxList(w);
        histStart = endIdx - window_size + 1;
        uWin = u(histStart:endIdx);
        yWin = y(histStart:endIdx);

        su = spectrogram(uWin, hammingWin, noverlap, nfft, Fs);
        sy = spectrogram(yWin, hammingWin, noverlap, nfft, Fs);
        s_mag_u = 10*log10(abs(su) + eps);
        s_mag_y = 10*log10(abs(sy) + eps);

        z_u = max(-nStd, min(nStd, (s_mag_u-mu_u)/sigma_u));
        z_y = max(-nStd, min(nStd, (s_mag_y-mu_y)/sigma_y));
        u_norm = (z_u+nStd)/(2*nStd);
        y_norm = (z_y+nStd)/(2*nStd);

        combined = [u_norm; y_norm];
        ind_img = gray2ind(combined, 256);
        images(:,:,:,w) = ind2rgb(ind_img, cmap);
        targets(w) = y(endIdx+horizon);
    end

    %% Predict (batched)
    predsNorm = predict(trainedNet, images);
    YPred = predsNorm * targetStd + targetMean;

    %% Metrics
    rmse = sqrt(mean((targets-YPred).^2));
    mae = mean(abs(targets-YPred));
    fit = 100*(1-norm(targets-YPred)/norm(targets-mean(targets)));

    [~, baseName, ~] = fileparts(fileList(i).name);
    results.Dataset(i) = baseName;
    results.RMSE(i) = rmse;
    results.MAE(i) = mae;
    results.Fit(i) = fit;

    fprintf('%s: RMSE=%.5f, MAE=%.5f, Fit=%.2f%%\n', baseName, rmse, mae, fit);

    %% Separate figure per dataset
    figure('Name', baseName, 'NumberTitle', 'off');
    plot(endIdxList+horizon, targets, 'b', 'LineWidth', 1.5); hold on;
    plot(endIdxList+horizon, YPred, 'r', 'LineWidth', 1.5);
    legend('Measured','Predicted');
    xlabel('Sample'); ylabel('Output');
    title(sprintf('%s  (RMSE=%.3f, MAE=%.3f, Fit=%.1f%%)', baseName, rmse, mae, fit), 'Interpreter', 'none');
    grid on;
end

%% Remove any skipped (empty) rows and show summary
results(results.Dataset == "", :) = [];
disp(results);
