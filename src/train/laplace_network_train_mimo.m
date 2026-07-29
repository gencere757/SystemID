%% laplace_network_train_mimo.m
%   Author: Arda Gencer
%   Functionality: True MIMO version of laplace_network_train.m -- one
%   network takes all 3 input channels + all 3 output channels' history
%   and predicts all 3 outputs at once (a single fullyConnectedLayer(3)
%   head), instead of training 3 separate SISO models.
%
%   Regressor window: feature_extraction_mimo.m selects *which* lags
%   matter per input/output pair, but those selections differ in length
%   and dead time across the 9 pairs, so there's no single sparse lag
%   pattern shared by all channels. Instead of trying to align 9
%   different sparse patterns into one input tensor, this script uses
%   the SAME simplification laplace_network_train.m already uses for
%   SISO: take the single largest lag needed across every pair (and
%   both level + rate variants) as a common window length L, and feed a
%   DENSE L-sample window of every channel into the sequence layer.
%   dead_time is intentionally NOT subtracted here (unlike the SISO
%   script) since dead times differ per pair -- using dead_time=0 keeps
%   the window conservative (includes everything any pair could need)
%   and lets the network's learned weights down-weight the truly-dead
%   portions itself. Revisit with a proper per-pair-aware window if this
%   turns out to hurt accuracy.

clc; clear; close all;

%% Parameters
maxEpochs = 100;
miniBatchSize = 512;
initialLearnRate = 1e-3;
numPoles = 16;
train_ratio = 0.7;
blockSize = 500;   % rows per block for the train/val split; keep >> L

load(fullfile("data", "features", "features_combined_mimo.mat"));   % top_output_lags, significant_input_lags, significant_output_lags, dead_time (+ dot variants), numChannels

%% Load every MIMO block file
dataFolder = fullfile("data", "train", "mimo");
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s". Run parse_data_mimo.m first.', dataFolder);
end

% Largest lag needed across every output channel's own past,
% input->output pairs, AND output->output coupling pairs (level and rate
% variants alike) -- see file header for why this becomes a single dense
% window instead of per-pair sparse regressors. Including
% significant_output_lags here is what actually lets the window reach
% far enough back to cover output-output coupling, not just the fact
% that y's other channels are present in yWindow below.
L = max([cellMax(top_output_lags), cellMax(significant_input_lags), cellMax(significant_output_lags), ...
          cellMax(top_output_dot_lags), cellMax(significant_input_dot_lags), cellMax(significant_output_dot_lags)]);
if L < 1
    error('Selected lag window came out empty -- check features_combined_mimo.mat.');
end
startIdx = L + 1;

Xall = {};
Yall = {};
datasetNames = strings(numel(fileList),1);

for fi = 1:numel(fileList)
    fpath = fullfile(fileList(fi).folder, fileList(fi).name);
    Dtmp = load(fpath);   % load into a struct to avoid name collisions across files

    if ~isfield(Dtmp,'y') || ~isfield(Dtmp,'u')
        warning('Skipping "%s": missing y or u variable.', fileList(fi).name);
        continue;
    end
    y_i = Dtmp.y;   % N x 3
    u_i = Dtmp.u;   % N x 3
    N = size(y_i, 1);

    if N <= startIdx
        warning('Skipping "%s": too short (length=%d) for L=%d.', fileList(fi).name, N, L);
        continue;
    end
    if size(y_i,2) ~= numChannels || size(u_i,2) ~= numChannels
        warning('Skipping "%s": expected %d channels.', fileList(fi).name, numChannels);
        continue;
    end

    Xi = cell(N-L, 1);
    Yi = zeros(N-L, numChannels);
    for k = startIdx:N
        row = k - L;
        uWindow = u_i(k-L:k-1, :);   % L x 3
        yWindow = y_i(k-L:k-1, :);   % L x 3
        Xi{row} = [uWindow'; yWindow'];   % 6 x L  (channels x time)
        Yi(row, :) = y_i(k, :);           % 1 x 3 -- all outputs at once
    end

    Xall{end+1} = Xi; %#ok<AGROW>
    Yall{end+1} = Yi; %#ok<AGROW>
    datasetNames(fi) = fileList(fi).name;

    fprintf('Loaded "%s": %d samples -> %d sequence rows\n', fileList(fi).name, N, size(Yi,1));
end

if isempty(Yall)
    error('No valid MIMO datasets were processed.');
end

X = cat(1, Xall{:});
Y = cat(1, Yall{:});   % (total rows) x 3

rowsPerDataset = cellfun(@(c) size(c,1), Yall);
boundarySamples = cumsum(rowsPerDataset);

%% Split for validation -- contiguous blocks *within each dataset file*,
% blocks randomly assigned to train/val. Same reasoning as
% multi_data_MLP.m / LSTM.m / laplace_network_train.m: avoids point-wise
% leakage between adjacent, mostly-overlapping windows, while never
% letting a block or excitation-level file bleed across the train/val
% boundary mid-block.
train_idx = [];
val_idx = [];
datasetStarts = [0; boundarySamples(:)];
for d = 1:numel(rowsPerDataset)
    rangeStart = datasetStarts(d) + 1;
    rangeEnd = datasetStarts(d+1);
    nRows = rangeEnd - rangeStart + 1;
    nBlocksD = max(1, floor(nRows / blockSize));
    blockOrder = randperm(nBlocksD);
    nTrainBlocks = round(train_ratio * nBlocksD);
    trainBlocks = blockOrder(1:nTrainBlocks);
    valBlocks = blockOrder(nTrainBlocks+1:end);
    for b = trainBlocks
        bStart = rangeStart + (b-1)*blockSize;
        bEnd = min(rangeStart + b*blockSize - 1, rangeEnd);
        train_idx = [train_idx, bStart:bEnd]; %#ok<AGROW>
    end
    for b = valBlocks
        bStart = rangeStart + (b-1)*blockSize;
        bEnd = min(rangeStart + b*blockSize - 1, rangeEnd);
        val_idx = [val_idx, bStart:bEnd]; %#ok<AGROW>
    end
    leftoverStart = rangeStart + nBlocksD*blockSize;
    if leftoverStart <= rangeEnd
        train_idx = [train_idx, leftoverStart:rangeEnd]; %#ok<AGROW>
    end
end
train_idx = sort(train_idx);
val_idx = sort(val_idx);

XTrain = X(train_idx);
YTrain = Y(train_idx, :);
XVal   = X(val_idx);
YVal   = Y(val_idx, :);

%% Normalize
% Channel-wise mean/std across all timesteps in the training set (6
% input channels: u1,u2,u3,y1,y2,y3)
allTrain = cell2mat(XTrain');           % 6 x (L*numTrainSamples)
muX = mean(allTrain, 2);
sigmaX = std(allTrain, 0, 2);

normalizeSeq = @(c) cellfun(@(s) (s - muX)./sigmaX, c, 'UniformOutput', false);
XTrain = normalizeSeq(XTrain);
XVal   = normalizeSeq(XVal);

% Per-output-channel mean/std (3 outputs)
muY = mean(YTrain, 1);
sigmaY = std(YTrain, 0, 1);
YTrain = (YTrain - muY)./sigmaY;
YVal   = (YVal - muY)./sigmaY;

% trainNetwork's pre-flight check can't tell a custom layer (like
% lastTimestepPooling1dLayer) reduces a sequence down to one value per
% observation -- it only trusts built-in constructs it recognizes (e.g.
% OutputMode="last" on an LSTM). Since it still treats this network as
% sequence-output, it demands Y in the same shape: a cell array of
% single-timestep sequences (numChannels x 1 per observation), not a
% plain M x numChannels matrix. Keep plain-matrix copies for evaluation
% after training, since predict() will likely return the same
% cell-of-sequences shape.
YValMat = YVal;   % keep a plain matrix copy for evaluation below
YTrain = num2cell(YTrain', 1)';   % Mx1 cell, each numChannels x 1
YVal   = num2cell(YVal', 1)';

%% MIMO Laplace architecture
% 6 input channels (u1,u2,u3,y1,y2,y3 history) -> 3 output channels
% (y1,y2,y3 next-step prediction), single shared network.
%
% Uses lastTimestepPooling1dLayer instead of globalAveragePooling1dLayer:
% averaging over the whole L-sample window (as global average pooling
% does) treats a sample from L steps back the same as the sample right
% before the prediction point, which can throw away exactly the recent-
% history information the regression head needs. Taking just the last
% timestep instead was the first thing tried after training plateaued at
% RMSE~0.65 (normalized) with average pooling and the plateau didn't move
% even after adding output-output coupling regressors -- see project
% memory / conversation history for the earlier diagnosis steps.
numInputChannels = 2 * numChannels;   % u + y history, stacked
numOutputs = numChannels;

layers = [
    sequenceInputLayer(numInputChannels, "Normalization", "none")
    laplaceLayer(numPoles, numInputChannels)
    lastTimestepPooling1dLayer
    fullyConnectedLayer(128)
    tanhLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(numOutputs)
    regressionLayer
];

options = trainingOptions("adam", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=initialLearnRate, ...
    L2Regularization=1e-4, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=60, ...
    Plots="training-progress", ...
    Verbose=true);

%% Train
net = trainNetwork(XTrain, YTrain, layers, options);

%% Evaluate (per output channel + overall)
YPred = predict(net, XVal);
% predict() on a network trained with cell-array (sequence) responses
% returns predictions in that same shape -- a cell array, one entry per
% observation, each numChannels x 1 (or numChannels x T with T=1).
% Unpack back into a plain M x numChannels matrix to match YValMat.
if iscell(YPred)
    YPred = cell2mat(cellfun(@(c) reshape(c(:,end), 1, []), YPred, 'UniformOutput', false));
end
YPred_actual = YPred .* sigmaY + muY;
YVal_actual  = YValMat .* sigmaY + muY;

rmse_per_ch = sqrt(mean((YVal_actual - YPred_actual).^2, 1));
mae_per_ch  = mean(abs(YVal_actual - YPred_actual), 1);
fit_per_ch  = 100 * (1 - vecnorm(YVal_actual - YPred_actual, 2, 1) ./ vecnorm(YVal_actual - mean(YVal_actual,1), 2, 1));

rmse = mean(rmse_per_ch);
mae  = mean(mae_per_ch);
fit  = mean(fit_per_ch);

% RMSE/MAE are in physical output units (meters here -- the FSM
% displacement signal is O(1e-5) m), so %.4f rounds them to 0.0000 and
% makes them look broken. Use scientific notation so the actual
% magnitude is visible. Fit% is unit-independent (a ratio) so it doesn't
% have this problem.
fprintf('MIMO LNO Validation RMSE per channel: [%s]  (mean %.6e)\n', num2str(rmse_per_ch, '%.6e  '), rmse);
fprintf('MIMO LNO Validation MAE  per channel: [%s]  (mean %.6e)\n', num2str(mae_per_ch, '%.6e  '), mae);
fprintf('MIMO LNO Validation Fit  per channel: [%s]%%  (mean %.2f%%)\n', num2str(fit_per_ch, '%.2f  '), fit);

predFig = figure('Name', 'MIMO_LNO_Validation_Predictions');
for c = 1:numChannels
    subplot(numChannels, 1, c);
    plot(YVal_actual(:,c), 'b', 'LineWidth', 1.2); hold on;
    plot(YPred_actual(:,c), 'r', 'LineWidth', 1.2);
    legend('True', 'Predicted');
    title(sprintf('y%d  (RMSE=%.3e, Fit=%.1f%%)', c, rmse_per_ch(c), fit_per_ch(c)));
    xlabel('Sample'); ylabel(sprintf('y%d', c));
    grid on;
end

modelsFolder = fullfile("data", "models");
if ~exist(modelsFolder, 'dir')
    mkdir(modelsFolder);
end
save(fullfile(modelsFolder, 'lno_mimo_model.mat'), 'net', 'datasetNames', 'L', 'numChannels', ...
     'muX', 'sigmaX', 'muY', 'sigmaY');

%% Archive this run (model + per-channel plot + mean metrics) so it isn't lost or overwritten next run
log_run("laplace_network_train_mimo", sprintf("%d MIMO block files, L=%d, last-timestep pooling", numel(fileList), L), ...
    rmse, mae, fit, predFig, fullfile(modelsFolder, 'lno_mimo_model.mat'));

%% --- Local functions ---
function m = cellMax(c)
% Max value across a cell array of numeric vectors (possibly of
% different, even empty, lengths) -- e.g. top_output_lags is 1x3 with a
% different-length lag vector per output channel.
    m = 0;
    for idx = 1:numel(c)
        if ~isempty(c{idx})
            m = max(m, max(c{idx}));
        end
    end
end
