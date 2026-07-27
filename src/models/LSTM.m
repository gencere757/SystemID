clc; clear; close all;
load("features_combined.mat");   % top_output_lags, significant_input_lags, dead_time (+ dot variants, unused here)
maxNumCompThreads(feature('numcores'));

%% Load every dataset in the "Training Data" folder
dataFolder = "Training Data";
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s". Check the folder name/path.', dataFolder);
end

max_lag = max([significant_input_lags(:); top_output_lags(:)]);
L = max_lag - dead_time;     % common window length for both channels
if L < 1
    error('dead_time must be smaller than max_lag.');
end
startIdx = max_lag + 1;

Xall = {};
Yall = {};
datasetNames = strings(numel(fileList),1);

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    Dtmp = load(fpath);   % load into a struct to avoid name collisions across files

    if ~isfield(Dtmp,'y') || ~isfield(Dtmp,'u')
        warning('Skipping "%s": missing y or u variable.', fileList(i).name);
        continue;
    end

    y_i = Dtmp.y(:);
    u_i = Dtmp.u(:);
    N = length(y_i);

    if N <= startIdx
        warning('Skipping "%s": too short (length=%d) for max_lag=%d.', fileList(i).name, N, max_lag);
        continue;
    end

    Xi = cell(N-max_lag, 1);
    Yi = zeros(N-max_lag, 1);
    for k = startIdx:N
        row = k - max_lag;
        uSeq = u_i(k-dead_time-L+1 : k-dead_time);   % delayed input window, length L
        ySeq = y_i(k-L : k-1);                       % past output window, length L
        Xi{row} = [uSeq(:)'; ySeq(:)'];               % 2 x L: channel 1 = u, channel 2 = y
        Yi(row) = y_i(k);
    end

    Xall{end+1} = Xi; %#ok<AGROW>
    Yall{end+1} = Yi; %#ok<AGROW>
    datasetNames(i) = fileList(i).name;

    fprintf('Loaded "%s": %d samples -> %d sequence rows\n', fileList(i).name, N, length(Yi));
end

if isempty(Yall)
    error('No valid datasets were processed.');
end

X = cat(1, Xall{:});
Y = cat(1, Yall{:});

% Cumulative row counts, for keeping blocks within one dataset and for
% plotting dataset boundaries later
rowsPerDataset = cellfun(@length, Yall);
boundarySamples = cumsum(rowsPerDataset);

%% Split data for validation — contiguous blocks *within each dataset*,
% blocks randomly assigned to train/val. Mirrors the "block" split mode in
% multi_data_MLP.m: avoids point-wise leakage between adjacent, mostly-
% overlapping windows while still sampling every dataset's full time range.
train_ratio = 0.7;
blockSize = 500;   % rows per block; keep >> max_lag

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
    % Leftover rows past the last full block go to train
    leftoverStart = rangeStart + nBlocksD*blockSize;
    if leftoverStart <= rangeEnd
        train_idx = [train_idx, leftoverStart:rangeEnd]; %#ok<AGROW>
    end
end
train_idx = sort(train_idx);
val_idx = sort(val_idx);

XTrain = X(train_idx);
YTrain = Y(train_idx);
XVal   = X(val_idx);
YVal   = Y(val_idx);

%% Normalize
% Compute channel-wise mean/std across all timesteps in the training set
allTrain = cell2mat(XTrain');           % 2 x (L*numTrainSamples)
muX = mean(allTrain, 2);
sigmaX = std(allTrain, 0, 2);

normalizeSeq = @(c) cellfun(@(s) (s - muX)./sigmaX, c, 'UniformOutput', false);
XTrain = normalizeSeq(XTrain);
XVal   = normalizeSeq(XVal);

muY = mean(YTrain);
sigmaY = std(YTrain);
YTrain = (YTrain - muY)/sigmaY;
YVal   = (YVal - muY)/sigmaY;

%% LSTM architecture (sequence-to-one regression)
numChannels = 2;   % u, y
layers = [
    sequenceInputLayer(numChannels, "Normalization", "none")
    lstmLayer(128, "OutputMode", "last")
    fullyConnectedLayer(64)
    tanhLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions("adam", ...
    MaxEpochs=30, ...
    MiniBatchSize=512, ...
    InitialLearnRate=1e-3, ...
    L2Regularization=1e-4, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=60, ...
    Plots="training-progress", ...
    Verbose=false);

%% Train
net = trainNetwork(XTrain, YTrain, layers, options);

%% Evaluate
YPred = predict(net, XVal);
YPred_actual = YPred*sigmaY + muY;
YVal_actual  = YVal*sigmaY + muY;

rmse = sqrt(mean((YVal_actual - YPred_actual).^2));
mae  = mean(abs(YVal_actual - YPred_actual));
fit  = 100 * (1 - norm(YVal_actual - YPred_actual) / norm(YVal_actual - mean(YVal_actual)));

fprintf('LSTM Validation RMSE: %.4f\n', rmse);
fprintf('LSTM Validation MAE:  %.4f\n', mae);
fprintf('LSTM Validation Fit:  %.2f%%\n', fit);

figure;
plot(YVal_actual, 'b', 'LineWidth', 1.2); hold on;
plot(YPred_actual, 'r', 'LineWidth', 1.2);
legend('True', 'Predicted');
title(sprintf('LSTM Validation Predictions (RMSE=%.3f, MAE=%.3f, Fit=%.1f%%)', rmse, mae, fit));
xlabel('Sample'); ylabel('Output');
grid on;

%% Predict whole sequence (all datasets, concatenated in load order)
Xnorm = normalizeSeq(X);
YPredWhole = predict(net, Xnorm);
YPredWhole = YPredWhole*sigmaY + muY;

figure;
plot(Y,'b','LineWidth',1.5); hold on;
plot(YPredWhole,'r','LineWidth',1.5);
for b = boundarySamples(1:end-1)
    xline(b, 'k--');
end
legend('True','Prediction');
xlabel('Sample (concatenated across all files)'); ylabel('Output');
title('LSTM prediction across all training datasets (dashed lines = dataset boundaries)');
grid on;

save('lstm_model.mat', 'net', 'datasetNames', 'max_lag', 'dead_time', 'muX', 'sigmaX', 'muY', 'sigmaY');
