clc; clear; close all;
load("data_real_2.mat");     % adjust filename if data.mat is genuinely different from before
load("features_2.mat");      % adjust filename if features.mat is genuinely different from before
maxNumCompThreads(feature('numcores'));

%% Build equal-length input/output windows for each sample
max_lag = max([significant_input_lags(:); top_output_lags(:)]);
L = max_lag - dead_time;     % common window length for both channels
if L < 1
    error('dead_time must be smaller than max_lag.');
end

N = length(y);
startIdx = max_lag + 1;      % first valid k, ensures all indices below are >= 1
X = cell(N-max_lag, 1);
Y = zeros(N-max_lag, 1);

for k = startIdx:N
    row = k - max_lag;
    uSeq = u(k-dead_time-L+1 : k-dead_time);   % delayed input window, length L
    ySeq = y(k-L : k-1);                       % past output window, length L
    X{row} = [uSeq(:)'; ySeq(:)'];             % 2 x L: channel 1 = u, channel 2 = y
    Y(row) = y(k);
end

%% Train/val split — block-random, same rationale as the MLP (chirp is non-stationary)
total_samples = numel(X);
train_ratio = 0.7;
blockSize = 500;   % keep >> max_lag
numBlocks = floor(total_samples/blockSize);
blockOrder = randperm(numBlocks);
train_blocks = blockOrder(1:round(train_ratio*numBlocks));
val_blocks   = blockOrder(round(train_ratio*numBlocks)+1:end);

train_idx = []; val_idx = [];
for b = train_blocks
    train_idx = [train_idx, (b-1)*blockSize+1 : min(b*blockSize, total_samples)]; %#ok<AGROW>
end
for b = val_blocks
    val_idx = [val_idx, (b-1)*blockSize+1 : min(b*blockSize, total_samples)]; %#ok<AGROW>
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

save('lstm_model.mat', 'net', 'muX', 'sigmaX', 'muY', 'sigmaY');
