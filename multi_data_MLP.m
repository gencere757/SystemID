clc;clear;close all;
load("features_combined.mat");
maxNumCompThreads(feature('numcores'));


%% Load every dataset in the "Training Data" folder
dataFolder = "Training Data";
fileList = dir(fullfile(dataFolder, "*.mat"));

if isempty(fileList)
    error('No .mat files found in "%s". Check the folder name/path.', dataFolder);
end

maxLag = max([significant_input_lags(:); top_output_lags(:)]);

Xall = {};
Yall = {};
datasetID = [];
datasetNames = strings(numel(fileList),1);
boundarySamples = [];   % cumulative row count at each dataset's start, for plotting

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    Dtmp = load(fpath);   % load into a struct to avoid name collisions across files

    if ~isfield(Dtmp,'y') || ~isfield(Dtmp,'u')
        warning('Skipping "%s": missing y or u variable.', fileList(i).name);
        continue;
    end

    y_i = Dtmp.y(:);
    u_i = Dtmp.u(:);

    if isfield(Dtmp,'T') && exist('T_ref','var') && abs(Dtmp.T - T_ref) > eps
        warning('"%s" has a different sampling time (T=%.6g) than the first file (T=%.6g).', ...
            fileList(i).name, Dtmp.T, T_ref);
    elseif isfield(Dtmp,'T') && ~exist('T_ref','var')
        T_ref = Dtmp.T;
    end

    [Xi, Yi] = buildRegressors(y_i, u_i, top_output_lags, significant_input_lags, maxLag);

    Xall{end+1} = Xi; %#ok<AGROW>
    Yall{end+1} = Yi; %#ok<AGROW>
    datasetID = [datasetID; i*ones(size(Yi))]; %#ok<AGROW>
    datasetNames(i) = fileList(i).name;

    fprintf('Loaded "%s": %d samples -> %d regressor rows\n', fileList(i).name, length(y_i), length(Yi));
end

X = cat(1, Xall{:});
Y = cat(1, Yall{:});

% Cumulative row counts, for marking dataset boundaries on the full-sequence plot later
rowsPerDataset = cellfun(@length, Yall);
boundarySamples = cumsum(rowsPerDataset);

%% Split data for validation (random or sequential)
randomSplit = true;   % true = random split, false = sequential split
total_samples = size(X,1);
train_ratio = 0.7;
train_amount = floor(total_samples * train_ratio);
if randomSplit
    idx = randperm(total_samples);
    train_idx = idx(1:train_amount);
    val_idx = idx(train_amount+1:end);
else
    train_idx = 1:train_amount;
    val_idx = train_amount+1:total_samples;
    idx = [train_idx val_idx];
end
XTrain = X(train_idx,:);
YTrain = Y(train_idx);
XVal = X(val_idx,:);
YVal = Y(val_idx);

%% Normalize the data
[XTrain,muX,sigmaX] = zscore(XTrain);
XVal = (XVal-muX)./sigmaX;
[YTrain,muY,sigmaY] = zscore(YTrain);
YVal = (YVal-muY)./sigmaY;

%% Architecture
numFeatures = size(XTrain,2);
layers = [
    featureInputLayer(numFeatures,"Normalization","none")
    fullyConnectedLayer(256)
    batchNormalizationLayer
    tanhLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    tanhLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    batchNormalizationLayer
    fullyConnectedLayer(1)
    regressionLayer
];
options = trainingOptions("adam",...
    MaxEpochs=100,...
    MiniBatchSize=4096,...
    L2Regularization=1e-3,...
    ValidationFrequency=60,...
    InitialLearnRate=1e-3,...
    Shuffle="every-epoch",...
    DispatchInBackground=true, ...
    ValidationData={XVal,YVal},...
    Plots="training-progress",...
    Verbose=false);

%% Train
net = trainNetwork(XTrain,YTrain,layers,options);

%% Prediction
YPred = predict(net, XVal);
YPred = YPred * sigmaY + muY;
YTrue = YVal * sigmaY + muY;

if randomSplit
    [val_idx_sorted, order] = sort(val_idx);
    YTrue_plot = YTrue(order);
    YPred_plot = YPred(order);
else
    val_idx_sorted = val_idx;
    YTrue_plot = YTrue;
    YPred_plot = YPred;
end

rmse = sqrt(mean((YTrue_plot - YPred_plot).^2));
mae = mean(abs(YTrue_plot - YPred_plot));
fit = 100 * (1 - norm(YTrue_plot - YPred_plot) / norm(YTrue_plot - mean(YTrue_plot)));

%% Predict whole sequence (all datasets, concatenated in load order)
XWhole = (X - muX)./sigmaX;
YPredWhole = predict(net, XWhole);
YPredWhole = YPredWhole*sigmaY + muY;

figure;
plot(Y,'b','LineWidth',1.5); hold on;
plot(YPredWhole,'r','LineWidth',1.5);
for b = boundarySamples(1:end-1)
    xline(b, 'k--');
end
legend('True','Prediction');
xlabel('Sample (concatenated across all files)'); ylabel('Output');
title('Prediction across all training datasets (dashed lines = dataset boundaries)');
grid on;

save('MLP_model.mat', 'net', 'datasetNames');

%% --- Local function: build regressors for one dataset ---
function [X, Y] = buildRegressors(y, u, top_output_lags, significant_input_lags, maxLag)
    N = length(y);
    X = zeros(N-maxLag, length(top_output_lags)+length(significant_input_lags));
    Y = zeros(N-maxLag,1);
    for k = maxLag+1:N
        yReg = y(k-top_output_lags);
        uReg = u(k-significant_input_lags);
        X(k-maxLag,:) = [yReg(:)' uReg(:)'];
        Y(k-maxLag) = y(k);
    end
end
