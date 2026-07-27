%% TrainAllModels.m
%   Author: Arda Gencer
%   Date: 24.07.2026
%   Functionality: Trains a separate ResNet-18 regression model for each
%   dataset folder found (Spectrogram, Scalogram), using the pre-saved
%   untrained network, and saves each trained model with a matching
%   filename. Intended to run unattended.

clc; clear; close all;

%% Parameters
trainSplitRatio = 0.8;
maxEpochs = 70;
miniBatchSize = 32;
initialLearnRate = 5e-5;

dataTypes = ["Spectrogram", "Scalogram"];
saveFolder = "Models";
untrainedModelPath = fullfile(saveFolder, "ResNet18_model_untrained.mat");

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

%% Load the untrained network once
M = load(untrainedModelPath);
untrainedNet = M.net;

%% Loop over each data type, train and save independently
for d = 1:numel(dataTypes)
    dataType = dataTypes(d);
    dataFolder = "Image Training Data " + dataType;

    fprintf("\n========== Training on %s data ==========\n", dataType);

    dataFile = dir(fullfile(dataFolder, "*.mat"));
    if isempty(dataFile)
        warning("No .mat file found in '%s' -- skipping.", dataFolder);
        continue;
    end

    %% Load dataset
    S = load(fullfile(dataFile.folder, dataFile.name));
    dataset = S.results;
    numSamples = numel(dataset);

    if numSamples == 0
        warning("Dataset in '%s' is empty -- skipping.", dataFolder);
        continue;
    end

    %% Extract images
    [H, W, C] = size(dataset(1).image);
    images = zeros(H, W, C, numSamples);
    for k = 1:numSamples
        images(:,:,:,k) = dataset(k).image;
    end

    %% Extract targets (handles scalar or vector horizon)
    targetLen = numel(dataset(1).target);
    targets = zeros(numSamples, targetLen);
    for k = 1:numSamples
        targets(k,:) = dataset(k).target(:)';
    end

    horizon = dataset(1).horizon;

    %% Normalize targets
    % Normalize targets before training
    targetMean = mean(targets);
    targetStd = std(targets);
    targets_norm = (targets - targetMean) / targetStd;


    %% Chronological split (no shuffling of the split itself)
    numTrain = round(trainSplitRatio * numSamples);
    trainImages = images(:,:,:,1:numTrain);
    trainTargets = targets_norm(1:numTrain,:);
    valImages = images(:,:,:,numTrain+1:end);
    valTargets = targets_norm(numTrain+1:end,:);

    fprintf("Training samples: %d | Validation samples: %d\n", numTrain, numSamples-numTrain);

    %% Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', initialLearnRate, ...
        'ValidationData', {valImages, valTargets}, ...
        'ValidationFrequency', 30, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', true);

    %% Train (fresh copy of the untrained network each time)
    trainedNet = trainNetwork(trainImages, trainTargets, untrainedNet, options);

    %% Evaluate
    predictions = predict(trainedNet, valImages);
    rmse = sqrt(mean((predictions - valTargets).^2, 'all'));
    fprintf("[%s] Validation RMSE: %.4f\n", dataType, rmse);

    %% Save trained model, tagged by data type
    modelSavePath = fullfile(saveFolder, sprintf("ResNet18_%s_trained.mat", dataType));
    save(modelSavePath, "trainedNet", "rmse", "horizon","targetMean", "targetStd");
    fprintf("Saved model to: %s\n", modelSavePath);
end

fprintf("\nAll training runs complete.\n");
