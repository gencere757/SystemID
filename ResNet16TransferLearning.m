%% ResNet16TransferLearning.m
%   Author: Arda Gencer
%   Date: 24.07.2026
clc; clear; close all;
saveFolder = "Models";
%% Loading the input data and targets
dataType = "Spectrogram";
dataFolder = "Image Training Data "+ dataType;
file = dir(fullfile(dataFolder,"*.mat"));
fpath = fullfile(file.folder, file.name);
S = load(fpath);
dataset = S.results;
numSamples = numel(dataset);
%Extract input images
[H, W, C] = size(dataset(1).image);   % get dimensions from the first image
images = zeros(H, W, C, numSamples);
for k = 1:numSamples
    images(:,:,:,k) = dataset(k).image;
end
%Extract outputs
targets = [dataset.target];
%Get horizon val
horizon = dataset(1).horizon;
%% Loading and Modifying Network
pre_net= resnet18;
lgraph = layerGraph(pre_net);

%Replace the input layer to match actual image size
newInput = imageInputLayer([H W C], 'Name', 'data', 'Normalization', 'none');
lgraph = replaceLayer(lgraph, 'data', newInput);

%Remove old layers
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
%The new layers
newFC = fullyConnectedLayer(horizon, 'Name', 'fc_regression');  %The new fully connected layer that will output our final value
newOutput = regressionLayer('Name', 'regression_output');
%Apply the new layers
lgraph = addLayers(lgraph, newFC);
lgraph = addLayers(lgraph, newOutput);
%Connect the modified network
lgraph = connectLayers(lgraph, 'pool5', 'fc_regression');
lgraph = connectLayers(lgraph, 'fc_regression', 'regression_output');
net = lgraph;
save(fullfile(saveFolder,"ResNet18_model_untrained.mat"), "net");
