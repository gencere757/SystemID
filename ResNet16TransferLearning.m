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

%Freeze the remaining layers
lgraph = freezeWeights(lgraph);
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

function lgraph = freezeWeights(lgraph)
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    for ii = 1:numel(layers)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end

    lgraph = createLgraphUsingConnections(layers, connections);
end

function lgraph = createLgraphUsingConnections(layers, connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph, layers(i));
    end
    for c = 1:height(connections)
        lgraph = connectLayers(lgraph, connections.Source{c}, connections.Destination{c});
    end
end
