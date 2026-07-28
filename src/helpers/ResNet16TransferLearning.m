%% ResNet16TransferLearning.m
%   Author: Arda Gencer
%   Date: 24.07.2026
clc; clear; close all;
saveFolder = fullfile("data", "models");
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

dataTypes = ["Spectrogram", "Scalogram"];
fineTuneFactor = 0.1;   %For conv5
fineTuneFactorEarlier = 0.03;   %For conv4

for d = 1:numel(dataTypes)
    dataType = dataTypes(d);
    dataFolder = fullfile("data", "images", lower(dataType));
    file = dir(fullfile(dataFolder,"dataset.mat"));
    fpath = fullfile(file.folder, file.name);
    S = load(fpath, 'results');
    dataset = S.results;

    [H, W, C] = size(dataset(1).image); %The size of input
    horizon = dataset(1).horizon;   %Prediction horizon

    pre_net = resnet18;
    lgraph = layerGraph(pre_net);

    %Modify the original input layer of ResNet18
    origInputLayer = pre_net.Layers(1);
    newInput = imageInputLayer([H W C], ...
        'Name', 'data', ...
        'Normalization', 'zerocenter', ...
        'Mean', imresize(origInputLayer.Mean, [H W]));
    lgraph = replaceLayer(lgraph, 'data', newInput);
    
    %Remove original output layers
    lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
    
    %Freeze weight except for conv4-5 
    lgraph = setLayerLearnRates(lgraph, fineTuneFactor, fineTuneFactorEarlier);
    
    %Add new output layers 
    newFC = fullyConnectedLayer(horizon, 'Name', 'fc_regression', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    newOutput = regressionLayer('Name', 'regression_output');
    lgraph = addLayers(lgraph, newFC);
    lgraph = addLayers(lgraph, newOutput);
    lgraph = connectLayers(lgraph, 'pool5', 'fc_regression');
    lgraph = connectLayers(lgraph, 'fc_regression', 'regression_output');

    net = lgraph;
    save(fullfile(saveFolder, sprintf("ResNet18_%s_model_untrained.mat", dataType)), "net");    %Save the network
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

function lgraph = setLayerLearnRates(lgraph, fineTuneFactor, fineTuneFactorEarlier)
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    for ii = 1:numel(layers)
        layerName = layers(ii).Name;

        isLastBlock = contains(layerName, 'res5') || contains(layerName, 'bn5');
        isEarlierBlock = contains(layerName, 'res4') || contains(layerName, 'bn4');

        if isLastBlock
            factor = fineTuneFactor;
            l2Factor = fineTuneFactor;
        elseif isEarlierBlock
            factor = fineTuneFactorEarlier;
            l2Factor = fineTuneFactorEarlier;
        else
            factor = 0;
            l2Factor = 0;
        end

        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = factor;
            end
            if ~isempty(regexp(propName, 'L2Factor$', 'once'))
                layers(ii).(propName) = l2Factor;
            end
        end
    end

    lgraph = createLgraphUsingConnections(layers, connections);
end
