%%% Converts timeseries data into scalogram data. Allows selection of
%%% dead_zone parameter that will only be used for guessing and not
%%% evaluation. Creates input regressors in thepe of rgb images and
%%% corresponding output vectors.

clc; clear; close all;

dataFolder = "Training Data";
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', dataFolder);
end

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
end
