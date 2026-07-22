%% Scalogram.m

%   Author: Arda Gencer
%   Date: 21.07.2026
%   Functionality: Converts timeseries data into scalogram data. Allows selection of
%   dead_zone parameter that will only be used for guessing and not
%   evaluation. Creates input regressors in thepe of rgb images and
%   corresponding output vectors.

clc; clear; close all;
%% Parameters
windowSize = 150;
intersection = 50;

cmap = jet(256);
u_shaped_combined = []; %The combined dataset across multiple files
%% Locate the main folder
dataFolder = "Training Data";
saveFolder = "Image Training Data";
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', dataFolder);
end

%% Slice the data into windows of pre determined size
for i = 1:numel(fileList)
    % Load the data
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    num_images = floor(size(u,1) / windowSize);
    newSize = num_images * windowSize;  %The truncated vector size
    % Reshape the data
    u_truncated = u(1:newSize);
    u_shaped = reshape(u_truncated,[],windowSize);
    u_shaped_combined = u_shaped_combined:u_shaped;
end

%% Generate image data
for i = 1:numel(u_shaped_combined,1) %Iterate through each data window
    elem = u_shaped_combined(i,:);  %Get the corresponding data window
    [s,f,t] = spectrogram(elem,"yaxis");    %Take stft of the input data window
    s_mag = 10*log10(abs(s));     %Convert complex numbers to their magnitude values

end
