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
step = windowSize - intersection;

cmap = jet(256);
u_shaped_combined = []; %The combined dataset across multiple files
base_img_name = "image";
dataFolder = "Training Data";
saveFolder = "Image Training Data";
%% Locate read and write folders
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', dataFolder);
end

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

%% Slice the data into windows of pre determined size
for i = 1:numel(fileList)
    % Load the data
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    [s,f,t] = spectrogram(u,"yaxis");    %Take stft of the input data
    s_mag = 10*log10(abs(s));     %Convert complex numbers to their magnitude values

    % Buffer the data
    s_mag_buffered = [];
    [numRows, numCols] = size(s_mag);
    numWindows = floor((numCols - windowSize)/step);
    startIdxs = 1:step:numCols-windowSize+1;
    
    for startIdx = startIdxs %Iterate through each start point
        window = s_mag(:,startIdx:startIdx+step);
        s_mag_buffered = [s_mag_buffered 
    end
end

%% Generate image data
% for i = 1:size(u_shaped_combined,1) %Iterate through each data window
%     elem = u_shaped_combined(i,:);  %Get the corresponding data window
%     [s,f,t] = spectrogram(elem,"yaxis");    %Take stft of the input data window
%     s_mag = 10*log10(abs(s));     %Convert complex numbers to their magnitude values
%     gray_img = mat2gray(s_mag);
%     ind_img = gray2ind(gray_img,256);
%     rgb_img = ind2rgb(ind_img,cmap);
%     img_path = sprintf("%s_%05d.png",base_img_name,i);
%     full_path = fullfile(saveFolder,img_path);
%     imwrite(rgb_img,full_path);
% end-
