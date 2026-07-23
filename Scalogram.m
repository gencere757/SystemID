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
intersection = 10;
step = windowSize - intersection;
%Spectrogram parameters
Fs = 1/6.6667e-05;   %15000 Hz

winLength = 128;              % 8.5 ms per window (128/15000)
noverlap = round(0.75*winLength);  % 75% overlap -> hop 32 samples (2.1 ms)
nfft = 128;

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
for i = 1:numel(fileList)   %Per file loop
    % Load the data
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    [s,f,t] = spectrogram(u,hamming(winLength),noverlap,nfft,Fs,"yaxis");    %Take stft of the input data
    s_mag = 10*log10(abs(s));     %Convert complex numbers to their magnitude values

   %Compute min/ max for normalization
   globalMin = min(s_mag(:));
   globalMax = max(s_mag(:));

    % Buffer the data
    [numRows, numCols] = size(s_mag);   %Get no of ros and columns
    numWindows = floor((numCols - windowSize)/step);    %no of samples to buffer the data into
    s_mag_buffered = zeros(numRows,windowSize,numWindows);     %Initialize buffered matrix
    startIdxs = 1:step:numCols-windowSize+1;
    
    for w = 1:numWindows %Iterate through each start window idx
        startIdx = startIdxs(w);
        window = s_mag(:,startIdx:startIdx+windowSize-1);   %Take the window 
        s_mag_buffered(:,:,w) = window; 
    end

    for img_idx = 1:size(s_mag_buffered,3)      %Iterate through each window in the 3D matrix
        %Convert to image and save
        matrix = s_mag_buffered(:,:,img_idx);
        gray_img = mat2gray(matrix);
        ind_img = gray2ind(gray_img,256);
        rgb_img = ind2rgb(ind_img,cmap);
        img_path = sprintf("%s_file%03d_win%05d.png", base_img_name, i, img_idx);        full_path = fullfile(saveFolder,img_path);
        imwrite(rgb_img,full_path);
    end
end
