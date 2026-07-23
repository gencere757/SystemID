%% Spectrogram.m

%   Author: Arda Gencer
%   Date: 21.07.2026
%   Functionality: Converts timeseries data into scalogram data. Allows selection of
%   dead_zone parameter that will only be used for guessing and not
%   evaluation. Creates input regressors in thepe of rgb images and
%   corresponding output vectors.

clc; clear; close all;
%% Parameters
window_size_u = 100;
intersection_u = 0;
step_u = window_size_u - intersection_u;
window_size_y = 100;
intersection_y = 0;
step_y = window_size_y - intersection_y;

%Spectrogram parameters
Fs = 1/6.6667e-05;   %15000 Hz

cmap = jet(256);    %Colormap
u_shaped_combined = []; %The combined dataset across multiple filesö
%Parameters for saving the data
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
    [cfs_u,f_u,t_u] = cwt(u,Fs);    %Take stft of the input data
    [cfs_y,f_y,t_y] = cwt(y,Fs);    %Take stft of the output data
    s_mag_u = 10*log10(abs(cfs_u));     %Convert complex numbers to their magnitude values
    s_mag_y = 10*log10(abs(cfs_y));

    %Compute min/ max for normalization
    u_global_min = min(s_mag_u(:));
    u_global_max = max(s_mag_u(:));
    y_global_min = min(s_mag_y(:));
    y_global_max = max(s_mag_y(:));

    % Buffer the data
    [numRows_u, numCols_u] = size(s_mag_u);   %Get no of ros and columns
    [numRows_y, numCols_y] = size(s_mag_y);

    numWindows_u = floor((numCols_u - window_size_u)/step_u);    %no of samples to buffer the data into
    numWindows_y = floor((numCols_y-window_size_y)/step_y);
    u_s_mag_buffered = zeros(numRows_u,window_size_u,numWindows_u);     %Initialize buffered matrix
    y_s_mag_buffered = zeros(numRows_y,window_size_y,numWindows_y);
    start_idxs_u = 1:step_u:numCols_u-window_size_u+1;
    start_idxs_y = 1:step_y:numCols_y-window_size_y;
    
    
    %Splitting input data
    for w = 1:numWindows_u %Iterate through each start window idx
        startIdx = start_idxs_u(w);
        window_u = s_mag_u(:,startIdx:startIdx+window_size_u-1);   %Take the window 
        u_s_mag_buffered(:,:,w) = window_u;   %Append it as 3rd dimension 
    end

    %Splitting output data
    for w = 1:numWindows_y %Iterate through each start window idx
        startIdx = start_idxs_y(w);
        window_y = s_mag_y(:,startIdx:startIdx+window_size_y-1);   %Take the window 
        y_s_mag_buffered(:,:,w) = window_y; 
    end
    
%Combining the input and output windows
numWindows = min(numWindows_u, numWindows_y);  % in case the u and y window no differ slightly
combined_buffered = zeros(numRows_u + numRows_y, window_size_u, numWindows);    %Pre allocate

for j = 1:numWindows    %Concat w,th ,nput on top output at bottom
    combined_buffered(:,:,j) = [u_s_mag_buffered(:,:,j); y_s_mag_buffered(:,:,j)];  %Concat. vertically on top of each other
end
    %Converting to image
    for img_idx = 1:size(combined_buffered,3)      %Iterate through each window in the 3D matrix
        %Convert to image and save
        matrix = combined_buffered(:,:,img_idx);
        gray_img = (matrix - u_global_min) / (u_global_max - u_global_min);
        ind_img = gray2ind(gray_img,256);
        rgb_img = ind2rgb(ind_img,cmap);
        img_path = sprintf("%s_file%03d_win%05d.png", base_img_name, i, img_idx);        full_path = fullfile(saveFolder,img_path);
        imwrite(rgb_img,full_path);
    end
end
