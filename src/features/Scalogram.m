%% Scalogram.m
%   Author: Arda Gencer
%   Date: 21.07.2026
%   Functionality: Converts timeseries data into scalogram data.
clc; clear; close all;
%% Parameters
window_size = 250;  %Window size of images, not scalogram
intersection_u = 0;
step_u = window_size - intersection_u;
intersection_y = 0;
step_y = window_size - intersection_y;
horizon = 1;

%Scalogram (CWT) parameters
Fs = 1/6.6667e-05;   %15000 Hz
cmap = parula(256);    %Colormap
u_shaped_combined = []; %The combined dataset across multiple files

%Parameters for saving the data
base_img_name = "image";
dataFolder = fullfile("data", "train");
saveFolder = fullfile("data", "images", "scalogram");
ImgFolder = fullfile(saveFolder, "previews");
results = struct('image', {}, 'target', {});
resultIdx = 1;
%% Locate read and write folders
fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', dataFolder);
end
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
if ~exist(ImgFolder, 'dir')
    mkdir(ImgFolder);
end

%% Pass 1 to compute mean/std for normalization
sum_u = 0; sumSq_u = 0; count_u = 0;
sum_y = 0; sumSq_y = 0; count_y = 0;

for i = 1:numel(fileList)   %Per file loop
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    n = length(u);
    lastValidStart = n - window_size - horizon + 1;
    startIdxs = 1:step_u:lastValidStart;
    for k = 1:length(startIdxs)
        [s_mag_u, s_mag_y, ~] = compute_window_scalograms(u, y, startIdxs(k), window_size, horizon, Fs);

        sum_u = sum_u + sum(s_mag_u(:));
        sumSq_u = sumSq_u + sum(s_mag_u(:).^2);
        count_u = count_u + numel(s_mag_u);

        sum_y = sum_y + sum(s_mag_y(:));
        sumSq_y = sumSq_y + sum(s_mag_y(:).^2);
        count_y = count_y + numel(s_mag_y);
    end
end

mu_u = sum_u / count_u;
sigma_u = sqrt(sumSq_u/count_u - mu_u^2);
mu_y = sum_y / count_y;
sigma_y = sqrt(sumSq_y/count_y - mu_y^2);

%% Pass 2 to normalize and save images
nStd = 3;   % clip at ±3 standard deviations

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    n = length(u);
    lastValidStart = n - window_size - horizon + 1;
    startIdxs = 1:step_u:lastValidStart;
    for k = 1:length(startIdxs) %Go through each window start
        [s_mag_u, s_mag_y, target] = compute_window_scalograms(u, y, startIdxs(k), window_size, horizon, Fs);

        z_u = (s_mag_u - mu_u) / sigma_u;
        z_y = (s_mag_y - mu_y) / sigma_y;

        z_u = max(-nStd, min(nStd, z_u));   % clip
        z_y = max(-nStd, min(nStd, z_y));

        u_norm = (z_u + nStd) / (2*nStd);   % remap to 0-1 range
        y_norm = (z_y + nStd) / (2*nStd);

        combined = [u_norm; y_norm];

        ind_img = gray2ind(combined, 256);
        rgb_img = ind2rgb(ind_img, cmap);

        img_path = sprintf("%s_file%03d_win%05d.png", base_img_name, i, k);
        full_path = fullfile(ImgFolder,img_path);

        results(resultIdx).image = rgb_img;
        results(resultIdx).target = target;
        results(resultIdx).horizon = horizon;
        resultIdx = resultIdx + 1;
        imwrite(rgb_img,full_path);
    end
end
save(fullfile(saveFolder, "dataset.mat"), "results", "-v7.3");

%Computes scalograms and also the target horizon
function [s_mag_u, s_mag_y, target] = compute_window_scalograms(u, y, startIdx, window_size, horizon, Fs)
    endIdx = startIdx + window_size - 1;
    u_window = u(startIdx:endIdx);
    y_window = y(startIdx:endIdx);
    target = y(endIdx + horizon);
    [cfs_u, ~] = cwt(u_window, Fs);
    [cfs_y, ~] = cwt(y_window, Fs);
    s_mag_u = 10*log10(abs(cfs_u) + eps);   % + eps avoids log(0) = -Inf, consistent for both
    s_mag_y = 10*log10(abs(cfs_y) + eps);
end
