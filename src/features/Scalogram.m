%% Scalogram.m
%   Author: Arda Gencer
%   Date: 21.07.2026
%   Functionality: Converts timeseries data into scalogram data.
clc; clear; close all;
%% Parameters
segment_width = 100;  %Width of image segments, in samples/CWT columns
horizon = 1;

%Scalogram (CWT) parameters
Fs = 1/6.6667e-05;   %15000 Hz
cmap = parula(256);    %Colormap

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

%% Pass 0 to compute full-signal CWT per file, cached for passes 1 and 2
%cwt doesn't decimate the time axis, so watch memory on long recordings
fileData = struct('s_mag_u', {}, 's_mag_y', {}, 'y', {});
for i = 1:numel(fileList)   %Per file loop
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);

    [cfs_u, ~] = cwt(u, Fs);
    [cfs_y, ~] = cwt(y, Fs);

    fileData(i).s_mag_u = 10*log10(abs(cfs_u) + eps);
    fileData(i).s_mag_y = 10*log10(abs(cfs_y) + eps);
    fileData(i).y = y;
end

%% Pass 1 to compute mean/std for normalization
sum_u = 0; sumSq_u = 0; count_u = 0;
sum_y = 0; sumSq_y = 0; count_y = 0;

for i = 1:numel(fileList)   %Per file loop
    startCols = valid_segment_starts(fileData(i), segment_width, horizon);
    for k = 1:numel(startCols)
        cEnd = startCols(k) + segment_width - 1;
        seg_u = fileData(i).s_mag_u(:, startCols(k):cEnd);
        seg_y = fileData(i).s_mag_y(:, startCols(k):cEnd);

        sum_u = sum_u + sum(seg_u(:));
        sumSq_u = sumSq_u + sum(seg_u(:).^2);
        count_u = count_u + numel(seg_u);

        sum_y = sum_y + sum(seg_y(:));
        sumSq_y = sumSq_y + sum(seg_y(:).^2);
        count_y = count_y + numel(seg_y);
    end
end

mu_u = sum_u / count_u;
sigma_u = sqrt(sumSq_u/count_u - mu_u^2);
mu_y = sum_y / count_y;
sigma_y = sqrt(sumSq_y/count_y - mu_y^2);

save(fullfile(saveFolder, "normalization_stats.mat"), "mu_u", "sigma_u", "mu_y", "sigma_y");

%% Pass 2 to normalize and save images
nStd = 3;   % clip at ±3 standard deviations

for i = 1:numel(fileList)
    startCols = valid_segment_starts(fileData(i), segment_width, horizon);
    y = fileData(i).y;

    for k = 1:numel(startCols) %Go through each segment start
        cStart = startCols(k);
        cEnd = cStart + segment_width - 1;
        seg_u = fileData(i).s_mag_u(:, cStart:cEnd);
        seg_y = fileData(i).s_mag_y(:, cStart:cEnd);

        target = y(cEnd + horizon);   %column index == sample index for cwt

        z_u = (seg_u - mu_u) / sigma_u;
        z_y = (seg_y - mu_y) / sigma_y;

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

%% Archive this run (dataset.mat + window/normalization summary) so it isn't lost or overwritten next run
% No RMSE/MAE/Fit here — this script builds a training set, it doesn't
% fit a model — so those are NaN.
log_run("Scalogram", ...
    sprintf("%d files, %d segments, mu_u=%.3g sigma_u=%.3g mu_y=%.3g sigma_y=%.3g", ...
        numel(fileList), numel(results), mu_u, sigma_u, mu_y, sigma_y), ...
    NaN, NaN, NaN, [], fullfile(saveFolder, "dataset.mat"));

%Returns valid segment start columns for one file
function startCols = valid_segment_starts(fd, segment_width, horizon)
    numCols = size(fd.s_mag_u, 2);
    lastValidStart = min(numCols, numel(fd.y) - horizon) - segment_width + 1;
    if lastValidStart < 1
        startCols = [];
        return;
    end
    startCols = 1:segment_width:lastValidStart;
end
