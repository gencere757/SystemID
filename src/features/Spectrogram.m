%% Spectrogram.m

%   Author: Arda Gencer
%   Date: 21.07.2026
%   Functionality: Converts timeseries data into spectrogram data. 

%Yapılacaklar: Spectrogram oluşturma süresini ölç

clc; clear; close all;
%% Parameters
window_size = 200;  %Window size of images, not spectrogram
intersection_u = 0;
step_u = window_size - intersection_u;
intersection_y = 0;
step_y = window_size - intersection_y;
horizon = 1;

%Spectrogram parameters
Fs = 1/6.6667e-05;   %15000 Hz
winLength = 128;              % 8.5 ms per window 
noverlap = round(0.9*winLength);  % Hop length of spectrogram
nfft = 128;

cmap = parula(256);    %Colormap
u_shaped_combined = []; %The combined dataset across multiple files
%Parameters for saving the data
base_img_name = "image";
dataFolder = fullfile("data", "train");
saveFolder = fullfile("data", "images", "spectrogram");
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


%% Pass 1 to get percentile-based normalization bounds

sum_u = 0; sumSq_u = 0; count_u = 0;
sum_y = 0; sumSq_y = 0; count_y = 0;

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);
    n = length(u);
    lastValidStart = n - window_size - horizon + 1;
    startIdxs = 1:step_u:lastValidStart;
    for k = 1:length(startIdxs)
        [s_mag_u, s_mag_y, ~] = compute_window_spectrograms(u, y, startIdxs(k), window_size, horizon, winLength, noverlap, nfft, Fs);
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

% Save normalization stats alongside the dataset — cnn_prediction.m needs
% these to normalize new windows the same way at inference time. This was
% previously never written even though cnn_prediction.m already expected
% it at "Image Training Data Spectrogram/normalization_stats.mat".
save(fullfile(saveFolder, "normalization_stats.mat"), "mu_u", "sigma_u", "mu_y", "sigma_y");

%% Pass 2 to normalize and save images

nStd = 3;   % clip at ±3 standard deviations

for i = 1:numel(fileList)
    % Load the data
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    y = S.y(:);
    u = S.u(:);

    n = length(u);
    lastValidStart = n - window_size - horizon + 1;
    startIdxs = 1:step_u:lastValidStart;

    for k = 1:length(startIdxs) %Go through each window start
        [s_mag_u, s_mag_y, target] = compute_window_spectrograms(u, y, startIdxs(k), window_size, horizon, winLength, noverlap, nfft, Fs);

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

%% Archive this run (dataset.mat + normalization stats + summary) so it isn't lost or overwritten next run
% No RMSE/MAE/Fit here — this script builds a training set, it doesn't
% fit a model — so those are NaN.
runFolder = log_run("Spectrogram", ...
    sprintf("%d files, %d windows, mu_u=%.3g sigma_u=%.3g mu_y=%.3g sigma_y=%.3g", ...
        numel(fileList), numel(results), mu_u, sigma_u, mu_y, sigma_y), ...
    NaN, NaN, NaN, [], fullfile(saveFolder, "dataset.mat"));
copyfile(fullfile(saveFolder, "normalization_stats.mat"), fullfile(runFolder, "normalization_stats.mat"));

%Computes spectrograms and also the targert horizon
function [s_mag_u, s_mag_y, target] = compute_window_spectrograms(u, y, startIdx, window_size, horizon, winLength, noverlap, nfft, Fs)
    endIdx = startIdx + window_size - 1;
    u_window = u(startIdx:endIdx);
    y_window = y(startIdx:endIdx);
    target = y(endIdx + horizon);

    [s_u,~,~] = spectrogram(u_window, hamming(winLength), noverlap, nfft, Fs, "yaxis");
    [s_y,~,~] = spectrogram(y_window, hamming(winLength), noverlap, nfft, Fs, "yaxis");

    s_mag_u = 10*log10(abs(s_u)+eps);
    s_mag_y = 10*log10(abs(s_y)+eps);
end
