clc; clear; close all;
maxNumCompThreads(feature('numcores'));

%% Config
dataFolder = "Training Data";
max_lag_output = 100;
max_lag_input = 150;
differentiate = true;
confidence_threshold = 0.05;
minSeparation = 5;   % minimum samples between kept output lags

fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', dataFolder);
end

%% Accumulate per-dataset correlation curves
acf_all = [];      % will be (numFiles x max_lag_output)
cc_all = [];        % will be (numFiles x numPositiveLags)
lags_pacf_ref = []; % lag axis, assumed same across files
lags_cc_pos_ref = [];

validFiles = 0;

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    if ~isfield(S,'y') || ~isfield(S,'u')
        warning('Skipping "%s": missing y or u.', fileList(i).name);
        continue;
    end
    y = S.y(:);
    u = S.u(:);

    %% PACF of output (per dataset)
    [acf_vals, lags_pacf] = parcorr(y, max_lag_output);
    acf_vals(lags_pacf == 0) = [];
    lags_pacf(lags_pacf == 0) = [];

    if isempty(lags_pacf_ref)
        lags_pacf_ref = lags_pacf;
    elseif ~isequal(lags_pacf_ref, lags_pacf)
        warning('"%s" has a different lag axis for PACF than earlier files — skipping.', fileList(i).name);
        continue;
    end

    %% Cross-correlation input/output (per dataset)
    if differentiate
        u_diff = diff(u);
        y_diff = diff(y);
    else
        u_diff = u;
        y_diff = y;
    end
    [cc_vals, lags_cc] = crosscorr(u_diff, y_diff, max_lag_input);
    positive_lags_idx = lags_cc > 0;
    cc_vals_pos = cc_vals(positive_lags_idx);
    lags_cc_pos = lags_cc(positive_lags_idx);

    if isempty(lags_cc_pos_ref)
        lags_cc_pos_ref = lags_cc_pos;
    elseif ~isequal(lags_cc_pos_ref, lags_cc_pos)
        warning('"%s" has a different lag axis for cross-correlation than earlier files — skipping.', fileList(i).name);
        continue;
    end

    acf_all = [acf_all; acf_vals(:)']; %#ok<AGROW>
    cc_all = [cc_all; cc_vals_pos(:)']; %#ok<AGROW>
    validFiles = validFiles + 1;
    fprintf('Processed "%s" (%d/%d)\n', fileList(i).name, validFiles, numel(fileList));
end

if validFiles == 0
    error('No valid datasets were processed.');
end

%% Aggregate correlation across datasets
% Mean absolute correlation across datasets at each lag — a lag only
% counts as strong if it's consistently strong across recordings, not
% just a fluke of one dataset.
acf_mean = mean(abs(acf_all), 1);
cc_mean  = mean(abs(cc_all), 1);

%% Output lag selection (same logic as before, on the aggregated curve)
[~, sort_idx] = sort(acf_mean, 'descend');
top_output_lags = lags_pacf_ref(sort_idx(1:min(50, length(sort_idx))));

top_output_lags_sorted = sort(top_output_lags);
keep = true(size(top_output_lags_sorted));
for i = 2:length(top_output_lags_sorted)
    if top_output_lags_sorted(i) - top_output_lags_sorted(find(keep(1:i-1),1,'last')) < minSeparation
        keep(i) = false;
    end
end
top_output_lags = top_output_lags_sorted(keep);

%% Input lag / dead-time selection (on aggregated cross-correlation)
significant_input_lags = lags_cc_pos_ref(cc_mean > confidence_threshold);
if ~isempty(significant_input_lags)
    dead_time = significant_input_lags(1);
else
    dead_time = 0;
end

%% Diagnostic plots (aggregated, across all datasets)
figure;
stem(lags_pacf_ref, acf_mean);
title('Mean |PACF| of Output Across All Datasets');
xlabel('Lags'); ylabel('Mean |Sample Autocorrelation|');

figure;
stem(lags_cc_pos_ref, cc_mean);
hold on;
yline(confidence_threshold, 'r--');
title('Mean |Cross-Correlation| Across All Datasets');
xlabel('Lags'); ylabel('Mean |Cross-Correlation|');
xlim([0, max_lag_input]);

save('features_combined.mat', 'top_output_lags', 'significant_input_lags', 'dead_time');
fprintf('Saved features_combined.mat using %d datasets.\n', validFiles);
