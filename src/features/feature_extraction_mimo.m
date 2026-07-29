%% feature_extraction_mimo.m
%   Author: Arda Gencer
%   Functionality: MIMO version of feature_extraction.m. Same method
%   (PACF for autoregressive lags, cross-correlation for input lags/dead
%   time), extended from the single SISO pair to all 3 output channels
%   and all 3x3 (input channel, output channel) pairs:
%     - top_output_lags{j}            : output j's own PACF-selected lags
%     - significant_input_lags{i,j}   : input i's cross-corr-selected
%                                        lags w.r.t. output j
%     - dead_time(i,j)                : first significant lag for that pair
%   ...and the same three again for the derivative (rate) signals
%   dy/dt, du/dt (the "_dot" variants).
%
%   Everything is aggregated (mean |.|) across every block-file in
%   data/train/mimo, exactly like the SISO version aggregates across
%   every file in data/train -- a lag only counts as strong if it's
%   consistently strong across blocks/excitation levels, not a fluke of
%   one recording.
%
%   Notes: each file in data/train/mimo must contain u, y as [N x 3]
%   matrices (rows = time, columns = channel) -- see parse_data_mimo.m.

clc; clear; close all;
maxNumCompThreads(feature('numcores'));

%% Config
dataFolder = fullfile("data", "train", "mimo");
featuresFolder = fullfile("data", "features");
if ~exist(featuresFolder, 'dir')
    mkdir(featuresFolder);
end
numChannels = 3;
max_lag_output = 100;
max_lag_input = 150;
differentiate = true;
confidence_threshold = 0.05;
minSeparation = 5;   % minimum samples between kept output lags

fileList = dir(fullfile(dataFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s". Run parse_data_mimo.m first.', dataFolder);
end

%% Accumulators
% Output PACF: one series per output channel j
acf_all = cell(numChannels, 1);
acf_dot_all = cell(numChannels, 1);
lags_pacf_ref = cell(numChannels, 1);
lags_pacf_dot_ref = cell(numChannels, 1);
for j = 1:numChannels
    acf_all{j} = [];
    acf_dot_all{j} = [];
end

% Cross-correlation: one series per (input i, output j) pair
cc_all = cell(numChannels, numChannels);
cc_dot_all = cell(numChannels, numChannels);
lags_cc_pos_ref = cell(numChannels, numChannels);
lags_cc_dot_pos_ref = cell(numChannels, numChannels);
for i = 1:numChannels
    for j = 1:numChannels
        cc_all{i,j} = [];
        cc_dot_all{i,j} = [];
    end
end

% Output-output cross-correlation: one series per (output i, output j)
% pair, i~=j -- does output i's past help predict output j? This is the
% coupling term the pure per-output PACF above can't see (PACF of y_j
% only looks at y_j's own past, never at y_i).
cc_yy_all = cell(numChannels, numChannels);
cc_yy_dot_all = cell(numChannels, numChannels);
lags_cc_yy_pos_ref = cell(numChannels, numChannels);
lags_cc_yy_dot_pos_ref = cell(numChannels, numChannels);
for i = 1:numChannels
    for j = 1:numChannels
        cc_yy_all{i,j} = [];
        cc_yy_dot_all{i,j} = [];
    end
end

validFiles = 0;

for fi = 1:numel(fileList)
    fpath = fullfile(fileList(fi).folder, fileList(fi).name);
    S = load(fpath);
    if ~isfield(S,'y') || ~isfield(S,'u')
        warning('Skipping "%s": missing y or u.', fileList(fi).name);
        continue;
    end
    y = S.y;   % N x 3
    u = S.u;   % N x 3
    if size(y,2) ~= numChannels || size(u,2) ~= numChannels
        warning('Skipping "%s": expected %d channels, got y=%d u=%d.', ...
            fileList(fi).name, numChannels, size(y,2), size(u,2));
        continue;
    end

    if isfield(S,'T')
        T_i = S.T;
    else
        T_i = 1;
        warning('"%s" has no T field; assuming unit sample time for derivative signals.', fileList(fi).name);
    end

    dy = columnGradient(y, T_i);   % N x 3, dy/dt per channel
    du = columnGradient(u, T_i);   % N x 3, du/dt per channel

    fileOk = true;

    %% Per-output PACF (level and rate)
    acf_vals_this = cell(numChannels,1);
    acf_dot_vals_this = cell(numChannels,1);
    for j = 1:numChannels
        [acf_vals, lags_pacf] = parcorr(y(:,j), max_lag_output);
        acf_vals(lags_pacf == 0) = [];
        lags_pacf(lags_pacf == 0) = [];

        if isempty(lags_pacf_ref{j})
            lags_pacf_ref{j} = lags_pacf;
        elseif ~isequal(lags_pacf_ref{j}, lags_pacf)
            warning('"%s" ch%d has a different lag axis for PACF than earlier files -- skipping file.', fileList(fi).name, j);
            fileOk = false;
            break;
        end
        acf_vals_this{j} = acf_vals;

        [acf_dot_vals, lags_pacf_dot] = parcorr(dy(:,j), max_lag_output);
        acf_dot_vals(lags_pacf_dot == 0) = [];
        lags_pacf_dot(lags_pacf_dot == 0) = [];

        if isempty(lags_pacf_dot_ref{j})
            lags_pacf_dot_ref{j} = lags_pacf_dot;
        elseif ~isequal(lags_pacf_dot_ref{j}, lags_pacf_dot)
            warning('"%s" ch%d has a different lag axis for dy PACF than earlier files -- skipping file.', fileList(fi).name, j);
            fileOk = false;
            break;
        end
        acf_dot_vals_this{j} = acf_dot_vals;
    end
    if ~fileOk
        continue;
    end

    %% Per (input i, output j) cross-correlation (level and rate)
    for i = 1:numChannels
        if differentiate
            u_diff = diff(u(:,i));
        else
            u_diff = u(:,i);
        end
        for j = 1:numChannels
            if differentiate
                y_diff = diff(y(:,j));
            else
                y_diff = y(:,j);
            end
            [cc_vals, lags_cc] = crosscorr(u_diff, y_diff, max_lag_input);
            pos = lags_cc > 0;
            cc_vals_pos = cc_vals(pos);
            lags_cc_pos = lags_cc(pos);

            if isempty(lags_cc_pos_ref{i,j})
                lags_cc_pos_ref{i,j} = lags_cc_pos;
            elseif ~isequal(lags_cc_pos_ref{i,j}, lags_cc_pos)
                warning('"%s" pair(u%d,y%d) has a different lag axis than earlier files -- skipping file.', fileList(fi).name, i, j);
                fileOk = false;
                break;
            end

            [cc_dot_vals, lags_cc_dot] = crosscorr(du(:,i), dy(:,j), max_lag_input);
            pos_dot = lags_cc_dot > 0;
            cc_dot_vals_pos = cc_dot_vals(pos_dot);
            lags_cc_dot_pos = lags_cc_dot(pos_dot);

            if isempty(lags_cc_dot_pos_ref{i,j})
                lags_cc_dot_pos_ref{i,j} = lags_cc_dot_pos;
            elseif ~isequal(lags_cc_dot_pos_ref{i,j}, lags_cc_dot_pos)
                warning('"%s" pair(u%d,y%d) has a different lag axis for du/dy than earlier files -- skipping file.', fileList(fi).name, i, j);
                fileOk = false;
                break;
            end

            cc_all{i,j} = [cc_all{i,j}; cc_vals_pos(:)']; %#ok<AGROW>
            cc_dot_all{i,j} = [cc_dot_all{i,j}; cc_dot_vals_pos(:)']; %#ok<AGROW>
        end
        if ~fileOk
            break;
        end
    end
    if ~fileOk
        continue;
    end

    %% Output-output cross-correlation (level and rate), i ~= j only --
    % i == j is already covered by the PACF step above; this captures
    % coupling BETWEEN different outputs (e.g. does y1's past help
    % predict y2?), which PACF-of-y_j-alone can never see.
    for i = 1:numChannels
        for j = 1:numChannels
            if i == j
                continue;
            end
            if differentiate
                yi_diff = diff(y(:,i));
                yj_diff = diff(y(:,j));
            else
                yi_diff = y(:,i);
                yj_diff = y(:,j);
            end
            [cc_vals, lags_cc] = crosscorr(yi_diff, yj_diff, max_lag_input);
            pos = lags_cc > 0;
            cc_vals_pos = cc_vals(pos);
            lags_cc_pos = lags_cc(pos);

            if isempty(lags_cc_yy_pos_ref{i,j})
                lags_cc_yy_pos_ref{i,j} = lags_cc_pos;
            elseif ~isequal(lags_cc_yy_pos_ref{i,j}, lags_cc_pos)
                warning('"%s" pair(y%d,y%d) has a different lag axis than earlier files -- skipping file.', fileList(fi).name, i, j);
                fileOk = false;
                break;
            end

            [cc_dot_vals, lags_cc_dot] = crosscorr(dy(:,i), dy(:,j), max_lag_input);
            pos_dot = lags_cc_dot > 0;
            cc_dot_vals_pos = cc_dot_vals(pos_dot);
            lags_cc_dot_pos = lags_cc_dot(pos_dot);

            if isempty(lags_cc_yy_dot_pos_ref{i,j})
                lags_cc_yy_dot_pos_ref{i,j} = lags_cc_dot_pos;
            elseif ~isequal(lags_cc_yy_dot_pos_ref{i,j}, lags_cc_dot_pos)
                warning('"%s" pair(y%d,y%d) has a different lag axis for dy/dy than earlier files -- skipping file.', fileList(fi).name, i, j);
                fileOk = false;
                break;
            end

            cc_yy_all{i,j} = [cc_yy_all{i,j}; cc_vals_pos(:)']; %#ok<AGROW>
            cc_yy_dot_all{i,j} = [cc_yy_dot_all{i,j}; cc_dot_vals_pos(:)']; %#ok<AGROW>
        end
        if ~fileOk
            break;
        end
    end
    if ~fileOk
        continue;
    end

    for j = 1:numChannels
        acf_all{j} = [acf_all{j}; acf_vals_this{j}(:)']; %#ok<AGROW>
        acf_dot_all{j} = [acf_dot_all{j}; acf_dot_vals_this{j}(:)']; %#ok<AGROW>
    end

    validFiles = validFiles + 1;
    fprintf('Processed "%s" (%d/%d)\n', fileList(fi).name, validFiles, numel(fileList));
end

if validFiles == 0
    error('No valid MIMO datasets were processed.');
end

%% Aggregate + select lags per output channel, per input/output pair
top_output_lags = cell(numChannels, 1);
top_output_dot_lags = cell(numChannels, 1);
for j = 1:numChannels
    acf_mean = mean(abs(acf_all{j}), 1);
    top_output_lags{j} = selectOutputLags(acf_mean, lags_pacf_ref{j}, minSeparation);

    acf_dot_mean = mean(abs(acf_dot_all{j}), 1);
    top_output_dot_lags{j} = selectOutputLags(acf_dot_mean, lags_pacf_dot_ref{j}, minSeparation);
end

significant_input_lags = cell(numChannels, numChannels);
significant_input_dot_lags = cell(numChannels, numChannels);
dead_time = zeros(numChannels, numChannels);
dead_time_dot = zeros(numChannels, numChannels);
for i = 1:numChannels
    for j = 1:numChannels
        cc_mean = mean(abs(cc_all{i,j}), 1);
        sig = lags_cc_pos_ref{i,j}(cc_mean > confidence_threshold);
        significant_input_lags{i,j} = sig;
        if ~isempty(sig)
            dead_time(i,j) = sig(1);
        end

        cc_dot_mean = mean(abs(cc_dot_all{i,j}), 1);
        sig_dot = lags_cc_dot_pos_ref{i,j}(cc_dot_mean > confidence_threshold);
        significant_input_dot_lags{i,j} = sig_dot;
        if ~isempty(sig_dot)
            dead_time_dot(i,j) = sig_dot(1);
        end
    end
end

% Output-output coupling: does output i's past help predict output j?
% (i ~= j only -- diagonal already covered by top_output_lags via PACF)
significant_output_lags = cell(numChannels, numChannels);
significant_output_dot_lags = cell(numChannels, numChannels);
dead_time_yy = zeros(numChannels, numChannels);
dead_time_yy_dot = zeros(numChannels, numChannels);
for i = 1:numChannels
    for j = 1:numChannels
        if i == j
            continue;
        end
        cc_mean = mean(abs(cc_yy_all{i,j}), 1);
        sig = lags_cc_yy_pos_ref{i,j}(cc_mean > confidence_threshold);
        significant_output_lags{i,j} = sig;
        if ~isempty(sig)
            dead_time_yy(i,j) = sig(1);
        end

        cc_dot_mean = mean(abs(cc_yy_dot_all{i,j}), 1);
        sig_dot = lags_cc_yy_dot_pos_ref{i,j}(cc_dot_mean > confidence_threshold);
        significant_output_dot_lags{i,j} = sig_dot;
        if ~isempty(sig_dot)
            dead_time_yy_dot(i,j) = sig_dot(1);
        end
    end
end

%% Diagnostic plots -- consolidated into 5 figures (one output/pair per subplot)
% rather than one figure per output/pair (which would be 33 figures for
% 3x3 + 3x2 pairs x level+rate), so the archive stays browsable.
pacfFig = figure('Name', 'MIMO_PACF_Outputs');
for j = 1:numChannels
    subplot(1, numChannels, j);
    stem(lags_pacf_ref{j}, mean(abs(acf_all{j}), 1));
    title(sprintf('Output y%d PACF', j));
    xlabel('Lag'); ylabel('Mean |PACF|');
end

ccFig = figure('Name', 'MIMO_CrossCorrelation');
for i = 1:numChannels
    for j = 1:numChannels
        subplot(numChannels, numChannels, (i-1)*numChannels + j);
        stem(lags_cc_pos_ref{i,j}, mean(abs(cc_all{i,j}), 1));
        hold on;
        yline(confidence_threshold, 'r--');
        title(sprintf('u%d -> y%d', i, j));
        xlim([0, max_lag_input]);
    end
end

pacfDotFig = figure('Name', 'MIMO_PACF_Output_Rates');
for j = 1:numChannels
    subplot(1, numChannels, j);
    stem(lags_pacf_dot_ref{j}, mean(abs(acf_dot_all{j}), 1));
    title(sprintf('dy%d/dt PACF', j));
    xlabel('Lag'); ylabel('Mean |PACF|');
end

ccDotFig = figure('Name', 'MIMO_CrossCorrelation_Rates');
for i = 1:numChannels
    for j = 1:numChannels
        subplot(numChannels, numChannels, (i-1)*numChannels + j);
        stem(lags_cc_dot_pos_ref{i,j}, mean(abs(cc_dot_all{i,j}), 1));
        hold on;
        yline(confidence_threshold, 'r--');
        title(sprintf('du%d/dt -> dy%d/dt', i, j));
        xlim([0, max_lag_input]);
    end
end

ccYYFig = figure('Name', 'MIMO_Output_Output_Coupling');
for i = 1:numChannels
    for j = 1:numChannels
        subplot(numChannels, numChannels, (i-1)*numChannels + j);
        if i == j
            axis off;   % diagonal: not computed here, see PACF plot instead
            title(sprintf('y%d -> y%d (see PACF)', i, j));
            continue;
        end
        stem(lags_cc_yy_pos_ref{i,j}, mean(abs(cc_yy_all{i,j}), 1));
        hold on;
        yline(confidence_threshold, 'r--');
        title(sprintf('y%d -> y%d', i, j));
        xlim([0, max_lag_input]);
    end
end

%% Save
save(fullfile(featuresFolder, 'features_combined_mimo.mat'), 'top_output_lags', 'significant_input_lags', 'dead_time', ...
     'top_output_dot_lags', 'significant_input_dot_lags', 'dead_time_dot', ...
     'significant_output_lags', 'significant_output_dot_lags', 'dead_time_yy', 'dead_time_yy_dot', 'numChannels');
fprintf('Saved %s using %d MIMO blocks.\n', fullfile(featuresFolder, 'features_combined_mimo.mat'), validFiles);

%% Archive this run (selected lags + diagnostic plots) so it isn't lost or overwritten next run
% No RMSE/MAE/Fit here -- this script selects regressors, it doesn't fit
% a model -- so those are NaN. The useful summary is the dead-time range
% across the 9 input->output pairs and the 6 output->output coupling pairs.
log_run("feature_extraction_mimo", ...
    sprintf("%d blocks, u->y dead_time [%d,%d], y->y coupling dead_time [%d,%d], mean %d output lags/channel", ...
        validFiles, min(dead_time(:)), max(dead_time(:)), ...
        min(dead_time_yy(~eye(numChannels))), max(dead_time_yy(~eye(numChannels))), ...
        round(mean(cellfun(@numel, top_output_lags)))), ...
    NaN, NaN, NaN, [pacfFig, ccFig, pacfDotFig, ccDotFig, ccYYFig], fullfile(featuresFolder, 'features_combined_mimo.mat'));

%% --- Local functions ---
function dX = columnGradient(X, T)
% Applies gradient() independently to each column of X (time down rows,
% channel across columns). Needed because gradient(X, T) on a matrix
% computes a 2-D spatial gradient by default, not a per-column time
% derivative -- looping per column avoids that trap.
    dX = zeros(size(X));
    for c = 1:size(X, 2)
        dX(:,c) = gradient(X(:,c), T);
    end
end

function selected = selectOutputLags(acf_mean, lags_ref, minSeparation)
% Same top-N + minimum-separation logic as the SISO feature_extraction.m.
    [~, sort_idx] = sort(acf_mean, 'descend');
    top = lags_ref(sort_idx(1:min(50, length(sort_idx))));

    top_sorted = sort(top);
    keep = true(size(top_sorted));
    for k = 2:length(top_sorted)
        if top_sorted(k) - top_sorted(find(keep(1:k-1),1,'last')) < minSeparation
            keep(k) = false;
        end
    end
    selected = top_sorted(keep);
end
