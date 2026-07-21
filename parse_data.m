%% parse_data.m
%   Author: Arda Gencer 
%   Date Created: 13.07.2026
%   Functionality:Converts the raw data from the speedgoat simulink experiment into two
%   vectors of input and output timeseries. Can do the operation for multiple
%   datasets.
%   Notes: The initial speedgoat data should be stored in a file named "Raw
%   Data" (this name can be changed manually in the script). The resulting
%   data files will be saved under a folder named "Training Data". Each
%   data file contains 3 variables. The input and output vectors and the
%   timestep T.


clc; clear; close all;

%% Config
rawFolder = "Raw Data";
outFolder = "Training Data";
graphFolder = "Graphs";
A = 1;                  % fraction of data to keep (1 = all)
transientEnd = 2500;    % samples to cut from the start for transient settling

if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end
if ~exist(graphFolder, 'dir')
    mkdir(graphFolder);
end

fileList = dir(fullfile(rawFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s". Check the folder name/path.', rawFolder);
end

%% Process every file in Raw Data
for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    fprintf('Processing "%s"...\n', fileList(i).name);

    S = load(fpath);   % load into struct to avoid collisions across files

    if ~isfield(S, 'data')
        warning('Skipping "%s": no variable named "data" found.', fileList(i).name);
        continue;
    end
    data = S.data;

    try
        outputData = data{1};
        ts = outputData.Values;
        T = ts.Time(2) - ts.Time(1);
        y = outputData.Values.Data;

        inputData = data{2};
        u = squeeze(inputData.Values.Data);
    catch ME
        warning('Skipping "%s": failed to parse expected structure (%s).', fileList(i).name, ME.message);
        continue;
    end

    %% Keep only first A*100%
    N = length(y);
    endIndex = floor(N * A);
    y = y(1:endIndex);
    u = u(1:endIndex);

    %% Cut transient oscillation
    if transientEnd >= length(y)
        warning('"%s": transientEnd (%d) >= signal length (%d). Skipping transient cut for this file.', ...
            fileList(i).name, transientEnd, length(y));
    else
        y = y(transientEnd+1:end);
        u = u(transientEnd+1:end);
    end

    %% Save under Training Data, using the original filename as a base
    [~, baseName, ~] = fileparts(fileList(i).name);
    outPath = fullfile(outFolder, baseName + "_processed.mat");
    save(outPath, 'y', 'u', 'T');

    fprintf('  -> saved "%s" (%d samples)\n', outPath, length(y));

    %% Plot input/output signals for this dataset and save to Graphs folder
    timeVector = (0:length(y)-1)' * T;

    fig = figure('Visible', 'off');
    subplot(2,1,1);
    plot(timeVector, y);
    title(sprintf('%s — Output (y)', baseName), 'Interpreter', 'none');
    xlabel('Time'); ylabel('y'); grid on;

    subplot(2,1,2);
    plot(timeVector, u);
    title(sprintf('%s — Input (u)', baseName), 'Interpreter', 'none');
    xlabel('Time'); ylabel('u'); grid on;

    graphPath = fullfile(graphFolder, baseName + ".png");
    saveas(fig, graphPath);
    close(fig);

    fprintf('  -> saved graph "%s"\n', graphPath);
end

fprintf('Done. Processed files are in "%s".\n', outFolder);
