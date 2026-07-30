clc; clear; close all;

%% Set up project paths so scripts in src/ can be called by name,
%% and so relative data folders resolve correctly regardless of
%% MATLAB's current folder.
projectRoot = fileparts(fileparts(mfilename('fullpath')));   % scripts/ -> project root
addpath(genpath(fullfile(projectRoot, 'src')));
cd(projectRoot);

%% Clear out any leftover processed test files before regenerating them
testFolder = fullfile("data", "test");
oldFiles = dir(fullfile(testFolder, "*.mat"));
for i = 1:numel(oldFiles)
    delete(fullfile(oldFiles(i).folder, oldFiles(i).name));
end
if ~isempty(oldFiles)
    fprintf('Cleared %d old .mat file(s) from "%s".\n', numel(oldFiles), testFolder);
end

parse_test_data
SimulinkDataPrepLno
run_lno_simulink_and_compare
