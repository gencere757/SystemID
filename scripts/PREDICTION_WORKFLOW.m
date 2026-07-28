clc; clear; close all;

%% Set up project paths so scripts in src/ can be called by name,
%% and so relative data folders resolve correctly regardless of
%% MATLAB's current folder.
projectRoot = fileparts(fileparts(mfilename('fullpath')));   % scripts/ -> project root
addpath(genpath(fullfile(projectRoot, 'src')));
cd(projectRoot);

parse_test_data
SimulinkDataPrep
simulink_predict
