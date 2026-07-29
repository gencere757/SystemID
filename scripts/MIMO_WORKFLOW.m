%% MIMO_WORKFLOW.m
%   Entry point for the MIMO (CubeSpec FSM) pipeline -- mirrors
%   MAIN_WORKFLOW.m's structure, kept entirely separate from the SISO
%   scripts it calls (parse_data_mimo, feature_extraction_mimo,
%   laplace_network_train_mimo all live alongside their SISO
%   counterparts but never touch data/train, data/test, or
%   features_combined.mat -- they read/write the *_mimo files instead).

%% Set up project paths so scripts in src/ can be called by name,
%% and so relative data folders ("data/raw/mimo", etc.) resolve
%% correctly no matter where MATLAB's current folder is.
projectRoot = fileparts(fileparts(mfilename('fullpath')));   % scripts/ -> project root
addpath(genpath(fullfile(projectRoot, 'src')));
cd(projectRoot);

parse_data_mimo
feature_extraction_mimo
%%
laplace_network_train_mimo
