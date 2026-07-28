%% Set up project paths so scripts in src/ can be called by name,
%% and so relative data folders ("Training Data", "Raw Data", etc.)
%% resolve correctly no matter where MATLAB's current folder is.
projectRoot = fileparts(fileparts(mfilename('fullpath')));   % scripts/ -> project root
addpath(genpath(fullfile(projectRoot, 'src')));
cd(projectRoot);

parse_data
feature_extraction
%% Long Part
multi_data_MLP
mlp_predict   % renamed from prediction.m during the repo reorg
