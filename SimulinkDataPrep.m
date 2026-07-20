clc; clear; close all;
load("Controllers\controller_base.mat");
load("Controllers\notch_base.mat");

%% Config
cleanedDataFile = "Test Processed Data\ELEVATION_SQUARE_WAVE_25HZ_10URAD_processed.mat";   % dataset for normalization stats AND for u timeseries
modelFile = "Multi_Data_MLP_model.mat";
outputFile = "SimulinkParams.mat";

%% Load model file (expects: net, maxLag, top_output_lags, significant_input_lags,
%% and ideally muX, sigmaX, muY, sigmaY already saved from training)
M = load(modelFile);

requiredForRegressor = {'maxLag','top_output_lags','significant_input_lags'};
for i = 1:numel(requiredForRegressor)
    if ~isfield(M, requiredForRegressor{i})
        error('"%s" is missing "%s". Re-save your training script''s model file with this variable included.', ...
            modelFile, requiredForRegressor{i});
    end
end

maxLag = M.maxLag;
top_output_lags = M.top_output_lags;
significant_input_lags = M.significant_input_lags;
net = M.net;

%% Always load the cleaned data file — needed for u (and possibly for stats recompute)
D = load(cleanedDataFile);
if ~isfield(D,'y') || ~isfield(D,'u')
    error('"%s" must contain variables "y" and "u".', cleanedDataFile);
end
y = D.y(:);
u = D.u(:);

if isfield(D, 'T')
    T = D.T;
else
    error('"%s" must contain sampling time "T" to build the u timeseries.', cleanedDataFile);
end

%% Get muX/sigmaX/muY/sigmaY — either already in the model file, or recompute from cleaned data
haveAllNormStats = all(isfield(M, {'muX','sigmaX','muY','sigmaY'}));

if haveAllNormStats
    fprintf('Normalization stats found in "%s" — using those directly.\n', modelFile);
    muX = M.muX;
    sigmaX = M.sigmaX;
    muY = M.muY;
    sigmaY = M.sigmaY;
else
    fprintf('Normalization stats not found in "%s" — recomputing from "%s".\n', modelFile, cleanedDataFile);

    % Rebuild the same regressor matrix used at training time
    N = length(y);
    if N <= maxLag
        error('"%s" is shorter than maxLag (%d) — cannot rebuild regressors.', cleanedDataFile, maxLag);
    end
    X = zeros(N-maxLag, length(top_output_lags)+length(significant_input_lags));
    Y = zeros(N-maxLag,1);
    for k = maxLag+1:N
        yReg = y(k-top_output_lags);
        uReg = u(k-significant_input_lags);
        X(k-maxLag,:) = [yReg(:)' uReg(:)'];
        Y(k-maxLag) = y(k);
    end

    % IMPORTANT: this recomputes stats from THIS dataset only. If your
    % original training used a combined/multi-dataset X, these will NOT
    % exactly match what the network was actually trained on — only use
    % this fallback path if the model file truly never saved the real
    % training-time stats. Prefer fixing the training script's save line
    % over relying on this recompute.
    [~, muX, sigmaX] = zscore(X);
    muX = muX(:);
    sigmaX = sigmaX(:);
    [~, muY, sigmaY] = zscore(Y);

    warning(['Recomputed muX/sigmaX/muY/sigmaY from "%s" instead of using saved training stats. ' ...
        'These may not exactly match the original training normalization if training used ' ...
        'a different or combined dataset.'], cleanedDataFile);
end

%% Package u as a timeseries, ready for a "From Workspace" block
timeVector = (0:length(u)-1)' * T;
u_timeseries = timeseries(u, timeVector);
u_timeseries.Name = 'u';
timeVector = (0:length(u)-1)' * T;
u_matrix = [timeVector, u(:)];   % first column = time, second column = data

%% Package everything Simulink needs into one file
save(outputFile, 'net', 'maxLag', 'top_output_lags', 'significant_input_lags', ...
     'muX', 'sigmaX', 'muY', 'sigmaY', 'u', 'u_timeseries', 'T','C_optd','notchD');

fprintf(['Saved "%s" with: net, maxLag, top_output_lags, significant_input_lags, ' ...
    'muX, sigmaX, muY, sigmaY, u, u_timeseries, T\n'], outputFile);
