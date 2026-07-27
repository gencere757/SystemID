clc; clear; close all;

%% Load model + normalization stats
load("MLP_model.mat");   % must contain: net, maxLag, top_output_lags, significant_input_lags,
                          % top_output_dot_lags, significant_input_dot_lags, muX, sigmaX, muY, sigmaY

%% Config: folder containing datasets to evaluate
evalFolder = "Training Data";   % change to a separate "Test Data" folder if you have one
fileList = dir(fullfile(evalFolder, "*.mat"));
if isempty(fileList)
    error('No .mat files found in "%s".', evalFolder);
end

%% Results table
results = table('Size', [numel(fileList), 4], ...
    'VariableTypes', {'string','double','double','double'}, ...
    'VariableNames', {'Dataset','RMSE','MAE','Fit'});

for i = 1:numel(fileList)
    fpath = fullfile(fileList(i).folder, fileList(i).name);
    S = load(fpath);
    if ~isfield(S,'y') || ~isfield(S,'u')
        warning('Skipping "%s": missing y or u.', fileList(i).name);
        continue;
    end
    y = S.y(:);
    u = S.u(:);

    if isfield(S,'T')
        T_i = S.T;
    else
        T_i = 1;
        warning('"%s" has no T field; assuming unit sample time for derivative regressors.', fileList(i).name);
    end
    % Optional: smooth before differentiating — gradient() amplifies sensor
    % noise. Delete the two lines below to activate. Reassigns y/u, so it
    % also changes Ynew (the measured target used for RMSE/MAE/Fit) and
    % the level regressors — you'd be scoring against smoothed ground
    % truth, not the raw measurement.
    % y = smoothdata(y, "sgolay", 31);
    % u = smoothdata(u, "sgolay", 31);

    dy = gradient(y, T_i);   % dy/dt
    du = gradient(u, T_i);   % du/dt

    %% Build regressors — must match buildRegressors() in multi_data_MLP.m exactly
    Nnew = length(y);
    if Nnew <= maxLag
        warning('Skipping "%s": too short (length=%d, maxLag=%d).', fileList(i).name, Nnew, maxLag);
        continue;
    end
    nFeat = length(top_output_lags) + length(significant_input_lags) + ...
            length(top_output_dot_lags) + length(significant_input_dot_lags);
    Xnew = zeros(Nnew-maxLag, nFeat);
    Ynew = zeros(Nnew-maxLag,1);
    for k = maxLag+1:Nnew
        yReg = y(k-top_output_lags);
        uReg = u(k-significant_input_lags);
        dyReg = dy(k-top_output_dot_lags);
        duReg = du(k-significant_input_dot_lags);
        Xnew(k-maxLag,:) = [yReg(:)' uReg(:)' dyReg(:)' duReg(:)'];
        Ynew(k-maxLag) = y(k);
    end

    %% Normalize using TRAINING statistics
    Xnew = (Xnew - muX)./sigmaX;

    %% Predict
    YPred = predict(net, Xnew);
    YPred = YPred*sigmaY + muY;

    %% Metrics
    rmse = sqrt(mean((Ynew-YPred).^2));
    mae = mean(abs(Ynew-YPred));
    fit = 100*(1-norm(Ynew-YPred)/norm(Ynew-mean(Ynew)));

    [~, baseName, ~] = fileparts(fileList(i).name);
    results.Dataset(i) = baseName;
    results.RMSE(i) = rmse;
    results.MAE(i) = mae;
    results.Fit(i) = fit;

    fprintf('%s: RMSE=%.5f, MAE=%.5f, Fit=%.2f%%\n', baseName, rmse, mae, fit);

    %% Separate figure per dataset
    figure('Name', baseName, 'NumberTitle', 'off');
    plot(Ynew,'b','LineWidth',1.5); hold on;
    plot(YPred,'r','LineWidth',1.5);
    legend('Measured','Predicted');
    xlabel('Sample'); ylabel('Output');
    title(sprintf('%s  (RMSE=%.3f, MAE=%.3f, Fit=%.1f%%)', baseName, rmse, mae, fit), 'Interpreter', 'none');
    grid on;
end

%% Remove any skipped (empty) rows and show summary
results(results.Dataset == "", :) = [];
disp(results);
