function runFolder = log_run(scriptName, datasetDesc, rmse, mae, fit, figHandles, modelPath)
%LOG_RUN Archive a training/eval run so results survive the next run.
%   log_run(scriptName, datasetDesc, rmse, mae, fit, figHandles, modelPath)
%
%   Call this at the end of any training or evaluation script. It:
%     1. Creates a timestamped folder under data/results/
%     2. Copies the given model file into it (if provided)
%     3. Saves every figure in figHandles as a PNG into it
%     4. Appends one summary row to data/results/results_log.csv
%
%   Inputs:
%     scriptName  - string, e.g. "multi_data_MLP"
%     datasetDesc - string describing what was run on, e.g.
%                   "12 files, block split" or "3 test files"
%     rmse/mae/fit - scalars. Use NaN for any that don't apply.
%     figHandles  - array of figure handles to save (use [] for none)
%     modelPath   - path to a saved model .mat to copy into the archive
%                   (use "" to skip)
%
%   Returns the path to this run's archive folder (runFolder), so callers
%   that have extra per-run data to save (e.g. a per-dataset results
%   table) can write it into the same folder.
%
%   Because results_log.csv is a single running log across every script,
%   opening it answers "what have I gotten so far" without re-running
%   anything — that's the whole point.

    resultsFolder = fullfile("data", "results");
    if ~exist(resultsFolder, 'dir')
        mkdir(resultsFolder);
    end

    runTag = string(datetime('now', 'Format', 'yyyy-MM-dd_HHmmss'));
    runFolder = fullfile(resultsFolder, runTag + "_" + string(scriptName));
    mkdir(runFolder);

    if nargin >= 7 && strlength(string(modelPath)) > 0 && isfile(modelPath)
        [~, mName, mExt] = fileparts(modelPath);
        copyfile(modelPath, fullfile(runFolder, mName + mExt));
    end

    for f = 1:numel(figHandles)
        figName = figHandles(f).Name;
        if isempty(figName)
            figName = sprintf('figure_%d', f);
        end
        saveas(figHandles(f), fullfile(runFolder, matlab.lang.makeValidName(figName) + ".png"));
    end

    logPath = fullfile(resultsFolder, "results_log.csv");
    logRow = table(datetime('now'), string(scriptName), string(datasetDesc), rmse, mae, fit, string(runFolder), ...
        'VariableNames', {'Timestamp','Script','Dataset','RMSE','MAE','Fit','RunFolder'});
    if isfile(logPath)
        writetable(logRow, logPath, 'WriteMode', 'append');
    else
        writetable(logRow, logPath);
    end

    fprintf('Archived run to "%s"\n', runFolder);
end
