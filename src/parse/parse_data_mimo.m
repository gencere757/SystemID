%% parse_data_mimo.m
%   Author: Arda Gencer
%   Functionality: Converts the raw CubeSpec FSM MIMO dataset into one
%   file per (excitation level, block), each containing a continuous
%   multi-channel u/y time series. Train/test stay exactly as pre-split
%   in the raw file (no re-splitting here -- avoids the leakage bug
%   already fixed in the SISO pipeline's multi_data_MLP.m).
%
%   Raw file variables: u_<level>_<split>, y_<level>_<split>
%     level in {100mV, 200mV, 300mV}, split in {train, test}
%     shape: [8192 time samples x 3 channels x numBlocks x 2 periods]
%     numBlocks = 6 for train, 3 for test.
%
%   Axis meaning (decoded from the CubeSpec FSM paper -- see project
%   memory "project_cubespec_fsm_dataset" for full derivation/confidence
%   notes):
%     dim1 = time samples within one period (N=8192, fs=6400 Hz)
%     dim2 = physical channel (3 piezo inputs / 3 probe outputs)
%     dim3 = block-experiment index
%     dim4 = usable steady-state period (2 near-identical repeats, after
%            the paper's authors already discarded the transient period)
%
%   The 2 periods within a block are concatenated into one continuous
%   16384-sample run: they're repeat measurements of the same periodic
%   input (u differs between the two indices by ~0.04%, i.e. floating-
%   point noise), so concatenating doesn't introduce a real discontinuity
%   and gives more usable samples per lag-safe segment. Different blocks
%   and different excitation levels are NOT concatenated together --
%   each stays its own output file, so feature_extraction_mimo.m's lag
%   windows never bridge across a real experiment boundary.

clc; clear; close all;

%% Config
rawFile = fullfile("data", "raw", "mimo", "mimo_data_fsm.mat");
outFolderTrain = fullfile("data", "train", "mimo");
outFolderTest = fullfile("data", "test", "mimo");
Fs = 6400;   % Hz, from the CubeSpec FSM paper (N=8192 samples/period @ this rate)
T = 1/Fs;

if ~exist(outFolderTrain, 'dir')
    mkdir(outFolderTrain);
end
if ~exist(outFolderTest, 'dir')
    mkdir(outFolderTest);
end

if ~isfile(rawFile)
    error('Raw MIMO file not found at "%s". Drop mimo_data_fsm.mat there first.', rawFile);
end
S = load(rawFile);

levels = ["100mV", "200mV", "300mV"];
splitNames = ["train", "test"];
splitOutFolders = [outFolderTrain, outFolderTest];

totalSaved = 0;
for lv = 1:numel(levels)
    level = levels(lv);
    for sp = 1:numel(splitNames)
        splitName = splitNames(sp);
        outFolder = splitOutFolders(sp);

        uVar = sprintf('u_%s_%s', level, splitName);
        yVar = sprintf('y_%s_%s', level, splitName);
        if ~isfield(S, uVar) || ~isfield(S, yVar)
            warning('Skipping %s/%s: variables "%s"/"%s" not found in raw file.', ...
                level, splitName, uVar, yVar);
            continue;
        end

        uAll = S.(uVar);   % [8192 x 3 x numBlocks x 2]
        yAll = S.(yVar);
        if size(uAll,4) < 2
            error('Expected at least 2 usable periods (dim 4) in "%s" -- got %d.', uVar, size(uAll,4));
        end
        numBlocks = size(uAll, 3);

        for b = 1:numBlocks
            % Concatenate the 2 usable steady-state periods -> one
            % continuous run. Rows = time, columns = channel (1..3).
            u = [uAll(:,:,b,1); uAll(:,:,b,2)];   % 16384 x 3
            y = [yAll(:,:,b,1); yAll(:,:,b,2)];   % 16384 x 3

            excitationLevel = level;   %#ok<NASGU> saved for provenance
            blockIdx = b;              %#ok<NASGU>

            outName = sprintf('%s_block%d.mat', level, b);
            outPath = fullfile(outFolder, outName);
            save(outPath, 'u', 'y', 'T', 'excitationLevel', 'blockIdx');

            fprintf('Saved "%s" (%d samples x %d channels)\n', outPath, size(u,1), size(u,2));
            totalSaved = totalSaved + 1;
        end
    end
end

if totalSaved == 0
    error('No MIMO blocks were saved -- check that "%s" contains the expected variables.', rawFile);
end

fprintf('Done. %d files saved -- train in "%s", test in "%s".\n', totalSaved, outFolderTrain, outFolderTest);
