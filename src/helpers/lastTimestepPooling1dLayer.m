classdef lastTimestepPooling1dLayer < nnet.layer.Layer
    % lastTimestepPooling1dLayer   Takes only the final timestep of a
    % [Channels x Batch x TimeSteps] sequence, instead of averaging over
    % every timestep the way globalAveragePooling1dLayer does.
    %
    % Why: after a sequence-to-sequence layer like laplaceLayer, the
    % output is still [Channels x Batch x TimeSteps] -- one value per
    % input timestep, not a single summary. globalAveragePooling1dLayer
    % collapses that by averaging across all T timesteps, which treats a
    % sample from L steps ago the same as the sample right before the
    % prediction point. If recent history matters more than the time-
    % averaged signal, that averaging can throw away exactly the
    % information the regression head needs. This layer instead passes
    % through just the last timestep -- the state closest to the moment
    % being predicted -- with no averaging.
    %
    % UNVERIFIED: written without access to MATLAB in this environment,
    % same caveat as laplaceLayer.m. The first version of this layer left
    % a leftover singleton time dimension after slicing, which made
    % trainNetwork treat the whole network as sequence-output and error
    % demanding sequence-shaped responses -- that's fixed below by
    % explicitly reshaping the dimension away and updating the dlarray's
    % format label, but this still hasn't been run end-to-end, so treat
    % the next training attempt as the real test.

    methods
        function layer = lastTimestepPooling1dLayer(args)
            arguments
                args.Name string = ""
            end
            layer.Name = args.Name;
            layer.Type = "Last-Timestep Pooling";
            layer.Description = "Selects the final timestep of the input sequence (no averaging)";
        end

        function Z = predict(~, X)
            % X: [Channels x Batch x TimeSteps], as produced by laplaceLayer.
            %
            % Slicing out the last timestep leaves a leftover singleton
            % dimension where T used to be. If that dimension isn't
            % actually removed (and, for a formatted dlarray, its 'T'
            % label dropped from the format), MATLAB still treats the
            % layer's output as a sequence -- which made trainNetwork
            % demand sequence-shaped responses instead of a plain numeric
            % matrix. So this explicitly reshapes the singleton away and
            % rebuilds the format string without 'T', rather than relying
            % on indexing or squeeze (squeeze would also drop the Batch
            % dimension whenever batch size happens to be 1, which is
            % wrong -- this only ever removes the time dimension).
            if isa(X, 'dlarray') && ~isempty(dims(X))
                fmt = dims(X);
                tPos = strfind(fmt, 'T');
                if isempty(tPos)
                    tPos = numel(fmt);   % fallback: assume time is the last dimension
                end

                idx = repmat({':'}, 1, ndims(X));
                idx{tPos} = size(X, tPos);
                Zsub = X(idx{:});          % same ndims as X, singleton at tPos

                data = stripdims(Zsub);    % plain array, same shape as Zsub
                shp = size(data);
                shp(tPos) = [];            % drop that specific (singleton) dimension
                if isscalar(shp)
                    shp(2) = 1;            % dlarray needs at least 2 dims
                end
                data = reshape(data, shp);

                newFmt = fmt;
                newFmt(tPos) = [];
                Z = dlarray(data, newFmt);
            else
                % Not a formatted dlarray -- plain numeric fallback.
                % Assume time is the last dimension and drop it the same way.
                tPos = ndims(X);
                idx = repmat({':'}, 1, ndims(X));
                idx{tPos} = size(X, tPos);
                Zsub = X(idx{:});
                shp = size(Zsub);
                shp(tPos) = [];
                if isscalar(shp)
                    shp(2) = 1;
                end
                Z = reshape(Zsub, shp);
            end
        end
    end
end
