function P = extractLaplaceParams(net, L)
%EXTRACTLAPLACEPARAMS  Pull trained laplaceLayer + FC-head weights into a
%   lightweight struct for fast streaming inference (no dlarray/dlnetwork
%   overhead, no full window reconstruction).
%
%   P = extractLaplaceParams(net, L)
%
%   net : trained SeriesNetwork/DAGNetwork containing a laplaceLayer
%   L   : fixed window length used to build training rows (must match
%         the L used in laplace_network_train.m -- max_lag - dead_time)
%
%
%       P = extractLaplaceParams(net, L);
%       save('lno_fast_params.mat','P','muX','sigmaX','muY','sigmaY');

    lIdx = find(arrayfun(@(l) isa(l,'laplaceLayer'), net.Layers));
    if isempty(lIdx)
        error('extractLaplaceParams:notFound', 'No laplaceLayer found in the network.');
    end
    lay = net.Layers(lIdx(1));

    K = lay.NumPoles;
    C = lay.NumChannels;

    RawSigma = localExtract(lay.RawSigma);   % K x 1
    Omega    = localExtract(lay.Omega);      % K x 1
    Rr = localExtract(lay.Rr);               % C x C x K
    Ri = localExtract(lay.Ri);               % C x C x K
    Wlocal = localExtract(lay.Wlocal);       % C x C
    Blocal = localExtract(lay.Blocal);       % C x 1

    sigma = -log(1 + exp(RawSigma));        % matches softplusL(x) then negation
    a = -sigma + 1i*Omega;                  % K x 1, forward exponent
    b =  sigma + 1i*Omega;                  % K x 1, inverse/reconstruction exponent

    dt = 1/(L-1);
    t  = linspace(0, 1, L);

    Abar  = zeros(K,1);
    for p = 1:K
        Abar(p) = mean(exp(b(p)*t));        % pooled inverse-basis constant per pole
    end
    decay = exp(-a/(L-1));                  % K x 1, per-step decay of running coeff
    expA  = exp(a);                         % K x 1, weight for the newest sample

    Rcplx = complex(Rr, Ri);                % C x C x K, packed residue matrices

    % Locate the two fullyConnectedLayer objects following the pooling stage
    fcIdx = find(arrayfun(@(l) isa(l,'nnet.cnn.layer.FullyConnectedLayer'), net.Layers));
    if numel(fcIdx) < 2
        error('extractLaplaceParams:fcNotFound', ...
            'Expected two fullyConnectedLayer objects after the pooling layer.');
    end
    W1 = localExtract(net.Layers(fcIdx(1)).Weights);
    B1 = localExtract(net.Layers(fcIdx(1)).Bias);
    W2 = localExtract(net.Layers(fcIdx(2)).Weights);
    B2 = localExtract(net.Layers(fcIdx(2)).Bias);

    P = struct('K',K, 'C',C, 'L',L, 'dt',dt, ...
                'a',a, 'decay',decay, 'expA',expA, 'Abar',Abar, ...
                'Rcplx',Rcplx, 'Wlocal',Wlocal, 'Blocal',Blocal, ...
                'W1',W1, 'B1',B1, 'W2',W2, 'B2',B2);
end

function v = localExtract(v)
    % Unwrap only if it's actually a dlarray -- trainNetwork (as opposed
    % to dlnetwork/custom training loops) returns plain numeric arrays
    % for learnables once training is finished, and extractdata errors
    % on non-dlarray input.
    if isa(v, 'dlarray')
        v = extractdata(v);
    end
    % Deep Learning Toolbox layers/dlarrays default to single precision,
    % while muX/sigmaX/muY/sigmaY (from your data) are typically double.
    % Force double everywhere so laplaceFastPredictor's output type is
    % consistent across all code paths -- Simulink's code generator
    % rejects a variable that's single on one branch and double on
    % another (e.g. the startup-NaN branch vs. the steady-state branch).
    v = double(v);
end
