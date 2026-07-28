classdef laplaceLayer < nnet.layer.Layer
    % laplaceLayer   Learnable Laplace (pole-residue) neural-operator layer.
    %
    % Computes, for input X of size [NumChannels x Batch x TimeSteps]:
    %
    %   laplaceOut = pole-residue branch (global, transform-based)
    %   localOut   = K_phi branch (pointwise 1x1 channel mixing, "local")
    %   Z          = laplaceOut + localOut
    %
    % Poles (decay rate + frequency) are shared across channels, like
    % FNO shares Fourier modes; each pole has its own learned
    % [NumChannels x NumChannels] residue matrix mixing input channels
    % into output channels. No nonlinearity is applied inside this
    % layer -- compose it with a separate reluLayer/etc. in your graph,
    % the same way spectralConv + activation are separate steps in FNO.
    %
    % Time is assumed to span a normalized [0,1] grid based on the
    % actual sequence length seen at runtime (T = size(X,3)).
    %
    % UNVERIFIED: written without access to MATLAB in this environment.

    properties
        NumPoles
        NumChannels
    end

    properties (Learnable)
        RawSigma   % NumPoles x 1        (decay rate before -softplus)
        Omega      % NumPoles x 1        (frequency)
        Rr         % NumChannels x NumChannels x NumPoles   (residue, real part)
        Ri         % NumChannels x NumChannels x NumPoles   (residue, imag part)
        Wlocal     % NumChannels x NumChannels               (K_phi weight)
        Blocal     % NumChannels x 1                          (K_phi bias)
    end

    methods
        function layer = laplaceLayer(numPoles, numChannels, args)
            arguments
                numPoles (1,1) double
                numChannels (1,1) double
                args.Name string = ""
            end
            layer.Name = args.Name;
            layer.Type = "Laplace Pole-Residue Layer";
            layer.Description = sprintf("Laplace operator layer (%d poles, %d channels)", ...
                numPoles, numChannels);
            layer.NumPoles = numPoles;
            layer.NumChannels = numChannels;

            layer.RawSigma = dlarray(0.5 + rand(numPoles,1));
            layer.Omega    = dlarray(40*rand(numPoles,1));
            scale = 1/numChannels;
            layer.Rr = dlarray(scale*randn(numChannels, numChannels, numPoles));
            layer.Ri = dlarray(scale*randn(numChannels, numChannels, numPoles));
            layer.Wlocal = dlarray(eye(numChannels) + 0.1*randn(numChannels));
            layer.Blocal = dlarray(zeros(numChannels,1));
        end

        function Z = predict(layer, X)
            % X: [C x B x T] dlarray. Strip any "CBT"-style format labels
            % up front so all indexing/reshape below is unambiguous.
            if isa(X,'dlarray') && ~isempty(dims(X))
                X = stripdims(X);
            end
            C = size(X,1); B = size(X,2); Tn = size(X,3);
            t = linspace(0, 1, Tn);          % normalized time grid, 1 x T
            dt = t(2) - t(1);

            K = layer.NumPoles;
            sigma = -softplusL(layer.RawSigma);   % K x 1, guaranteed negative
            omega = layer.Omega;                   % K x 1

            omegaT   = omega * t;                  % K x T (outer product)
            decayFwd = exp((-sigma) * t);           % K x T
            decayInv = exp(sigma * t);               % K x T
            cosOT = cos(omegaT);
            sinOT = sin(omegaT);

            % X reshaped to [T x (C*B)] for basis projection over time
            Xp = reshape(permute(X, [3 1 2]), Tn, C*B);   % T x (C*B)

            laplaceOut = dlarray(zeros(C, B, Tn, 'like', X));

            for k = 1:K
                basisFwdReal_k = decayFwd(k,:) .* cosOT(k,:);   % 1 x T
                basisFwdImag_k = decayFwd(k,:) .* sinOT(k,:);   % 1 x T

                coeffReal_k = reshape(basisFwdReal_k * Xp * dt, C, B);  % C x B
                coeffImag_k = reshape(basisFwdImag_k * Xp * dt, C, B);  % C x B

                Rr_k = layer.Rr(:,:,k);   % C_out x C_in
                Ri_k = layer.Ri(:,:,k);
                outReal_k = Rr_k * coeffReal_k - Ri_k * coeffImag_k;    % C_out x B
                outImag_k = Rr_k * coeffImag_k + Ri_k * coeffReal_k;    % C_out x B

                basisInvReal_k = decayInv(k,:) .* cosOT(k,:);   % 1 x T
                basisInvImag_k = decayInv(k,:) .* sinOT(k,:);   % 1 x T

                % outer-product accumulate: [C_out x B x 1] .* [1 x 1 x T] -> [C_out x B x T]
                term = reshape(outReal_k, C, B, 1) .* reshape(basisInvReal_k, 1, 1, Tn) ...
                     - reshape(outImag_k, C, B, 1) .* reshape(basisInvImag_k, 1, 1, Tn);
                laplaceOut = laplaceOut + term;
            end

            % K_phi branch: pointwise (1x1) channel mixing, applied identically at each time step
            Xflat = reshape(X, C, B*Tn);                       % C x (B*T)
            localFlat = layer.Wlocal * Xflat + layer.Blocal;   % C x (B*T)
            localOut = reshape(localFlat, C, B, Tn);

            Z = laplaceOut + localOut;
        end
    end
end

function y = softplusL(x)
    y = log(1 + exp(x));
end
