classdef laplaceLayer < nnet.layer.Layer
    % laplaceLayer   Learnable Laplace (pole-residue) neural-operator layer,
    % with amplitude-gated (RBF-scheduled) pole activation.
    %
    % Computes, for input X of size [NumChannels x Batch x TimeSteps]:
    %
    %   laplaceOut = pole-residue branch (global, transform-based),
    %                with each pole's contribution scaled by a learned
    %                Gaussian "gate" over the window's signal amplitude
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
    % --- Amplitude gating (new) ---
    % Each pole p additionally has a learnable center GateMu(p) and
    % width GateRawSigma(p) (positive after softplus). For each batch
    % element, a windowed-RMS amplitude summary "a" is computed from X,
    % and pole p's contribution to the output is scaled by
    %   gate_p(a) = exp( -(a - GateMu(p))^2 / (2*softplus(GateRawSigma(p))^2) )
    % This is a Takagi-Sugeno / LPV-style gain-scheduling mechanism:
    % each pole is a local linear mode that "turns on" near its own
    % learned operating-point amplitude, letting the layer's effective
    % gain vary with signal magnitude instead of being fixed (a strictly
    % linear pole-residue model cannot represent amplitude-dependent
    % behavior like disproportionate overshoot growth -- this is the fix
    % for that).
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
        GateMu        % NumPoles x 1   (learned amplitude center per pole)
        GateRawSigma  % NumPoles x 1   (gate width before softplus)
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

            % Spread gate centers across a plausible amplitude range.
            % NOTE: X arriving here is whatever this layer receives from
            % sequenceInputLayer -- in laplace_network_train.m that's
            % z-scored (muX/sigmaX normalized) data, so amplitudes are
            % roughly in the [-3, 3] range for a well-behaved signal, but
            % check your actual training data's windowed-RMS spread and
            % adjust this range if it's off -- centers that don't span
            % where your data actually lives will leave some poles
            % permanently "dead" (near-zero gradient, see class notes).
            layer.GateMu = dlarray(linspace(0, 3, numPoles)');
            % Initialize width comparable to the spacing between centers
            % so adjacent gates overlap somewhat rather than leaving gaps.
            spacing = 3/max(numPoles-1, 1);
            initSigma = max(spacing, 0.25);
            layer.GateRawSigma = dlarray(log(exp(initSigma) - 1) * ones(numPoles,1));  % inverse-softplus
        end

        function Z = predict(layer, X)
            if isa(X,'dlarray') && ~isempty(dims(X))
                X = stripdims(X);
            end
            C = size(X,1); B = size(X,2); Tn = size(X,3);
            t = linspace(0, 1, Tn);
            if Tn > 1
                dt = 1 / (Tn - 1);
            else
                dt = 1;
            end
        
            K = layer.NumPoles;
            sigma = -softplusL(layer.RawSigma);
            omega = layer.Omega;
        
            omegaT   = omega * t;
            decayFwd = exp((-sigma) * t);
            decayInv = exp(sigma * t);
            cosOT = cos(omegaT);
            sinOT = sin(omegaT);
        
            basisFwdReal = decayFwd .* cosOT;
            basisFwdImag = decayFwd .* sinOT;
            basisInvReal = decayInv .* cosOT;
            basisInvImag = decayInv .* sinOT;
        
            Xp = reshape(permute(X, [3 1 2]), Tn, C*B);
        
            coeffReal = permute(reshape(basisFwdReal * Xp * dt, K, C, B), [2 3 1]);
            coeffImag = permute(reshape(basisFwdImag * Xp * dt, K, C, B), [2 3 1]);
        
            outReal = pagemtimes(layer.Rr, coeffReal) - pagemtimes(layer.Ri, coeffImag);
            outImag = pagemtimes(layer.Rr, coeffImag) + pagemtimes(layer.Ri, coeffReal);

            % --- amplitude-gated pole activation -----------------------
            % Windowed-RMS amplitude summary per batch element, averaged
            % over channels and time within this window: 1 x B.
            ampRMS = sqrt(mean(X.^2, [1 3]));

            sigmaG = softplusL(layer.GateRawSigma);                    % K x 1, positive
            % (GateMu - ampRMS): (K x 1) - (1 x B) broadcasts to K x B
            gate = exp( -((layer.GateMu - ampRMS).^2) ./ (2*sigmaG.^2) );  % K x B

            % Reshape to 1 x B x K so it broadcasts against the C
            % dimension of outReal/outImag (C x B x K) via elementwise
            % multiply -- this scales pole p's ENTIRE residue-matrix
            % contribution for batch element b by gate(p,b), applied
            % after the Rr/Ri matrix multiply (mathematically identical
            % to scaling Rr/Ri themselves, since gate is a scalar here).
            gateBCT = permute(gate, [3 2 1]);   % 1 x B x K
            outReal = outReal .* gateBCT;
            outImag = outImag .* gateBCT;
            % -------------------------------------------------------------
        
            combinedReal = reshape(outReal, C*B, K) * basisInvReal;
            combinedImag = reshape(outImag, C*B, K) * basisInvImag;
            laplaceOut = reshape(combinedReal - combinedImag, C, B, Tn);
        
            Xflat = reshape(X, C, B*Tn);
            localFlat = layer.Wlocal * Xflat + layer.Blocal;
            localOut = reshape(localFlat, C, B, Tn);
        
            Z = laplaceOut + localOut;
        end
    end
end

function y = softplusL(x)
    y = log(1 + exp(x));
end
