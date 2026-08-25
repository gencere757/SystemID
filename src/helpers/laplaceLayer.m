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
