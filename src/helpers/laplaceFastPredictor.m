classdef laplaceFastPredictor < matlab.System
    %LAPLACEFASTPREDICTOR  O(1)-per-step streaming predictor for the
    %   trained Laplace network, for use as a MATLAB System block in
    %   Simulink closed-loop simulation (code generation).
    %
    %   Replaces calling predict(net, X) on a full L-length window every
    %   timestep. Instead maintains a running per-pole coefficient state
    %   that is updated incrementally as new samples arrive, and uses a
    %   closed-form pooled reconstruction (valid because the trained
    %   architecture is laplaceLayer -> globalAveragePooling1dLayer).
    %
    %   *** Fixed architecture sizes ***
    %   C=2 channels (u,y), K=16 poles, L=87 window length (= max_lag -
    %   dead_time from training), H=64 hidden units (fullyConnectedLayer(64)
    %   in training -- unrelated to L, just a coincidentally-different
    %   constant). These are hardcoded as literal sizes below -- NOT loaded
    %   dynamically -- because MATLAB Coder needs every variable's size
    %   fixed at COMPILE TIME, before setupImpl ever runs. If you retrain
    %   with different dimensions, update every window-length "87" below
    %   (buffer, coeffRe/coeffIm are C x K so unaffected, the startup
    %   loop's bufCount checks and linspace) to match the new L -- do NOT
    %   touch the "64" in W1/B1/W2, that's the FC layer width, not L.
    %
    %   SETUP (once, offline, after training):
    %       P = extractLaplaceParams(net, L);
    %       save('lno_fast_params.mat','P','muX','sigmaX','muY','sigmaY');
    %
    %   SIMULINK USAGE:
    %       Add this as a MATLAB System block. Set ParamsFile to the
    %       .mat file above. Feed it ONE raw [u; y] sample (C x 1) per
    %       timestep via a Mux -- not a window; the block keeps the
    %       window internally.
    %
    %   *** CRITICAL: channel alignment ***
    %   The training script builds each row as
    %       uSeq = u(k-dead_time-L+1 : k-dead_time)
    %       ySeq = y(k-L : k-1)
    %   i.e. the u-channel window is offset from the y-channel window by
    %   dead_time samples. This block does NOT know about that offset --
    %   it just streams whatever C x 1 vector you feed it. u must be
    %   pre-delayed by dead_time samples (see delayU.m) before being
    %   muxed with y and fed in here.
    %
    %   The first L-1 calls only fill the initial window and return NaN;
    %   hold your last valid control action until real predictions start
    %   flowing.
    %
    %   UNVERIFIED IN MATLAB (no MATLAB available in this environment).
    %   The pooled closed-form and the incremental coefficient recursion
    %   were both cross-checked numerically in Python/numpy against a
    %   direct translation of laplaceLayer.predict() +
    %   globalAveragePooling1dLayer, matching to floating-point precision
    %   (~1e-13 relative error) over 200+ simulated sliding-window steps.
    %   Still: before trusting this in closed loop, validate it against
    %   predict(net, X) on several real recorded windows.

    properties (Nontunable)
        ParamsFile = 'lno_fast_params.mat'
    end

    properties (Access = private)
        % Normalization (C x 1 / scalar) -- fixed size so Coder knows
        % xRaw must be 2 x 1, which is what forces the Mux upstream to
        % actually produce 2 elements instead of collapsing to 1.
        muX = zeros(2,1)
        sigmaX = zeros(2,1)
        muY = 0
        sigmaY = 1

        % Pole constants (K x 1 each), split into real/imag parts --
        % complex numbers are avoided in stored state for codegen
        % simplicity; arithmetic below does real/imag manually.
        decayRe = zeros(16,1)
        decayIm = zeros(16,1)
        expARe  = zeros(16,1)
        expAIm  = zeros(16,1)
        AbarRe  = zeros(16,1)
        AbarIm  = zeros(16,1)
        aRe     = zeros(16,1)   % forward exponent (real part), startup only
        aIm     = zeros(16,1)   % forward exponent (imag part), startup only

        % Residue matrices (C x C x K), local branch, FC head
        Rr = zeros(2,2,16)
        Ri = zeros(2,2,16)
        Wlocal = zeros(2,2)
        Blocal = zeros(2,1)
        W1 = zeros(64,2)
        B1 = zeros(64,1)
        W2 = zeros(1,64)
        B2 = 0

        dt = 1/86

        % Running state
        coeffRe = zeros(2,16)
        coeffIm = zeros(2,16)
        buffer  = zeros(2,87)     % oldest sample first
        bufCount = 0
    end

    methods (Access = protected)
        function setupImpl(obj)
            S = load(obj.ParamsFile);
            P = S.P;
            if P.C ~= 2 || P.K ~= 16 || P.L ~= 87
                error('laplaceFastPredictor:sizeMismatch', ...
                    ['This block is hardcoded for C=2, K=16, L=87 (code ' ...
                     'generation requires compile-time-fixed sizes). ' ...
                     'Loaded params have C=%d, K=%d, L=%d -- update every ' ...
                     'zeros(...) size in laplaceFastPredictor.m to match.'], ...
                     P.C, P.K, P.L);
            end

            obj.muX = double(S.muX(:));
            obj.sigmaX = double(S.sigmaX(:));
            obj.muY = double(S.muY);
            obj.sigmaY = double(S.sigmaY);

            obj.decayRe = real(P.decay(:));  obj.decayIm = imag(P.decay(:));
            obj.expARe  = real(P.expA(:));   obj.expAIm  = imag(P.expA(:));
            obj.AbarRe  = real(P.Abar(:));   obj.AbarIm  = imag(P.Abar(:));
            obj.aRe     = real(P.a(:));      obj.aIm     = imag(P.a(:));

            obj.Rr = real(P.Rcplx);
            obj.Ri = imag(P.Rcplx);
            obj.Wlocal = P.Wlocal;
            obj.Blocal = P.Blocal(:);

            obj.W1 = P.W1;
            obj.B1 = P.B1(:);
            obj.W2 = P.W2;
            obj.B2 = P.B2;

            obj.dt = P.dt;

            obj.coeffRe = zeros(2,16);
            obj.coeffIm = zeros(2,16);
            obj.buffer  = zeros(2,87);
            obj.bufCount = 0;
        end

        function yhat = stepImpl(obj, xRaw)
            % Explicitly pin the input size to 2x1. Without this, MATLAB's
            % implicit expansion (broadcasting) makes "xRaw(:) - obj.muX"
            % satisfiable with xRaw as EITHER 1x1 or 2x1 -- Coder's size
            % inference can then resolve the port to the less-constrained
            % size (1), which is what was fighting the Mux upstream. The
            % assert() form below is recognized by MATLAB Coder as an
            % explicit compile-time size declaration.
            assert(isequal(size(xRaw), [2 1]));
            xNorm = (double(xRaw) - obj.muX) ./ obj.sigmaX;   % 2 x 1

            if obj.bufCount < 87
                % --- startup: fill the first window, one-time direct sum ---
                obj.buffer(:, obj.bufCount+1) = xNorm;
                obj.bufCount = obj.bufCount + 1;
                if obj.bufCount == 87
                    t = linspace(0, 1, 87);   % 1 x 87
                    for p = 1:16
                        decayT = exp(obj.aRe(p)*t);       % 1 x 87
                        basisFwdRe = decayT .* cos(obj.aIm(p)*t);
                        basisFwdIm = decayT .* sin(obj.aIm(p)*t);
                        obj.coeffRe(:,p) = (obj.buffer * basisFwdRe.') * obj.dt;
                        obj.coeffIm(:,p) = (obj.buffer * basisFwdIm.') * obj.dt;
                    end
                end
                yhat = NaN(size(obj.muY));
                return;
            end

            % --- steady state: O(1) incremental coefficient update ---
            xOld = obj.buffer(:,1);
            obj.buffer = [obj.buffer(:,2:end), xNorm];

            for p = 1:16
                cr = obj.coeffRe(:,p);  ci = obj.coeffIm(:,p);
                dr = obj.decayRe(p);    di = obj.decayIm(p);
                % (cr+j ci)*(dr+j di)
                newRe = cr*dr - ci*di;
                newIm = cr*di + ci*dr;
                % remove departing sample's (decayed) contribution
                newRe = newRe - xOld*obj.dt*dr;
                newIm = newIm - xOld*obj.dt*di;
                % add entering sample at the newest position
                newRe = newRe + xNorm*obj.dt*obj.expARe(p);
                newIm = newIm + xNorm*obj.dt*obj.expAIm(p);
                obj.coeffRe(:,p) = newRe;
                obj.coeffIm(:,p) = newIm;
            end

            % pooled laplace branch (closed form, no reconstruction)
            pooled = zeros(2,1);
            for p = 1:16
                cr = obj.coeffRe(:,p);  ci = obj.coeffIm(:,p);
                % Op = (Rr+jRi)*(cr+jci), C x C times C x 1
                Opre = obj.Rr(:,:,p)*cr - obj.Ri(:,:,p)*ci;
                Opim = obj.Rr(:,:,p)*ci + obj.Ri(:,:,p)*cr;
                % real(Op * Abar_p)
                pooled = pooled + (Opre*obj.AbarRe(p) - Opim*obj.AbarIm(p));
            end
            localPooled = obj.Wlocal * mean(obj.buffer,2) + obj.Blocal;
            z = pooled + localPooled;

            % remaining head: FC(64) -> tanh -> FC(1)
            % (dropoutLayer is identity at inference, so it's omitted here)
            h = tanh(obj.W1*z + obj.B1);
            yNorm = obj.W2*h + obj.B2;

            yhat = yNorm*obj.sigmaY + obj.muY;
        end

        function resetImpl(obj)
            obj.coeffRe(:) = 0;
            obj.coeffIm(:) = 0;
            obj.buffer(:) = 0;
            obj.bufCount = 0;
        end
        function sz = getOutputSizeImpl(~)
            sz = [1 1];
        end
        
        function fixed = isOutputFixedSizeImpl(~)
            fixed = true;
        end
        
        function dt = getOutputDataTypeImpl(~)
            dt = 'double';
        end
        
        function cplx = isOutputComplexImpl(~)
            cplx = false;
        end
    end
end
