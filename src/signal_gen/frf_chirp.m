%% generate_frf_chirp.m
% Generates a swept-sine (chirp) timeseries with user-defined sampling
% time, magnitude, and frequency sweep range. Useful as a synthetic
% input signal for FRF (frequency response function) estimation / system
% identification testing -- a chirp excites a continuous band of
% frequencies in one run, unlike a square wave which only excites a
% fundamental + odd harmonics.
clear; clc; close all;
%% --- User-defined parameters ---
Ts      = 6.6667e-05;   % Sampling time [s]
simTime = 10;            % Total simulation time [s]
mag     = 100;           % Chirp magnitude (amplitude), signal will swing +/- mag/2 about offset
offset  = 0;             % DC offset (mean level) of the chirp
f0      = 1;              % Start frequency [Hz]
f1      = 500;            % End frequency [Hz]
sweepMethod = 'logarithmic';   % 'linear' | 'logarithmic' | 'quadratic' -- see doc chirp
%% --- Time vector ---
t = (0:Ts:simTime)';   % column vector of time samples
%% --- Nyquist check -----------------------------------------------------
% Warn (don't silently alias) if f1 is set above what Ts can resolve.
fNyquist = 1/(2*Ts);
if f1 > fNyquist
    warning(['End frequency f1=%.2f Hz exceeds the Nyquist frequency ' ...
        '%.2f Hz for Ts=%.6g s. Content above Nyquist will alias. ' ...
        'Lower f1 or Ts to fix.'], f1, fNyquist);
end
%% --- Generate chirp ------------------------------------------------------
% chirp(t, f0, t1, f1, method) sweeps from f0 at t=0 to f1 at t=t1, in
% [-1, 1]; here t1 = simTime so the sweep spans the whole signal.
u = offset + (mag/2) * chirp(t, f0, simTime, f1, sweepMethod);
%% --- Package as timeseries object ---
u_timeseries = timeseries(u, t, 'Name', 'FRFChirpInput');
u_timeseries.DataInfo.Units = 'amplitude';
u_timeseries.TimeInfo.Units = 'seconds';
%% --- Plot: time domain ---
figure;
plot(t, u, 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('Amplitude');
title(sprintf('FRF Chirp: %s sweep %.2f-%.2f Hz, mag=%.2f, Ts=%.3g s', ...
    sweepMethod, f0, f1, mag, Ts));
grid on;
ylim([offset - mag/2 - 0.1*mag, offset + mag/2 + 0.1*mag]);
%% --- Plot: spectrogram, to visually confirm the sweep covers the intended
%% frequency band and there's no unexpected aliasing/rolloff ---
figure;
spectrogram(u, kaiser(256,5), 220, 512, 1/Ts, 'yaxis');
title('FRF Chirp: Spectrogram');
%% --- (Optional) Save to .mat file ---
save('data\unknown_test\frf_chirp_data.mat', 't', 'u', 'u_timeseries');
%% --- (Optional) Export to CSV ---
% writematrix([t, u], 'frf_chirp_data.csv');
