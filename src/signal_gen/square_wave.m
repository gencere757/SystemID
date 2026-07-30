%% generate_square_wave.m
% Generates a square wave timeseries with user-defined sampling time,
% magnitude, and simulation duration. Useful as a synthetic input signal
% for system identification testing.

clear; clc; close all;

%% --- User-defined parameters ---
Ts      = 6.6667e-05;   % Sampling time [s]
simTime = 10;      % Total simulation time [s]
mag     = 400;      % Square wave magnitude (amplitude), signal will swing +/- mag/2 about offset
offset  = 0;       % DC offset (mean level) of the square wave
freq    = 5;       % Square wave frequency [Hz]
duty    = 50;       % Duty cycle [%] (percentage of period the signal is "high")

%% --- Time vector ---
t = (0:Ts:simTime)';   % column vector of time samples

%% --- Generate square wave ---
% square(t, duty) returns a wave in [-1, 1] with given duty cycle (%)
u = offset + (mag/2) * square(2*pi*freq*t, duty);

%% --- Package as timeseries object ---
u_timeseries = timeseries(u, t, 'Name', 'SquareWaveInput');
u_timeseries.DataInfo.Units = 'amplitude';
u_timeseries.TimeInfo.Units = 'seconds';

%% --- Plot ---
figure;
plot(t, u, 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Amplitude');
title(sprintf('Square Wave: freq=%.2f Hz, mag=%.2f, duty=%d%%, Ts=%.3f s', ...
    freq, mag, duty, Ts));
grid on;
ylim([offset - mag/2 - 0.1*mag, offset + mag/2 + 0.1*mag]);

%% --- (Optional) Save to .mat file ---
save('data\unknown_test\square_wave_data.mat', 't', 'u', 'u_timeseries');

%% --- (Optional) Export to CSV ---
% writematrix([t, u], 'square_wave_data.csv');
