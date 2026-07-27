clc; clear; close all;
%% Creating PRBS Signal With Increasing Frequency and -1 1 amplitude
%Parameters
%Bunlar struct yapılıp ayrı dosyaya çevrilecek yönetim için
%Ayrıca pozitiften pozitife geçen bir forma çevrilecek. Bunun için geçeceği
%min ve max time belirlenecek
simTime = 55;
segmentNumber = 10; %Number of segments with different prbs signals
T = 6.6667e-02; %Sampling time
minFreq = 0.2;  %Minimum frequency of signals
freqIncrement = 0.1;   %Frequency increment rate
maxMagnitude = 500;  %Maximum magnitude of signals
minMagnitude = 0;


segmentTime = simTime / segmentNumber;  %Time for a single segment in seconds
segTimestepNo = round(segmentTime / T);  %number of timesteps in a segment
numSamples = round(simTime/(T*segmentNumber))+1; %Number of samples for a single segment
signals = zeros(segmentNumber,numSamples);   %Vector to store signals
for i = 1:segmentNumber
    freq = min(minFreq + freqIncrement * (i-1),1); %Determine frequency for the current signal
    signals(i,:) = idinput(numSamples,"prbs",[0 freq],[-1 1]);  %Generate PRBS with current frequency
end

prbs = reshape(signals',1,[]);   %The final PRBS Signal

%% Adding amplitude modulation
N = length(prbs);

% Find indices where prbs switches level (start of each constant-value run)
switchPoints = [1, find(diff(prbs) ~= 0) + 1, N + 1];
numHolds = length(switchPoints) - 1;

% Draw one random amplitude per PRBS "pulse" (i.e., per hold interval)
holdAmplitudes = rand(1, numHolds) * (maxMagnitude - minMagnitude) + minMagnitude;

% Quantize
num_levels = 256;
holdAmplitudes = round(holdAmplitudes * (num_levels - 1)) / (num_levels - 1);

% Build the envelope by repeating each amplitude across its hold interval
amplitude_envelope = zeros(1, N);
for k = 1:numHolds
    amplitude_envelope(switchPoints(k):switchPoints(k+1)-1) = holdAmplitudes(k);
end

% Multiply PRBS by amplitude envelope
aprbs = prbs .* amplitude_envelope;

%% Plot
figure;
subplot(2,1,1);
plot(prbs);
xlabel("Timestep");
ylabel("Input Signal");
title("Varying Frequency Constant Amplitude Prbs Signal");
grid on;

subplot(2,1,2);
plot(aprbs);
xlabel("Timestep");
ylabel("Input Signal");
title("Varying Amplitude PRBS Signal");
yline(0);
grid on;
