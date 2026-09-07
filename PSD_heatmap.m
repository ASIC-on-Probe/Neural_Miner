clc;
clear all;
close all;

% Load the Excel file
filename = 'E:\QQPCmgr\Desktop\16ch-day 1-1.xlsx';
data = readtable(filename);

% Extract time and signal channels
time = data.time; % Assuming the first column is time
signal_data = table2array(data(:, 2:end)); % Extract all channel data

% Estimate the sampling rate (assuming uniform time steps)
fs = 20000; % Sampling frequency in Hz

% Number of channels
num_channels = size(signal_data, 2);

nfft = 8192;
psd_matrix = [ ];
freq_vector = [ ];
 for ch = 1:num_channels
     signal = signal_data(: , ch);
     [Pxx, f] = pwelch(signal, hamming(nfft), [], [], fs);
     
     if isempty(freq_vector)
         freq_vector = f;
     end
     
     psd_matrix(:, ch) = 10*log10(Pxx);
 end
 
 freq_limit = 1500;
 freq_idx = freq_vector <= freq_limit;
 psd_matrix = psd_matrix(freq_idx, :);
 freq_vector = freq_vector(freq_idx, :);
 
 figure;
 imagesc(1:num_channels, freq_vector, psd_matrix);
 colorbar;
 xlabel('Channel');
 ylabel('Frequency');
 set(gca, 'YDir', 'normal');
 ylim([0, 1500]);
 colormap jet;