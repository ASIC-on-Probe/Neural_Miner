clc;
clear all;
close all;

% Load the Excel file
filename = 'E:\QQPCmgr\Desktop\SNR-1.xlsx';
data = readtable(filename);

% Extract signal data (assuming first column is time, rest are channels)
signal_data = table2array(data(:, 2:end));

% Define original sampling rate and new downsampled rate
fs_original = 20000; % Original sampling frequency (Hz)
fs_down = 1000; % Target downsampling frequency (Hz)
down_factor = fs_original / fs_down; % Compute downsampling factor

% Low-pass filter (200 Hz) using Butterworth filter
[b_low, a_low] = butter(4, 200 / (fs_original / 2), 'low'); % 4th order low-pass filter

% Define bandpass filters for Theta, Alpha, Beta, Gamma bands
bands = struct('Theta', [4, 8], ...
               'Alpha', [8, 12], ...
               'Beta', [12, 30], ...
               'Gamma', [30, 120]);

num_channels = size(signal_data, 2);
num_bands = length(fieldnames(bands));

% Initialize band power storage
band_power = zeros(num_channels, num_bands);

% Process each channel
for ch = 1:num_channels
    signal = signal_data(:, ch);
    
    % Apply low-pass filter
    signal_filtered = filtfilt(b_low, a_low, signal);
    
    % Downsample the signal to 1 kHz
    signal_down = downsample(signal_filtered, down_factor);
    
    % Compute band power for each frequency band
    b_idx = 1;
    for band_name = fieldnames(bands)'
        band_range = bands.(band_name{1});

        % Apply bandpass filter for this band
        [b_band, a_band] = butter(4, band_range / (fs_down / 2), 'bandpass');
        signal_band = filtfilt(b_band, a_band, signal_down);
        
        % Compute band power
        band_power(ch, b_idx) = bandpower(signal_band, fs_down, band_range);
        b_idx = b_idx + 1;
    end
end

% Convert results to table
band_power_table = array2table(band_power, 'VariableNames', fieldnames(bands));
band_power_table.Channel = (1:num_channels)'; % Add channel numbers

% Display results
disp('Band Power (in ¦ÌV^2) for each channel:');
disp(band_power_table);
writetable(band_power_table, 'E:\QQPCmgr\Desktop\band_power.xlsx');