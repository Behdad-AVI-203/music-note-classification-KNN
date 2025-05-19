clc;
clear;
close all;

training_data_path = 'Project_Part2/learning_data';
test_data_path = 'Project_Part2/test_data';

training_files = dir(fullfile(training_data_path, '*.wav'));
num_training = length(training_files);

test_files = dir(fullfile(test_data_path, '*.wav'));
num_test = length(test_files);

training_peaks = zeros(1, num_training);
training_labels = strings(1, num_training);
test_peaks = zeros(1, num_test);

for i = 1:num_training
    [audio, fs] = audioread(fullfile(training_data_path, training_files(i).name));
    N = length(audio);
    Y = fft(audio);
    f = (0:N-1) * (fs / N);
    magnitude = abs(Y(1:floor(N/2)));
    [~, peak_idx] = max(magnitude);
    training_peaks(i) = f(peak_idx);
    [~, name, ~] = fileparts(training_files(i).name);
    training_labels(i) = name;
end

for i = 1:num_test
    [audio, fs] = audioread(fullfile(test_data_path, test_files(i).name));
    N = length(audio);
    Y = fft(audio);
    f = (0:N-1) * (fs / N);
    magnitude = abs(Y(1:floor(N/2)));
    [~, peak_idx] = max(magnitude);
    test_peaks(i) = f(peak_idx);
end

K = 5;

test_labels = strings(1, num_test);
for i = 1:num_test
    distances = abs(training_peaks - test_peaks(i));
    [~, sorted_indices] = sort(distances);
    nearest_labels = training_labels(sorted_indices(1:K));
    test_labels(i) = mode(categorical(nearest_labels));
end

disp('Test Data Classification Results:');
for i = 1:num_test
    fprintf('File: %s, Predicted Label: %s\n', test_files(i).name, test_labels(i));
end

figure;
subplot(2, 1, 1);
bar(1:num_training, training_peaks, 'b');
title('Frequency Peaks of Training Data');
xlabel('Training File Index');
ylabel('Frequency (Hz)');
grid on;

subplot(2, 1, 2);
bar(1:num_test, test_peaks, 'r');
title('Frequency Peaks of Test Data');
xlabel('Test File Index');
ylabel('Frequency (Hz)');
grid on;
