%% For testing correctness of values obtained from feature extraction

% Load actual Fp1 data
data = load('Fp1_data.mat');
Fp1_data = data.Fp1_data;
% Load few data from Fp1 data
data_test = load('Fp1_test.mat');
Fp1_test = data_test.Fp1_test;

%% Entropy
channel_entropy = {};
% Compute histogram with a specified number of bins
[n, edges] = histcounts(Fp1_data, 125, 'Normalization', 'probability');
% Compute entropy for each bin and sum them up
entropy_value = -sum(n .* log2(n + eps));
channel_entropy{1} = entropy_value;
% Display the entropy value
disp(['Channel Entropy:', num2str(entropy_value)]);

%% Power
% Compute the Fourier transform of the signal
f = fft(Fp1_data);
% Compute the complex conjugate of the Fourier transform
f_conj = conj(f);
% Compute the power using the provided equation
power = sum(f .* f_conj);
% Display the computed power
disp('EEG Signal Power:');
disp(power);

%% Moment
% Set order
order = 4;
% Compute the moment of the data vector
m = moment(Fp1_data, order);
% Display the computed moment
disp(['Moment with order of ', num2str(order), ': ', num2str(m)]);

%% Kurtosis
% Compute the mean (mu) and standard deviation (sigma) of the data
mu = mean(Fp1_data);
sigma = std(Fp1_data);
% Compute the kurtosis using mu and sigma
kurtosis = mean((Fp1_data - mu).^4) / sigma^4;
disp("Kurtosis:");
disp(kurtosis);

%% Skewness
mu = mean(Fp1_data);
sigma = std(Fp1_data);
% Compute the skewness using the formula derived from below source which our project's feature extraction heavily relies on
% https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7881082/#CR17
s = mean((Fp1_data - mu).^3) / sigma^3;
disp("Skewness:");
disp(s);

%% Mean
disp("Mean:");
disp(mu);

%% Standard deviation
% Compute the square of the differences between each sample and the mean
diff_squared = (Fp1_data - mu).^2;
% Compute the sum of the squared differences
sum_diff_squared = sum(diff_squared);
% Compute the standard deviation using the provided formula
N = length(Fp1_data);
sigma = sqrt(sum_diff_squared / (N - 1));
% Display the computed standard deviation
disp(['Standard deviation:', num2str(sigma)]);

%% Median
% Sort the data vector in ascending order
sorted_data = sort(Fp1_data);
% Compute the median using the formula
if mod(N, 2) == 0
    % If the number of elements is even, take the average of the middle two elements
    median_value = (sorted_data(N/2) + sorted_data(N/2 + 1)) / 2;
else
    % If the number of elements is odd, take the middle element
    median_value = sorted_data((n + 1) / 2);
end
% Display the computed median
disp(['Median:', num2str(median_value)]);

%% Maximum EEG signal
% Compute the maximum value of the signal using the provided formula
M = max(Fp1_data);
% Display the computed maximum value
disp(['Maximum EEG signal:', num2str(M)]);

%% Minimum EEG signal
% Compute the minimum value of the signal using the provided formula
M = min(Fp1_data);
% Display the computed minimum value
disp(['Minimum EEG signal:', num2str(M)]);

%% Variance
% Calculate the variance using the equation v = sigma^2
variance = sigma^2;
% Display the result
disp(['Variance: ', num2str(variance)]);

