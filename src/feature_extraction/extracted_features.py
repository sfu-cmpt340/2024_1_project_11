import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy, moment
import mne


# std_calculation

def calculate_std(sig1, channels_lists):
    """
    Calculate the standard deviation for specified channels in a given signal.
    
    Parameters:
    - sig1: DataFrame or structured array containing the signal data.
    - channels_lists: List of channels for which to calculate the standard deviation.
    
    Returns:
    - Dictionary with channels as keys and their standard deviation as values.
    """
    channel_std_sig1 = {}
    for channel in channels_lists:
        channel_data = sig1[channel].values
        std_value = np.nanstd(channel_data)
        channel_std_sig1[channel] = std_value
    return channel_std_sig1


# mean_calculation

def calculate_mean(sig, channels):
    """
    Calculate the mean for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the mean.

    Returns:
    - Dictionary with channels as keys and their mean as values.
    """
    channel_means = {}
    for channel in channels:
        channel_data = sig[channel].values
        mean_value = np.nanmean(channel_data)
        channel_means[channel] = mean_value
    return channel_means


# Add to calculations.py or create a new file, e.g., median_calculation.py

def calculate_med(sig, channels):
    """
    Calculate the median for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the median.

    Returns:
    - Dictionary with channels as keys and their median as values.
    """
    channel_median = {}
    for channel in channels:
        channel_data = sig[channel].values
        median_value = np.nanmedian(channel_data)
        channel_median[channel] = median_value
    return channel_median


def calculate_kurtosis(sig, channels):
    """
    Calculate the kurtosis for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the kurtosis.

    Returns:
    - Dictionary with channels as keys and their kurtosis as values.
    """
    channel_kurtosis = {}
    for channel in channels:
        channel_data = sig[channel].values  
        # 'fisher=True' returns Fisher's definition of kurtosis (kurtosis of normal == 0.0).
        # 'bias=False' uses the unbiased estimator.
        kurtosis_value = kurtosis(channel_data, fisher=True, bias=False)
        channel_kurtosis[channel] = kurtosis_value
    return channel_kurtosis


# skewness

def calculate_skewness(sig, channels):
    """
    Calculate the skewness for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the skewness.

    Returns:
    - Dictionary with channels as keys and their skewness as values.
    """
    channel_skewness = {}
    for channel in channels:
        channel_data = sig[channel].values
        skewness_value = skew(channel_data, bias=False)  # 'bias=False' for unbiased skewness calculation
        channel_skewness[channel] = skewness_value
    return channel_skewness

# extracted_features.py

def calculate_max_signal(sig, channels):
    """
    Calculate the maximum value for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the maximum value.

    Returns:
    - Dictionary with channels as keys and their maximum value as values.
    """
    channel_max_signal = {}
    for channel in channels:
        channel_data = sig[channel].values
        max_signal_value = np.nanmax(channel_data)  # Use nanmax to ignore NaN values
        channel_max_signal[channel] = max_signal_value
    return channel_max_signal


def calculate_min_signal(sig, channels):
    """
    Calculate the minimum value for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the minimum value.

    Returns:
    - Dictionary with channels as keys and their minimum value as values.
    """
    channel_min_values = {}
    for channel in channels:
        channel_data = sig[channel].values
        min_value = np.nanmin(channel_data)  
        channel_min_values[channel] = min_value
    return channel_min_values

# variance_calculation
def calculate_variance(sig, channels):
    """
    Calculate the entropy for specified channels in a given signal.

    Parameters:
    - sig: DataFrame or structured array containing the signal data.
    - channels: List of channels for which to calculate the minimum value.

    Returns:
    - Dictionary with channels as keys and their entropy as values.
    """
    channel_variance = {}
    for channel in channels:
        channel_data = sig[channel].values
        variance_value = np.nanvar(channel_data)
        channel_variance[channel] = variance_value
    return channel_variance

# all_features_calculations
def calculate_features_table(sig, channels):
    """
    Calculate various statistical features for specified channels in a given signal.
    Parameters:
    - sig: DataFrame containing the signal data.
    - channels: List of channels for which to calculate the features.
    Returns:
    - Numpy array with each row containing features of a single channel.
    """
    # Define the features you will calculate
    features = ['STD', 'Mean', 'Max', 'Min', 'Var', 'Med', 'SKW', 'ENT', 'KRT', 'MOM', 'POW']
    data = []

    for channel in channels:
        channel_data = sig[channel].dropna().values  

        # Compute probability distribution for entropy
        hist, bin_edges = np.histogram(channel_data, bins='auto', density=True)

        # Compute frequency vector, FFT (Fast Fourier Transform) and its conjugate for power
        f = np.fft.fftfreq(len(channel_data), 1/200) # Sampling frequency: 200 Hz
        f_prime = np.fft.fft(channel_data)
        f_prime_conj = np.conj(f_prime)

        std_value = np.nanstd(channel_data)
        mean_value = np.nanmean(channel_data)
        max_value = np.nanmax(channel_data)
        min_value = np.nanmin(channel_data)
        var_value = np.nanvar(channel_data)
        med_value = np.nanmedian(channel_data)
        skew_value = skew(channel_data, bias=False)
        ent_value = entropy(hist, base=2)
        kurt_value = kurtosis(channel_data, fisher=True, bias=False)
        mom_value = moment(channel_data, moment=4, nan_policy='omit') # Quantitave measure of distribution shape, fourth order moment is related to kurtosis
        pow_value = np.sum(f_prime * f_prime_conj)
        
        # Create a features vector for the channel and append it to the data list
        channel_features = np.array([std_value, mean_value, max_value,min_value, var_value, med_value, skew_value, ent_value, kurt_value, mom_value, pow_value])
        data.append(channel_features)

    # Convert the list of feature vectors into a 2D numpy array
    features_array = np.array(data)
    
    return features_array
