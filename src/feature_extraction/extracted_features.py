import pandas as pd
import numpy as np
import mne


# std_calculation.py

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


# mean_calculation.py

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
