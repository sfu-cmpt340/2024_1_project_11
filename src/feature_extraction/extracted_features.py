import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy, moment
import mne

def calculate_features_table(channel_data):
    """Calculate features for a single EEG channel

    Parameters
    ----------
    channel_data : array-like
        EEG data for a single channel

    Returns
    -------
    list
        List of calculated features including standard deviation, mean, maximum, minimum,
        variance, median, skewness, kurtosis, entropy, 4th moment, and power.
    """
    channel_features = []
    channel_data = np.nan_to_num(channel_data)  

    # For entropy
    # Compute probability distribution
    hist, bin_edges = np.histogram(channel_data, bins='auto', density=True) 
    # Normalize the histogram counts to form a probability distribution
    hist = hist / np.sum(hist)
    # Compute entropy for each bin and sum them up
    entropy_value = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    # For moment
    # Choose the order of interest to compute the nth central moment
    order = 4

    # For power
    # Compute frequency vector, representing the corresponding frequencies for each f_prime
    f_prime = np.fft.fft(channel_data)
    f_prime_conj = np.conj(f_prime)
    # Useful variable for frequency band analysis and filtering operations (Sampling frequency: 200 Hz) for future use
    # f = np.fft.fftfreq(len(channel_data), 1/200) 
    channel_features.append(np.nanstd(channel_data))
    channel_features.append(np.nanmean(channel_data))
    channel_features.append(np.nanmax(channel_data))
    channel_features.append(np.nanmin(channel_data))
    channel_features.append(np.nanvar(channel_data))
    channel_features.append(np.nanmedian(channel_data))
    channel_features.append(skew(channel_data, bias=False))
    channel_features.append(kurtosis(channel_data, fisher=False, bias=False))
    channel_features.append(entropy_value)
    channel_features.append(np.mean((channel_data - np.mean(channel_data))**order))
    channel_features.append(np.sum(f_prime * f_prime_conj).real) # power feature

    return channel_features


def extract_features_all_samples(df, top_channels_df):
    """Extract average features for each sample from the top channels

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing EEG data for all samples, with samples as rows and channels as columns
    top_channels_df : pd.DataFrame
        DataFrame containing the top channels for each sample

    Returns
    -------
    pd.DataFrame
        DataFrame containing the average features for each sample, with samples as rows and features as columns
    """
    feature_columns = ['std', 'mean', 'max', 'min', 'var', 
                       'med', 'skew', 'kurt', 'ent', 'mom', 'pow']
    features_df = pd.DataFrame(columns=feature_columns)
    for sample_id in top_channels_df.index: 
        top_channels = top_channels_df.loc[sample_id].dropna().values.tolist() 
        if not top_channels:  
            continue
        sample_data = df.loc[[str(sample_id)]]
        channel_features = []
        for channel in top_channels:
            channel_data = sample_data[channel]
            channel_features.append(calculate_features_table(channel_data)) # 33 features per sample

        channel_features_transposed = np.array(channel_features).T # Transposing feature list for Averaging
        mean_features = np.mean(channel_features_transposed, axis=1) # 11 features per sample
        features_df.loc[sample_id, :] = mean_features

    return features_df


