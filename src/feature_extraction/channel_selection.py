import pandas as pd
import numpy as np
import itertools
import mne

def calculate_top_channels(sample_data, n=3):
    """Calculate the top channels with the highest variance for a single sample

    Parameters
    ----------
    sample_data : pd.DataFrame
        EEG data for a single sample, with channels as columns
    n : int, optional
        Number of top channels to retrieve (default is 3)

    Returns
    -------
    list
        List of top channels with highest variance for the single sample
    """
    sample_array = sample_data.to_numpy() # Convert DataFrame to NumPy array
    variances = np.var(sample_array, axis=0) # Calculate variances for each channel
    top_n_indices = np.argsort(variances)[::-1][:n] # Get indices of top 3 channels based on variance
    top_n_channels = sample_data.columns[top_n_indices].tolist() # top 3 channels of the sample to list
    
    return top_n_channels

def calculate_all_samples(df, sample_ids, n):
    """
    Select the top channels for each sample in the DataFrame.

    Parameters:
    -------------------------
    df : pd.DataFrame
        DataFrame containing EEG data with sample IDs as index.
    sample_ids : list
        List of sample IDs to process.
    n : int, optional
        Number of samples to select

    Returns:
    -------------------------
    pd.DataFrame
        DataFrame where each row represents a sample and columns represent top channels.
    """
    top_channels_df = pd.DataFrame(index=sample_ids, columns=range(3))  # Initialize DataFrame with 3 columns
    for sample_id in itertools.islice(sample_ids, n):
        eeg_sig = df.loc[[str(sample_id)]]
        top_channels = calculate_top_channels(eeg_sig)  # Calculate top channels for the sample
        top_channels_df.loc[sample_id, :3] = top_channels[:3]  # Store top 3 channels in DataFrame
    return top_channels_df

    




