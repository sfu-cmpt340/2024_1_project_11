import pandas as pd
import numpy as np
import mne

def calculate_top_channels(sample_data, n=3):
    """Calculate the top channels with the highest variance for a single sample
    
    Parameters
    -------------------------
    sample_data : pd.DataFrame
        EEG data for a single sample, with channels as columns
    n : int, optional
        Number of top channels to retrieve (default is 3)
        
    Returns
    -------------------------
    list
        List of top channels with highest variance for the single sample
    """
    top_channels = []
    for column in sample_data.columns:  
        variances = np.var(sample_data[column])  
        top_channels.append((column, variances))  # Append channel name and its variance
    
    # Sort channels by variance in descending order and get top n channels
    top_channels.sort(key=lambda x: x[1], reverse=True)
    print("All channels:", top_channels)  
    top_n_channels = [channel[0] for channel in top_channels[:n]]
    
    return top_n_channels

