import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy, moment

def calculate_features_table(channel_data):
    channel_features = []
    channel_data = np.nan_to_num(channel_data)  # Replacing NaN values with zeros
    hist, bin_edges = np.histogram(channel_data, bins='auto', density=True)
    f = np.fft.fftfreq(len(channel_data), 1/200)  # Sampling frequency: 200 Hz
    f_prime = np.fft.fft(channel_data)
    f_prime_conj = np.conj(f_prime)
    channel_features.append(np.nanstd(channel_data))
    channel_features.append(np.nanmean(channel_data))
    channel_features.append(np.nanmax(channel_data))
    channel_features.append(np.nanmin(channel_data))
    channel_features.append(np.nanvar(channel_data))
    channel_features.append(np.nanmedian(channel_data))
    channel_features.append(skew(channel_data, bias=False))
    channel_features.append(kurtosis(channel_data, fisher=True, bias=False))
    channel_features.append(entropy(hist, base=2))
    channel_features.append(moment(channel_data, moment=4, nan_policy='omit'))
    channel_features.append(np.sum(f_prime * f_prime_conj).real)
    
    return channel_features


def extract_features_all_samples(df, top_channels_df):
    feature_columns = ['std', 'mean', 'max', 'min', 'var', 
                       'med', 'skew', 'kurt', 'ent', 'mom', 'pow']
    features_df = pd.DataFrame(columns=feature_columns)
    for sample_id in top_channels_df.index: 
        top_channels = top_channels_df.loc[sample_id].dropna().values.tolist() 
        if not top_channels:  
            continue
        sample_data = df.loc[[str(sample_id)]]
        print(f"Processing sample_id: {sample_id}")

        channel_features = []
        for channel in top_channels:
            channel_data = sample_data[channel]
            channel_features.append(calculate_features_table(channel_data))

        print(f"channel features for one sample: {channel_features}")
        channel_features_transposed = np.array(channel_features).T
        print(f"Transposed features: {channel_features_transposed}")
        mean_features = np.mean(channel_features_transposed, axis=1)
        print(f"averaged features for one sample: {mean_features}")
        features_df.loc[sample_id, :] = mean_features
    return features_df


