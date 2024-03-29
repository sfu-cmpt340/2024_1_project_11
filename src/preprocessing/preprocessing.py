import mne
import numpy as np
import pandas as pd
from mne.decoding import Scaler

def convert_to_mne(df):
  mne_info = mne.create_info(ch_names=df.columns.tolist(), sfreq=200, ch_types='eeg')
  mne_info.set_montage('standard_1020')
      
  data = np.array(df.transpose())
  data = np.nan_to_num(data)
      
  raw = mne.io.RawArray(data, mne_info)
  raw.apply_function(lambda x: x / 20e6, picks='eeg')

  return raw

def notch_filter(df, freqs):
  raw = convert_to_mne(df)
  eeg_picks = mne.pick_types(raw.info, eeg=True)
  raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks)
  
  data = raw_notch.get_data().reshape(df.shape)
  return pd.DataFrame(data, index=df.index, columns=df.columns)

def bp_filter(df, l_freq, h_freq, method='fir'):
  raw = convert_to_mne(df)
  raw_filtered = raw.copy().filter(l_freq, h_freq, method=method)
  
  data = raw_filtered.get_data().reshape(df.shape)
  return pd.DataFrame(data, index=df.index, columns=df.columns)

def standardize(df):
  data = df.to_numpy()
  scaler = Scaler(scalings='mean')
  scaled = scaler.fit_transform(data.reshape(1, *data.shape))
  scaled = scaled.reshape(data.shape)
  df_scaled = pd.DataFrame(scaled, index = df.index, columns = df.columns)
  return df_scaled
