import mne
import numpy as np
import pandas as pd
import pywt
from mne.decoding import Scaler
from torch import threshold

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

def soft_threshold(data, threshold):
  return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

def estimate_noise(detail_coeffs):
  return np.median(np.abs(detail_coeffs)) / 0.6745

def universal_threshold(signal_length, sigma):
  return sigma * np.sqrt(2 * np.log(signal_length))

def wavelet_transform(df, level, basis='db4'):
  signals = list(set(df.index.to_list()))

  df_copy = df.copy()

  for signal_id in signals:
    curr_signal = df.loc[signal_id].copy()
    
    for channel in curr_signal:
      chan_np = curr_signal[channel].to_numpy()
      coeffs = pywt.wavedec(chan_np, basis, level=level)

      sigma = estimate_noise(coeffs[-1])
      threshold = universal_threshold(chan_np.size, sigma)

      # Apply soft thresholding to detail coefficients
      coeffs[1:] = [soft_threshold(detail_coeff, threshold) for detail_coeff in coeffs[1:]]
      cleaned_channel = pywt.waverec(coeffs, basis)

      if len(cleaned_channel) > len(chan_np):
        cleaned_channel = cleaned_channel[:len(chan_np)]
      elif len(cleaned_channel) < len(chan_np):
        cleaned_channel = np.append(cleaned_channel, np.zeros(len(chan_np) - len(cleaned_channel)))
        
      curr_signal[channel] = cleaned_channel
    df_copy.loc[signal_id] = curr_signal
  return df_copy