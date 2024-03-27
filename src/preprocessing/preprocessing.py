import mne
import numpy as np
import pandas as pd
from mne.decoding import Scaler

def notch_filter(raw, freqs):
  """
  params
  ------------------------
  raw: MNE Raw Array object
  freqs: List of frequencies to filter out
  """
  eeg_picks = mne.pick_types(raw.info, eeg=True)
  raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks)
  return raw_notch

def standardize(signal):
  """
  params
  -------------------------
  signal: pd.DataFrame
  """
  data = signal.to_numpy()
  scaler = Scaler(scalings='mean')
  scaled = scaler.fit_transform(data.reshape(1, *data.shape))
  scaled = scaled.reshape(data.shape)
  df_scaled = pd.DataFrame(scaled, index = signal.index, columns = signal.columns)
  return df_scaled