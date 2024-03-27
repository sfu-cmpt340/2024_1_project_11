import mne
import numpy as np
import pandas as pd
from mne.decoding import Scaler

def notch_filter(raw, freqs):
  '''
  params
  ------------------------
  raw: MNE Raw Array object
  freqs: List of frequencies to filter out
  '''
  eeg_picks = mne.pick_types(raw.info, eeg=True)
  raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks)
  return raw_notch