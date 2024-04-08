import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

def plot_signal(df, duration, channels):
  sig_duration = 50
  fs = 200
  N = fs*sig_duration

  samples = np.arange(N)
  time = samples/fs
  base_offset = 150

  # Apply an increasing offset for each channel
  for i, channel in enumerate(channels):
    offset = i * base_offset  # Increase the offset for each channel
    plt.plot(time, df[channel] + offset, label=channel)

  plt.xlim(0, duration)
  plt.show()
  plt.clf()

class VisualizeEEG:
  """Visualization class for given EEG signals

  Parameters
  -------------------------
  df: DataFrame
    EEG signal in a Pandas DataFrame format

  Attributes
  -------------------------
  _info: mne.Info 
    Measurement information/recording metadata
  _data: {array-like}, shape = [num_channels, num_rows]
    NumPy representation of EEG signals
  _raw: mne.RawArray
    Raw object from Numpy array. Used for MNE visualization methods
  """
  def __init__(self, df):
    self.info_ = mne.create_info(ch_names=df.columns.tolist(), sfreq=200, ch_types='eeg')
    self.info_.set_montage('standard_1020')
    
    self.data_ = np.array(df.transpose())
    self.data_ = np.nan_to_num(self.data_)
    
    self.raw_ = mne.io.RawArray(self.data_, self.info_)
    self.raw_.apply_function(lambda x: x / 20e6, picks='eeg')

  def plot_signal(self, start, duration, scaling=None, n_channels=19):
    """ Multi-channel time series plot

    Parameters
    -------------------------
    start: Float
      Start time
    duration: Float
      Duration of time
    n_channels: Int
      Number of channels to plot
    """



  def plot_topomap(self, start, end, delta):
    """ Topographic mapping of evoked signal
    
    Parameters
    -------------------------
    start: Float
      Start time
    end: Float
      End time
    delta: Float
      Length of time steps
    """
    evoked = mne.EvokedArray(self.raw_.get_data(), self.info_, baseline=(0,0))
    times = np.arange(start, end, delta)
    evoked.plot_topomap(ch_type="eeg", times=times)

