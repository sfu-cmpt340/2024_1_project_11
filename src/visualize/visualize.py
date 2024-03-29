import pandas as pd
import numpy as np
import mne

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
    self.raw_.plot(start=start, duration=duration, scalings={'eeg': 'auto'}, n_channels=n_channels);

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

