#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_metadata():
  file_path = Path("data/train.csv")
  return pd.read_csv(file_path)
  
metadata = load_metadata()

def extract_eeg():
  eeg_dir = Path("../data/eeg")
  tarball_path = Path("data/eeg.tar.gz")
  if not tarball_path.is_file():
    url = 'https://dl.dropboxusercontent.com/scl/fi/5sina48c4naaxv6uze0fv/eeg.tar.gz?rlkey=r7ec191extynfcm8fy0tsiws5&dl=0'
    urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as eeg_tarball:
      eeg_tarball.extractall()
    
extract_eeg()

metadata


# In[4]:


# Add a unique id to each sample
from src.utils.utils import compute_signal_hash

def compute_unique_id(row):
  return str(compute_signal_hash(row))

metadata['unique_id'] = metadata.apply(compute_unique_id, axis=1)

cols = metadata.columns.tolist()
cols = [cols[-1]] + cols[:-1]
metadata = metadata[cols]
metadata


# In[5]:


import dask.dataframe as dd
from src.utils import compute_signal_hash

channel_order = ['Fp1', 'Fp2',
            'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T3', 'C3', 'Cz', 'C4', 'T4', 
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2',
          ]

def load_signals(metadata):
  rows = len(metadata)
  eeg_list = []

  for row in range(0,rows):
    sample = metadata.iloc[row]
    f_name = f'data/eeg/{sample.eeg_id}.parquet'
    eeg = pd.read_parquet(f_name)[channel_order]
    eeg_offset = int(sample.eeg_label_offset_seconds)

    eeg['unique_id'] = sample['unique_id']
    eeg = eeg.set_index('unique_id')

    eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
    eeg_list.append(eeg)

  return dd.concat(eeg_list)

ddf = load_signals(metadata)


# In[6]:


df = ddf.compute()
eeg_ids = list(set(df.index))
df


# In[7]:


df = df.interpolate(method='linear', axis=0)


# In[8]:


# Apply filtering
from src.preprocessing.preprocessing import bp_filter, notch_filter
from scipy.signal import iirnotch, filtfilt, butter

def apply_notch_filter(df, fs, f0):
  filtered = df.copy()
  b,a = iirnotch(f0, 30, fs)
  for column in filtered.columns:
    filtered[column] = filtfilt(b, a, df[column])
  return filtered

def apply_bp_filter(df, fs, lowcut, highcut):
  filtered = df.copy()
  nyq = 0.5*fs
  low = lowcut / nyq
  high = highcut/nyq
  b,a = butter(5, [low,high], btype='band')
  for channel in filtered.columns:
    filtered[channel] = filtfilt(b,a, df[channel])
  return filtered


filtered_data = apply_notch_filter(df, 200, 60)
filtered_data = apply_bp_filter(filtered_data, 200, 0.5, 50)


# In[10]:


from src.preprocessing.preprocessing import wavelet_transform
preprocessed_data = wavelet_transform(filtered_data, 2, basis='bior3.3')


# Extracting top 3 channels based on max variance for all samples
# - 1000 samples computation duration = approx. 15 minutes

# In[11]:


from src.feature_extraction import calculate_all_samples

top_channels_df = calculate_all_samples(filtered_data, eeg_ids, len(eeg_ids))
top_channels_df


# Extracting Statistical Features from every sample with extraction function
# - 1000 samples computation duration = approx. 20 minutes

# In[12]:


from src.feature_extraction import extract_features_all_samples

features_df = extract_features_all_samples(df, top_channels_df)
features_df


# Setting up Feature Data and Target Data for correct format to split data and Input to Microsoft's Light Gradient Boosting Machine (LGBM)

# In[13]:


# Setup feature table
# input_df = pd.merge(features_df, metadata[['unique_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']], left_index=True, right_on='unique_id')
input_df = pd.merge(features_df, metadata[['unique_id', 'expert_consensus']], left_index=True, right_on='unique_id')
input_df = input_df.set_index('unique_id')
input_df


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


x = input_df.iloc[:, :11].astype(float)
y = input_df[['expert_consensus']].to_numpy().flatten()

categories = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

le = LabelEncoder()
le.fit(categories)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=92)


# In[39]:


import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

params = {
    'objective': 'multiclass',
    'num_class': 6,
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 121,
    'learning_rate': 0.018623105710769177,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.756777580360579,
    'max_depth': 8,
    'verbose': -1
}
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train, y_train)


# Inputting Parameters for LGBM Model
# - parameters were obtained by observing similiar implementation in same competition project using LGBM library. (see report doc --> citations/acknowledgements for more details) 
# - Slight adjustments to parameters applied to fit our implementation

# In[40]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy


# Splitting Data 80/20 and adjusting params to obtain training and testing sets

# Training our LGBM model on the training data and evaluating it on the test data - Probabilities for each target label are obtained and Displayed

# Generating Confusion Matrix with predicted labels and true labels

# In[37]:


# Generate confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = lgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm)

cm_df.columns = le.inverse_transform([0,1,2,3,4,5]).tolist()
cm_df.index = le.inverse_transform([0,1,2,3,4,5]).tolist()

hm = sns.heatmap(cm_df, annot=True, cmap='Blues')
hm.set_xlabel('Predicted')
hm.set_ylabel('Actual')

hm.xaxis.set_label_position('top')
hm.xaxis.tick_top()


# Generating result metrics to evaluate our multiclassification model

# In[35]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[36]:


y_pred_proba = lgb_model.predict_proba(X_test)

# Predicted probabilities to DataFrame
pred_df = pd.DataFrame(y_pred_proba, columns=['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'])
pred_df['eeg_id'] = X_test.index
pred_df = pred_df[['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
pred_df


# In[20]:


# File created to test the correctness of extracted values using MATLAB
# Save Fp1 channel data into a MATLAB file
# import scipy.io
# scipy.io.savemat('Fp1_data.mat', {'Fp1_data': sig1['Fp1']})


# In[ ]:


import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(lgb_model, file)

