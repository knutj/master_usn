import os
import xgboost as xgb
import seaborn as sms
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from optuna.logging import set_verbosity, WARNING
os.system('cls' if os.name == 'nt' else 'clear')
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm
from optuna.logging import set_verbosity, CRITICAL
import sys
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import TimeWarp, Drift, Reverse, Quantize, AddNoise
#%%
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
optuna.seed = seed

os.environ["OMP_NUM_THREADS"] = "1"    # limit OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

figure_path=os.path.join(os.getcwd(), 'figures','xgo_boost_time_ret')
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

model_path=os.path.join(os.getcwd(), 'models', 'xgo_boost_time_ret')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_path=os.path.join(os.getcwd(), 'data', 'xgo_boost_time_ret')
if not os.path.exists(data_path):
    os.makedirs(data_path)
from dataclass import *
from oversample import * 
from prosess_data import *
from helper import *
from data_import import * 
from model import *
#%%
# --- Step 1: Load the Data ---
#df = pd.read_csv("data/cleaned_data.csv.gz")
df = pd.read_csv("data/cleaned_data.csv.gz",nrows=5000)
df = startimportdata_re(df,figure_path,model_path,data_path,"cnn_re")
#%%

#%%
""" class ReadmissionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y,lengths):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, T, F)
        self.y = torch.tensor(y, dtype=torch.long)     # long for classification
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if idx < len(self.X): 
            x_seq = self.X[idx]
            y_val = self.y[idx]
            length = self.lengths[idx]
            return x_seq, length, x_seq, y_val  # return twice for LSTM and CNN input
        else:
            print(f"index {idx} is larger thatn {len(self.X)}")
#%% """



#%%
from sklearn.model_selection import StratifiedGroupKFold
# Create subject-level labels for stratification
subject_labels = df.groupby('id')['label'].max().reset_index()
subject_ids = subject_labels['id'].values
subject_y = subject_labels['label'].values

# Stratified group split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(subject_ids, subject_y, groups=subject_ids))

# Get train/test subject IDs
train_subjects = subject_ids[train_idx]
test_subjects = subject_ids[test_idx]

# Filter data
train_mask = df['id'].isin(train_subjects)
test_mask = df['id'].isin(test_subjects)

# Extract features and labels
X_train_all_ = df[train_mask].drop(columns=['label']).reset_index(drop=True)
X_test_ = df[test_mask].drop(columns=['label']).reset_index(drop=True)

y_train_all_ = df.loc[train_mask, ['label']].reset_index(drop=True)
y_test_ = df.loc[test_mask, ['label']].reset_index(drop=True)




#%%
sequence_length = 8
(X_test, y_test, lengths_test, group_test) = prepare_data(X_test_, y_test_, sequence_length)
print(X_test.shape)
test_data = ReadmissionDataset(X_test, y_test, lengths_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
num_features = X_test.shape[2]
#%%
(X_train_all,y_train_all,lengths_train,group_train) = prepare_data(X_train_all_, y_train_all_,sequence_length)

x_df = X_train_all
y_df = y_train_all
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import contextlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc

class_weight_scale = 1 
param={
    'objective':'binary:logistic',
    'nthread':4,
    'scale_pos_weight':class_weight_scale,
    'seed' : 1,
    'device':'cpu'
}
    
xgb1 = XGBClassifier(enable_categorical = True)
xgb1.set_params(**param)

def interp(x_new, x_old, y_old):
    return np.interp(x_new, x_old, y_old)


# Initialize XGBClassifier with categorical support
xgb1 = XGBClassifier(enable_categorical=True, eval_metric='logloss', device = "cuda")

# Parameters for cross-validation and ROC plotting
K = 5
skf = StratifiedKFold(n_splits=K)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
i = 0
roc_aucs_xgb1 = []

fig = plt.figure(figsize=(7, 7))

# Cross-validation loop
for train_indices, test_indices in skf.split(x_df, y_df['label']):
    X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
    X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]

    

    # Compute class weight scale for imbalanced data
    class_weight_scale = 1. * y_train.value_counts()[0] / y_train.value_counts()[1]
    print('class weight scale : {}'.format(class_weight_scale))
    class_weight_scale = 1
    xgb1.set_params(scale_pos_weight=class_weight_scale)

    #dtrain = xgb.DMatrix(X_train, label=y_train, device='cuda')
    #dtest = xgb.DMatrix(X_test, label=y_test, device='cuda')
    # Train the model
    #X_train = cudf.DataFrame.from_pandas(X_train)
    #X_valid = cudf.DataFrame.from_pandas(X_valid)
    #y_train = cudf.DataFrame.from_pandas(y_train)
    #y_valid = cudf.DataFrame.from_pandas(y_valid)

    #dtrain = xgb.DMatrix(x_df,label=y_df['label'],enable_categorical=True)

    xgb1.fit(X_train,y_train)

    # Predict probabilities for ROC curve
    xgb1_pred_prob = xgb1.predict_proba(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid, xgb1_pred_prob[:, 1])

    # Use numpy.interp to interpolate TPR values
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    # Calculate AUC for this fold
    roc_auc = auc(fpr, tpr)
    roc_aucs_xgb1.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f'ROC fold {i} (area = {roc_auc:0.2f})')

    i += 1

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

# Compute mean ROC curve across folds
mean_tpr /= K
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label=f'Mean ROC (area = {mean_auc:0.2f})', lw=lw)

# Finalize plot
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Initial estimator ROC curve')
plt.legend(loc="lower right")

# Save the figure
fig.savefig(os.path.join(figure_path,'initial_ROC_rand.png'))
plt.show()  # Optionally display the plot
plt.close()

# Option to perform hyperparameter optimization. Otherwise loads pre-defined xgb_opt params
optimize = True
op = False
import psutil
import optuna

# Get logical CPU count (includes hyperthreads)
logical_cores = psutil.cpu_count(logical=True)

# Get physical core count
physical_cores = psutil.cpu_count(logical=False)

print("Logical cores (threads):", logical_cores)
print("Physical cores:", physical_cores)

X_train = x_df
y_train = y_df['label']
#X_train = cudf.DataFrame.from_pandas(X_train)

