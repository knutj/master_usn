import os

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

figure_path=os.path.join(os.getcwd(), 'figures','cnn_lsm_ret')
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

model_path=os.path.join(os.getcwd(), 'models', 'cnn_lsm_ret')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_path=os.path.join(os.getcwd(), 'data', 'cnn_lsm_ret')
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
df = pd.read_csv("data/cleaned_data.csv.gz")
#df = pd.read_csv("data/cleaned_data.csv.gz",nrows=5000)
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
#%%
""" class LSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout=0.3):
        super(LSTMStack, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(packed)
        #return hn[-1]  # (batch_size, hidden_size)  <-- 2D
        # Use the last hidden state from the top layer
        lstm_out = hn[-1]  # shape: (batch_size, hidden_size)
         # Apply LayerNorm and dropout
        lstm_out = self.layernorm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        return lstm_out  # (B, hidden_size)
        
class CNNExtractor(nn.Module):
    def __init__(self, input_size, cnn_channels, kernel_size, dropout):
        super(CNNExtractor, self).__init__()
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Dropout(dropout)
        )
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x expected as (B, S, F); Conv1d wants (B, F, S)
        x = x.permute(0, 2, 1)  # (B, F, S)
        x = self.cnn(x)  # (B, C, S)
        # Collapse the time dimension so we return 2D features
        # Global average and max pooling
        avg_pool = self.global_avg(x)  # (B, C, 1)
        max_pool = self.global_max(x)  # (B, C, 1)

        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2C, 1)
        return combined.squeeze(-1)  # (B, 2 * cnn_channels)


class DiagnosisModel(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_channels, kernel_size, num_layers, dropout, num_class=2):
        super(DiagnosisModel, self).__init__()
        self.lstm = LSTMStack(input_size, hidden_size, num_layers,dropout)
        self.cnn = CNNExtractor(input_size, cnn_channels, kernel_size, dropout)

        in_features = hidden_size + 2 * cnn_channels  # match what we actually concatenate
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 64),
            nn.Dropout(dropout / 3),
            nn.ReLU(),
            nn.Linear(64, num_class),  # 2 classe
        )

    def forward(self, lstm_input, lengths, cnn_input):
        lstm_features = self.lstm(lstm_input, lengths)  # (B, hidden_size)
        cnn_features = self.cnn(cnn_input)  # (B, cnn_channels)

        # Both branches are 2D now, so concatenation works
        combined = torch.cat([lstm_features, cnn_features], dim=1)  # (B, hidden_size + cnn_channels)
        return self.mlp(combined)  # (batch_size, 2)
 """

#%%

#%%

#%%

# Try importing DTW from tslearn, fallback to Euclidean distance if not installed

#%%

#%%
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import contextlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc

# ------------------------------
# Optuna Objective Function
# ------------------------------



def objective(trial):
    import torch
    import torch.nn as nn


    torch.backends.cudnn.benchmark = True
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)



    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    kernel_size = trial.suggest_int("kernel_size", 1, 3)
    cnn_channels = trial.suggest_int("cnn_channels", 16, 64)

    input_size= num_features
    model = DiagnosisModel(input_size=input_size, hidden_size=hidden_dim,cnn_channels=cnn_channels,kernel_size=kernel_size ,num_layers=num_layers,dropout=dropout,num_class=2)

    #print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss()

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best_val_auc = 0
    aucs = []
    patience = 3
    for fold,(train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all,groups=group_train)):

        X_train, X_val = X_train_all[train_index], X_train_all[val_index]
        y_train, y_val = y_train_all[train_index], y_train_all[val_index]
        len_train = lengths_train[train_index]
        len_val = lengths_train[val_index]

        (X_train, y_train, len_train) = oversample_sequences_smotets(X_train, y_train, len_train)


        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))

      
        # 📊 Class distribution AFTER oversampling
        new_counts = Counter(y_train)

        print("  Sample:")
        for cls, count in new_counts.items():
            print(f"  Class {cls}: {count} samples")

                # --------------------------
        # Compute and apply class weights
        # --------------------------
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)

        
        try:
            train_data = ReadmissionDataset(X_train, y_train,len_train)
            val_data = ReadmissionDataset(X_val, y_val,len_val)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
        except:
            print(len(X_train))
            print(len(y_train))
            print(len(len_train))
            continue
        try:
            (_,loss) = train_re(model, train_loader, optimizer, criterion, device,epochs=5)
        except:
            print(len(X_train))
            print(len(y_train))
            print(len(len_train))
            continue
        try:
        # Evaluate
            (_,y_true,y_score)= evaluate_re(model, val_loader, device)
        except:
            print(len(y_val))
            print(len(X_val))
            print(len(len_val))
            continue

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)                        # binary ROC AUC
        aucs.append(auc)

        # --------------------------
        # Final AUC Score
        # --------------------------
        if len(aucs) == 0:
            return 0.0  # all folds failed

        if auc > best_val_auc:
            best_val_auc = auc
            epochs_no_improve = 0
            torch.save(model.state_dict(),os.path.join(model_path,"best.pt"))



        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at  {auc}")
            break

        # Cleanup
        #del train_loader, val_loader
        #torch.cuda.empty_cache()
        #gc.collect()

    mean_auc = float(np.mean(aucs))
    return mean_auc

loadmodel=False
if loadmodel==False:
    #%%
    # Ensure output folder exists
    #os.makedirs("model", exist_ok=True)

    # Silence Optuna's logs, but keep tqdm bar
    set_verbosity(CRITICAL)
    optuna.logging.disable_default_handler()

    print("🔍 Starting Optuna optimization with progress bar...\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=10,
        n_jobs=16,
        show_progress_bar=True  # 👈 tqdm-style global progress bar
    )

    print("\n✅ Optimization complete!")

    # Save and report best result
    joblib.dump(study, os.path.join(model_path,"optuna_studyv4.pkl"))
    with open(os.path.join(model_path,"best_paramsv4.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
else:
    study = joblib.load(os.path.join(model_path,"optuna_studyv4.pkl"))
    # Access best trial or results
    print("Best trial:")
    print(study.best_trial)
    print("Best value (objective):", study.best_value)
    print("Best parameters:", study.best_trial.params)
    with open(os.path.join(model_path,"best_paramsv4.json"), "r") as f:
        best_params = json.load(f)

    
# Access best trial or results
print("Best trial:")
print(study.best_trial)
print("Best value (objective):", study.best_value)
print("Best parameters:", study.best_trial.params)

print("\n🏆 Best trial:")
print(study.best_trial)

#from IPython.display import display, HTML
#display(HTML("<style>.output_scroll {height: 100px; overflow-y: scroll;}</style>"))
#%%
import optuna.visualization as vis

import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis_matplotlib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    def save_and_show(ax_or_axes, filename, title=None):
        # Support for single or multiple axes
        ax = ax_or_axes[0] if isinstance(ax_or_axes, np.ndarray) else ax_or_axes
        if title:
            ax.set_title(title)
        fig = ax.figure
        fig.savefig(filename)
        plt.show()
        plt.close(fig)

    # Optimization History
    ax1 = vis_matplotlib.plot_optimization_history(study)
    save_and_show(ax1, os.path.join(figure_path,"cnn_optimization_history.png"), "Optimization History")

    # Parameter Importances
    ax2 = vis_matplotlib.plot_param_importances(study)
    save_and_show(ax2, os.path.join(figure_path,"cnn_param_importances.png"), "Parameter Importances")

    # Parameter Slices (may return multiple axes)
    ax3 = vis_matplotlib.plot_slice(study)
    save_and_show(ax3, os.path.join(figure_path,"cnn_param_slices.png"), "Parameter Slices")

    # Parallel Coordinates (may return multiple axes)
    ax4 = vis_matplotlib.plot_parallel_coordinate(study)
    save_and_show(ax4, os.path.join(figure_path,"cnn_parallel_coordinates.png"), "Parallel Coordinates")

    print("✅ All plots displayed and saved successfully.")

except Exception as e:
    print("❌ Error generating Optuna plots.")
    print(e)



def build_model(params):
    model = DiagnosisModel(
            input_size=num_features,
            cnn_channels=params["cnn_channels"],
            kernel_size=params["kernel_size"],
            hidden_size=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
    return model
#%%
best_params = study.best_trial.params
print("Best parameters:", best_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(best_params).to(device)
torch.save(model.state_dict(), os.path.join(model_path,"best_model_v4.pt"))

#%%
lr=best_params["lr"]

optimizer_name = best_params["optimizer"]
#criterion = nn.CrossEntropyLoss()

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optimizer_name == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
elif optimizer_name == "SGD":
    optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))



thresholds = np.linspace(0, 1, 101)


fold_thresholds = []
fold_f1s = []
all_folds_labels=[]
all_folds_scores=[]
aucs = []
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


losses =[]
for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all, groups=group_train)):

    X_train, X_val = X_train_all[train_index], X_train_all[val_index]
    y_train, y_val = y_train_all[train_index], y_train_all[val_index]
    len_train = lengths_train[train_index]
    len_val = lengths_train[val_index]

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))



    # 📊 Class distribution AFTER oversampling
    new_counts = Counter(y_train)

    print("  Sample:")
    for cls, count in new_counts.items():
        print(f"  Class {cls}: {count} samples")

    #indices = np.arange(len(X_train))
    #np.random.shuffle(indices)
    #X_train_ = X_train[indices]
    #y_train_ = y_train[indices]
    #len_train_ = len_train[indices]

    # 📊 Class distribution AFTER oversampling
    new_counts = Counter(y_train)
    (X_train, y_train, len_train) = oversample_sequences_smotets(X_train, y_train, len_train)
    print("  Sample:")
    for cls, count in new_counts.items():
        print(f"  Class {cls}: {count} samples")


                # --------------------------
        # Compute and apply class weights
        # --------------------------
    #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_), y=y_train_)
    #class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)

    train_data = ReadmissionDataset(X_train, y_train,len_train)
    val_data = ReadmissionDataset(X_val, y_val,len_val)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    (_,loss) = train_re(model, train_loader, optimizer, criterion, device,epochs=5)

    # Evaluate
    (_,y_true,y_score)= evaluate_re(model, val_loader, device)
    all_folds_labels.append(y_true)
    all_folds_scores.append(y_score)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)                        # binary ROC AUC
    aucs.append(auc)
    losses.append(loss)

    f1_scores = [f1_score(y_true, y_score >= t) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    fold_f1s.append(best_f1)
    fold_thresholds.append(best_thresh)
    print(f"Best F1 threshold = {best_thresh:.3f}, F1 = {best_f1:.4f}")

    



print(f"\nMean best threshold = {np.mean(fold_thresholds):.3f}")
print(f"Mean F1 across folds = {np.mean(fold_f1s):.4f}")
mean_auc = float(np.mean(aucs))
print(f"Mean 5-fold ROC-AUC: {mean_auc:.4f}")
#%%
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Combine all folds
y_true_all = np.concatenate(all_folds_labels)
y_score_all = np.concatenate(all_folds_scores)
y_pred_all = (y_score_all > 0.5).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(figure_path,"cnn_roc_auc.png"))
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)

plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(figure_path,"cnn_precsion.png"))            
plt.show()

# Plot training loss per fold
plt.figure(figsize=(8, 5))
for i, loss_history in enumerate(losses):
    plt.plot(loss_history, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figure_path,"cnn_loss.png"))             
plt.show()


# Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(figure_path,"cnn_cm.png"))             
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true_all, y_pred_all))

text=classification_report(y_true_all, y_pred_all) 
with open(os.path.join(data_path,"classification_report_train_cnn.txt"), "w") as f:
    f.write(text)


#%%
def best_f1_threshold(y_true_list, y_score_list, plot=True):
    """
    Find the threshold that maximizes F1-score.

    Parameters:
        y_true  : array-like, true binary labels
        y_score : array-like, predicted probabilities for positive class
        plot    : bool, if True plots F1 vs threshold

    Returns:
        best_thresh (float): threshold that gives best F1
        best_f1 (float): best F1-score value
    """
    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_true, y_score >= t) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, f1_scores, marker='.')
        plt.axvline(best_thresh, color='r', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
        plt.xlabel("Threshold")
        plt.ylabel("F1-score")
        plt.title("F1-score vs Decision Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/cnn_best_f1.png") 
        plt.show()

    return best_thresh, best_f1


(t,f1)= best_f1_threshold(all_folds_labels, all_folds_scores,True)
print(f"Optimal global F1 threshold = {t:.3f} (F1 = {f1:.4f})")

#%%
# After evaluating validation data
best_thresh, best_f1 = best_f1_threshold(all_folds_labels, all_folds_scores, plot=False)
print(best_thresh,best_f1)
# Save the threshold for later
with open("model/best_threshold.json", "w") as f:
    json.dump({"threshold": best_thresh}, f)

#%%
(X_test, y_test, lengths_test, group_test) = prepare_data(X_test_, y_test_, sequence_length)
print(X_test.shape)
test_data = ReadmissionDataset(X_test, y_test, lengths_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

(model,y_true,y_score)=evaluate_re(model, test_loader, device)
print("Test ROC AUC:", roc_auc_score(y_true, y_score))

print("Final shapes:")
print("y_true:", len(y_true))
print("y_score:", len(y_score))
print("Test ROC AUC:", roc_auc_score(y_true, y_score))
#%%
# after you've filled all_labels (list of tensors) and all_pos_scores (list of tensors)


# turn probabilities into predicted labels (use a clear threshold, not round)
y_pred = (y_score >= best_thresh).astype(np.int64)

print("Test F1:", f1_score(y_true, y_pred))
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Pick a threshold (0.5, or your tuned one)
thr = 0.5
y_pred = (y_score >= thr).astype(np.int64)

# Confusion matrix (counts)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix (counts):\n", cm)

# Normalized per true class (i.e., recall per class)
cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
print("Confusion matrix (normalized by true class):\n", cm_norm)

# Quick report (precision/recall/F1 per class)
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

report = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(data_path,"classification_report_test_cnn.txt"), "w") as f:
    f.write(report)



# Optional: plot (if you’re in a notebook)
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(values_format='d')
plt.title(f'Confusion Matrix (threshold={thr})')
plt.savefig(os.path.join(figure_path,"cnn_cm_test.png")) 
plt.show()
#%%
print(best_thresh)
#
#
# flatten lists of tensors -> 1D numpy arrays
from sklearn.metrics import f1_score, classification_report

y_pred = (y_score >=   best_thresh).astype(np.int64)

# optional: nice class names
target_names = ["Negative (0)", "Positive (1)"]

print("Classification report @ threshold =", best_thresh)
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
#%%
y_pred = (y_score >= best_thresh).astype(np.int64)
target_names = ["Negative (0)", "Positive (1)"]

print(f"📊 Classification report @ threshold = {best_thresh:.3f}")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# 🧮 Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:")
print(cm)

# Optional: make it pretty
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix @ Threshold = {best_thresh:.2f}")
plt.savefig(os.path.join(figure_path,"cnn_cm_evalu_best.png"))             
plt.show()
#%%
from sklearn.metrics import precision_recall_curve
prec, rec, thr_list = precision_recall_curve(y_true, y_score)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_thr = 0.5 if best_idx == len(thr_list) else thr_list[best_idx]

y_pred_best = (y_score >= best_thr).astype(np.int64)

print(f"\nClassification report @ best F1 threshold = {best_thr:.3f}")
print(classification_report(y_true, y_pred_best, target_names=target_names, digits=4))
#%%
from torchinfo import summary
summary(model)
