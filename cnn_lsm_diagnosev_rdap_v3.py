# %%
import os

from optuna.logging import set_verbosity, WARNING
import dask.dataframe as dd
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
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from optuna.logging import set_verbosity, CRITICAL
import sys
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import TimeWarp, Drift, Reverse, Quantize, AddNoise
# %%
import random
from data_import import *
from dataclass import *
from prosess_data import *
from helper import train,evaluate
from oversample import * 
from ml import *
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
# %%
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
optuna.seed = seed

#os.environ["OMP_NUM_THREADS"] = "1"  # limit OpenMP threads
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

# %%
figure_path=os.path.join(os.getcwd(), 'figures','cnn_lsm_diag_removev2')
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

model_path=os.path.join(os.getcwd(), 'models', 'cnn_lsm_diag_removev2')
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_path=os.path.join(os.getcwd(), 'data', 'cnn_lsm_diag_removev2')
if not os.path.exists(data_path):
    os.makedirs(data_path)

# %%
# --- Step 1: Load the Data ---
df = pd.read_csv("data/cleaned_data.csv.gz")

import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Current memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

df,tranlate_dic=startimportdata(df,figure_path,model_path,data_path,"cnn_lstm",30000)
number_class = np.unique(df["label"])
    # %%


admission_counts = df['id'].value_counts().sort_index()
admission_stats = admission_counts.describe()
print(admission_stats)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(admission_counts, bins=range(1, admission_counts.max() + 2), edgecolor='black')
plt.title("Number of Admissions per Patient")
plt.xlabel("Admissions")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.show()
plt.savefig(os.path.join(figure_path,"diagnsosi.png"))
plt.close()


num_class = len(number_class)
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
sequence_length = 15
(X_test, y_test, lengths_test, group_test) = prepare_data(X_test_, y_test_, sequence_length)
print(X_test.shape)
test_data = ReadmissionDataset(X_test, y_test, lengths_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
num_features = X_test.shape[2]
#%%
(X_train_all,y_train_all,lengths_train,group_train) = prepare_data(X_train_all_, y_train_all_,sequence_length)

import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Current memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")



# %%
print(y_test)

# %%


# %%

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# %%

# %%
# %%
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
    from collections import Counter
    from sklearn.metrics import roc_auc_score
    import numpy as np

    torch.backends.cudnn.benchmark = True

    # ---- Hyperparameter suggestions ----
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    kernel_size = trial.suggest_int("kernel_size", 1, 3)
    cnn_channels = trial.suggest_int("cnn_channels", 16, 64)

    # ---- Model ----
    input_size = num_features
    

    # ---- Cross-validation ----
    min_class_count = min(Counter(y_train_all).values())
    n_splits = min(5, min_class_count)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs = []
    best_trial_auc = -np.inf
    best_model_file = None

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all, groups=group_train)):
        print(f"\n📦 Fold {fold + 1}/{n_splits}")
        print(" oversample  Sample:")
        
        try:
            model = DiagnosisModel(
        input_size=input_size,
        hidden_size=hidden_dim,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout=dropout,
        num_class=num_class
    ).to(device)
        except:
            print("class do not load")
        X_train, X_val = X_train_all[train_index], X_train_all[val_index]
        y_train, y_val = y_train_all[train_index], y_train_all[val_index]
        len_train, len_val = lengths_train[train_index], lengths_train[val_index]


        counts = Counter(y_train)
        for cls, count in counts.items():
            print(f"  Class {cls}: {count} samples")
        # ---- Oversample ----
        X_train, y_train, len_train = balance_sequences_hybrid_tsmote(X_train, y_train, len_train)

        new_counts = Counter(y_train)

        print(" oversample  Sample:")
        for cls, count in new_counts.items():
            print(f"  Class {cls}: {count} samples")
        # ---- Optimizer ----
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        
        # ---- DataLoaders ----
        train_data = ReadmissionDataset(X_train, y_train, len_train)
        val_data = ReadmissionDataset(X_val, y_val, len_val)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

        # ---- Loss ----
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        # early stopping
        early_stopper = EarlyStopping(patience=5)
        # ---- Train ----
        try:
            model, _ = train(model, train_loader, optimizer, criterion, device, epochs=5, num_class=num_class)
        except:
            print("train do not load")
        # ---- Evaluate ----
        try:
            model, y_true, y_score = evaluate(model, val_loader, device)
        except:
            print("evalalute do not start")
        
        if y_true is None or y_score is None:
            print("⚠️ Skipping fold due to empty validation.")
            continue

        if len(np.unique(y_true)) < num_class:
            print(f"⚠️ Missing classes in fold {fold + 1}: {set(range(num_class)) - set(np.unique(y_true))}")
            continue

        unique_in_fold = np.unique(y_true)
        print(f"✅ Classes in fold: {unique_in_fold}")
        print(f"y_true shape: {y_true.shape}, y_score shape: {y_score.shape}")
        print("Unique labels in y_true:", np.unique(y_true))
        print("Any NaN in y_score?", np.isnan(y_score).any())
        print("Sum of probs per sample (should be ~1):", np.mean(np.sum(y_score, axis=1)))
        if np.isnan(y_score).any():
            print("⚠️ NaNs detected in model outputs — skipping this fold.")
            continue

        if len(np.unique(y_true)) < 2:
            print("⚠️ Only one class present in this fold — skipping.")
            continue

        try:
            auc = roc_auc_score(
                y_true,
                y_score,
                multi_class='ovr',
                labels=np.arange(num_class)
            )
            if np.isnan(auc).any():
                continue
            aucs.append(auc)
        except ValueError as e:
            print("⚠️ roc_auc_score failed:", e)
            continue

        aucs.append(auc)
        print(f"✅ Fold {fold + 1} AUC: {auc:.4f}")


    # ---- Save checkpoint if this is the best fold-model so far for this trial ----
    if auc > best_trial_auc:
        best_trial_auc = auc
        best_model_file = os.path.join(model_path, f"optuna_trial_{trial.number}_fold_{fold+1}_auc_{auc:.4f}.pt"
            )
        try:
            torch.save(model.state_dict(), best_model_file)
            # attach path and best AUC to the trial so it is persisted in study storage
            trial.set_user_attr("best_model", best_model_file)
            trial.set_user_attr("best_model_auc", float(best_trial_auc))
            print(f"💾 Saved best model for trial {trial.number}, fold {fold+1} -> {best_model_file}")
        except Exception as e:
            print("⚠️ Failed to save model checkpoint:", e)


    # ---- Post-CV Handling ----
    if len(aucs) == 0:
        print("❌ All folds failed or skipped. Pruning trial.")
        raise optuna.exceptions.TrialPruned()




    mean_auc = float(np.mean(aucs))
    print(f"✅ Mean AUC: {mean_auc:.4f}")
    return mean_auc


# %%
import optuna, joblib, os, json
from optuna.logging import set_verbosity, CRITICAL
from multiprocessing import Process

os.makedirs("model", exist_ok=True)
set_verbosity(CRITICAL)
optuna.logging.disable_default_handler()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    load_if_exists=True
)

from joblib import parallel_backend

# SAFE: avoid shell-dependent progress bar

def is_jupyter_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (NameError, ImportError):
        return False


if is_jupyter_notebook():
    n_job=1
else:
    n_job=20

study.optimize(
            objective,
            n_trials=20,
            n_jobs=n_job,
            show_progress_bar=True  # disable tqdm if not a real terminal
)


joblib.dump(study, os.path.join(model_path,"optuna_studyv4.pkl"))
with open(os.path.join(model_path,"best_paramsv4.json"), "w") as f:
    json.dump(study.best_trial.params, f, indent=2)

print("Best trial:", study.best_trial)


# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import optuna

if is_jupyter_notebook():
    import optuna.visualization as plotf  # interactive
else:
    import optuna.visualization.matplotlib as plotf  # static

def save_and_show(fig_or_ax, filename, title=None, show_plot=True):
    """
    Saves and optionally shows an Optuna plot (works with both Plotly and Matplotlib).
    """
    import plotly.graph_objects as go

    # --- Handle Plotly figure ---
    if isinstance(fig_or_ax, go.Figure):
        if title:
            fig_or_ax.update_layout(title=title)
        fig_or_ax.write_image(filename)  # requires kaleido installed
        if show_plot:
            fig_or_ax.show()
        return

    # --- Handle Matplotlib Axes or Figure ---
    if isinstance(fig_or_ax, plt.Figure):
        fig = fig_or_ax
    elif hasattr(fig_or_ax, "figure"):  # Axes or array of Axes
        fig = fig_or_ax.figure
    elif isinstance(fig_or_ax, (list, np.ndarray)):  # list of Axes
        fig = fig_or_ax[0].figure
    else:
        raise TypeError(f"Unsupported plot object type: {type(fig_or_ax)}")

    if title:
        try:
            fig.suptitle(title)
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)



try:

    show_plots = is_jupyter_notebook()
    # Optimization History
    ax1 = plotf.plot_optimization_history(study)
    save_and_show(ax1, os.path.join(figure_path, "cnn_optimization_history.png"), "Optimization History", show_plots)

    # Parameter Importances
    ax2 = plotf.plot_param_importances(study)
    save_and_show(ax2, os.path.join(figure_path, "cnn_param_importances.png"), "Parameter Importances", show_plots)

    # Parameter Slices
    ax3 = plotf.plot_slice(study)
    save_and_show(ax3, os.path.join(figure_path, "cnn_param_slices.png"), "Parameter Slices", show_plots)

    # Parallel Coordinates
    ax4 = plotf.plot_parallel_coordinate(study)
    save_and_show(ax4, os.path.join(figure_path, "cnn_parallel_coordinates.png"), "Parallel Coordinates", show_plots)

    print("✅ All Optuna plots generated, saved, and displayed (if Jupyter).")

except Exception as e:
    print("❌ Error generating Optuna plots.")
    print(e)

# %%
def build_model(params):
    model = DiagnosisModel(
        input_size=num_features,
        cnn_channels=params["cnn_channels"],
        kernel_size=params["kernel_size"],
        hidden_size=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_class=num_class
    )
    return model

# %%
best_params = study.best_trial.params
best_trial = study.best_trial
print("Best parameters:", best_params)
print("Best trial number:", best_trial.number)
print("Best trial value (objective):", best_trial.value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model_path = best_trial.user_attrs.get("best_model", None)
if best_model_path is None or not os.path.exists(best_model_path):
    print("⚠️ No best model file saved for best trial. You may inspect trial.user_attrs for candidates.")
    model = build_model(best_params).to(device)
    torch.save(model.state_dict(), os.path.join(model_path, "best_model_untrained_v4.pt"))
else:
    # rebuild model with best params and load weights
    model = build_model(best_trial.params).to(device)
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    print(f"✅ Loaded best trial model from {best_model_path}")
    # optionally save as canonical file
    torch.save(model.state_dict(), os.path.join(model_path, "best_model_from_study.pt"))



# %%
lr = best_params["lr"]

optimizer_name = best_params["optimizer"]
# criterion = nn.CrossEntropyLoss()

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optimizer_name == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
elif optimizer_name == "SGD":
    optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))

thresholds = np.linspace(0, 1, 101)

fold_thresholds = []
fold_f1s = []
all_folds_labels = []
all_folds_scores = []
aucs = []
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

losses = []
for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all, groups=group_train)):

    X_train, X_val = X_train_all[train_index], X_train_all[val_index]
    y_train, y_val = y_train_all[train_index], y_train_all[val_index]
    len_train = lengths_train[train_index]
    len_val = lengths_train[val_index]
    new_counts = Counter(y_train)
    print("  Sample:")
    for cls, count in new_counts.items():
        print(f"  Class {cls}: {count} samples")
    # ---- Oversample ----
    X_train, y_train, len_train = balance_sequences_hybrid_tsmote(X_train, y_train, len_train)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))

    # 📊 Class distribution AFTER oversampling
    new_counts = Counter(y_train)

    print(" over sample Sample:")
    for cls, count in new_counts.items():
        print(f"  Class {cls}: {count} samples")


    # 📊 Class distribution AFTER oversampling
    new_counts = Counter(y_train)

    print("  Sample:")
    for cls, count in new_counts.items():
        print(f"  Class {cls}: {count} samples")

        # --------------------------
        # Compute and apply class weights
        # --------------------------
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_), y=y_train_)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_), y=y_train_)
    #class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_data = ReadmissionDataset(X_train, y_train, len_train)
    val_data = ReadmissionDataset(X_val, y_val, len_val)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

    (model, loss) = train(model, train_loader, optimizer, criterion, device, epochs=5,num_class=num_class)

    # Evaluate
    (model, y_true, y_score) = evaluate(model, val_loader, device)
    all_folds_labels.append(y_true)
    all_folds_scores.append(y_score)
    if len(np.unique(y_true)) < num_class:
        print(f"⚠️ Missing classes in fold {fold + 1}: {set(range(num_class)) - set(np.unique(y_true))}")
        continue
    if len(np.unique(y_true)) < num_class:
        print(f"⚠️ Missing classes in fold {fold + 1}: {set(range(num_class)) - set(np.unique(y_true))}")
        continue

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        aucs.append(auc)
        losses.append(loss)
        y_pred = np.argmax(y_score, axis=1)
        f1_scores = f1_score(y_true, y_pred, average='weighted')

        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores
        fold_f1s.append(best_f1)
        fold_thresholds.append(best_thresh)
        print(f"Best F1 threshold = {best_thresh:.3f}, F1 = {best_f1:.4f}")
    except:
        continue
print(f"\nMean best threshold = {np.mean(fold_thresholds):.3f}")
print(f"Mean F1 across folds = {np.mean(fold_f1s):.4f}")
mean_auc = float(np.mean(aucs))
print(f"Mean 5-fold ROC-AUC: {mean_auc:.4f}")

# %%
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Combine all folds
y_true_all = np.concatenate(all_folds_labels)
y_score_all = np.concatenate(all_folds_scores)
y_pred_all = np.argmax(y_score_all, axis=1)

# ROC Curve
roc_auc = roc_auc_score(
    y_true_all,
    y_score_all,
    multi_class='ovr',         # or 'ovo'
    average='macro'            # or 'weighted'
)
print(f"Multiclass ROC AUC (OvR, macro): {roc_auc:.4f}")

# %%
# Binarize the true labels
y_true_bin = label_binarize(y_true_all, classes=np.arange(num_class))

# Plot ROC curve per class
for i in range(num_class):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_all[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curves")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figure_path,"roc_curve.png"))
plt.close()


from sklearn.metrics import precision_recall_curve, average_precision_score

# Binarize the true labels
y_true_bin = label_binarize(y_true_all, classes=np.arange(num_class))  # shape: (n_samples, n_classes)

# Plot precision-recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(num_class):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_all[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_score_all[:, i])
    plt.plot(recall, precision, label=f"Class {i} (AP = {ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multi-class Precision-Recall Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figure_path,"precsion.png"))
plt.close()

# Plot training loss per fold
plt.figure(figsize=(8, 5))
for i, loss_history in enumerate(losses):
    plt.plot(loss_history, label=f"Fold {i + 1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figure_path,"loss.png"))
plt.close()

# Optional: class names (if using LabelEncoder before)
# target_names = le.classes_
# OR: use default numerical labels
target_names = [f"Class {i}" for i in np.unique(y_true_all)]

# Create confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)

# Optional: normalize per true class
normalize = True
if normalize:
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

sorted_labels = [tranlate_dic[i] for i in sorted(tranlate_dic)]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
            xticklabels=sorted_labels, yticklabels=sorted_labels)
plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(figure_path,"confusion.png"))
plt.close()

# %%
# Classification Report
print("\nClassification Report: training\n")
train_text,_=pretty_classification_report(y_true_all, y_pred_all, tranlate_dic)
print(classification_report(y_true_all, y_pred_all))
print(train_text)
with open(os.path.join(data_path,"training.txt"),'w') as f:
    f.write(train_text)

# %%
def best_f1_threshold(y_true_list, y_score_list, plot=False, average="weighted"):
    """
        Computes the best F1 score using argmax for multi-class predictions.

        Parameters:
            y_true_list: List of arrays with true labels
            y_score_list: List of arrays with predicted probabilities (shape: [n_samples, n_classes])
            average: Type of averaging for F1 ('micro', 'macro', 'weighted')
            plot: Ignored (kept for compatibility)

        Returns:
            best_f1 (float): Best F1-score using argmax predictions
        """
    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    y_pred = np.argmax(y_score, axis=1)

    best_f1 = f1_score(y_true, y_pred, average=average)

    print(f"✅ Best F1 score (multi-class, {average}): {best_f1:.4f}")
    return None, best_f1  # No threshold returned — just F1


(t, f1) = best_f1_threshold(all_folds_labels, all_folds_scores, True)
print(f"global F1 threshold =  (F1 = {f1:.4f})")

# %%
# After evaluating validation data
best_thresh, best_f1 = best_f1_threshold(all_folds_labels, all_folds_scores, plot=False)
print(best_thresh, best_f1)
# Save the threshold for later
with open(os.path.join(model_path,"best_threshold.json"), "w") as f:
    json.dump({"threshold": best_thresh}, f)

# %%
(X_test, y_test, lengths_test, group_test) = prepare_data(X_test_, y_test_, sequence_length)
print(X_test.shape)
test_data = ReadmissionDataset(X_test, y_test, lengths_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

(model, y_true, y_score) = evaluate(model, test_loader, device)

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ✅ Step 1: Predict class with highest probability
y_pred = np.argmax(y_score, axis=1)  # shape: (n_samples,)

# ✅ Step 2: F1 score
print("Test F1 (weighted):", f1_score(y_true, y_pred, average='weighted'))

# ✅ Step 3: Classification report
print("\nClassification report:")
formatted_text,report_with_name=pretty_classification_report(y_true, y_pred, tranlate_dic)

with open(os.path.join(data_path,"test.txt"),'w') as f:
    f.write(formatted_text)
print(formatted_text)


# %%
training_data = confusion_matrix(y_true,y_pred,normalize='true')

labels_ordered = [tranlate_dic[str(i)] for i in range(len(tranlate_dic))]
diagonal = np.diag(training_data)
plt.figure(figsize=(12, 8))
plt.bar(range(len(tranlate_dic)), diagonal)
plt.xlabel('Class')
plt.ylabel('Correct predictions')
plt.xticks(ticks=range(len(labels_ordered)), labels=labels_ordered, rotation=45)
plt.tight_layout()
plt.close()

# %%
# ✅ Step 5: Confusion matrix (normalized by true class)
cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
print("\nConfusion matrix (normalized by true class):")
print(cm_norm)

# Step 2: Display with labels from true_dic
sorted_labels = [tranlate_dic[i] for i in sorted(tranlate_dic)]


# ✅ Step 6: Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted_labels)
disp.plot(cmap="Blues", values_format="0.2f")
plt.title("Confusion Matrix (multi-class)")
plt.xlabel("Predicted Label evalutaion")
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(figure_path,"confusion_matrix_eval.png"))
plt.close()

# %%
cm_norm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (normalized by true class):")
print(cm_norm)

# Step 2: Display with labels from true_dic
sorted_labels = [tranlate_dic[i] for i in sorted(tranlate_dic)]



plt.figure(figsize=(12, 8)) 
# ✅ Step 6: Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sorted_labels)
disp.plot(cmap="Blues", values_format="0.2f")
plt.title("Confusion Matrix (multi-class)")
plt.xlabel("Predicted Label evalutaion")
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(figure_path,"confusion_matrix_eval_numbers.png"))
plt.close()

# %%



